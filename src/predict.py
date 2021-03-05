#!/usr/bin/env python3
"""Generate NBA predictions using an Elo regressor algorithm (elora)"""
import json
import os
import requests

import pandas as pd

from .model import EloraNBA
from .data import games, upcoming_games


def rank(spread_model, total_model, datetime, slack_report=False):
    """Rank NBA teams at a certain point in time. The rankings are based on all
    available data preceding that moment in time.
    """
    df = pd.DataFrame(
        spread_model.ratings(datetime),
        columns=['team', 'point spread']
    ).sort_values(by='point spread', ascending=False).reset_index(drop=True)

    df['point spread'] = df[['point spread']].applymap('{:4.1f}'.format)
    spread_col = '  ' + df['team'] + '  ' + df['point spread']

    df = pd.DataFrame(
        total_model.ratings(datetime),
        columns=['team', 'point total']
    ).sort_values(by='point total', ascending=False).reset_index(drop=True)

    df['point total'] = df[['point total']].applymap('{:4.1f}'.format)
    total_col = '  ' + df['team'] + '  ' + df['point total']

    rankings = pd.concat([spread_col, total_col], axis=1).round(decimals=1)
    rankings.columns = [6*' ' + 'spread', 7*' ' + 'total']
    rankings.index += 1

    rankings.index.name = 'rank'
    timestamp = datetime.floor('Min')

    report = '\n'.join([
        f'*RANKING  |  @{timestamp}*\n',
        '```',
        rankings.to_string(header=True, justify='left'),
        '```',
        '*against average team on neutral field'])

    if slack_report is True:
        requests.post(
            os.getenv('SLACK_WEBHOOK'),
            data=json.dumps({'text': report}),
            headers={'Content-Type': 'application/json'})

    print(report)


def forecast(spread_model, total_model, games, slack_report=False):
    """Forecast outcomes for the list of games specified.
    """
    games = upcoming_games()

    report = pd.DataFrame({
        "date": games.datetime,
        "fav": games.team_away,
        "und": "@" + games.team_home,
        "odds": spread_model.sf(
            0, games.datetime, games.team_away, games.team_home),
        "spread": spread_model.mean(
            games.datetime, games.team_away, games.team_home),
        "total": total_model.mean(
            games.datetime, games.team_away, games.team_home)
    }).round({'spread': 1, 'total': 1})

    report[["fav", "und"]] = report[["und", "fav"]].where(
        report["odds"] < 0.5, report[["fav", "und"]].values)

    report["spread"] = report["spread"].where(
        report["odds"] < 0.5, -report["spread"].values)

    report["one minus odds"] = 1 - report["odds"]
    report["odds"] = report[["odds", "one minus odds"]].max(axis=1)
    report["odds"] = (100*report.odds).astype(int).astype(str) + '%'

    report.drop(columns="one minus odds", inplace=True)
    report.sort_values('spread', inplace=True)

    timestamp = pd.Timestamp('now')
    report = '\n'.join([
        f'*FORECAST  |  @{timestamp}*\n',
        '```',
        report.to_string(index=False),
        '```'])

    if slack_report is True:
        requests.post(
            os.getenv('SLACK_WEBHOOK'),
            data=json.dumps({'text': report}),
            headers={'Content-Type': 'application/json'})

    print(report)


def bet(spread_model, total_model, games):
    """Quantify the expected profit (or losses) incurred by betting
    on either side of each game
    """
    # TODO implement this


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NBA prediction model')

    subparsers = parser.add_subparsers(
        dest='subparser',
        help='base functionality',
        required=True)

    rank_p = subparsers.add_parser('rank')

    rank_p.add_argument(
        '--time', type=pd.Timestamp, default=pd.Timestamp.now(),
        help='evaluation index time to make predictions; default is now')

    rank_p.add_argument(
        '--slack-report', help='send predictions to Slack webhook address',
        action='store_true')

    forecast_p = subparsers.add_parser('forecast')

    forecast_p.add_argument(
        '--slack-report', help='send predictions to Slack webhook address',
        action='store_true')

    args = parser.parse_args()
    kwargs = vars(args)
    subparser = kwargs.pop('subparser')

    spread_model = EloraNBA.from_cache(games, 'spread')
    total_model = EloraNBA.from_cache(games, 'total')

    if subparser == 'rank':
        rank(spread_model, total_model, args.time,
             slack_report=args.slack_report)
    elif subparser == 'forecast':
        games = upcoming_games(days=1)
        forecast(spread_model, total_model, games,
                 slack_report=args.slack_report)
    #elif subparser == 'bet':
    #    # TODO implement this
    else:
        raise(ValueError, 'No such argument {}'.format(subparser))
