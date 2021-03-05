#!/usr/bin/env python3
""" Source latest NBA game data and store in a sqlite database """
from datetime import datetime

import pandas as pd
from sportquery.nba import engine


def get_upcoming_games(days=1):
    """Return a pandas dataframe of upcoming games
    """
    df = pd.read_sql("""
        SELECT season, datetime, team as team_home, opponent as team_away
        FROM schedule WHERE is_home = 1 ORDER BY datetime""", engine)

    df['datetime'] = pd.to_datetime(df.datetime)

    time_diff = (df.datetime - datetime.now())

    return df[(0 <= time_diff.dt.days) & (time_diff.dt.days <= days)]


def get_training_data():
    """Preprocess historical NBA boxscore data into a form that is suitable for
    modelling
    Returns:
        Pandas datafrme of NBA game data
    """
    pairs = pd.read_sql("""
        SELECT DISTINCT
            game_id, season, datetime, team as team_home, opponent as team_away
        FROM schedule
            WHERE (datetime IS NOT NULL) AND (is_home = 1)""", engine)

    boxscore = pd.read_sql("""
        SELECT DISTINCT
            game_id,
            team,
            pts as mf_pts,
            ast as mf_ast,
            tov as mf_tov,
            trb as mf_trb
        FROM boxscore WHERE player = 'All'""", engine)

    model_features = pairs.merge(
        boxscore.rename({'team': 'team_home'}, axis=1),
        on=['game_id', 'team_home'], how='inner'
    ).merge(
        boxscore.rename({'team': 'team_away'}, axis=1),
        on=['game_id', 'team_away'], how='inner',
        suffixes=('_home', '_away'))

    model_features['datetime'] = pd.to_datetime(model_features.datetime)

    return model_features


upcoming_games = get_upcoming_games()
games = get_training_data()

if __name__ == '__main__':
    print(games)
