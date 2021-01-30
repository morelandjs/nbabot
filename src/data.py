#!/usr/bin/env python3
""" Source latest NBA game data and store in a sqlite database """
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import prefect
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
from prefect.run_configs import LocalRun
from sportsipy.nba.boxscore import Boxscore
from sportsipy.nba.schedule import Schedule
from sportsipy.nba.nba_utils import _retrieve_all_teams
import sqlalchemy

from . import cachedir

db_path = cachedir / 'nba.db'
db_path.parent.mkdir(parents=True, exist_ok=True)


@task
def initialize_database():
    """Initialize a sqlite database, and create a table with the correct
    schema if it doesn't already exist. Returns the database connection engine.
    """
    engine = sqlalchemy.create_engine(f'sqlite:///{db_path.expanduser()}')

    engine.execute(
        """CREATE TABLE IF NOT EXISTS games (
            date                                   TEXT,
            location                               TEXT,
            losing_abbr                            TEXT,
            losing_name                            TEXT,
            winning_abbr                           TEXT,
            winning_name                           TEXT,
            winner                                 TEXT,
            pace                                   REAL,
            away_assist_percentage                 REAL,
            away_assists                           BIGINT,
            away_block_percentage                  REAL,
            away_blocks                            BIGINT,
            away_defensive_rating                  REAL,
            away_defensive_rebound_percentage      REAL,
            away_defensive_rebounds                BIGINT,
            away_effective_field_goal_percentage   REAL,
            away_field_goal_attempts               BIGINT,
            away_field_goal_percentage             REAL,
            away_field_goals                       BIGINT,
            away_free_throw_attempt_rate           REAL,
            away_free_throw_attempts               BIGINT,
            away_free_throw_percentage             REAL,
            away_free_throws                       BIGINT,
            away_losses                            BIGINT,
            away_minutes_played                    BIGINT,
            away_offensive_rating                  REAL,
            away_offensive_rebound_percentage      REAL,
            away_offensive_rebounds                BIGINT,
            away_personal_fouls                    BIGINT,
            away_points_q1                         BIGINT,
            away_points_q2                         BIGINT,
            away_points_q3                         BIGINT,
            away_points_q4                         BIGINT,
            away_points                            BIGINT,
            away_steal_percentage                  REAL,
            away_steals                            BIGINT,
            away_three_point_attempt_rate          REAL,
            away_three_point_field_goal_attempts   BIGINT,
            away_three_point_field_goal_percentage REAL,
            away_three_point_field_goals           BIGINT,
            away_total_rebound_percentage          REAL,
            away_total_rebounds                    BIGINT,
            away_true_shooting_percentage          REAL,
            away_turnover_percentage               REAL,
            away_turnovers                         BIGINT,
            away_two_point_field_goal_attempts     BIGINT,
            away_two_point_field_goal_percentage   REAL,
            away_two_point_field_goals             BIGINT,
            away_wins                              BIGINT,
            home_assist_percentage                 REAL,
            home_assists                           BIGINT,
            home_block_percentage                  REAL,
            home_blocks                            BIGINT,
            home_defensive_rating                  REAL,
            home_defensive_rebound_percentage      REAL,
            home_defensive_rebounds                BIGINT,
            home_effective_field_goal_percentage   REAL,
            home_field_goal_attempts               BIGINT,
            home_field_goal_percentage             REAL,
            home_field_goals                       BIGINT,
            home_free_throw_attempt_rate           REAL,
            home_free_throw_attempts               BIGINT,
            home_free_throw_percentage             REAL,
            home_free_throws                       BIGINT,
            home_losses                            BIGINT,
            home_minutes_played                    BIGINT,
            home_offensive_rating                  REAL,
            home_offensive_rebound_percentage      REAL,
            home_offensive_rebounds                BIGINT,
            home_personal_fouls                    BIGINT,
            home_points_q1                         BIGINT,
            home_points_q2                         BIGINT,
            home_points_q3                         BIGINT,
            home_points_q4                         BIGINT,
            home_points                            BIGINT,
            home_steal_percentage                  REAL,
            home_steals                            BIGINT,
            home_three_point_attempt_rate          REAL,
            home_three_point_field_goal_attempts   BIGINT,
            home_three_point_field_goal_percentage REAL,
            home_three_point_field_goals           BIGINT,
            home_total_rebound_percentage          REAL,
            home_total_rebounds                    BIGINT,
            home_true_shooting_percentage          REAL,
            home_turnover_percentage               REAL,
            home_turnovers                         BIGINT,
            home_two_point_field_goal_attempts     BIGINT,
            home_two_point_field_goal_percentage   REAL,
            home_two_point_field_goals             BIGINT,
            home_wins                              BIGINT,
        UNIQUE(date, winning_abbr, losing_abbr))""")

    engine.execute(
        """CREATE TABLE IF NOT EXISTS schedule (
            boxscore_index  TEXT,
            season          BIGINT,
            datetime        TEXT,
            team_away       TEXT,
            team_home       TEXT,
        UNIQUE(boxscore_index))""")

    logger = prefect.context.get('logger')
    logger.info('successfully connected to database')

    return engine


@task
def update_schedules(engine, current_season, start_season=2002):
    """Sync NBA game schedules. Rerunning this task overwrites
    and updates existing schedules for the current season.

    Returns boxscore_index of all games that are in the schedule
    table but not yet in the games table.
    """
    logger = prefect.context.get('logger')

    start_season = pd.read_sql(
        'SELECT MAX(season) FROM schedule', engine
    ).squeeze() or start_season

    engine.execute(f"DELETE FROM schedule WHERE season == {start_season}")

    for season in range(start_season, current_season + 1):
        for team in _retrieve_all_teams(season)[0].keys():
            logger.info(f'syncing {season} {team}')
            sched = Schedule(team, season)
            df = sched.dataframe

            df = df[df.location == 'Home']
            df['team_home'] = team
            df['team_away'] = df.opponent_abbr
            df['season'] = season
            df['time'] = (df.time + 'm').str.upper()
            df['datetime'] = pd.to_datetime(
                df.datetime.astype(str).str.cat(df.time, sep=' '),
                format='%Y-%m-%d %I:%M%p')

            cols = [
                'boxscore_index',
                'season',
                'datetime',
                'team_away',
                'team_home']

            df[cols].to_sql(
                'schedule', engine, if_exists='append', index=False)

    date_text = pd.read_sql("SELECT date FROM games", engine).squeeze()

    max_date = max(
        pd.to_datetime(date_text, format='%I:%M %p, %B %d, %Y'),
        default=datetime(2000, 1, 1))

    boxscore_ids = pd.read_sql(
        f"""SELECT DISTINCT boxscore_index FROM schedule
        WHERE ('{max_date}' < datetime) AND (datetime < '{datetime.now()}')
        ORDER BY datetime""", engine)

    return boxscore_ids


@task
def update_boxscores(engine, boxscore_ids):
    """Sync NBA game data and store in a sqlite database
    """
    logger = prefect.context.get('logger')

    for boxscore_id in boxscore_ids.values:
        logger.info(f'syncing {boxscore_id}')
        boxscore = Boxscore(boxscore_id)

        df = boxscore.dataframe
        summary = boxscore.summary

        if (df is not None and summary is not None):
            try:
                away_qtrs, home_qtrs = [[
                    f'{tm}_points_q{k}' for k in [1, 2, 3, 4]
                ] for tm in ['away', 'home']]
                df[away_qtrs] = summary['away'][:4]
                df[home_qtrs] = summary['home'][:4]
                df.to_sql('games', engine, if_exists='append', index=False)
            except sqlalchemy.exc.IntegrityError:
                logger.info(f'{boxscore_id} already stored in database')
                continue


def upcoming_games(days=1):
    """Return a pandas dataframe of upcoming games
    """
    engine = sqlalchemy.create_engine(f'sqlite:///{db_path.expanduser()}')

    df = pd.read_sql("SELECT * FROM schedule ORDER BY datetime", engine)

    df['datetime'] = pd.to_datetime(df.datetime)

    time_diff = (df.datetime - datetime.now())

    return df[(0 <= time_diff.dt.days) & (time_diff.dt.days <= days)]


def preprocess_data():
    """Preprocess historical NBA boxscore data into a form that is suitable for
    modelling

    Returns:
        Pandas dataframe of NBA game data
    """
    engine = sqlalchemy.create_engine(f'sqlite:///{db_path.expanduser()}')

    df = pd.read_sql("""
        SELECT
            date,
            losing_abbr,
            winning_abbr,
            winner,
            home_points,
            home_points_q1,
            home_points_q2,
            home_points_q3,
            home_points_q4,
            away_points,
            away_points_q1,
            away_points_q2,
            away_points_q3,
            away_points_q4
            FROM games""", engine)

    df['date'] = pd.to_datetime(df.date, format='%H:%M %p, %B %d, %Y')

    df['team_home'] = np.where(
        df.winner == 'Home', df.winning_abbr, df.losing_abbr)

    df['team_away'] = np.where(
        df.winner == 'Away', df.winning_abbr, df.losing_abbr)

    df['home_points_1h'] = df.home_points_q1 + df.home_points_q2
    df['away_points_1h'] = df.away_points_q1 + df.away_points_q2

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='update NBA gamedata')

    parser.add_argument(
        '--schedule', help='register the prefect flow with its backend',
        action='store_true')

    args = parser.parse_args()

    tz = 'America/New_York'

    schedule = IntervalSchedule(
        start_date=datetime(2020, 12, 2, 8, 0),
        interval=timedelta(days=7),
        end_date=datetime(2021, 2, 3, 8, 0))

    with Flow('update NBA game data', schedule=schedule) as flow:
        current_season = Parameter('current_season', default=2021)
        engine = initialize_database()
        boxscore_ids = update_schedules(engine, current_season)
        update_boxscores(engine, boxscore_ids)

    flow.run_config = LocalRun(working_dir=Path.cwd())

    if args.schedule is True:
        flow.register(project_name='nbabot')

    flow.run(current_season=2021, run_on_schedule=args.schedule)
