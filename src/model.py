#!/usr/bin/env python3
"""Trains team model and exposes predictor class objects"""
from functools import partial
import operator
from pathlib import Path
import pickle

from elora import Elora
import numpy as np
import optuna
import pandas as pd

from . import cachedir


class EloraNBA(Elora):
    def __init__(self, games, mode, learning_rate, regress_frac, rest_coeff,
                 burnin=1260):
        """Generate NBA point-spread or point-total predictions using the Elo
        regressor algorithm (elora).

        Args:
            games (pd.DataFrame): pandas dataframe containing comparisons
                (datetime, team_home, team_away, *feat_home, *feature_away).
            mode (str): comparison type, equal to 'spread' or 'total'.
            learning_rate (float): gradient descent learning rate
            regress_frac (float): one minus the fractional amount used to
                regress ratings to the mean each offseason
            rest_coeff (float): prefactor that modulates the strength of rest
                effects, i.e. how much better/worse a team plays as a function
                of days between games. values can be positive or negative.
            burnin (int, optional): number of games to ignore from the
                beginning of the games dataframe when computing performance
                metrics. default is 512.
        """

        # training data
        self.games = games.sort_values(
            by=['datetime', 'team_home', 'team_away']).dropna()

        # NOTE: for debugging only
        self.games.drop(columns=[
            'mf_ast_home', 'mf_tov_home', 'mf_trb_home',
            'mf_ast_away', 'mf_tov_away', 'mf_trb_away'], inplace=True)

        # hyperparameters
        self.mode = mode
        self.learning_rate = learning_rate
        self.regress_frac = regress_frac
        self.rest_coeff = rest_coeff
        self.burnin = burnin

        # model operation mode: "spread" or "total"
        if self.mode not in ["spread", "total"]:
            raise ValueError(
                "Unknown mode; valid options are 'spread' and 'total'")

        # mode-specific training hyperparameters
        self.commutes, self.compare = {
            "total": (True, operator.add),
            "spread": (False, operator.sub),
        }[mode]

        # prepare model training data
        mf_home = self.games.filter(regex='^mf.*_home$', axis=1)
        mf_away = self.games.filter(regex='^mf.*_away$', axis=1)
        mf_home.columns = mf_home.columns.str.replace('_home', '')
        mf_away.columns = mf_away.columns.str.replace('_away', '')

        # initialize base class with chosen hyperparameters
        super().__init__(
            self.games.datetime,
            self.games.team_home,
            self.games.team_away,
            self.compare(mf_home, mf_away),
            self.bias(self.games))

        # calibrate the model
        self.fit(self.learning_rate, self.commutes)

        # compute performance metrics
        self.residuals_ = self.residuals()[:, 0]
        self.mean_abs_error = np.mean(np.abs(self.residuals_[burnin:]))
        self.rms_error = np.sqrt(np.mean(self.residuals_[burnin:]**2))
        print(np.mean(self.residuals_))
        print(self.mean_abs_error)

    def regression_coeff(self, elapsed_time):
        """Regress ratings to the mean as a function of elapsed time.

        Regression fraction equals:

            self.regress_frac if elapsed_days > 90, else 1

        Args:
            elapsed_time (datetime.timedelta): elapsed time since last update

        Returns:
            coefficient used to regress a rating to its mean value
        """
        elapsed_days = elapsed_time / np.timedelta64(1, 'D')

        tiny = 1e-6
        arg = np.clip(self.regress_frac, tiny, 1 - tiny)
        factor = np.log(arg)/365.

        return np.exp(factor * elapsed_days)

    def compute_rest_days(self, games):
        """Compute rest days for home and away teams

        Args:
            games (pd.DataFrame): dataframe of NBA game records

        Returns:
            pd.DataFrame including rest day columns
        """
        game_dates = pd.concat([
            games[["datetime", "team_home"]].rename(
                columns={"team_home": "team"}),
            games[["datetime", "team_away"]].rename(
                columns={"team_away": "team"}),
        ]).sort_values(by="datetime")

        game_dates['datetime_prev'] = game_dates.datetime

        game_dates = pd.merge_asof(
            game_dates[['team', 'datetime']],
            game_dates[['team', 'datetime', 'datetime_prev']],
            on='datetime', by='team', allow_exact_matches=False)

        for team in ["home", "away"]:
            game_dates_team = game_dates.rename(columns={
                'datetime_prev': f'datetime_{team}_prev', 'team': f'team_{team}'})
            games = games.merge(game_dates_team, on=['datetime', f'team_{team}'])

        one_day = pd.Timedelta("1 days")

        games["rest_days_home"] = np.clip(
            (games.datetime - games.datetime_home_prev) / one_day, 3, 16).fillna(7)
        games["rest_days_away"] = np.clip(
            (games.datetime - games.datetime_away_prev) / one_day, 3, 16).fillna(7)

        return games.drop(columns=['datetime_home_prev', 'datetime_away_prev'])

    def bias(self, games):
        """Circumstantial bias factors which apply to a single game.

        Args:
            games (pd.DataFrame): dataframe of NBA game records

        Returns:
            pd.Series of game bias correction coefficients
        """
        games = self.compute_rest_days(games)

        rest_adv = self.rest_coeff * self.compare(
            games.rest_days_away, games.rest_days_home)

        return rest_adv

    @classmethod
    def from_cache(cls, games, mode, n_trials=100, retrain=False):
        """Instantiate the regressor using cached hyperparameters if available,
        otherwise train and cache a new instance.

        Args:
            games (pd.DataFrame): dataframe of NBA game records
            mode (str): comparison type, equal to 'spread' or 'total'.
            n_trials (int, optional): number of optuna steps to use for
                hyperparameter optimization. default value is 100.
            retrain (bool, optional): load hyperparameters from cache if
                available and retrain is False, recalibrate hyperparameters
                otherwise. default is False.
        """
        cachefile = cachedir / f'{mode}_model.pkl'
        cachefile.parent.mkdir(exist_ok=True, parents=True)

        if not retrain and cachefile.exists():
            params = pickle.load(cachefile.open(mode='rb'))
            learning_rate = params['learning_rate']
            regress_frac = params['regress_frac']
            rest_coeff = params['rest_coeff']
            return cls(games, mode, learning_rate, regress_frac, rest_coeff)

        def objective(trial):
            """hyperparameter objective function
            """
            learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
            regress_frac = trial.suggest_uniform('regress_frac', 0.0, 1.0)
            rest_coeff = trial.suggest_uniform('rest_coeff', -0.2, 0.2)
            regressor = cls(games, mode, learning_rate, regress_frac, rest_coeff)
            return regressor.mean_abs_error

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)

        params = study.best_params.copy()

        pickle.dump(params, cachefile.open(mode='wb'))

        return cls(games, mode, *params.values())


if __name__ == '__main__':
    """Minimal example of how to use this module
    """
    import argparse
    from .data import games

    parser = argparse.ArgumentParser(description='calibrate hyperparameters')

    parser.add_argument(
        '--steps', type=int, default=100,
        help='number of Optuna calibration steps')

    args = parser.parse_args()

    for mode in ['spread', 'total']:
        EloraNBA.from_cache(games, mode, n_trials=args.steps, retrain=True)
