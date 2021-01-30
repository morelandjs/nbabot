"""Graphically validate model predictions."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .data import preprocess_data
from .model import EloraNBA
from . import workdir


def assess_predictions(mode):
    """
    Plot several statistical tests of the model prediction accuracy.
    """
    # figure style and layout
    width, height = plt.rcParams["figure.figsize"]
    fig, (axl, axr) = plt.subplots(ncols=2, figsize=(2*width, height))

    # load nfl model predictions
    nba_model = EloraNBA.from_cache(games, mode)

    # standard normal distribution
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)
    axl.plot(x, y, color="black")

    # raw residuals
    residuals = nba_model.residuals(standardize=False)
    logging.info("{} residual mean: {:.2f}"
                 .format(mode, residuals.mean()))
    logging.info("{} residual mean absolute error: {:.2f}"
                 .format(mode, nba_model.mean_abs_error))

    # standardized residuals
    std_residuals = nba_model.residuals(standardize=True)
    axl.hist(std_residuals, bins=40, histtype="step", density=True)

    # residual figure attributes
    axl.set_xlim(-4, 4)
    axl.set_ylim(0, .45)
    axl.set_xlabel(r"$(y_\mathrm{obs}-y_\mathrm{pred})/\sigma_\mathrm{pred}$")
    axl.set_title("Standardized residuals")

    # quantiles
    quantiles = nba_model.sf(
        nba_model.examples.value,
        nba_model.examples.time,
        nba_model.examples.label1,
        nba_model.examples.label2,
        nba_model.examples.bias
    )[nba_model.burnin:]

    axr.axhline(1, color="black")
    axr.hist(quantiles, bins=20, histtype="step", density=True)

    # quantile figure attributes
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1.5)
    axr.set_xlabel(' '.join([r"$\int_{-\infty}^{y_\mathrm{obs}}$",
                             r"$P(y_\mathrm{pred})$" r"$dy_\mathrm{pred}$"]))
    axr.set_title("Quantiles")
    plt.tight_layout()

    filepath = workdir / f'plots/validate_{mode}.png'
    filepath.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(filepath, dpi=200)


if __name__ == '__main__':
    games = preprocess_data()
    for mode in ['spread', 'total']:
        assess_predictions(mode)
