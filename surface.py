"""The ball-level expected-value surface, and the two columns stage XW needs.

This is an xwOBAcon model that uses contact quality metrics to predict the expected value 
of a batted ball. It is independent from the shipped model and can be used to directly
replace the BIP stage.

The two inputs are exit speed and launch angle. Batted ball direction was tested three 
ways (raw spray angle, a binary pull flag, and a pull/center/oppo classifier) and rejected 
each time. 
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import data

INPUTS = ["ExitSpeed", "LaunchAngle"]
MAX_TRAINING_HALF = 2      # fitted on SeasonID < 2, i.e. half 1 alone


def build_bip(statcast_df): 
    """The surface is a ball-level model, so it cannot be fed the player-half panel which 
    necessitates this function.

    Of note, Statcast does not have exit speed for every ball in play. Those balls are kept, 
    because the per-half calibration scalar needs every ball's observed value in its total.
    HistGradientBoostingRegressor routes the missing inputs natively.

    Args:
        statcast_df (DataFrame): the frame data.load_statcast returns
    Returns:
        bip (DataFrame): one row per ball in play: identity, SeasonID, the two inputs,
            its wOBA
    """
    sc = statcast_df[statcast_df["type"] == "X"]
    bip = pd.DataFrame({
        "BatterID":    sc["BatterID"],
        "ExitSpeed":   sc["launch_speed"].astype(float),
        "LaunchAngle": sc["launch_angle"].astype(float),
        "wOBA":        sc["events"].map(data.WOBA_WEIGHTS).fillna(0.0),
        "SeasonID":    data.season_id(sc["game_year"], sc["season_half"]),
    })
    return bip.reset_index(drop=True)


def fit_surface(bip, before=MAX_TRAINING_HALF, **kw): 
    """Expected wOBA value of a batted ball, from halves strictly before `before`.

    Args:
        bip (DataFrame): build_bip output
        before (int): train on halves with SeasonID strictly below this
        kw (keyword args): passed through to HistGradientBoostingRegressor
    Returns:
        model (HistGradientBoostingRegressor): the fitted regressor
    """
    train = bip[bip["SeasonID"] < before]
    m = HistGradientBoostingRegressor(random_state=2018, **kw)
    m.fit(train[INPUTS], train["wOBA"])
    return m


def ball_values(model, bip): 
    """Per-ball expected value, calibrated within each half. A Series on bip's index.

    Args:
        model (HistGradientBoostingRegressor): a fitted surface
        bip (DataFrame): the balls to value, from build_bip
    Returns:
        values (Series): calibrated expected value per ball, on bip's index
    """
    expected = pd.Series(model.predict(bip[INPUTS]), index=bip.index)
    observed = bip.groupby("SeasonID")["wOBA"].sum()
    scale = observed / expected.groupby(bip["SeasonID"]).sum()
    # One scalar per half, mapped back onto the balls by the half each was struck in.
    return expected * bip["SeasonID"].map(scale)


def attach(player_half, bip, model=None): 
    """player_half plus the expected/observed pair stage XW is built on.

    Args:
        player_half (DataFrame): the panel
        bip (DataFrame): build_bip output
        model (HistGradientBoostingRegressor or None): a fitted surface, or None
            to fit one on the default window
    Returns:
        out (DataFrame): player_half plus nXWcon and nWcon
    """
    if model is None:
        model = fit_surface(bip)
    valued = bip.assign(nXWcon=ball_values(model, bip))
    per_half = (valued.groupby(["BatterID", "SeasonID"])
                      .agg(nXWcon=("nXWcon", "sum"), nWcon=("wOBA", "sum")).reset_index())
    out = player_half.merge(per_half, how="left", on=["BatterID", "SeasonID"])
    out[["nXWcon", "nWcon"]] = out[["nXWcon", "nWcon"]].fillna(0.0)
    return out


def heatmap(model, ev=(60, 115), la=(-40, 60), step=0.5): 
    """The surface itself, as a launch-angle x exit-speed grid of expected wOBA.

    Args:
        model (HistGradientBoostingRegressor): the fitted surface
        ev (tuple of int): (min, max) exit speed
        la (tuple of int): (min, max) launch angle
        step (float): grid spacing, in both units
    Returns:
        grid (DataFrame): expected wOBA, rows launch angle by columns exit speed
    """
    evs = np.arange(ev[0], ev[1] + step, step)
    las = np.arange(la[0], la[1] + step, step)
    ev_grid, la_grid = np.meshgrid(evs, las)
    points = pd.DataFrame({"ExitSpeed": ev_grid.ravel(),
                           "LaunchAngle": la_grid.ravel()})
    # meshgrid varies exit speed along a row, so the flat predictions reshape back as
    # (launch angles, exit speeds) 
    z = model.predict(points[INPUTS]).reshape(len(las), len(evs))
    return pd.DataFrame(z, index=pd.Index(las, name="LaunchAngle"),
                        columns=pd.Index(evs, name="ExitSpeed"))
