"""Estimator: data and fitted parameters in, predictions out. Fitting lives in tuning.py.

Every rate in the model is a shrunk rate:

    theta = (y + kappa*mu) / (n + kappa)

y and n are the hitter's pooled history before the half being predicted. kappa sets how
far he is pulled toward mu, mu sets where he is pulled to. The target is itself fitted,

    mu = expit(b0 + sum_f b_f x_f + gamma * nu/(n + nu))

and the features it reads are shrunk the same way one level down,

    x = (N + phi*L) / (D + phi)

which is why a hitter with no history needs no special case anywhere: D = 0 gives x = L,
and n = 0 gives theta = mu.

Two stages, composed to wOBA in evaluate.py. The PA stage splits the plate appearance
into K/BB/HBP/BIP over nPA; the BIP stage splits the ball in play into HR/1B/2B/3B/OUT
over nBIP.

`Params` is defined here rather than alongside the code that fills it, because this is
the module that reads it: tuning.py populates one per target half, predict consumes it.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass
class Params:
    """Everything needed to project one target half."""
    target_half: int
    phi: int
    beta: dict = field(default_factory=dict)    # category -> coefficients, or None
    kappa: dict = field(default_factory=dict)   # category -> concentration
    feat_use: dict = field(default_factory=dict)


def build_history(player_half, T, cols, lam=1.0): 
    """Counts summed over every half before T, one row per hitter with history.

    lam is the recency decay: half t is weighted lam ** (T - 1 - t), so the most recent
    half counts 1 and the pooled denominator stays readable as effective plate
    appearances. The shipped model sets lam = 1, but a future version could tune this
    parameter to incorporate changes in a hitter's talent level over time.

    Args:
        player_half (DataFrame): the panel
        T (int): target half; only halves strictly before it are summed
        cols (list of str): count columns to sum
        lam (float): recency decay; 1.0 pools flat
    Returns:
        hist (DataFrame): one row per hitter with history, indexed by BatterID
    """
    h = player_half[player_half["SeasonID"] < T]
    if lam == 1.0:
        return h.groupby("BatterID")[cols].sum()
    w = lam ** (T - 1 - h["SeasonID"])
    # BatterID goes in as a Series rather than a name, since h[cols] has already dropped
    # it. The row index survives .mul, so counts and labels still line up.
    return h[cols].mul(w, axis=0).groupby(h["BatterID"]).sum()


def recent_half(player_half, T): 
    """
    Args:
        player_half (DataFrame): the panel
        T (int): target half
    Returns:
        recent (DataFrame): the panel rows for half T-1
    """
    return player_half[player_half["SeasonID"] == T - 1]


def half_players(player_half, T): 
    """Everyone who appeared in half T. The default spine for scoring that half.

    Args:
        player_half (DataFrame): the panel
        T (int): the half to take the spine from
    Returns:
        players (Index): named BatterID
    """
    return pd.Index(player_half.loc[player_half["SeasonID"] == T, "BatterID"],
                    name="BatterID")


def league_ratio(recent, num, den): 
    """
    Args:
        recent (DataFrame): half T-1 rows, from recent_half
        num (str): numerator column
        den (str): denominator column
    Returns:
        ratio (float): the ratio of the two sums
    """
    return recent[num].sum() / recent[den].sum()


def league_rates(player_half, T, cats, denom): 
    """Each category's league rate in half T-1.

    Args:
        player_half (DataFrame): the panel
        T (int): target half
        cats (list of str): the stage's category columns
        denom (str): the denominator the rates will be applied on
    Returns:
        rates (dict): category column -> its league rate
    """
    recent = recent_half(player_half, T)
    return {c: league_ratio(recent, c, denom) for c in cats}


def shrink_features(hist, recent, phi, features): 
    """Pooled feature values, shrunk toward the league value.

    Ratio of sums, not mean of ratios: a hitter with no out-of-zone swing has no per-PA
    rate to average, and a mean over the rates that do exist runs high.

    phi is in each feature's own denominator units -- swings and pitches at the PA stage,
    balls in play at the BIP stage -- so the two are not comparable numbers. A zero
    denominator returns the league value exactly.

    Args:
        hist (DataFrame): pooled history, from build_history
        recent (DataFrame): half T-1 rows, for the league values
        phi (int): feature concentration
        features (dict): name -> (numerator column, denominator column)
    Returns:
        feats (DataFrame): one column per feature, on hist's index
    """
    out = {}
    for name, (num, den) in features.items():
        league = league_ratio(recent, num, den)
        out[name] = (hist[num] + phi * league) / (hist[den] + phi)
    return pd.DataFrame(out, index=hist.index)


def build_pairs(player_half, T, phi, stage, players=None): 
    """One row per (player, category): pooled history, features, and the outcome.

    A hitter with no prior half gets zero counts and lands on mu through the ordinary 
    formula.

    Suffixes: _p is pooled history, _t is the target half. n_p sets how far a hitter is
    shrunk, n_t sets how much his row weighs in a fit or a loss.

    Args:
        player_half (DataFrame): the panel
        T (int): target half
        phi (int): feature concentration
        stage (dict): the stage dict
        players (Index or None): player ids to project, or None to score against half T
    Returns:
        pairs (DataFrame): one row per (player, category): y_p, n_p, the features, green,
            league_rate, and n_t and act when a spine was not passed
    """
    cats, denom = stage["cats"], stage["denom"]

    # Per category, what the target half is scored on
    scored_on = dict(zip(cats, stage.get("act_cats", cats)))

    recent = recent_half(player_half, T)

    if players is None:
        tgt = player_half[player_half["SeasonID"] == T].set_index("BatterID")
        if tgt.empty:
            raise ValueError(f"half {T} has not been played; pass `players` to project it")
        idx = tgt.index
    else:
        tgt = None
        idx = pd.Index(players, name="BatterID")

    # Pool the stage's denominator, its categories, and the raw columns behind its
    # features. A feat_cols entry that repeats the denominator or a category would make
    # hist[denom] a DataFrame, and the reindex below would fail on duplicate labels.
    cols = [denom] + cats + stage["feat_cols"]
    assert len(set(cols)) == len(cols), "feat_cols repeats a category or the denominator"
    hist = build_history(player_half, T, cols, stage.get("lam", 1.0))
    hist = hist.reindex(idx).fillna(0.0)
    X = shrink_features(hist, recent, phi, stage["features"])

    # Greenness: nu/(n + nu), this is 1 at debut and near 0 for a veteran. It cannot be a
    # feature, since shrink_features needs a league ratio of two columns and this needs
    # L = 1 with N = 0. Present on every row, but only enters a category whose feat_use
    # names it, which is what keeps HBP and 3B on the flat league prior.
    X["green"] = stage["green"] / (hist[denom] + stage["green"])
    X["n_p"] = hist[denom]
    X["target_half"] = T

    # X, hist and tgt share the BatterID index, so these align by hitter.
    def for_category(c):
        r = X.assign(cat=c[1:], y_p=hist[c], league_rate=league_ratio(recent, c, denom))
        if tgt is None:
            return r
        return r.assign(n_t=tgt[denom], act=tgt[scored_on[c]] / tgt[denom])

    pairs = pd.concat([for_category(c) for c in cats]).reset_index()
    if tgt is None:
        return pairs

    # act is 0/0 for a hitter who took a plate appearance but never put a ball in play.
    return pairs[pairs["n_t"] > 0]


def shrink(y, n, kappa, mu): 
    """
    Args:
        y (array-like): pooled numerator
        n (array-like): pooled denominator
        kappa (float): shrinkage concentration
        mu (array-like): the target being shrunk toward
    Returns:
        theta (ndarray): the shrunk rate, elementwise
    """
    return ((np.asarray(y, float) + kappa * np.asarray(mu, float))
            / (np.asarray(n, float) + kappa))


def normalize(pred, keys): 
    """Rescale a player's categories to sum to one.

    Args:
        pred (array-like): one predicted rate per (player, category) row
        keys (list of Series): the columns identifying one player's set of categories
    Returns:
        scaled (ndarray): the same rates, each group scaled to sum to one
    """
    pred = pd.Series(np.asarray(pred, float))
    return (pred / pred.groupby(list(keys)).transform("sum")).to_numpy()


def design(d, cols): 
    """Intercept plus one column per selected feature.

    Args:
        d (DataFrame): the rows to build the matrix from
        cols (list of str): the selected feature columns
    Returns:
        X (ndarray): len(d) rows by 1 + len(cols) columns
    """
    return np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])


def apply_prior(beta, d, cols, link="logit"): 
    """mu per row: expit(x'b), or the league rate where there are no features (HBP, 3B).

    The logit link binds mu inside (0, 1), which the identity link cannot do.

    Args:
        beta (ndarray or None): fitted coefficients, or None for the flat league rate
        d (DataFrame): the rows to evaluate on
        cols (list of str): the feature columns beta was fitted on
        link (str): "logit", "identity" or "binomial"
    Returns:
        mu (ndarray): the prior value per row
    """
    if beta is None:
        return d["league_rate"].to_numpy()
    z = design(d, cols) @ beta
    if link == "identity":
        return z
    return expit(z)


def predict(player_half, T, params, stage, players=None): 
    """Project every category at target half T from already-fitted parameters.

    Renormalises when the stage has more than one category.

    Args:
        player_half (DataFrame): the panel
        T (int): target half
        params (Params): the fitted parameters for T
        stage (dict): the stage dict
        players (Index or None): spine to project, or None to score against half T
    Returns:
        preds (DataFrame): build_pairs' rows plus mu_i, kappa and pred
    """
    p = build_pairs(player_half, T, params.phi, stage, players=players)
    out = []
    for c in stage["cats"]:
        cat = c[1:]
        cur = p[p["cat"] == cat].copy()
        cur["mu_i"] = apply_prior(params.beta[cat], cur, params.feat_use[cat],
                                  stage.get("link", "logit"))
        cur["kappa"] = params.kappa[cat]
        out.append(cur)
    f = pd.concat(out, ignore_index=True)

    theta = shrink(f["y_p"], f["n_p"], f["kappa"], f["mu_i"])
    f["pred"] = (normalize(theta, [f["BatterID"], f["target_half"]])
                 if len(stage["cats"]) > 1 else theta)
    return f
