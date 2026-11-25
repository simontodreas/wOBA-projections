"""Composition to wOBA, scoring, and the tables the model is read off.

The loss throughout is PA-weighted mean squared error on observed wOBA, times 1000. The
weighting matters: half-seasons vary by an order of magnitude in plate appearances, so an
unweighted metric is dominated by rows carrying almost no information.

Only 2025 1H and 2025 2H can be scored, since a parameter for target T is fit on targets
strictly before T. Two folds is not many, which is why `score` returns the fold range
beside the mean.
"""

import numpy as np
import pandas as pd

import model as M

HALF_NAME = {2: "2024 2H", 3: "2025 1H", 4: "2025 2H", 5: "2026 1H"}

# The wOBA weights. data_final builds nwOBA from the delivered rate rather than from
# these, so a typo here cannot redefine the target, only make the prediction disagree
# with it.
WOBA_W = {"BB": 0.689, "HBP": 0.720, "1B": 0.882,
          "2B": 1.254, "3B": 1.590, "HR": 2.050}

# K and OUT carry weight zero but are kept, because the eight probabilities summing to
# one is the check that the composition is sound.
OUTCOMES = ["K", "BB", "HBP", "HR", "1B", "2B", "3B", "OUT"]


def wide(f, prefix): 
    """A predict() frame pivoted to one row per (BatterID, target_half).

    Args:
        f (DataFrame): a predict() frame
        prefix (str): string prepended to every category column
    Returns:
        w (DataFrame): one row per (BatterID, target_half), one column per category
    """
    w = f.pivot(index=["BatterID", "target_half"], columns="cat", values="pred")
    w.columns = [prefix + c for c in w.columns]
    return w


def compose_woba(player_half, T, params, stages, players): 
    """Turns the PA stage and the BIP stage to per-PA outcome probabilities and 
    predicted wOBA.

    Args:
        player_half (DataFrame): the panel
        T (int): target half
        params (tuple of Params): the (PA, BIP) pair
        stages (tuple of dict): the (PA, BIP) stage dicts
        players (Index): the spine to project both stages on
    Returns:
        out (DataFrame): p_ for each of the eight outcomes, plus pred_wOBA
    """
    pi = wide(M.predict(player_half, T, params[0], stages[0], players=players), "pi_")
    psi = wide(M.predict(player_half, T, params[1], stages[1], players=players), "psi_")
    p = pi.join(psi)

    out = pd.DataFrame({"p_K": p["pi_K"], "p_BB": p["pi_BB"], "p_HBP": p["pi_HBP"]})
    for x in ("HR", "1B", "2B", "3B", "OUT"):
        out["p_" + x] = p["pi_BIP"] * p["psi_" + x]
    out["pred_wOBA"] = sum(w * out["p_" + x] for x, w in WOBA_W.items())

    assert np.allclose(out[["p_" + c for c in OUTCOMES]].sum(axis=1), 1.0)
    return out


def compose_xwoba(player_half, T, params, stages, players): 
    """The PA stage and stage XW to predicted wOBA. The alternative to compose_woba.

    params and stages are (PA, XW) pairs. The five-category contact stage is replaced by
    xwOBAcon value:

        pred_wOBA = 0.689*p_BB + 0.720*p_HBP + p_BIP * wOBAcon

    p_K carries wOBA weight zero and OUT sits inside wOBAcon as a zero, so neither
    appears. 

    Args:
        player_half (DataFrame): the panel
        T (int): target half
        params (tuple of Params): the (PA, XW) pair
        stages (tuple of dict): the (PA, XW) stage dicts
        players (Index): the spine to project both stages on
    Returns:
        out (DataFrame): p_K, p_BB, p_HBP, p_BIP, wOBAcon and pred_wOBA
    """
    pi = wide(M.predict(player_half, T, params[0], stages[0], players=players), "pi_")
    xw = wide(M.predict(player_half, T, params[1], stages[1], players=players), "xw_")
    p = pi.join(xw)

    out = pd.DataFrame({"p_K": p["pi_K"], "p_BB": p["pi_BB"], "p_HBP": p["pi_HBP"],
                        "p_BIP": p["pi_BIP"],
                        "wOBAcon": p["xw_" + stages[1]["cats"][0][1:]]})
    out["pred_wOBA"] = (WOBA_W["BB"] * out["p_BB"] + WOBA_W["HBP"] * out["p_HBP"]
                        + out["p_BIP"] * out["wOBAcon"])
    return out


def compose_nested(player_half, T, params, stages, players): 
    """The PA stage, the home-run stage, and the four-way stage to predicted wOBA.

    The contact stage is split in two rather than resolved five ways at once, so a 
    ball in play is first a home run or not and the rest is a four-way choice on the 
    balls that stayed in the park:

        p_HR = pi_BIP * theta_HR
        p_x  = pi_BIP * (1 - theta_HR) * psi_x        x in 1B, 2B, 3B, OUT

    Same eight probabilities as compose_woba and the same assertion, since the two
    designs describe the same event.

    Args:
        player_half (DataFrame): the panel
        T (int): target half
        params (tuple of Params): the (PA, HR, 4-way) triple
        stages (tuple of dict): the (PA, HR, 4-way) stage dicts
        players (Index): the spine to project all three stages on
    Returns:
        out (DataFrame): p_ for each of the eight outcomes, plus pred_wOBA
    """
    pi = wide(M.predict(player_half, T, params[0], stages[0], players=players), "pi_")
    theta = wide(M.predict(player_half, T, params[1], stages[1], players=players), "theta_")
    psi = wide(M.predict(player_half, T, params[2], stages[2], players=players), "psi_")
    p = pi.join(theta).join(psi)

    in_play = p["pi_BIP"] * (1 - p["theta_HR"])
    out = pd.DataFrame({"p_K": p["pi_K"], "p_BB": p["pi_BB"], "p_HBP": p["pi_HBP"],
                        "p_HR": p["pi_BIP"] * p["theta_HR"]})
    for x in ("1B", "2B", "3B", "OUT"):
        out["p_" + x] = in_play * p["psi_" + x]
    out["pred_wOBA"] = sum(w * out["p_" + x] for x, w in WOBA_W.items())

    assert np.allclose(out[["p_" + c for c in OUTCOMES]].sum(axis=1), 1.0)
    return out


def project(player_half, targets, params_by_T, stages, compose=compose_woba): 
    """Composed wOBA at several scored target halves, stacked on (BatterID, target_half).

    Spined on everyone who appeared in each half. To project an unplayed half, call
    compose_woba directly with the list of players to project.

    `compose` defaults to the shipped composer. compose_xwoba has the same signature,
    so the XW comparison scores through this identical path.

    Args:
        player_half (DataFrame): the panel
        targets (tuple of int): the target halves to project
        params_by_T (dict): target half -> the fitted Params for it
        stages (tuple of dict): the stage dicts compose expects
        compose (callable): the composer to run at each target
    Returns:
        preds (DataFrame): compose's output at every target, stacked
    """
    return pd.concat([compose(player_half, t, params_by_T[t], stages,
                              M.half_players(player_half, t))
                      for t in targets])


def project_stage(player_half, targets, params_by_T, stage): 
    """One stage's projections at several scored halves, one column per category.

    This does what project does for the composed model, for a stage on its own. 
    
    Args:
        player_half (DataFrame): the panel
        targets (tuple of int): the target halves to project
        params_by_T (dict): target half -> the fitted Params for it
        stage (dict): the stage dict
    Returns:
        preds (DataFrame): one column per category, stacked over the targets
    """
    return pd.concat([wide(M.predict(player_half, t, params_by_T[t], stage,
                                     players=M.half_players(player_half, t)), "")
                      for t in targets])


def observed(player_half, targets=(3, 4)): 
    """Observed wOBA and its PA weight, one row per (player, scored half).

    Indexed to match what project() returns.

    Args:
        player_half (DataFrame): the panel
        targets (tuple of int): the halves to take actuals from
    Returns:
        actual (DataFrame): n_t and act per (BatterID, target_half)
    """
    d = player_half[player_half["SeasonID"].isin(targets)]
    return (d.rename(columns={"SeasonID": "target_half"})
             .set_index(["BatterID", "target_half"])[["nPA", "wOBA"]]
             .rename(columns={"nPA": "n_t", "wOBA": "act"}))


def score(actual, preds): 
    """PA-weighted MSE x1000 per fold, one row per prediction column.

    Args:
        actual (DataFrame): observed() output
        preds (DataFrame): one column per prediction to score, on the same index
    Returns:
        scores (DataFrame): one row per column: a column per fold, plus mean and spread
    """
    weight = actual["n_t"].groupby(level="target_half").sum()
    rows = {}
    for name in preds.columns:
        se = (preds[name] - actual["act"]) ** 2 * actual["n_t"]
        rows[name] = se.groupby(level="target_half").sum() / weight * 1000

    scores = pd.DataFrame(rows).T.rename(columns=HALF_NAME)
    scores["mean"] = scores.mean(axis=1)
    # :-1 drops the "mean" column just added, so the spread is over the folds alone.
    scores["spread"] = (scores.iloc[:, :-1].max(axis=1)
                        - scores.iloc[:, :-1].min(axis=1))
    return scores


def beta_table(params): 
    """Fitted prior coefficients, one row per (category, term).

    Args:
        params (Params): one stage's fitted parameters
    Returns:
        betas (DataFrame): coef, indexed by (category, term)
    """
    rows = []
    for cat, b in params.beta.items():
        if b is None:
            continue
        for name, coef in zip(["const"] + params.feat_use[cat], b):
            rows.append(dict(cat=cat, term=name, coef=coef))
    return pd.DataFrame(rows).set_index(["cat", "term"])


def shrinkage_table(player_half, T, params, stage, players=None): 
    """Table that contextualizes kappa 

    Args:
        player_half (DataFrame): the panel
        T (int): target half the parameters were fitted for
        params (Params): one stage's fitted parameters
        stage (dict): the stage dict
        players (Index or None): the spine the median is taken over, or None for half T
    Returns:
        shrinkage (DataFrame): kappa, features, units, median history and implied weight
            per category
    """
    if players is None:
        idx = M.half_players(player_half, T)
    else:
        idx = pd.Index(players)

    n = (M.build_history(player_half, T, [stage["denom"]])[stage["denom"]]
         .reindex(idx).fillna(0.0))
    med = n.median()

    shrinkage = pd.DataFrame({"kappa": {c[1:]: params.kappa[c[1:]]
                                        for c in stage["cats"]}})
    shrinkage["features"] = [", ".join(params.feat_use[c]) or "(league rate)"
                             for c in shrinkage.index]
    # Generic column names so two stages stack into one table instead of two NaN blocks.
    shrinkage["units"] = stage["denom"]
    shrinkage["median history"] = med
    shrinkage["implied weight"] = med / (med + shrinkage["kappa"])
    return shrinkage


def bias_by_depth(actual, pred, depth, bins): 
    """PA-weighted projected wOBA against actual, by how much history a hitter had.

    depth is pooled history per (player, target half); pooled_depth builds it.

    Args:
        actual (DataFrame): observed() output
        pred (Series): predicted wOBA on the same index
        depth (Series): pooled history per row, from pooled_depth
        bins (list of int): the depth bin edges
    Returns:
        bias (DataFrame): rows, proj, act and miss per bin
    """
    d = pd.DataFrame({"act": actual["act"], "n_t": actual["n_t"],
                      "pred": pred, "bin": pd.cut(depth, bins)}).dropna()
    # PA-weighted mean
    d["w_proj"] = d["pred"] * d["n_t"]
    d["w_act"] = d["act"] * d["n_t"]

    # observed=True: bin is a categorical, and the default emits a row per empty bin.
    by_bin = d.groupby("bin", observed=True)
    sums = by_bin[["w_proj", "w_act", "n_t"]].sum()
    bias = pd.DataFrame({"rows": by_bin.size(),
                         "proj": sums["w_proj"] / sums["n_t"],
                         "act": sums["w_act"] / sums["n_t"]})
    bias["miss"] = bias["proj"] - bias["act"]
    return bias


def pooled_depth(player_half, index, denom="nPA"): 
    """Pooled history per (player, target half), the x axis of the depth table.

    Taken from build_history rather than recomputed, so it is the same quantity the
    estimator sees. No history gives 0.

    Args:
        player_half (DataFrame): the panel
        index (MultiIndex): the (BatterID, target_half) index to build against
        denom (str): the history column to pool
    Returns:
        depth (Series): pooled history on that index
    """
    per_target = {}
    for t in index.get_level_values("target_half").unique():
        per_target[t] = M.build_history(player_half, t, [denom])[denom]

    depth = []
    for batter, t in index:
        depth.append(per_target[t].get(batter, 0.0))
    return pd.Series(depth, index=index)


def projection_table(player_half, T, params, stages, players, people): 
    """One row per player: every outcome probability, beside the actual if it exists.

    Args:
        player_half (DataFrame): the panel
        T (int): target half
        params (tuple of Params): the (PA, BIP) pair
        stages (tuple of dict): the (PA, BIP) stage dicts
        players (Index): the hitters to put in the table
        people (DataFrame): data.build_people output, for names and handedness
    Returns:
        out (DataFrame): one row per player: name, the p_ block, pred_wOBA,
            the a_ block, act_wOBA and resid
    """
    pred = compose_woba(player_half, T, params, stages, players).droplevel("target_half")

    tgt = player_half[player_half["SeasonID"] == T].set_index("BatterID")
    act = pd.DataFrame({"a_" + c: tgt["n" + c] / tgt["nPA"] for c in OUTCOMES})
    act.insert(0, "nPA", tgt["nPA"])
    act["act_wOBA"] = tgt["wOBA"]

    out = (people.reindex(pd.Index(players, name="BatterID"))
           .join(pred[["p_" + c for c in OUTCOMES] + ["pred_wOBA"]])
           .join(act))
    out["resid"] = out["pred_wOBA"] - out["act_wOBA"]
    out.insert(0, "target_half", T)
    return out
