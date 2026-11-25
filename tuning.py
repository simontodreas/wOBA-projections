"""Parameter estimation. Everything here is fit; nothing here predicts.

Three parameters per stage, fitted in this order because each changes the setup of the
next one:

    phi      feature concentration    grid search on prior error
    beta     prior coefficients       denominator-weighted squared error, logit link
    kappa    shrinkage concentration  bounded 1-D search on predictive error

Feature selection comes before all three, since phi is fitted on whichever features
survive it.

Every fit draws its rows through fitting_set, which is the one place the expanding
window is enforced.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

import model as M


def no_features(cats): 
    """A flat league prior for every category. Used for the wOBA floor.

    Keyed on the bare category name, since feat_use uses "HR" where the count column is
    "nHR".

    Args:
        cats (list of str): the stage's category columns
    Returns:
        feat_use (dict): bare category name -> an empty feature list
    """
    return {c[1:]: [] for c in cats}


def fitting_set(pairs, T, cat=None): 
    """Rows a parameter for target T may see: strictly earlier targets.

    This is where the expanding window is enforced. Target 3 fits on target 2, etc.
   
    Args:
        pairs (DataFrame): pair rows spanning several target halves
        T (int): the target half a parameter is being fitted for
        cat (str or None): a category column to keep, or None for all of them
    Returns:
        train (DataFrame): the rows a fit for T may see
    """
    m = pairs["target_half"] < T
    if cat is not None:
        # cat is a count column ("nHR"); the pairs frame stores the bare name.
        m &= pairs["cat"] == cat[1:]
    return pairs[m]


def pairs_all_targets(player_half, phi, stage): 
    """Pair rows for every scoreable target half, at one phi.

    Args:
        player_half (DataFrame): the panel
        phi (int): feature concentration
        stage (dict): the stage dict
    Returns:
        pairs (DataFrame): build_pairs output for targets 2, 3 and 4, stacked
    """
    return pd.concat([M.build_pairs(player_half, t, phi, stage) for t in (2, 3, 4)],
                     ignore_index=True)


def fit_prior(train, cols, link="logit"): 
    """beta: features to the target-half rate, minimizing weighted squared error.

    Returns None when the category takes no features, which apply_prior reads as a
    fallback to the league rate.

    Squared error through a logistic map rather than a binomial likelihood. The likelihood
    weights a hitter by n*p*(1-p) where the reported loss weights by n, so the two rank
    hitters differently. Keeping the objective aligned means the fit, feature selection
    and phi all optimise one thing.

    The other two links are alternatives the shipped model does not use, kept so the
    comparison can be run. "identity" is the unconstrained linear prior the logit
    replaced, on the same weighted squared error; "binomial" keeps the logistic map and
    swaps the objective for the likelihood the paragraph above objects to. 

    Args:
        train (DataFrame): the fitting rows, from fitting_set
        cols (list of str): feature columns to fit on
        link (str): "logit", "identity" or "binomial"
    Returns:
        beta (ndarray or None): coefficients, intercept first
    """
    if not cols:
        return None
    X = M.design(train, cols)
    a, n = train["act"].to_numpy(float), train["n_t"].to_numpy(float)
    # Clipped off 0 and 1 so the logit starting point below is finite for a rare category.
    mean_rate = np.clip(np.average(a, weights=n), 0.000001, 0.999999)

    # All three objectives return (value, gradient), which is what jac=True expects below.
    def squared_error(b):
        p = expit(X @ b)
        r = p - a
        return float((n * r ** 2).sum()), X.T @ (2.0 * n * r * p * (1.0 - p))

    def squared_error_identity(b):
        r = X @ b - a
        return float((n * r ** 2).sum()), X.T @ (2.0 * n * r)

    def negative_log_likelihood(b):
        p = np.clip(expit(X @ b), 1e-12, 1 - 1e-12)
        y = a * n                                  # exact integer counts: act = y / n_t
        return (float(-(y * np.log(p) + (n - y) * np.log(1 - p)).sum()),
                X.T @ (n * p - y))

    # Start at the intercept-only optimum, the weighted mean rate on the prior's own scale.
    if link == "identity":
        obj, b0 = squared_error_identity, np.r_[mean_rate, np.zeros(len(cols))]
    elif link == "binomial":
        obj, b0 = negative_log_likelihood, np.r_[logit(mean_rate), np.zeros(len(cols))]
    else:
        obj, b0 = squared_error, np.r_[logit(mean_rate), np.zeros(len(cols))]

    fit = minimize(obj, b0, jac=True, method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"prior fit failed: {fit.message}")
    return fit.x


def fit_kappa(train, mu): 
    """kappa minimizing weighted error against the next half.

    Kappa is bounded intentionally. The objective is not a likelihood, so an informative 
    prior can make kappa -> inf optimal

    Args:
        train (DataFrame): the fitting rows
        mu (ndarray): the prior value per row, from apply_prior
    Returns:
        kappa (float): the fitted shrinkage concentration
    """
    y = train["y_p"].to_numpy(float)
    n = train["n_p"].to_numpy(float)
    act = train["act"].to_numpy(float)
    weight = train["n_t"].to_numpy(float)

    # Searched in log kappa, since the bounds span four orders of magnitude.
    def weighted_error(log_kappa):
        theta = M.shrink(y, n, np.exp(log_kappa), mu)
        return np.average((theta - act) ** 2, weights=weight)

    fit = minimize_scalar(weighted_error, bounds=(np.log(1), np.log(20000)),
                          method="bounded")
    return np.exp(fit.x)


def prior_error(pairs, T, cat, cols): 
    """Weighted error of the fitted prior alone, ignoring shrinkage.

    The objective behind feature selection and phi. Fit in sample, which is fine 
    because the features come from the prior halves and the outcome from the target half,
    so their noise is independent.

    Args:
        pairs (DataFrame): pair rows spanning several target halves
        T (int): the target half being fitted for
        cat (str): the category column to score
        cols (list of str): the feature columns to fit
    Returns:
        mse (float): weighted mean squared error of the prior
    """
    d = fitting_set(pairs, T, cat)
    mu = M.apply_prior(fit_prior(d, cols), d, cols)
    return np.average((mu - d["act"].to_numpy()) ** 2, weights=d["n_t"].to_numpy(float))


def featured_cats(stage, feat_use): 
    """
    Args:
        stage (dict): the stage dict
        feat_use (dict): the category's selected features
    Returns:
        cats (list of str): the category columns that take at least one feature
    """
    return [c for c in stage["cats"] if feat_use[c[1:]]]


def feature_selection(player_half, T, phi, stage): 
    """Greedy forward selection by prior error, one pass per category.

    Step 0 is the flat league rate: every later step is read against it, and a category
    keeps it if nothing clears the 2% stopping rule.

    Marginal correlations mislead here because the candidates are close to collinear:
    whiffs and swings are the zone and out-of-zone pieces recombined.

    Args:
        player_half (DataFrame): the panel
        T (int): the target half being fitted for
        phi (int): the reference phi selection is scored at
        stage (dict): the stage dict
    Returns:
        steps (DataFrame): one row per step, columns (feature, mse) per category
    """
    pairs = pairs_all_targets(player_half, phi, stage)
    names = list(stage["features"])
    rows = []
    for c in stage["cats"]:
        cat = c[1:]
        chosen = []
        rows.append(dict(cat=cat, step=0, feature="(league avg)",
                         mse=prior_error(pairs, T, c, []) * 1000))
        while len(chosen) < len(names):
            best, best_mse = None, None
            for f in names:
                if f in chosen:
                    continue
                mse = prior_error(pairs, T, c, chosen + [f])
                if best_mse is None or mse < best_mse:
                    best, best_mse = f, mse
            chosen.append(best)
            rows.append(dict(cat=cat, step=len(chosen), feature=best,
                             mse=best_mse * 1000))
    # pivot over mixed dtypes returns object columns; infer_objects puts mse back to float
    t = pd.DataFrame(rows).pivot(index="step", columns="cat", values=["feature", "mse"])
    return t.infer_objects()


def fit_phi(player_half, T, feat_use, stage): 
    """Feature concentration, on a grid over the fitting halves.

    phi = 0 is undefined rather than just bad: a hitter with no out-of-zone swings in his
    pooled history would make owhiff 0/0.

    Args:
        player_half (DataFrame): the panel
        T (int): the target half being fitted for
        feat_use (dict): category's selected features
        stage (dict): the stage dict, which carries phi_grid
    Returns:
        phi (int): the grid value with the lowest summed prior error
    """
    cats = featured_cats(stage, feat_use)
    err = []
    for ph in stage["phi_grid"]:
        pairs = pairs_all_targets(player_half, ph, stage)
        total = 0.0
        for c in cats:
            total += prior_error(pairs, T, c, feat_use[c[1:]])
        err.append(total)
    return stage["phi_grid"][int(np.argmin(err))]


def phi_surface(player_half, T, feat_use, stage): 
    """Prior error against phi, per category and summed, as percent above the minimum.

    Same objective fit_phi minimizes, so its argmin is the phi that ships.

    Args:
        player_half (DataFrame): the panel
        T (int): the target half being fitted for
        feat_use (dict): category -> its selected features
        stage (dict): the stage dict, which carries phi_grid
    Returns:
        surface (DataFrame): one row per grid phi, one column per featured
            category plus TOTAL
    """
    cats = featured_cats(stage, feat_use)
    rows = []
    for ph in stage["phi_grid"]:
        pairs = pairs_all_targets(player_half, ph, stage)
        row = {}
        for c in cats:
            row[c[1:]] = prior_error(pairs, T, c, feat_use[c[1:]])
        rows.append(row)
    err = pd.DataFrame(rows, index=pd.Index(stage["phi_grid"], name="phi"))
    err["TOTAL"] = err.sum(axis=1)
    # min is per column, so each category and TOTAL is read against its best phi.
    # That makes the columns comparable in shape but not in level.
    return (err / err.min() - 1) * 100


def fit_all(player_half, T, feat_use, stage): 
    """Fit every parameter for target half T on halves strictly before it.

    Order is phi, beta, kappa. An informative prior raises kappa, so beta has to be
    settled first.

    beta and kappa are fit on the same rows, which inflates kappa: the prior looks better
    in sample than it is. With only one or two fitting transitions there is no room to
    cross-fit inside them.

    Args:
        player_half (DataFrame): the panel
        T (int): the target half to fit for
        feat_use (dict): category's selected features
        stage (dict): the stage dict
    Returns:
        params (Params): phi, beta and kappa for T
    """
    # phi and the feature picks are not refit per link: the link comparison varies one
    # thing, so both stay at their shipped values and only beta and kappa see the link.
    link = stage.get("link", "logit")
    phi = (fit_phi(player_half, T, feat_use, stage)
           if any(feat_use.values()) else stage["phi_grid"][0])
    pairs = pairs_all_targets(player_half, phi, stage)

    beta, kappa = {}, {}
    for c in stage["cats"]:
        cat = c[1:]
        past = fitting_set(pairs, T, c)
        if not len(past):
            beta[cat], kappa[cat] = None, np.nan
            continue
        b = fit_prior(past, feat_use[cat], link)
        beta[cat] = b
        kappa[cat] = fit_kappa(past, M.apply_prior(b, past, feat_use[cat], link))

    return M.Params(target_half=T, phi=phi, beta=beta, kappa=kappa, feat_use=feat_use)
