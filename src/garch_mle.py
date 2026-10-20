"""Gaussian conditional maximum likelihood for percent returns."""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


def fit_garch11(returns_pct, initialization="sample"):
    """Return estimates, BHHH standard errors, log likelihood and fitted variances.

    Input must already contain returns in percent (e.g. 1.2 means 1.2%).
    The same estimator applies to daily or weekly return observations.
    initialization="sample": fixed sample variance, with divisor n.
    initialization="long-run": omega / (1 - alpha - beta), recomputed at
    every likelihood evaluation, including its derivatives in BHHH scores.
    Standard errors assume a correctly specified conditional Gaussian model.
    Their covariance is the inverse sum of observation-score outer products,
    including the score for the estimated constant mean.
    """
    if initialization not in {"sample", "long-run"}:
        raise ValueError("initialization must be 'sample' or 'long-run'.")
    y = np.asarray(returns_pct, dtype=float)
    if y.ndim != 1 or y.size < 10 or not np.isfinite(y).all():
        raise ValueError("Provide at least 10 finite, one-dimensional returns.")
    v = float(np.var(y, ddof=0))
    if v <= 0 or not np.isfinite(v):
        raise ValueError("Returns must have a positive, finite sample variance.")
    s = np.sqrt(v)

    def unpack(z):
        # The third softmax weight is 1 - alpha - beta.
        a, b, slack = softmax([z[2], z[3], 0.0])
        with np.errstate(over="ignore", under="ignore"):
            w = v * np.exp(z[1])
        return np.array([s * z[0], w, a, b])

    def nll(z):
        mu, w, a, b = unpack(z)
        if not np.isfinite([mu, w, a, b]).all() or w <= 0 or a + b >= 1:
            return np.inf
        e = y - mu
        h = np.empty(y.size)
        h[0] = v if initialization == "sample" else w / (1 - a - b)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for t in range(1, y.size):
                h[t] = w + a * e[t - 1] ** 2 + b * h[t - 1]
            value = 0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + e**2 / h)
        return float(value) if np.isfinite(value) else np.inf

    # Multiple starts reduce sensitivity to local optima.
    fits = []
    for a, b in [(0.05, 0.90), (0.10, 0.70), (0.15, 0.40)]:
        slack = 1 - a - b
        start = [np.mean(y) / s, np.log(slack), np.log(a / slack), np.log(b / slack)]
        fit = minimize(
            nll,
            start,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6},
        )
        if fit.success and np.isfinite(fit.fun):
            fits.append(fit)
    if not fits:
        raise RuntimeError("GARCH optimization failed to converge.")
    fit = min(fits, key=lambda result: result.fun)
    z = fit.x

    mu, w, a, b = unpack(z)
    e = y - mu
    h = np.empty(y.size)
    h[0] = v if initialization == "sample" else w / (1 - a - b)
    # dh[t] = derivative of h[t] with respect to (mu, omega, alpha, beta).
    dh = np.zeros((y.size, 4))
    # Sample initialization is fixed; long-run initialization is parameter-dependent.
    if initialization == "long-run":
        slack = 1 - a - b
        dh[0] = [0, 1 / slack, w / slack**2, w / slack**2]
    for t in range(1, y.size):
        h[t] = w + a * e[t - 1] ** 2 + b * h[t - 1]
        dh[t] = (
            np.array([-2 * a * e[t - 1], 1, e[t - 1] ** 2, h[t - 1]]) + b * dh[t - 1]
        )
    scores = 0.5 * ((e**2 / h - 1) / h)[:, None] * dh
    scores[:, 0] += e / h  # Direct derivative of the mean in e[t].

    J = np.array(
        [
            [s, 0, 0, 0],
            [0, w, 0, 0],
            [0, 0, a * (1 - a), -a * b],
            [0, 0, -a * b, b * (1 - b)],
        ]
    )
    # BHHH in transformed coordinates improves numerical scaling.
    # B is a SUM, so its inverse already includes the sample-size factor.
    scores_z = scores @ J
    B = scores_z.T @ scores_z
    if np.isfinite(B).all() and np.linalg.matrix_rank(B) == 4:
        covariance = J @ np.linalg.solve(B, J.T)
        se = np.sqrt(np.diag(covariance))
    else:
        warnings.warn(
            "BHHH information is singular or nonfinite; standard errors unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
        se = np.full(4, np.nan)

    names = ["mu", "omega", "alpha", "beta"]
    return {
        "estimates": dict(zip(names, [mu, w, a, b])),
        "standard_errors": dict(zip(names, se)),
        "loglik": -float(fit.fun),
        "initialization": initialization,
        "initial_variance": float(h[0]),
        "conditional_variance": h.copy(),
    }
