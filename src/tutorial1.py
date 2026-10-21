"""Reproducible Tutorial 1 setup and blocks A/B. Block C is excluded."""

import hashlib
from pathlib import Path

from arch import arch_model
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import het_arch

from .garch_mle import fit_garch11


def arch_lm(residuals, lags=12):
    """ARCH-LM: residual squares regressed on an intercept and lagged squares.

    Pass residuals, NOT their squares: het_arch squares them internally.
    The course convention uses n = T - lags and LM = n R-squared (ddof=0).
    """
    e = np.asarray(residuals, dtype=float)
    if e.ndim != 1 or not np.isfinite(e).all() or len(e) <= 2 * lags:
        raise ValueError("ARCH-LM needs finite residuals and enough observations.")
    lm, pvalue, f, fp = het_arch(e, nlags=lags, ddof=0, result_object=False)
    return {
        "lm": float(lm),
        "pvalue": float(pvalue),
        "f": float(f),
        "f_pvalue": float(fp),
        "lags": lags,
        "auxiliary_n": len(e) - lags,
    }


def leverage_correlation(returns):
    """Descriptive Pearson correlation of r[t] with abs(r[t+1]), without wraparound."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 3 or not np.isfinite(r).all():
        raise ValueError("Supply at least three finite returns.")
    return {
        "correlation": float(np.corrcoef(r[:-1], np.abs(r[1:]))[0, 1]),
        "pairs": len(r) - 1,
    }


def news_impact_curve(parameters, shocks, gjr=False):
    """Next variance at each model's own unconditional state; shocks are standardised.

    Gaussian innovation symmetry implies E[z^2 1(z<0)] = 1/2 for GJR.
    This state is unrelated to the variance used to initialise the fitted filter.
    """
    w, a, b = (parameters[name] for name in ("omega", "alpha", "beta"))
    gamma = parameters.get("gamma", 0.0) if gjr else 0.0
    persistence = a + b + gamma / 2
    if w <= 0 or a < 0 or a + gamma < 0 or b < 0 or persistence >= 1:
        raise ValueError(
            "A positive finite unconditional variance requires stationary parameters."
        )
    hbar = w / (1 - persistence)
    z = np.asarray(shocks, dtype=float)
    curve = w + (a + gamma * (z < 0)) * z**2 * hbar + b * hbar
    return curve, float(hbar)


def load_tutorial_data(workbook="data/NLNPE_TD_data.xlsx", start="1981-01-01", end="2026-10-09"):
    """Recompute percent log returns from the included, frozen workbook."""
    workbook = Path(workbook)
    if not workbook.exists():
        raise FileNotFoundError(f"The frozen input workbook is required: {workbook}")
    with pd.ExcelFile(workbook) as book:
        raw = pd.read_excel(book, sheet_name="weekly_spx_ust10")
        info = pd.read_excel(book, sheet_name="provenance") if "provenance" in book.sheet_names else None
    if "spx_close" not in raw:
        raise ValueError("The weekly sheet must contain spx_close.")
    date_column = next((c for c in raw if "date" in str(c).lower()), raw.columns[0])
    dates = pd.to_datetime(raw[date_column], errors="coerce")
    close = pd.to_numeric(raw["spx_close"], errors="coerce")
    if dates.isna().any() or not np.isfinite(close).all() or (close <= 0).any() or dates.duplicated().any():
        raise ValueError("Workbook dates or prices are missing, duplicated or invalid.")
    frame = pd.DataFrame({"close": close.to_numpy()}, index=pd.DatetimeIndex(dates)).sort_index()
    returns = 100*np.log(frame.close/frame.close.shift(1))
    if "spx_logret_pct" in raw:
        provided = pd.Series(pd.to_numeric(raw.spx_logret_pct, errors="coerce").to_numpy(), index=pd.DatetimeIndex(dates)).sort_index()
        valid = returns.notna() & provided.notna()
        if not valid.any():
            raise ValueError("No embedded log returns can be compared with prices.")
        np.testing.assert_allclose(returns[valid], provided[valid], atol=1e-6, rtol=1e-6)
    returns = returns.dropna().loc[lambda x: (x.index >= pd.Timestamp(start)) & (x.index < pd.Timestamp(end))]
    if len(returns) < 30 or not np.isfinite(returns).all():
        raise ValueError("Insufficient or nonfinite weekly returns.")
    if returns.index.to_series().diff().iloc[1:].dt.days.max() > 8:
        raise ValueError("Missing weekly observations require explicit treatment before fitting.")
    reconstructed = info is not None and any(str(v).lower()=="reconstructed" for v in info.to_numpy().ravel())
    provenance = {
        "official": not reconstructed, "reconstructed": reconstructed,
        "source": str(workbook.resolve()), "sheet": "weekly_spx_ust10",
        "definition": "100 log(P_t/P_(t-1))", "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "requested_start": start, "exclusive_end": end,
    }
    if reconstructed:
        provenance["limitation"] = "Public-source reconstruction matches published summary checks; original rows not available for identity verification."
    return returns.rename("return_pct"), provenance


def package_fit(returns, asymmetric=False, backcast=None, cov_type="robust"):
    """Constant-mean Gaussian arch fit, optional presample backcast, robust SE by default."""
    model = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=int(asymmetric),
        q=1,
        dist="normal",
        rescale=False,
    )
    result = model.fit(
        disp="off", cov_type=cov_type, options={"maxiter": 2000, "ftol": 1e-10},
        backcast=backcast,
    )
    if result.convergence_flag != 0:
        raise RuntimeError(
            f"arch optimization did not converge: {result.optimization_result.message}"
        )
    return result


def package_parameters(result):
    mapping = {
        "mu": "mu",
        "omega": "omega",
        "alpha[1]": "alpha",
        "beta[1]": "beta",
        "gamma[1]": "gamma",
    }
    return {mapping[k]: float(v) for k, v in result.params.items()}


def analyze(returns, provenance, initialization="sample"):
    """All reported numbers are calculated from this exact weekly series."""
    r = returns.to_numpy()
    custom = fit_garch11(r, initialization=initialization)
    benchmark_backcast = float(np.var(r, ddof=0))
    benchmark = package_fit(returns, backcast=benchmark_backcast)
    # Hessian-based (classic) covariance, to cross-check our numerical Hessian.
    benchmark_classic = package_fit(returns, backcast=benchmark_backcast, cov_type="classic")
    gjr = package_fit(returns, asymmetric=True)
    bp, gp = package_parameters(benchmark), package_parameters(gjr)
    comparison = pd.DataFrame(
        {
            "Custom estimate": custom["estimates"],
            "Custom BHHH SE": custom["standard_errors"],
            "arch estimate": bp,
        }
    )
    comparison["Abs. difference"] = abs(
        comparison["Custom estimate"] - comparison["arch estimate"]
    )
    comparison["Relative difference (%)"] = (
        100 * comparison["Abs. difference"] / abs(comparison["arch estimate"])
    )
    comparison["Same to 2 decimals"] = comparison["Custom estimate"].round(
        2
    ) == comparison["arch estimate"].round(2)
    z = np.linspace(-4, 4, 401)
    gn, gh = news_impact_curve(custom["estimates"], z)
    jn, jh = news_impact_curve(gp, z, gjr=True)
    checkpoints = [
        {"checkpoint": title, "tag": "correct"}
        for title in [
            "Residual, not return",
            "Initial variance",
            "Likelihood sign",
            "Positive, stationary parameters",
            "BHHH standard errors",
            "Returns in percent",
        ]
    ]
    result = {
        "scope": "Tutorial 1 setup, A and B only; Block C excluded",
        "number_labels": {
            "results": "computed",
            "innovation_law": "assumed Gaussian",
            "annualisation": "assumed sqrt(52)",
            "TD_reference": "stated in handout",
        },
        "provenance": provenance,
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "n": len(r),
        "statistics": {
            "mean_pct": float(r.mean()),
            "weekly_sd_pct": float(np.std(r, ddof=1)),
            "annualised_sd_pct": float(np.std(r, ddof=1) * np.sqrt(52)),
            "excess_kurtosis": float(kurtosis(r, fisher=True, bias=False)),
            "skewness": float(skew(r, bias=False)),
        },
        "arch_lm": arch_lm(r - r.mean()),
        "post_garch_arch_lm": arch_lm(
            (r - custom["estimates"]["mu"]) / np.sqrt(custom["conditional_variance"])
        ),
        "post_gjr_arch_lm": arch_lm(np.asarray(gjr.std_resid)),
        "custom": {k: v for k, v in custom.items() if k != "conditional_variance"},
        "arch": {
            "estimates": bp,
            "loglik": float(benchmark.loglikelihood),
            "initialization": "sample variance (divisor T) used as presample backcast; first variance still follows package recursion",
            "backcast": benchmark_backcast,
            "initial_variance": float(benchmark.conditional_volatility.iloc[0] ** 2),
            "standard_errors": {
                k: float(v) for k, v in package_parameters_se(benchmark).items()
            },
            "classic_standard_errors": {
                k: float(v) for k, v in package_parameters_se(benchmark_classic).items()
            },
        },
        "gjr": {
            "estimates": gp,
            "loglik": float(gjr.loglikelihood),
            "gamma_se": float(gjr.std_err["gamma[1]"]),
            "gamma_t": float(gjr.tvalues["gamma[1]"]),
            "gamma_p": float(gjr.pvalues["gamma[1]"]),
            "covariance": "Bollerslev-Wooldridge robust sandwich",
        },
        "comparison": comparison.reset_index(names="Parameter").to_dict(
            orient="records"
        ),
        "benchmark_pass": bool(comparison["Same to 2 decimals"].all()),
        "leverage": leverage_correlation(r),
        "checkpoints": checkpoints,
        "nic": {
            "garch_hbar": gh,
            "gjr_hbar": jh,
            "z_min": -4,
            "z_max": 4,
            "units": "next-period variance in percent squared",
            "fixed_state": "each model's own unconditional variance",
            "positive_shock_equality": "not required for separately fitted models",
        },
    }
    paths = pd.DataFrame(
        {
            "return_pct": r,
            "GARCH volatility (%)": np.sqrt(custom["conditional_variance"]),
            "GJR volatility (%)": np.asarray(gjr.conditional_volatility),
        },
        index=returns.index,
    )
    curves = pd.DataFrame(
        {"z": z, "GARCH variance (%^2)": gn, "GJR variance (%^2)": jn}
    )
    return result, paths, curves


def package_parameters_se(result):
    mapping = {
        "mu": "mu",
        "omega": "omega",
        "alpha[1]": "alpha",
        "beta[1]": "beta",
        "gamma[1]": "gamma",
    }
    return {mapping[k]: float(v) for k, v in result.std_err.items()}
