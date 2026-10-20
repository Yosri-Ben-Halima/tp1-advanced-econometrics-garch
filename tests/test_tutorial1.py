"""Scientific checks for Tutorial 1 A/B, without network access."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import chi2


class Tutorial1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("src.tutorial1") is None:
            raise AssertionError(
                "The Tutorial 1 A/B analysis module is not implemented."
            )
        from src.tutorial1 import arch_lm, leverage_correlation, news_impact_curve

        cls.arch_lm = staticmethod(arch_lm)
        cls.leverage = staticmethod(leverage_correlation)
        cls.nic = staticmethod(news_impact_curve)

    def test_arch_lm_matches_independent_auxiliary_regression(self):
        e = np.random.default_rng(15).normal(size=150)
        q = 12
        squares = e**2
        target = squares[q:]
        design = np.column_stack(
            [np.ones(len(target))]
            + [squares[q - j : len(e) - j] for j in range(1, q + 1)]
        )
        residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
        r2 = 1 - (residual @ residual) / np.sum((target - target.mean()) ** 2)
        result = self.arch_lm(e, q)
        self.assertAlmostEqual(result["lm"], len(target) * r2, places=9)
        self.assertAlmostEqual(
            result["pvalue"], chi2.sf(len(target) * r2, q), places=10
        )

    def test_nic_uses_unconditional_state_and_negative_indicator(self):
        p = {"omega": 0.1, "alpha": 0.05, "beta": 0.8, "gamma": 0.1}
        z = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
        garch, hg = self.nic(p, z)
        gjr, hj = self.nic(p, z, gjr=True)
        self.assertAlmostEqual(hg, 0.1 / (1 - 0.05 - 0.8))
        self.assertAlmostEqual(hj, 0.1 / (1 - 0.05 - 0.8 - 0.1 / 2))
        np.testing.assert_allclose(garch[:2], garch[:2:-1])
        self.assertAlmostEqual(gjr[1] - gjr[3], 0.1 * hj)
        self.assertAlmostEqual(gjr[3], 0.1 + (0.05 + 0.8) * hj)
        self.assertNotAlmostEqual(garch[3], gjr[3])

    def test_nonstationary_nic_is_rejected(self):
        with self.assertRaises(ValueError):
            self.nic({"omega": 0.1, "alpha": 0.1, "beta": 0.9}, np.array([0.0, 1.0]))

    def test_leverage_correlation_has_correct_lead(self):
        r = np.array([3.0, -2.0, 1.0, -4.0, 5.0, 2.0])
        result = self.leverage(r)
        self.assertAlmostEqual(
            result["correlation"], np.corrcoef(r[:-1], abs(r[1:]))[0, 1]
        )
        self.assertEqual(result["pairs"], len(r) - 1)

    def test_bhhh_for_both_initializations_matches_numerical_scores(self):
        from src.garch_mle import fit_garch11

        rng = np.random.default_rng(123)
        y = np.empty(900)
        h, e = 2.0, 0.0
        for t in range(len(y)):
            h = 0.1 + 0.08 * e**2 + 0.87 * h
            e = np.sqrt(h) * rng.normal()
            y[t] = 0.2 + e
        y = y[200:]
        for mode in ("sample", "long-run"):
            fit = fit_garch11(y, initialization=mode)
            p = np.array(list(fit["estimates"].values()))

            def contributions(theta):
                mu, w, a, b = theta
                e = y - mu
                h = np.empty(len(y))
                h[0] = np.var(y) if mode == "sample" else w / (1 - a - b)
                for i in range(1, len(y)):
                    h[i] = w + a * e[i - 1] ** 2 + b * h[i - 1]
                return -0.5 * (np.log(2 * np.pi) + np.log(h) + e**2 / h)

            step = 1e-6 * np.maximum(1, abs(p))
            scores = np.column_stack(
                [
                    (
                        contributions(p + np.eye(4)[i] * step[i])
                        - contributions(p - np.eye(4)[i] * step[i])
                    )
                    / (2 * step[i])
                    for i in range(4)
                ]
            )
            reference = np.sqrt(np.diag(np.linalg.inv(scores.T @ scores)))
            np.testing.assert_allclose(
                list(fit["standard_errors"].values()), reference, rtol=1e-5
            )
            self.assertAlmostEqual(fit["loglik"], contributions(p).sum(), places=7)

    def test_supplied_workbook_reproduces_published_loading_checks(self):
        from scipy.stats import kurtosis, skew
        from src.tutorial1 import load_tutorial_data
        root = Path(__file__).resolve().parents[1]
        r, provenance = load_tutorial_data(root/"NLNPE_TD_data.xlsx")
        self.assertTrue(provenance["official"])
        self.assertFalse(provenance["reconstructed"])
        self.assertEqual(len(r), 2387)
        self.assertEqual(round(r.std(ddof=1), 2), 2.30)
        self.assertEqual(round(r.std(ddof=1)*np.sqrt(52), 1), 16.6)
        self.assertEqual(round(kurtosis(r, fisher=True, bias=False), 1), 6.4)
        self.assertEqual(round(skew(r, bias=False), 2), -.76)

    def test_independent_arch_benchmark_matches_required_precision(self):
        from src.tutorial1 import load_tutorial_data, package_fit, package_parameters
        from src.garch_mle import fit_garch11
        root = Path(__file__).resolve().parents[1]
        r, _ = load_tutorial_data(root/"NLNPE_TD_data.xlsx")
        custom = fit_garch11(r)
        fitted = package_fit(r, backcast=float(np.var(r, ddof=0)))
        package = package_parameters(fitted)
        expected_first_variance = package["omega"] + (package["alpha"] + package["beta"]) * np.var(r, ddof=0)
        self.assertAlmostEqual(fitted.conditional_volatility.iloc[0] ** 2, expected_first_variance, places=8)
        for name, estimate in custom["estimates"].items():
            self.assertEqual(round(estimate, 2), round(package[name], 2), name)


if __name__ == "__main__":
    unittest.main()
