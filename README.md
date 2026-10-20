# Tutorial 1: GARCH audit and leverage

This project covers GARCH estimation, diagnostics, leverage, and news impact curves.
The one-page report is written in `report.tex`; Python reproduces the calculations
and figure, then compiles the manuscript without generating its prose.

```powershell
uv sync
uv run python main.py
```

Outputs are in the repository root: `report.pdf`, `news_impact.pdf` and
`results.json`. The report PDF and input workbook are included in the repository.
Other PDFs, generated results, caches, and `prompt_log.md` are excluded from Git.

The included `NLNPE_TD_data.xlsx` is the instructor's workbook supplied by the
user on 16 October 2026. It replaces the earlier reconstructed input.
The analysis recomputes 2,387 weekly log returns, multiplied by 100, from
the S&P 500 closes (1981-01-09 to 2026-10-02), checks them against the supplied
return column, and matches all five published checks at displayed precision.
The workbook also contains its README, data dictionary and daily data sheet.
Its SHA-256 is recorded in `results.json` to identify the input used.

`garch_mle.py` implements Gaussian constant-mean GARCH without `arch`. It uses
residuals, exponential/softmax constraints and BHHH standard errors from the
inverse sum of observation-score outer products, with the delta method.
The default initial variance is the fixed sample variance (divisor T).
The alternative long-run initial variance is recalculated at every evaluation.
`arch` is used only for the independent benchmark and GJR estimation; GJR
inference uses robust covariance.
The GARCH benchmark passes the sample variance (divisor T) as `backcast`.
This initializes presample terms, so its first variance differs from the fixed
sample variance used by our estimator. The GJR fit keeps its default backcast.

```powershell
uv run python main.py --results-only
uv run python main.py --results-only --initialization long-run
uv run python -m unittest discover -s tests -v
```

PDF compilation requires Tectonic on PATH, or the existing compiler at
`../garch_course/tools/tectonic.exe`. Numerical calculations do not require it.
`report_evidence.json` records the results discussed in the manuscript. If a
different sample or initialization changes them, review the manuscript and
its evidence before compiling; the script prevents compiling stale prose.
