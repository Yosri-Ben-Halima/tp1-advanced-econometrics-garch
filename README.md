# Tutorial 1: GARCH audit and leverage

This project covers GARCH estimation, diagnostics, leverage, and news impact curves.
The completed one-page report is included as `report/report.pdf`.
Run `main.py` to reproduce the numerical analysis and news impact curve values.
Analysis modules are under `src/`; report files are under `report/`.

```powershell
uv sync
uv run python main.py
```

Running the analysis writes `report/results.json`.
The report source (`report/report.tex`, `report/report_render.py`,
`report/report_evidence.json`), the report PDF, the annotated prompt log
(`context/prompt_log.md`) and the input workbook are included in the repository.
`report/deliverable.pdf` is the printed hand-in: the one-page report followed by
the prompt log, the output of `main.py` and code excerpts. Rebuild it with a
LaTeX compiler (`tectonic`) on the PATH:

```powershell
uv run python -c "from src.tutorial1 import analyze, load_tutorial_data; from report.report_render import compile_report; r, p = load_tutorial_data('data/NLNPE_TD_data.xlsx'); compile_report(analyze(r, p)[0], 'report')"
uv run python report/build_deliverable.py
```

The included `data/NLNPE_TD_data.xlsx` is the instructor's workbook supplied by the
user on 16 October 2026. It replaces the earlier reconstructed input.
The analysis recomputes 2,387 weekly log returns, multiplied by 100, from
the S&P 500 closes (1981-01-09 to 2026-10-02), checks them against the supplied
return column, and matches all five published checks at displayed precision.
The workbook also contains its README, data dictionary and daily data sheet.
Its SHA-256 is recorded in `report/results.json` to identify the input used.

`src/garch_mle.py` implements Gaussian constant-mean GARCH without `arch`. It uses
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
uv run python main.py --initialization long-run
uv run python -m unittest discover -s tests -v
```
