# Tutorial 1 — prompt log

## Block A — GARCH implementation, audit and benchmark

| User prompt / request | Relevant response and audit outcome |
|---|---|
| “Write a Python function that estimates a GARCH(1,1) with a constant mean on weekly returns in percent, by maximum likelihood, without the arch package. State explicitly: how the first conditional variance is initialised, how positivity and stationarity are enforced, how standard errors are computed. Return estimates, standard errors and the maximised log-likelihood.” | **Correct** Gaussian ML implementation with demeaned residuals, fixed sample-variance initialization and parameter transformations. **Wrong against checkpoint 5:** the initial numerical observed-Hessian covariance was valid in principle but did not use the specified expected-information/BHHH methods. |
| “does the implementation check these checkpoints ?” | Audit found checkpoints 1–4 and 6 satisfied under the stated assumptions. Checkpoint 5 required correction. Softmax constraints exclude exact zero values of alpha and beta. |
| “adjust to match checkpoint 5” | Replaced the covariance calculation with BHHH: inverse of the sum of observation-score outer products, including the constant mean and recursive variance derivatives, then the delta method. Verified against numerical scores. |
| “is it long run var or sample var” | **Correct:** initialization uses the fixed sample variance, divisor T. It is distinct from the fitted long-run variance `omega / (1 - alpha - beta)`. Earlier quoted values belonged to earlier samples. |
| “make both possible, up to user's selection” | Added sample and long-run initialization choices. Long-run initialization is recalculated during optimization and its derivatives enter BHHH. This was a later enhancement, not part of the initial implementation. |
| “add student tests 5% confidence (add a * for significant coefs)” | Added two-sided Wald statistics with a Student-t reference and T−4 degrees of freedom. **Correct only as a labelled approximation:** these are not exact finite-sample Student tests. GJR inference in the report uses robust asymptotic Wald inference. |
| “by "arch estimate" you mean the garch implementation of arch module ? also add relative difference column in %” | Clarified that the benchmark uses the Python `arch` package's Gaussian constant-mean GARCH(1,1). Added absolute relative differences, using the absolute package estimate as denominator and unrounded values for calculation. |
| “why is Standard errors a "pass" when in first implementation it failed to use BHHH” | **Corrected audit presentation:** distinguish the initial failure from the corrected estimator. The final audit identifies the initial standard-error choice as wrong and documents the BHHH correction. |
| **Summary:** explain the initialization difference between our estimator and `arch`, try a sample-variance backcast, then update the report. | `arch` uses a weighted backcast by default. Adopted sample variance as its benchmark backcast, but this initializes presample terms: its first fitted variance is `omega + (alpha + beta) * backcast`, subject to variance bounds. It therefore still differs from fixing our first variance to the sample variance. Updated the benchmark and evidence. |

## Block B — asymmetry and news-impact curves

| Prompt / request | Relevant response and qualification |
|---|---|
| **Assignment prompt, summary (slide 14):** derive and plot GARCH/GJR news-impact curves for standardized shocks from −4 to 4, evaluated at unconditional variance, and explain the fixed state. | For each fitted model, fix the current variance at its own unconditional variance. GARCH uses `h_bar = omega / (1 - alpha - beta)` and `NIC(z) = omega + (alpha*z² + beta)*h_bar`. GJR under symmetric Gaussian innovations uses `h_bar = omega / (1 - alpha - beta - gamma/2)` and `NIC(z) = omega + (alpha + gamma*1(z<0))*z²*h_bar + beta*h_bar`. The extra GJR term vanishes for positive shocks, but separately fitted curves need not coincide there. |
