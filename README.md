# SSM-Based Private Asset NAV Nowcasting

A high-performance Go implementation of State-Space Model (SSM) nowcasting for private asset Net Asset Values. The system produces weekly unsmoothed NAV estimates for illiquid assets by fusing sparse quarterly NAV reports with high-frequency returns from matched exchange-traded proxies and market indices.

The underlying methodology builds on [Brown, Ghysels and Gredil (2022)](#references), with several material extensions described in [Key Departures from BGG (2022)](#key-departures-from-bgg-2022).

## Motivation

Institutional portfolios holding private assets (private equity, private credit, real estate, infrastructure) face a persistent problem: reported NAVs arrive quarterly, with multi-week delays, and carry appraisal smoothing bias. Between reporting dates, investors are effectively blind to the true mark-to-market exposure of a significant share of their portfolio.

This creates concrete downstream failures. Portfolio VaR and volatility estimates understate true risk because stale NAVs artificially suppress measured variance and correlation with public markets. Allocation weights drift undetected between rebalancing decisions, producing unintended factor exposures. Solvency and capital adequacy calculations that rely on quarter-end snapshots can be materially stale within weeks of the reporting date.

The system addresses this by producing **weekly log-return estimates** for each private asset, requiring only quarterly NAV data and daily prices for a matched public proxy and market index. No fund-level cash flow or distribution data is needed.

## Input Data

All input files are placed in the `inputs/` directory:

| File | Format | Description |
|------|--------|-------------|
| `private_assets_prices_quarterly_wide.csv` | Date x Asset | Quarterly NAV prices. Missing values treated as unobserved. |
| `etp_and_index_mapping_table.csv` | Row per asset | Links each asset to its ETP proxy and market index. Required columns: `asset_label`, `etp_ticker`, `broad_index_ticker`. Optional: `lambda_init`. |
| `etp_proxy_idio_daily.csv` | Date x Ticker | Daily prices for all ETP proxies referenced in the mapping table. |
| `index_prices_daily.csv` | Date x Ticker | Daily prices for all market indices referenced in the mapping table. |

## Project Structure

```
ssm_go_project/
|-- cmd/ssm/main.go              # Entry point, parallel orchestration, output assembly
|-- internal/
|   |-- config/config.go         # Constants, parameter bounds, environment config
|   |-- csvio/
|   |   |-- reader.go            # CSV reading, column lookup
|   |   |-- writer.go            # Wide-format CSV output, weekly resampling
|   |-- garch/garch.go           # GARCH(1,1) conditional variance (separate MLE, pre-Kalman)
|   |-- kalman/
|   |   |-- filter.go            # 4-state Kalman filter NLL (float64 + complex128)
|   |   |-- smoother.go          # Forward Kalman filter + RTS backward smoother
|   |-- mathutil/mathutil.go     # OLS, 4x4 matrix ops, bound maps, sigmoids, series cleaning
|   |-- mle/estimator.go         # L-BFGS parameter estimation for the 8 SSM parameters
|   |-- pipeline/
|   |   |-- prepare.go           # Data ingestion, alignment, GARCH pre-estimation
|   |   |-- compute.go           # Per-asset SSM pipeline
|   |-- timeutil/                # Date parsing, weekly Friday grid, cutoff logic
|-- inputs/                      # Input CSV files
|-- outputs/                     # Generated return panels
|-- docs/                        # Reference papers and presentations
|-- go.mod / go.sum              # Go module (depends on gonum for L-BFGS)
|-- Makefile                     # build, run, clean targets
```

## Data Preparation

Before any per-asset estimation, `PrepareFullData` reads all inputs and builds aligned weekly panels:

1. **Read quarterly NAVs** into a date-keyed map per asset. Values are converted to log-levels internally.
2. **Read the mapping table** linking each asset to its proxy ticker, market index ticker, and optional lambda_init.
3. **Read daily proxy and index prices** into per-ticker date-keyed maps.
4. **Build a weekly Friday grid.** All daily prices are resampled to weekly frequency (last observation on or before each Friday). Weekly log-returns are computed from the resampled prices.
5. **Compute comparable-to-market betas.** For each proxy ticker, beta_c2m is estimated by OLS of the proxy weekly returns on the market index (minimum 60 observations).
6. **Estimate GARCH conditional variance.** For each asset, OLS residuals of the proxy on the market are fit with GARCH(1,1) via its own separate MLE to produce the time-varying h_t series. This is a pre-estimation step run once per asset before the Kalman filter MLE begins. Falls back to constant sample variance if GARCH fails or has insufficient data (< 250 observations).
7. **Compute full-sample OLS alpha and beta** for each asset from the proxy-on-market regression.
8. **Assemble base panels.** All series (log-NAVs, masks, returns, h_t, OLS estimates, beta_c2m) are aligned to a unified date axis merging the weekly grid with NAV reporting dates.

## Model

### State-Space Formulation

The filter tracks four latent states at each weekly time step t:

| State | Symbol | Description |
|-------|--------|-------------|
| Idiosyncratic return | eta_t | Weekly idiosyncratic log-return shock |
| True log-NAV | v*_t | Unsmoothed (mark-to-market) log-NAV level |
| Reported log-NAV | m_t | Smoothed (reported) log-NAV level |
| Distribution intensity | d_t | Low-frequency mean shift in the proxy relationship |

**Transition equations:**

```
eta_t = alpha + beta * r_m,t + F * sqrt(h_t) * eps_t
v*_t  = v*_{t-1} + eta_t
m_t   = (1 - lambda) * v*_t + lambda * m_{t-1}
d_t   = delta * d_{t-1} + sigma_d * sqrt(h_t) * eps_d,t
```

The smoothing equation for m_t is the critical structural assumption: when lambda is close to 1, the fund manager places heavy weight on the prior reported value, producing the artificially smooth return series characteristic of private assets. The filter estimates lambda jointly with the other parameters and uses it to recover the true v*_t.

**Observation equations** (fire only when data is available at t):

NAV observation (quarterly):
```
y_nav,t = m_t + sigma_nav * eps_nav,t
```

Comparable-return observation (weekly):
```
r_c,t = beta_c * eta_t + psi_c + d_t + F_c * sqrt(h_t) * eps_c,t
```

All noise variances scale with the time-varying GARCH variance h_t, allowing the model to adapt to changing market regimes.

### GARCH Conditional Variance

The h_t series is estimated from proxy-market OLS residuals using GARCH(1,1):

```
u_t = r_proxy,t - (alpha_ols + beta_ols * r_market,t)
h_t = omega + alpha_g * u^2_{t-1} + beta_g * h_{t-1}
```

The GARCH parameters are estimated via their own separate MLE (also using L-BFGS with complex-step gradients), run once per asset during data preparation, before the Kalman filter MLE begins. Initial variance uses the ARCH backcast convention. The resulting h_t series is cleaned: NaN gaps are interpolated, non-positive values replaced with the series median, and all values floored at 1e-10.

### Maximum Likelihood Estimation

The 8 free SSM parameters are estimated by minimising the Kalman filter negative log-likelihood using **L-BFGS** (gonum/optimize). Gradients are computed via the **complex-step method**: each parameter is perturbed by i*eps (eps = 1e-20) in the complex plane, and the gradient is recovered as Im[f(theta + i*eps)] / eps. This requires a full complex128 implementation of the filter NLL but produces derivatives exact to machine precision with no step-size tuning.

All parameters are mapped to unconstrained space via sigmoid transforms before optimisation, ensuring the likelihood is never evaluated at inadmissible values (e.g. negative variances).

Because beta_c enters both the likelihood and the proxy-adjustment formula that prepares the input data, the estimation iterates internally: estimate theta, update the proxy adjustment with the new beta_c, re-estimate theta, repeating until beta_c stabilises (tolerance 1e-5, typically 2-3 iterations, maximum 6).

### Kalman Filter and RTS Smoother

The forward Kalman filter processes both observation channels sequentially at each time step: first the NAV update (if observed), then the comparable-return update (if available). The state is initialised at v*_0 = m_0 = mean of observed log-NAVs with a large diagonal covariance (1e4), reflecting diffuse prior uncertainty.

After the forward pass, a **Rauch-Tung-Striebel backward smoother** refines each state estimate using future observations. This is critical because quarterly NAV reports retroactively sharpen the weekly estimates between reporting dates.

The output is the smoothed true log-NAV path v*_{t|T}.

### NAV Anchoring

A uniform level shift is applied so the smoothed trajectory passes through the most recent observed NAV:

```
shift = log(NAV_anchor) - v*_smooth(t_anchor)
v*_anchored(t) = v*_smooth(t) + shift   for all t
```

This preserves the return dynamics (period-to-period differences) while ensuring consistency with the last audited figure. Weekly log-returns are then: r_t = v*_anchored(t) - v*_anchored(t-1).

### Per-Asset Pipeline Summary

For each asset, `ComputeSSMReturns` executes:

1. **Window extraction**: slice data to the as-of date; determine calibration (~2,000 days) and evaluation (~500 days) windows.
2. **NAV sufficiency check**: require at least 12 NAV observations; skip otherwise.
3. **Rolling OLS**: re-estimate (alpha, beta) over the calibration window (minimum 60 pairs; fall back to full-sample).
4. **MLE**: estimate theta via L-BFGS with complex-step gradients.
5. **Kalman filter + RTS smoother**: forward filter then backward smooth to produce v*_{t|T}.
6. **NAV anchoring**: level-shift to the last observed NAV.
7. **Return extraction**: difference the anchored log-NAV series within the evaluation window.

All assets are processed in parallel (goroutines bounded by SSM_WORKERS, default: all CPUs).

## Parameters

### Estimated Parameters (8-dimensional theta)

| Index | Parameter | Description | Bounds |
|-------|-----------|-------------|--------|
| 0 | beta_c | Proxy-to-fund idiosyncratic return loading | [0.01, 3.0] |
| 1 | psi_c | Proxy return intercept | [-0.20, 0.20] |
| 2 | log(F) | Idiosyncratic volatility scale | [log(0.3), log(4.5)] |
| 3 | log(sigma_nav) | NAV reporting noise | [log(1e-4), log(2.0)] |
| 4 | log(F_c) | Proxy observation noise scale | [log(0.01), log(2.0)] |
| 5 | lambda | NAV smoothing intensity | [0.40, 0.99] |
| 6 | delta | Distribution AR(1) persistence | [0.00, 0.995] |
| 7 | log(sigma_d) | Distribution-intensity noise | [log(1e-6), log(5.0)] |

### Fixed Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| alpha | Rolling OLS (calibration window) | Jensen's alpha of comparable on market |
| beta | Rolling OLS (calibration window) | Market beta of comparable on market |

### Initial Values

| Parameter | Initial | Rationale |
|-----------|---------|-----------|
| beta_c | 0.20 | Conservative proxy-to-fund loading |
| psi_c | 0.00 | No assumed return differential |
| F | log(0.20) | Moderate idiosyncratic vol. relative to GARCH |
| sigma_nav | log(0.10) | Meaningful reporting noise |
| F_c | log(0.50) | Reflects imperfect proxy match |
| lambda | 0.90 | Strong smoothing, typical for PE appraisals |
| delta | 0.50 | Moderate distribution persistence |
| sigma_d | log(0.20) | Moderate distribution noise |

Lambda can be overridden per asset via the `lambda_init` column in the mapping table. If the optimiser fails from a warm-started point, it retries from these defaults. If that also fails, the asset is excluded.

### Algorithm Constants

| Constant | Value | Description |
|----------|-------|-------------|
| WindowDays | 500 | Evaluation window (trading days) |
| CalibDays | 2,000 | Calibration window for MLE |
| MinNAVForMLE | 12 | Minimum NAV obs. to attempt estimation |
| GARCHMinObs | 250 | Minimum obs. for GARCH fitting |
| OuterMaxIter | 6 | Max beta_c iterations |
| OuterTolBetaC | 1e-5 | beta_c convergence tolerance |
| CSEps | 1e-20 | Complex-step perturbation |
| OptMaxIter | 400 | L-BFGS iteration limit |
| OptGTol | 1e-6 | Gradient norm threshold |

## Output

```
outputs/ssm_returns_<YYYY-MM-DD>.csv
```

Wide-format CSV (Date x Asset) of weekly log-returns over the evaluation window. Only assets passing the minimum NAV threshold are included.

## Usage

### Prerequisites

- Go 1.21+
- Input CSVs in `inputs/`

### Build and Run

```bash
make run                          # default as-of: 2025-11-14
make run ASOF=2026-03-03          # custom as-of date
make build                        # compile only
make clean                        # remove binary
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| SSM_ASOF_DATE | 2025-11-14 | Target nowcasting date (YYYY-MM-DD) |
| SSM_WORKERS | all CPUs | Parallelism level |
| SSM_DEBUG | (unset) | Enable debug logging |

## Key Departures from BGG (2022)

The model builds on Brown, Ghysels and Gredil (2022). Six design decisions depart from the original paper, each motivated by production deployment requirements.

**1. ETP-based comparable assets.** BGG matches funds to Fama-French 12 industry portfolios. This implementation uses asset-specific ETPs (sector ETFs, strategy ETFs) paired with broad indices, capturing finer cross-sectional risk variation. The mapping is explicit and auditable per asset.

**2. NAV-level state vector.** BGG tracks cumulative log-returns with a separate mapping function M_t converting between returns and asset values. This implementation works directly in log-NAV levels, eliminating the iterative return-to-value mapping loop (BGG Section 3.3.1, Appendix A.2.1) and simplifying NAV anchoring. The trade-off is a mild loss of interpretability for cumulative-return decomposition, acceptable when the deliverable is a NAV series.

**3. OLS-fixed (alpha, beta).** BGG profiles alpha and beta on a 15x15 grid (225 evaluations) with a PME-based penalty. Here they are fixed to rolling OLS estimates from weekly proxy-on-market regressions, which are well-identified at weekly frequency. This avoids the grid cost and focuses the optimiser on the parameters genuinely hard to identify from sparse NAV data (lambda, F, sigma_nav), reducing per-asset estimation time by roughly an order of magnitude.

**4. NAV anchoring.** BGG does not anchor filtered states; the output is self-consistent via the return-to-value mapping. This implementation applies a uniform level shift so the smoothed trajectory passes through the last audited NAV, ensuring consistency in portfolio reporting without affecting estimated returns.

**5. Complex-step gradients.** BGG uses numerical Hessians for standard errors but does not discuss gradient computation. This implementation uses complex-step derivatives (perturbation i*eps, eps = 1e-20), which are exact to machine precision and require no step-size tuning. The entire Kalman NLL is dual-implemented in float64 and complex128 for this purpose.

**6. No cash-flow data required.** BGG jointly models distributions and NAVs. This implementation treats the distribution state d_t as fully latent, estimated from NAV and proxy data alone. This allows deployment without waiting for fund-level cash-flow reconciliation, which is frequently unavailable in timely machine-readable format.

## References

- **Brown, G. W., Ghysels, E., and Gredil, O. (2022).** *Nowcasting Net Asset Values: The Case of Private Equity.* Review of Financial Studies, 36(3), 1093-1140. [SSRN 3507873](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3507873). See `docs/ssrn-3507873.pdf`.
- **Miranda, M. J. and Fackler, P. L. (2004).** *Applied Computational Economics and Finance.* MIT Press.

Additional materials in `docs/`:

| File | Description |
|------|-------------|
| ssrn-3507873.pdf | BGG original paper |
| Private_Assets_KF_Model.pdf | Model overview and implementation notes |
| Flow.pdf | Pipeline flow documentation |
| kalman_filter.webp | Kalman filter diagram |

## License

Internal use. Not for redistribution without permission.
