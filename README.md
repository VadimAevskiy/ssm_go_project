# SSM-Based Private Asset NAV Nowcasting

A high-performance Go implementation of State-Space Model (SSM) nowcasting for private asset Net Asset Values, adapted from the methodology of [Brown, Ghysels & Gredil (2022)](#references) and re-engineered for live portfolio monitoring of illiquid assets using publicly traded proxies.

## Motivation

Institutional portfolios holding private assets—private equity, private credit, real estate, infrastructure—face a persistent problem: reported NAVs arrive quarterly, with multi-week delays, and carry appraisal smoothing bias. Between reporting dates, investors are effectively blind to the true mark-to-market exposure of a significant share of their portfolio.

This project produces **weekly unsmoothed NAV estimates** for each private asset by fusing sparse quarterly NAV reports with high-frequency returns from matched exchange-traded proxies (ETPs) and market indices. The output is a panel of weekly log-returns suitable for risk attribution, allocation rebalancing, and regulatory reporting.

## Methodological Foundation

The core insight, due to Brown, Ghysels & Gredil (BGG), is that a private fund's true asset value can be modelled as a **latent state** in a linear Gaussian SSM, with two distinct observation channels:

1. **Quarterly NAV reports** — informative but smoothed and delayed.
2. **Comparable public asset returns** — available at high frequency but only partially correlated with the fund.

A Kalman filter optimally blends these two signals, weighting each by its estimated precision, to extract the unobservable true NAV path at the weekly frequency. See `docs/ssrn-3507873.pdf` for the full treatment.

## Adaptations from the Original Paper

This implementation departs from BGG in several deliberate ways, motivated by the practical requirements of a live portfolio system where fund-level cash flow data may not be available.

### 1. ETP-Based Comparable Assets (vs. Industry Benchmarks)

**BGG** matches each PE fund to a Fama-French 12 industry portfolio as the comparable asset.

**This implementation** matches each private asset to a specific exchange-traded product (ETP) — typically an ETF tracking the relevant sub-sector, geography, or strategy — paired with a broad market index.

**Rationale.** ETPs capture finer cross-sectional variation in risk exposures than broad industry portfolios. A European mid-cap infrastructure fund, for instance, is better proxied by a dedicated infrastructure ETF than by the FF "Utilities" bucket. The mapping table (`inputs/etp_and_index_mapping_table.csv`) makes this pairing explicit and auditable per asset.

### 2. NAV-Level State Vector (vs. Cumulative Returns)

**BGG** defines the latent state as the fund's cumulative log-return from inception, with a separate mapping function *M_t* converting between returns and asset values that must be iteratively re-estimated.

**This implementation** works directly in **log-NAV levels** with a four-dimensional state:

| State | Description |
|-------|-------------|
| η_t | Weekly idiosyncratic log-return shock |
| v\*_t | True (unsmoothed) log-NAV level |
| m_t | Reported (smoothed) log-NAV level |
| d_t | Distribution-intensity state |

**Rationale.** Operating in NAV levels eliminates the need for the iterative return-to-value mapping loop (BGG Section 3.3.1, Appendix A.2.1), which is a potential source of convergence issues and computational cost. The level parameterisation also simplifies the NAV-anchoring step (see §4 below), since the filter state is directly comparable to observed NAVs without back-transforming through *M_t*. The trade-off is a mild loss of structural interpretability for the cumulative-return decomposition, which is acceptable when the primary deliverable is a NAV time series rather than fund-level risk attribution.

### 3. OLS-Fixed (α, β) with Full MLE on Remaining Parameters

**BGG** profiles α and β on a 15×15 grid (225 evaluations), penalises via a PME-based pricing-error criterion, and iterates the remaining 8 parameters to convergence.

**This implementation** fixes α and β to their rolling OLS estimates from the comparable-asset regression over the calibration window, then runs **unconstrained L-BFGS** on the remaining 8-parameter vector {β_c, ψ_c, F, σ_nav, F_c, λ, δ, σ_d} in a single pass.

**Rationale.** The grid-profile approach is designed for settings where α and β are weakly identified from sparse cash-flow data. In our setting, the comparable-asset return series is observed weekly, so the OLS regression of proxy returns on the market index provides a well-identified and stable estimate of the systematic risk loading. Fixing (α, β) at these values avoids the combinatorial cost of the 225-point grid and focuses the optimiser's budget on the parameters that are genuinely hard to identify from quarterly NAV data — principally λ (smoothing intensity), F (idiosyncratic volatility scale), and σ_nav (reporting noise). In practice, this reduces per-asset estimation time by roughly an order of magnitude, enabling the full panel to be processed in seconds.

### 4. NAV Anchoring

**BGG** does not anchor filtered states to observed NAVs; the model output is self-consistent by construction via the return-to-value mapping.

**This implementation** applies a **level shift** to the smoothed state trajectory so that it passes through the most recent observed NAV at the boundary of the evaluation window.

**Rationale.** For portfolio reporting, it is essential that the nowcasted NAV series is consistent with the last audited figure. Without anchoring, numerical drift in the Kalman smoother can produce a level offset that, while statistically harmless for return estimation, creates an unexplainable discrepancy in reported portfolio value. The shift is applied uniformly across all time steps, preserving the return dynamics extracted by the filter.

### 5. Complex-Step Gradient Computation

**BGG** uses numerical Hessians (Miranda & Fackler, 2004) for standard errors but does not discuss gradient computation for the optimiser.

**This implementation** computes exact gradients via the **complex-step method**: each parameter is perturbed by iε in the complex plane, and the gradient is recovered as Im[f(θ + iε)] / ε. The entire Kalman filter NLL is implemented in both `float64` and `complex128` arithmetic for this purpose.

**Rationale.** The complex-step derivative is analytically exact to machine precision (no finite-difference truncation error) and requires no step-size tuning. This makes the L-BFGS optimiser substantially more robust on the ill-conditioned likelihood surfaces typical of sparse-data SSMs, where finite-difference gradients often trigger premature convergence or line-search failures.

### 6. No Cash-Flow or Distribution Data Required

**BGG** jointly models fund distributions and NAV reports, using the distribution channel to identify the smoothing function and the return-to-value mapping.

**This implementation** treats the distribution-intensity state *d_t* as latent rather than observed.

**Rationale.** Fund-level cash-flow data (capital calls, distributions) is frequently unavailable to the LP's risk team in a timely, machine-readable format. By relying solely on quarterly NAVs and the public proxy, this system can be deployed without waiting for cash-flow reconciliation. The distribution state is still present in the model to absorb low-frequency mean shifts in the comparable-asset relationship, but it is estimated from the NAV and proxy data alone, which is sufficient when the primary goal is NAV nowcasting rather than fund-level parameter recovery.

### 7. Outer Fixed-Point Loop for β_c

**BGG** estimates the comparable-asset parameters (β_c, ψ, F_c) jointly with the fund parameters inside the Kalman MLE.

**This implementation** wraps the MLE in an **outer fixed-point iteration** over β_c, the proxy-to-fund return loading. At each outer step, the proxy return series is adjusted for the current β_c estimate before the inner optimisation is run.

**Rationale.** β_c enters the observation equation nonlinearly (it scales the Kalman innovation for the comparable-return channel and simultaneously appears in the proxy-adjustment formula involving the market return lead). Iterating over β_c in an outer loop avoids the ill-conditioning that arises when the optimiser must simultaneously navigate the likelihood surface over β_c and the smoothing/noise parameters. Convergence is typically achieved in 2–3 outer iterations.

## Architecture

```
ssm_go_project/
├── cmd/ssm/main.go              # CLI entry point, parallel orchestration
├── internal/
│   ├── config/config.go         # Constants, bounds, environment config
│   ├── csvio/                   # CSV I/O and weekly resampling
│   ├── garch/garch.go           # GARCH(1,1) MLE for conditional variance
│   ├── kalman/
│   │   ├── filter.go            # 4-state Kalman filter (real + complex)
│   │   └── smoother.go          # Forward filter + RTS backward smoother
│   ├── mathutil/mathutil.go     # Numerics: OLS, 4×4 linalg, sigmoids
│   ├── mle/estimator.go         # L-BFGS parameter estimation
│   ├── pipeline/
│   │   ├── prepare.go           # Data ingestion and alignment
│   │   └── compute.go           # Per-asset SSM pipeline
│   └── timeutil/                # Date parsing and weekly grid construction
├── inputs/                      # Input data (CSV)
├── outputs/                     # Generated return panels (CSV)
├── docs/                        # Reference papers and presentations
├── go.mod
├── go.sum
└── Makefile
```

### Per-Asset Pipeline

For each private asset, `ComputeSSMReturns` executes:

1. **Window extraction** — Slice data to the as-of date; determine calibration and evaluation windows from the daily date grid.
2. **Rolling OLS** — Estimate (α, β) from proxy vs. market returns over the calibration window.
3. **MLE** — Estimate the 8-parameter θ vector via L-BFGS with complex-step gradients, iterated over the β_c fixed point.
4. **Kalman filter + RTS smoother** — Forward-filter the full history, then backward-smooth via Rauch-Tung-Striebel to produce the latent NAV path.
5. **NAV anchoring** — Shift the smoothed trajectory to pass through the last observed NAV.
6. **Return extraction** — Difference the anchored log-NAV series to obtain weekly log-returns.

All assets are processed in parallel across available CPU cores.

## Usage

### Prerequisites

- Go 1.21+
- Input CSVs in `inputs/` (see below)

### Build & Run

```bash
make run                          # default as-of date: 2025-11-14
make run ASOF=2026-03-03          # custom as-of date
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SSM_ASOF_DATE` | `2025-11-14` | Target nowcasting date (YYYY-MM-DD) |
| `SSM_WORKERS` | all CPUs | Parallelism level |
| `SSM_DEBUG` | (unset) | Enable debug logging |

### Input Data

| File | Description |
|------|-------------|
| `private_assets_prices_quarterly_wide.csv` | Quarterly NAVs, wide format (Date × Asset) |
| `etp_and_index_mapping_table.csv` | Asset → ETP proxy → Market index mapping |
| `etp_proxy_idio_daily.csv` | Daily ETP proxy prices |
| `index_prices_daily.csv` | Daily market index prices |

### Output

`outputs/ssm_returns_<YYYY-MM-DD>.csv` — Wide-format CSV of weekly log-returns (Date × Asset) for the evaluation window ending at the as-of date.

## Model Parameters

The 8-dimensional θ vector estimated by MLE:

| Index | Parameter | Description | Bounds |
|-------|-----------|-------------|--------|
| 0 | β_c | Proxy-to-fund idiosyncratic return loading | [0.01, 3.0] |
| 1 | ψ_c | Proxy return intercept | [−0.20, 0.20] |
| 2 | log F | Idiosyncratic volatility scale | [log 0.3, log 4.5] |
| 3 | log σ_nav | NAV reporting noise | [log 1e-4, log 2.0] |
| 4 | log F_c | Proxy idiosyncratic vol. scale | [log 0.01, log 2.0] |
| 5 | λ | NAV smoothing intensity | [0.40, 0.99] |
| 6 | δ | Distribution AR(1) coefficient | [0.00, 0.995] |
| 7 | log σ_d | Distribution-intensity noise | [log 1e-6, log 5.0] |

## References

- **Brown, G. W., Ghysels, E., & Gredil, O. (2022).** *Nowcasting Net Asset Values: The Case of Private Equity.* Review of Financial Studies. [SSRN 3507873](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3507873). See `docs/ssrn-3507873.pdf`.

Additional reference materials are available in the `docs/` folder.

## License

Internal use. Not for redistribution without permission.
