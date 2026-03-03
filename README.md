# SSM Private Equity NAV Nowcasting

A production-grade Go implementation of the state-space model (SSM) for nowcasting private equity fund net asset values (NAV) at weekly frequency.

Based on: **Brown, Ghysels & Gredil (2022)** — *"Nowcasting Net-Asset-Values: The Case of Private Equity"*, SSRN 3507873.

## Overview

Private equity funds report NAV quarterly with significant lags. This model uses a 4-state Kalman filter to estimate weekly unsmoothed NAV by combining:

- **Quarterly NAV reports** (sparse, lagged, smoothed)
- **Public market index returns** (weekly, systematic risk)
- **Comparable ETP returns** (weekly, idiosyncratic signal)
- **GARCH(1,1) conditional variance** from proxy residuals

The model jointly estimates NAV smoothing intensity (λ), systematic risk (β), idiosyncratic volatility (F), and other parameters via maximum-likelihood with L-BFGS and complex-step analytic gradients.

## Project Structure

```
ssm_go_project/
├── cmd/
│   └── ssm/
│       └── main.go              # Entry point, CLI, parallel orchestration
├── internal/
│   ├── config/
│   │   └── config.go            # Constants, bounds, env configuration
│   ├── csvio/
│   │   ├── reader.go            # CSV parsing, weekly resampling
│   │   └── writer.go            # Output CSV writing
│   ├── mathutil/
│   │   └── mathutil.go          # Numeric helpers, OLS, matrix ops
│   ├── timeutil/
│   │   └── timeutil.go          # Date parsing, weekly grids
│   ├── garch/
│   │   └── garch.go             # GARCH(1,1) fitting via L-BFGS
│   ├── kalman/
│   │   ├── filter.go            # Kalman NLL (real + complex-step)
│   │   └── smoother.go          # Forward filter + RTS smoother
│   ├── mle/
│   │   └── estimator.go         # MLE with outer β_c fixed-point
│   └── pipeline/
│       ├── prepare.go           # Data loading and panel construction
│       └── compute.go           # Per-asset SSM return computation
├── inputs/                       # Input CSV data files
├── outputs/                      # Generated output files
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

### Package Responsibilities

| Package | Role |
|---------|------|
| `config` | All tunable constants, parameter bounds, environment variable parsing |
| `csvio` | CSV I/O, weekly resampling, log-return computation |
| `mathutil` | `IsFinite`, `CleanReturnSeries`, `CleanHt`, `OLSBetaAlpha`, `Sigmoid`, bound mapping, 4×4 matrix ops |
| `timeutil` | Date parsing, Friday-grid construction, as-of index lookup, window computation |
| `garch` | GARCH(1,1) estimation with backcast initialisation and complex-step gradients |
| `kalman` | 4-state Kalman filter NLL (real and complex), forward filter + RTS backward smoother |
| `mle` | Maximum-likelihood estimation with L-BFGS, bounded parameters, outer β_c loop |
| `pipeline` | End-to-end data preparation and per-asset SSM computation |

## Requirements

- **Go 1.22+** (tested with 1.25.1)
- **gonum** v0.17.0 (optimisation library)

## Quick Start

```bash
# Build
make build

# Run with default as-of date (2025-11-14)
make run

# Run with custom date
make run ASOF=2025-06-30

# Or directly:
go build -o ssm_go ./cmd/ssm
SSM_ASOF_DATE=2025-11-14 ./ssm_go
```

## Configuration

All runtime settings are controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SSM_ASOF_DATE` | `2025-11-14` | Target valuation date |
| `SSM_WORKERS` | `NumCPU()` | Parallel worker count |
| `SSM_DEBUG` | *(empty)* | Enable debug logging (any non-empty value) |

## Input Files

Place these in the `inputs/` directory:

| File | Description |
|------|-------------|
| `private_assets_prices_quarterly_wide.csv` | Quarterly NAV observations (date × asset wide format) |
| `etp_and_index_mapping_table.csv` | Asset → ETP proxy → market index mapping with initial parameters |
| `etp_proxy_idio_daily.csv` | Daily ETP proxy prices |
| `index_prices_daily.csv` | Daily broad market index prices |

## Output

The model writes a single CSV to `outputs/ssm_returns_YYYY-MM-DD.csv` containing weekly log-returns for each asset within the evaluation window.

## Algorithm Summary

For each asset at the target date:

1. **Data windowing**: Extract calibration (2000 trading days) and evaluation (500 trading days) windows
2. **GARCH(1,1)**: Fit conditional variance on proxy residuals (proxy − β·market) with complex-step gradients
3. **Rolling OLS**: Estimate α, β from comparable vs. market returns over the calibration window
4. **MLE estimation**: Optimise 8 SSM parameters via L-BFGS with:
   - Sigmoid-bounded parameter space
   - Complex-step analytic gradients (machine-precision, no finite-difference noise)
   - Outer fixed-point iteration for β_c (comparable-to-fund loading)
5. **Kalman filter**: Forward pass with dual observation equations (NAV + comparable return)
6. **RTS smoother**: Backward pass for optimal smoothed state estimates
7. **NAV anchoring**: Shift smoothed log-NAV to match the last observed quarterly NAV before the evaluation window
8. **Return extraction**: Compute weekly log-return differences from the anchored smoothed states

## Performance Notes

- **Parallelism**: Assets are processed concurrently with a bounded semaphore (default: all CPU cores)
- **Memory**: Pre-allocated scratch buffers per goroutine avoid allocation pressure in the inner loop
- **Gradients**: Complex-step differentiation provides machine-precision gradients without the instability of finite differences
- **Matrix ops**: All 4×4 Kalman covariance operations use stack-allocated fixed-size arrays (zero heap allocation in hot path)
- **Bound mapping**: Sigmoid transform converts bounded optimisation to unconstrained L-BFGS, matching scipy's L-BFGS-B behaviour

## License

Internal / proprietary. Not for redistribution.
