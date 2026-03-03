// Package config centralises all tunable constants, parameter bounds,
// and environment-driven settings for the SSM nowcasting system.
package config

import (
	"math"
	"os"
	"strconv"
	"strings"

	"ssm_go/internal/mathutil"
)

// ── Algorithm constants ─────────────────────────────────────────────────────

const (
	WindowDays    = 500   // rolling window for return extraction
	CalibDays     = 2000  // calibration window for MLE
	MinNAVForMLE  = 12    // minimum NAV observations for estimation
	GARCHMinObs   = 250   // minimum observations for GARCH fitting
	OuterMaxIter  = 6     // outer fixed-point iterations for β_c
	OuterTolBetaC = 1e-5  // convergence tolerance for β_c
	CSEps         = 1e-20 // complex-step epsilon for gradient
)

// ── Optimiser settings ──────────────────────────────────────────────────────

const (
	OptMaxIter = 400  // L-BFGS major iteration limit
	OptGTol    = 1e-6 // gradient norm convergence threshold
)

// ── Parameter bounds for the 8-dimensional θ vector ─────────────────────────

// Bound defines a box constraint [Lo, Hi] for one parameter.
type Bound struct {
	Lo, Hi float64
}

// ThetaBounds defines the 8 parameter bounds used by the Kalman MLE.
//
// Index mapping:
//
//	0: β_c  (comp-to-fund beta)       [0.01, 3.0]
//	1: ψ_c  (comp intercept)          [-0.20, 0.20]
//	2: logF (idiosyncratic vol scale)  [log(0.30), log(4.5)]
//	3: logσ_nav (NAV noise)            [log(1e-4), log(2.0)]
//	4: logF_c (comp idio vol scale)    [log(0.01), log(2.0)]
//	5: λ    (NAV smoothing)            [0.40, 0.99]
//	6: δ    (distribution AR(1))       [0.00, 0.995]
//	7: logσ_d (distribution noise)     [log(1e-6), log(5.0)]
var ThetaBounds = [8]Bound{
	{0.01, 3.0},
	{-0.20, 0.20},
	{math.Log(0.30), math.Log(4.5)},
	{math.Log(1e-4), math.Log(2.0)},
	{math.Log(0.01), math.Log(2.0)},
	{0.40, 0.99},
	{0.00, 0.995},
	{math.Log(1e-6), math.Log(5.0)},
}

// DefaultTheta returns sensible starting values for the 8-parameter vector.
// If lambdaInit is non-nil, it overrides the default λ initial value.
func DefaultTheta(lambdaInit *float64) [8]float64 {
	th := [8]float64{
		0.20, 0.00,
		math.Log(0.20), math.Log(0.10), math.Log(0.50),
		0.90, 0.50,
		math.Log(0.20),
	}
	if lambdaInit != nil && mathutil.IsFinite(*lambdaInit) {
		lam := *lambdaInit
		if lam < 0.40 {
			lam = 0.40
		}
		if lam > 0.99 {
			lam = 0.99
		}
		th[5] = lam
	}
	return th
}

// ── Runtime configuration ───────────────────────────────────────────────────

const DefaultAsofDate = "2025-11-14"

// RunConfig holds runtime settings parsed from the environment.
type RunConfig struct {
	BaseDir  string
	AsofDate string
	Workers  int
	Debug    bool
}

// LoadRunConfig reads configuration from the environment and working directory.
func LoadRunConfig(baseDir string) RunConfig {
	cfg := RunConfig{
		BaseDir:  baseDir,
		AsofDate: DefaultAsofDate,
		Workers:  0, // 0 means "use runtime.NumCPU()"
		Debug:    false,
	}
	if v := strings.TrimSpace(os.Getenv("SSM_ASOF_DATE")); v != "" {
		cfg.AsofDate = v
	}
	if strings.TrimSpace(os.Getenv("SSM_DEBUG")) != "" {
		cfg.Debug = true
	}
	if v := strings.TrimSpace(os.Getenv("SSM_WORKERS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Workers = n
		}
	}
	return cfg
}
