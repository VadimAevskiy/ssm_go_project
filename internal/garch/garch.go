// Package garch implements GARCH(1,1) conditional variance estimation
// using maximum likelihood with L-BFGS optimisation and complex-step
// analytic gradients.
package garch

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/optimize"

	"ssm_go/internal/config"
	"ssm_go/internal/mathutil"
)

// Fit holds the estimated GARCH(1,1) parameters and conditional variances.
type Fit struct {
	Mu      float64
	Omega   float64
	Alpha   float64
	Beta    float64
	CondVar []float64 // h_t for t = 0..T-1
}

// ── Internal helpers ────────────────────────────────────────────────────────

// archBackcast computes the backcast initial variance per ARCH convention.
func archBackcast(y []float64, mu float64) float64 {
	n := len(y)
	if n == 0 {
		return 1e-6
	}
	tau := math.Min(75.0, float64(n)) / 100.0
	sumW := 1.0 - math.Pow(tau, float64(n))
	if sumW <= 0 {
		sumW = 1.0
	}
	v0 := 0.0
	wt := (1.0 - tau) / sumW
	for t := n - 1; t >= 0; t-- {
		d := y[t] - mu
		v0 += wt * d * d
		wt *= tau
	}
	if !mathutil.IsFinite(v0) || v0 <= 1e-10 {
		return 1e-6
	}
	return v0
}

// nllReal computes the GARCH(1,1) negative log-likelihood (real-valued).
func nllReal(p []float64, y []float64, v0 float64) float64 {
	mu := p[0]
	omega := math.Exp(p[1])
	alpha := mathutil.Sigmoid(p[2]) * 0.999
	beta := (0.999 - alpha) * mathutil.Sigmoid(p[3])
	n := len(y)
	h := v0
	ll := 0.0
	const log2pi = 1.8378770664093453
	for t := 0; t < n; t++ {
		eps := y[t] - mu
		if h < 1e-12 {
			h = 1e-12
		}
		ll += -0.5 * (log2pi + math.Log(h) + eps*eps/h)
		h = omega + alpha*eps*eps + beta*h
	}
	return -ll
}

// nllComplex computes the GARCH(1,1) NLL using complex arithmetic
// to support complex-step gradient evaluation.
func nllComplex(p [4]complex128, y []float64, v0 float64) complex128 {
	mu := p[0]
	omega := cmplx.Exp(p[1])
	alpha := mathutil.SigmoidC(p[2]) * 0.999
	beta := (0.999 - alpha) * mathutil.SigmoidC(p[3])
	n := len(y)
	h := complex(v0, 0)
	ll := complex(0, 0)
	const log2pi = 1.8378770664093453
	for t := 0; t < n; t++ {
		eps := complex(y[t], 0) - mu
		if cmplx.Abs(h) < 1e-12 {
			h = complex(1e-12, 0)
		}
		ll += -0.5 * (complex(log2pi, 0) + cmplx.Log(h) + (eps*eps)/h)
		h = omega + alpha*(eps*eps) + beta*h
	}
	return -ll
}

// ── Public API ──────────────────────────────────────────────────────────────

// FitNormal estimates a GARCH(1,1) model with Gaussian innovations
// on a centred & scaled return series. Returns the fitted model or an error.
func FitNormal(yScaled []float64) (*Fit, error) {
	n := len(yScaled)
	if n < config.GARCHMinObs {
		return nil, fmt.Errorf("not enough observations for GARCH (%d < %d)", n, config.GARCHMinObs)
	}

	mu0 := 0.0
	for _, v := range yScaled {
		mu0 += v
	}
	mu0 /= float64(n)
	v0 := archBackcast(yScaled, mu0)

	alpha0 := 0.10
	beta0 := 0.85
	omega0 := math.Max(v0*(1.0-alpha0-beta0), 1e-8)
	p2init := math.Log((alpha0 / 0.999) / (1.0 - alpha0/0.999))
	bfrac0 := beta0 / (0.999 - alpha0)
	p3init := math.Log(bfrac0 / (1.0 - bfrac0))
	p0 := []float64{mu0, math.Log(omega0), p2init, p3init}

	obj := func(p []float64) float64 { return nllReal(p, yScaled, v0) }
	grad := func(g, p []float64) {
		var pc [4]complex128
		for i := 0; i < 4; i++ {
			pc[i] = complex(p[i], 0)
		}
		for k := 0; k < 4; k++ {
			pp := pc
			pp[k] += complex(0, config.CSEps)
			g[k] = imag(nllComplex(pp, yScaled, v0)) / config.CSEps
		}
	}

	prob := optimize.Problem{Func: obj, Grad: grad}
	res, err := optimize.Minimize(prob, p0,
		&optimize.Settings{MajorIterations: 500, GradientThreshold: 1e-6},
		&optimize.LBFGS{})
	if err != nil && (res == nil || res.X == nil) {
		return nil, fmt.Errorf("GARCH optimisation failed: %w", err)
	}
	if res == nil || res.X == nil {
		return nil, fmt.Errorf("GARCH: no result")
	}

	pp := res.X
	muH := pp[0]
	omegaH := math.Exp(pp[1])
	alphaH := mathutil.Sigmoid(pp[2]) * 0.999
	betaH := (0.999 - alphaH) * mathutil.Sigmoid(pp[3])

	condVar := make([]float64, n)
	h := v0
	for t := 0; t < n; t++ {
		eps := yScaled[t] - muH
		if h < 1e-12 {
			h = 1e-12
		}
		condVar[t] = h
		h = omegaH + alphaH*eps*eps + betaH*h
	}
	return &Fit{Mu: muH, Omega: omegaH, Alpha: alphaH, Beta: betaH, CondVar: condVar}, nil
}

// ComputeFundHt estimates the fund-level idiosyncratic conditional variance
// series from the proxy and market daily return series.
//
// Steps: (1) OLS residuals proxy − β·market, (2) GARCH(1,1) on residuals,
// (3) fallback to constant variance if GARCH fails.
func ComputeFundHt(rProxyRaw, rMktRaw []float64) ([]float64, error) {
	rp := mathutil.CleanReturnSeries(rProxyRaw)
	rm := mathutil.CleanReturnSeries(rMktRaw)
	a, b := mathutil.OLSBetaAlpha(rp, rm, 60)
	if !mathutil.IsFinite(a) {
		a = 0.0
	}
	if !mathutil.IsFinite(b) {
		b = 0.0
	}
	u := make([]float64, len(rp))
	for i := range u {
		u[i] = rp[i] - (a + b*rm[i])
	}
	u = mathutil.CleanReturnSeries(u)

	uFin := make([]float64, 0, len(u))
	for _, v := range u {
		if mathutil.IsFinite(v) {
			uFin = append(uFin, v)
		}
	}

	fallback := func() []float64 {
		varU := mathutil.SampleVariance(uFin)
		if !mathutil.IsFinite(varU) || varU <= 0 {
			varU = 1e-4
		}
		ht := make([]float64, len(u))
		for i := range ht {
			ht[i] = varU
		}
		return mathutil.CleanHt(ht)
	}

	if len(uFin) < config.GARCHMinObs {
		return fallback(), nil
	}

	yScaled := make([]float64, len(uFin))
	for i, v := range uFin {
		yScaled[i] = v * 100.0
	}
	fit, err := FitNormal(yScaled)
	if err != nil {
		return fallback(), nil
	}

	condVar := make([]float64, len(fit.CondVar))
	for i, v := range fit.CondVar {
		condVar[i] = v / 10000.0
	}

	hFull := make([]float64, len(u))
	for i := range hFull {
		hFull[i] = math.NaN()
	}
	start := len(u) - len(condVar)
	if start < 0 {
		start = 0
	}
	for i := 0; i < len(condVar) && start+i < len(hFull); i++ {
		hFull[start+i] = condVar[i]
	}

	// Forward-fill, then backward-fill
	last := math.NaN()
	for i := 0; i < len(hFull); i++ {
		if mathutil.IsFinite(hFull[i]) {
			last = hFull[i]
		} else if mathutil.IsFinite(last) {
			hFull[i] = last
		}
	}
	last = math.NaN()
	for i := len(hFull) - 1; i >= 0; i-- {
		if mathutil.IsFinite(hFull[i]) {
			last = hFull[i]
		} else if mathutil.IsFinite(last) {
			hFull[i] = last
		}
	}
	return mathutil.CleanHt(hFull), nil
}
