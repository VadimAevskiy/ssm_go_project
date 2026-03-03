// Package mle implements maximum-likelihood estimation of the 8 SSM
// parameters using L-BFGS with complex-step gradients, wrapped in an
// outer fixed-point iteration for the β_c proxy-adjustment coefficient.
package mle

import (
	"math"

	"gonum.org/v1/gonum/optimize"

	"ssm_go/internal/config"
	"ssm_go/internal/kalman"
	"ssm_go/internal/mathutil"
)

// OptDiag stores diagnostic information from the optimisation.
type OptDiag struct {
	OptSuccess   int
	OptNit       float64
	OptNfev      float64
	FinalNll     float64
	UsedFallback int
}

// Scratch holds pre-allocated work buffers to reduce allocation pressure
// during repeated MLE calls. Callers should create one per goroutine.
type Scratch struct {
	rcAdj     []float64
	hClean    []float64
	driftBase []float64
	rmLead    []float64
}

// ensureScratch grows or re-slices the scratch buffers to length n.
func ensureScratch(s *Scratch, n int) {
	grow := func(p *[]float64) {
		if cap(*p) < n {
			*p = make([]float64, n)
		} else {
			*p = (*p)[:n]
		}
	}
	grow(&s.rcAdj)
	grow(&s.hClean)
	grow(&s.driftBase)
	grow(&s.rmLead)
}

// clipToBounds clips initial values to be strictly within bounds.
// This mirrors scipy's L-BFGS-B which auto-clips x0 to the bound box.
func clipToBounds(x0 [8]float64) [8]float64 {
	for i := 0; i < 8; i++ {
		lo := config.ThetaBounds[i].Lo
		hi := config.ThetaBounds[i].Hi
		margin := (hi - lo) * 0.005
		if x0[i] < lo+margin {
			x0[i] = lo + margin
		}
		if x0[i] > hi-margin {
			x0[i] = hi - margin
		}
	}
	return x0
}

// EstimateParams runs the full MLE estimation for one asset at one date.
//
// It iterates an outer fixed-point loop to jointly solve for β_c
// (which enters the proxy-adjustment) and the remaining 7 parameters.
//
// Arguments:
//   - s:           pre-allocated scratch buffers (one per goroutine)
//   - y, navMask:  log-NAV observations and mask
//   - rm, rc, h:   market returns, comparable returns, conditional variance
//   - alphaFixed, betaFixed: OLS-estimated risk-return profile
//   - betaC2M:     comparable-to-market beta
//   - lambdaInit:  optional initial λ from the mapping table
//   - x0Override:  optional warm-start parameter vector
func EstimateParams(
	s *Scratch,
	y []float64, navMask []bool,
	rm []float64, rc []float64, h []float64,
	alphaFixed, betaFixed, betaC2M float64,
	lambdaInit *float64,
	x0Override *[8]float64,
) (thetaHat [8]float64, diag OptDiag) {

	n := len(y)
	ensureScratch(s, n)

	copy(s.hClean, mathutil.CleanHt(h))
	for t := 0; t < n; t++ {
		s.driftBase[t] = alphaFixed + betaFixed*rm[t]
	}
	for t := 0; t < n-1; t++ {
		s.rmLead[t] = rm[t+1]
	}
	s.rmLead[n-1] = math.NaN()

	x0Default := config.DefaultTheta(lambdaInit)
	x0 := x0Default
	if x0Override != nil {
		x0 = *x0Override
	}

	// runOpt performs a single L-BFGS optimisation from the given starting point.
	runOpt := func(x0 [8]float64) (th [8]float64, d OptDiag, ok bool) {
		x0 = clipToBounds(x0)
		z0 := make([]float64, 8)
		for i := 0; i < 8; i++ {
			z0[i] = mathutil.FromBound(x0[i], config.ThetaBounds[i].Lo, config.ThetaBounds[i].Hi)
		}

		nfev := 0
		f := func(z []float64) float64 {
			nfev++
			var theta [8]float64
			for i := 0; i < 8; i++ {
				theta[i] = mathutil.ToBound(z[i], config.ThetaBounds[i].Lo, config.ThetaBounds[i].Hi)
			}
			return kalman.NLLReal(y, navMask, s.rcAdj, s.hClean, s.driftBase, theta)
		}
		grad := func(g, z []float64) {
			var zc [8]complex128
			for i := 0; i < 8; i++ {
				zc[i] = complex(z[i], 0)
			}
			for k := 0; k < 8; k++ {
				pp := zc
				pp[k] += complex(0, config.CSEps)
				var thC [8]complex128
				for i := 0; i < 8; i++ {
					thC[i] = mathutil.ToBoundC(pp[i], config.ThetaBounds[i].Lo, config.ThetaBounds[i].Hi)
				}
				g[k] = imag(kalman.NLLComplex(y, navMask, s.rcAdj, s.hClean, s.driftBase, thC)) / config.CSEps
			}
		}

		settings := optimize.Settings{
			MajorIterations:   config.OptMaxIter,
			GradientThreshold: config.OptGTol,
		}
		prob := optimize.Problem{Func: f, Grad: grad}
		res, err := optimize.Minimize(prob, z0, &settings, &optimize.LBFGS{})

		if res == nil || res.X == nil || !mathutil.IsFinite(res.F) {
			return x0, OptDiag{OptSuccess: 0, OptNfev: float64(nfev)}, false
		}
		for i := 0; i < 8; i++ {
			th[i] = mathutil.ToBound(res.X[i], config.ThetaBounds[i].Lo, config.ThetaBounds[i].Hi)
		}
		d.OptSuccess = 1
		if err != nil {
			d.OptSuccess = 0
		}
		d.OptNit = float64(res.Stats.MajorIterations)
		d.OptNfev = float64(nfev)
		d.FinalNll = res.F
		d.UsedFallback = 0
		return th, d, true
	}

	// ── Outer fixed-point iterations for β_c ────────────────────────────────

	bestF := math.Inf(1)
	bestTheta := x0
	bestDiag := OptDiag{OptSuccess: 0}

	betaCPrev := x0[0]
	x0Curr := x0

	for outer := 0; outer < config.OuterMaxIter; outer++ {
		adjCoef := betaCPrev*betaFixed - betaC2M
		for i := 0; i < n; i++ {
			s.rcAdj[i] = rc[i] - adjCoef*s.rmLead[i]
		}

		th, d, ok := runOpt(x0Curr)

		// Fallback: retry with default starting point
		if !ok {
			th, d, ok = runOpt(x0Default)
			if ok {
				d.UsedFallback = 1
			}
		}

		if !ok {
			break
		}

		if mathutil.IsFinite(d.FinalNll) && d.FinalNll < bestF {
			bestF = d.FinalNll
			bestTheta = th
			bestDiag = d
		}

		betaCNew := th[0]
		x0Curr = th

		if mathutil.IsFinite(betaCNew) && math.Abs(betaCNew-betaCPrev) < config.OuterTolBetaC {
			break
		}
		betaCPrev = betaCNew
	}

	thetaHat = bestTheta
	diag = bestDiag

	if !mathutil.IsFinite(bestF) {
		for i := range thetaHat {
			thetaHat[i] = math.NaN()
		}
		diag = OptDiag{OptSuccess: 0, UsedFallback: 1}
	}

	return
}
