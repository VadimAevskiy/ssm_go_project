package pipeline

import (
	"math"
	"sort"
	"time"

	"ssm_go/internal/config"
	"ssm_go/internal/kalman"
	"ssm_go/internal/mathutil"
	"ssm_go/internal/mle"
	"ssm_go/internal/timeutil"
)

// SortedNavObs returns the NAV observation dates and values in chronological order.
func SortedNavObs(base *BasePanel) ([]time.Time, []float64) {
	dates := make([]time.Time, 0, len(base.NavObsByDate))
	for d := range base.NavObsByDate {
		dates = append(dates, d)
	}
	sort.Slice(dates, func(i, j int) bool { return dates[i].Before(dates[j]) })
	vals := make([]float64, len(dates))
	for i, d := range dates {
		vals[i] = base.NavObsByDate[d]
	}
	return dates, vals
}

// ComputeSSMReturns runs the full SSM pipeline for one asset at one as-of date:
//  1. Window extraction and calibration period selection
//  2. Rolling OLS for α, β
//  3. MLE parameter estimation (with outer β_c loop)
//  4. Kalman filter + RTS smoother
//  5. NAV anchoring and return extraction
//
// Returns (dates, logReturns) within the evaluation window, or (nil, nil) if
// the asset has insufficient NAV observations for estimation.
func ComputeSSMReturns(base *BasePanel, prep *Prepared, asofIdx int) (dates []time.Time, returns []float64) {
	allDates := prep.AllDates
	dailyDatesFull := prep.DailyDatesFull

	yFull := base.Y
	navMaskFull := base.NavMask
	rmFull := base.Rm
	rcFull := base.Rc
	hFull := base.Ht

	betaBase := base.BetaBase
	alphaBase := base.AlphaBase
	betaC2M := base.BetaC2M
	lambdaInit := base.LambdaInit

	asofDate := allDates[asofIdx]

	// ── Determine evaluation window ─────────────────────────────────────────
	cutoffRef := timeutil.CutoffDateFromDaily(dailyDatesFull, asofDate)
	idxCut := sort.Search(asofIdx+1, func(i int) bool { return !allDates[i].Before(cutoffRef) })
	if idxCut < 0 {
		idxCut = 0
	}
	if idxCut > asofIdx {
		idxCut = asofIdx
	}
	cutoffDate := allDates[idxCut]

	// ── Slice data up to as-of date ─────────────────────────────────────────
	yAsof := yFull[:asofIdx+1]
	navMaskAsof := navMaskFull[:asofIdx+1]
	rmAsof := rmFull[:asofIdx+1]
	rcAsof := rcFull[:asofIdx+1]
	hAsof := hFull[:asofIdx+1]

	rmLeadAsof := make([]float64, len(rmAsof))
	for i := range rmLeadAsof {
		rmLeadAsof[i] = math.NaN()
	}
	if len(rmAsof) >= 2 {
		for i := 0; i < len(rmAsof)-1; i++ {
			rmLeadAsof[i] = rmAsof[i+1]
		}
	}

	// ── Determine calibration window ────────────────────────────────────────
	calibStartRef := timeutil.StartDateFromDaily(dailyDatesFull, asofDate, config.CalibDays)
	idxMle := sort.Search(asofIdx+1, func(i int) bool { return !allDates[i].Before(calibStartRef) })
	if idxMle < 0 {
		idxMle = 0
	}
	if idxMle > asofIdx {
		idxMle = asofIdx
	}

	yMle := yAsof[idxMle:]
	navMaskMle := navMaskAsof[idxMle:]
	rmMle := rmAsof[idxMle:]
	rcMle := rcAsof[idxMle:]
	hMle := hAsof[idxMle:]

	// Check minimum NAV observations
	navCnt := 0
	for _, m := range navMaskMle {
		if m {
			navCnt++
		}
	}
	if navCnt < config.MinNAVForMLE {
		return nil, nil
	}

	// ── Rolling OLS for α, β ────────────────────────────────────────────────
	alphaFixedT := alphaBase
	betaFixedT := betaBase
	{
		yy := make([]float64, 0, len(rcMle))
		xx := make([]float64, 0, len(rcMle))
		for i := range rcMle {
			if mathutil.IsFinite(rcMle[i]) && mathutil.IsFinite(rmMle[i]) {
				yy = append(yy, rcMle[i])
				xx = append(xx, rmMle[i])
			}
		}
		if len(yy) >= 60 {
			a, b := mathutil.OLSBetaAlpha(yy, xx, 2)
			if mathutil.IsFinite(b) {
				betaFixedT = b
			}
			if mathutil.IsFinite(a) {
				alphaFixedT = a
			}
		}
		if !mathutil.IsFinite(betaFixedT) {
			betaFixedT = betaBase
		}
		if !mathutil.IsFinite(alphaFixedT) {
			alphaFixedT = alphaBase
		}
	}

	// ── MLE estimation ──────────────────────────────────────────────────────
	var scratch mle.Scratch
	thetaHat, _ := mle.EstimateParams(
		&scratch,
		yMle, navMaskMle, rmMle, rcMle, hMle,
		alphaFixedT, betaFixedT, betaC2M,
		lambdaInit,
		nil,
	)

	// ── Kalman filter + smoother ────────────────────────────────────────────
	rcAdjAsof := make([]float64, len(rcAsof))
	copy(rcAdjAsof, rcAsof)
	if mathutil.IsFinite(thetaHat[0]) {
		adjCoef := thetaHat[0]*betaFixedT - betaC2M
		for i := range rcAdjAsof {
			rcAdjAsof[i] = rcAsof[i] - adjCoef*rmLeadAsof[i]
		}
	}

	vs, _ := kalman.FilterAndSmooth(
		thetaHat, yAsof, navMaskAsof, rmAsof, rcAdjAsof,
		mathutil.CleanHt(hAsof), alphaFixedT, betaFixedT,
	)

	vAnch := make([]float64, len(vs))
	copy(vAnch, vs)

	// ── Anchor smoothed states to actual NAV ────────────────────────────────
	navDatesSorted, navValsSorted := SortedNavObs(base)
	anchorIdx := -1
	for i, d := range navDatesSorted {
		if !d.Before(cutoffDate) {
			anchorIdx = i
			break
		}
	}
	if anchorIdx < 0 && len(navDatesSorted) > 0 {
		anchorIdx = len(navDatesSorted) - 1
		for i := len(navDatesSorted) - 1; i >= 0; i-- {
			if navDatesSorted[i].Before(cutoffDate) {
				anchorIdx = i
				break
			}
		}
	}
	if anchorIdx >= 0 {
		anchorDate := navDatesSorted[anchorIdx]
		anchorNav := navValsSorted[anchorIdx]
		modelPos := -1
		for i := 0; i <= asofIdx; i++ {
			if allDates[i].Equal(anchorDate) {
				modelPos = i
				break
			}
		}
		if modelPos >= 0 && modelPos < len(vAnch) &&
			mathutil.IsFinite(vAnch[modelPos]) &&
			mathutil.IsFinite(anchorNav) && anchorNav > 0 {
			shift := math.Log(anchorNav) - vAnch[modelPos]
			for i := range vAnch {
				if mathutil.IsFinite(vAnch[i]) {
					vAnch[i] += shift
				}
			}
		}
	}

	// ── Compute log-returns from anchored states ────────────────────────────
	rAnch := make([]float64, len(vAnch))
	for i := range rAnch {
		rAnch[i] = math.NaN()
	}
	for i := 1; i < len(vAnch); i++ {
		if mathutil.IsFinite(vAnch[i]) && mathutil.IsFinite(vAnch[i-1]) {
			rAnch[i] = vAnch[i] - vAnch[i-1]
		}
	}

	// ── Collect returns within the evaluation window ────────────────────────
	for i := 0; i <= asofIdx; i++ {
		if !allDates[i].Before(cutoffDate) {
			dates = append(dates, allDates[i])
			returns = append(returns, rAnch[i])
		}
	}

	return dates, returns
}
