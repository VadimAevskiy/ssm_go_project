// Package mathutil provides numerical helper functions used throughout
// the SSM estimation pipeline: safe floating-point operations, series
// cleaning, OLS regression, and statistical summaries.
package mathutil

import (
	"math"
	"math/cmplx"
	"sort"
	"strconv"
)

// ── Safe floating-point operations ──────────────────────────────────────────

// IsFinite returns true if x is neither NaN nor ±Inf.
func IsFinite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// ToFloat parses a trimmed string to float64; returns NaN on failure.
func ToFloat(s string) float64 {
	if len(s) == 0 {
		return math.NaN()
	}
	// inline trimming of common whitespace
	start, end := 0, len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\r' || s[start] == '\n') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\r' || s[end-1] == '\n') {
		end--
	}
	if start == end {
		return math.NaN()
	}
	// fast-path: manual float parsing for common patterns would add
	// complexity; strconv.ParseFloat is already efficient.
	v, err := strconv.ParseFloat(s[start:end], 64)
	if err != nil {
		return math.NaN()
	}
	return v
}

// SafeLog returns log(x), clamped to log(1e-12) for non-positive or non-finite x.
func SafeLog(x float64) float64 {
	if !IsFinite(x) || x < 1e-12 {
		return math.Log(1e-12)
	}
	return math.Log(x)
}

// Sigmoid computes the logistic sigmoid 1/(1+exp(-z)).
func Sigmoid(z float64) float64 {
	if z >= 0 {
		e := math.Exp(-z)
		return 1.0 / (1.0 + e)
	}
	e := math.Exp(z)
	return e / (1.0 + e)
}

// SigmoidC computes the logistic sigmoid for complex arguments.
func SigmoidC(z complex128) complex128 {
	return 1.0 / (1.0 + cmplx.Exp(-z))
}

// ── Bound mapping ───────────────────────────────────────────────────────────

// ToBound maps an unconstrained z to the interval [lo, hi] via sigmoid.
func ToBound(z, lo, hi float64) float64 {
	return lo + (hi-lo)*Sigmoid(z)
}

// ToBoundC maps an unconstrained complex z to [lo, hi] via sigmoid.
func ToBoundC(z complex128, lo, hi float64) complex128 {
	return complex(lo, 0) + complex(hi-lo, 0)*SigmoidC(z)
}

// FromBound maps a bounded x ∈ [lo, hi] to unconstrained z via logit.
func FromBound(x, lo, hi float64) float64 {
	p := (x - lo) / (hi - lo)
	if p < 1e-12 {
		p = 1e-12
	}
	if p > 1.0-1e-12 {
		p = 1.0 - 1e-12
	}
	return math.Log(p / (1.0 - p))
}

// ── Series operations ───────────────────────────────────────────────────────

// SampleVariance computes the unbiased sample variance of x.
func SampleVariance(x []float64) float64 {
	if len(x) < 2 {
		return math.NaN()
	}
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	ss := 0.0
	for _, v := range x {
		d := v - mean
		ss += d * d
	}
	return ss / float64(len(x)-1)
}

// CleanReturnSeries fills NaN gaps by linear interpolation, with
// constant extrapolation at the edges. Returns a new slice.
func CleanReturnSeries(arr []float64) []float64 {
	n := len(arr)
	out := make([]float64, n)
	copy(out, arr)

	// Normalise non-finite to NaN
	for i := range out {
		if !IsFinite(out[i]) {
			out[i] = math.NaN()
		}
	}

	// Check if any finite values exist
	anyFinite := false
	for _, v := range out {
		if IsFinite(v) {
			anyFinite = true
			break
		}
	}
	if !anyFinite {
		for i := range out {
			out[i] = 0.0
		}
		return out
	}

	// Collect finite anchor points
	idx := make([]int, 0, n)
	val := make([]float64, 0, n)
	for i, v := range out {
		if IsFinite(v) {
			idx = append(idx, i)
			val = append(val, v)
		}
	}

	// Constant extrapolation at edges
	for i := 0; i < idx[0]; i++ {
		out[i] = val[0]
	}
	for i := idx[len(idx)-1] + 1; i < n; i++ {
		out[i] = val[len(val)-1]
	}

	// Linear interpolation between anchors
	for k := 0; k < len(idx)-1; k++ {
		i0, i1 := idx[k], idx[k+1]
		v0, v1 := val[k], val[k+1]
		out[i0] = v0
		step := float64(i1 - i0)
		for i := i0 + 1; i < i1; i++ {
			t := float64(i-i0) / step
			out[i] = v0*(1-t) + v1*t
		}
		out[i1] = v1
	}
	return out
}

// CleanHt cleans a conditional-variance series: interpolates NaN,
// replaces non-positive with median, and floors at 1e-10.
func CleanHt(h []float64) []float64 {
	n := len(h)
	out := make([]float64, n)
	copy(out, h)

	for i := range out {
		if !IsFinite(out[i]) {
			out[i] = math.NaN()
		}
	}
	out = CleanReturnSeries(out)
	for i := range out {
		if !IsFinite(out[i]) || out[i] <= 0 {
			out[i] = math.NaN()
		}
	}

	// Compute median of finite positive values
	fin := make([]float64, 0, n)
	for _, v := range out {
		if IsFinite(v) && v > 0 {
			fin = append(fin, v)
		}
	}
	med := 1e-6
	if len(fin) > 0 {
		sort.Float64s(fin)
		med = fin[len(fin)/2]
		if !IsFinite(med) || med <= 0 {
			med = 1e-6
		}
	}

	for i := range out {
		if !IsFinite(out[i]) || out[i] <= 0 {
			out[i] = med
		}
		if out[i] < 1e-10 {
			out[i] = 1e-10
		}
	}
	return out
}

// ── OLS regression ──────────────────────────────────────────────────────────

// OLSBetaAlpha computes intercept (alpha) and slope (beta) of y = α + β·x
// using only observations where both y[i] and x[i] are finite.
// Returns (NaN, NaN) if fewer than minObs valid pairs.
func OLSBetaAlpha(y, x []float64, minObs int) (alpha, beta float64) {
	if len(y) != len(x) {
		return math.NaN(), math.NaN()
	}
	yy := make([]float64, 0, len(y))
	xx := make([]float64, 0, len(y))
	for i := range y {
		if IsFinite(y[i]) && IsFinite(x[i]) {
			yy = append(yy, y[i])
			xx = append(xx, x[i])
		}
	}
	if len(yy) < minObs || len(yy) < 2 {
		return math.NaN(), math.NaN()
	}
	mx, my := 0.0, 0.0
	for i := range yy {
		mx += xx[i]
		my += yy[i]
	}
	n := float64(len(xx))
	mx /= n
	my /= n
	vx, cov := 0.0, 0.0
	for i := range yy {
		dx := xx[i] - mx
		dy := yy[i] - my
		vx += dx * dx
		cov += dx * dy
	}
	vx /= n - 1
	if !IsFinite(vx) || vx <= 0 {
		return math.NaN(), math.NaN()
	}
	cov /= n - 1
	beta = cov / vx
	alpha = my - beta*mx
	return alpha, beta
}

// ── 4×4 matrix operations (stack-allocated for hot loops) ───────────────────

// Sym4Fast symmetrises a 4×4 matrix in-place.
func Sym4Fast(P *[4][4]float64) {
	P01 := 0.5 * (P[0][1] + P[1][0])
	P02 := 0.5 * (P[0][2] + P[2][0])
	P03 := 0.5 * (P[0][3] + P[3][0])
	P12 := 0.5 * (P[1][2] + P[2][1])
	P13 := 0.5 * (P[1][3] + P[3][1])
	P23 := 0.5 * (P[2][3] + P[3][2])
	P[0][1], P[1][0] = P01, P01
	P[0][2], P[2][0] = P02, P02
	P[0][3], P[3][0] = P03, P03
	P[1][2], P[2][1] = P12, P12
	P[1][3], P[3][1] = P13, P13
	P[2][3], P[3][2] = P23, P23
}

// Sym4 returns a symmetrised copy of a 4×4 matrix.
func Sym4(A [4][4]float64) [4][4]float64 {
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			v := 0.5 * (A[i][j] + A[j][i])
			A[i][j] = v
			A[j][i] = v
		}
	}
	return A
}

// Inv4 computes the inverse of a 4×4 matrix via Gauss-Jordan elimination.
// Returns (result, true) on success, or (zero, false) if singular.
func Inv4(A [4][4]float64) ([4][4]float64, bool) {
	var aug [4][8]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			aug[i][j] = A[i][j]
		}
		aug[i][4+i] = 1.0
	}
	for i := 0; i < 4; i++ {
		p := i
		for r := i; r < 4; r++ {
			if math.Abs(aug[r][i]) > math.Abs(aug[p][i]) {
				p = r
			}
		}
		if math.Abs(aug[p][i]) < 1e-18 {
			return [4][4]float64{}, false
		}
		if p != i {
			aug[i], aug[p] = aug[p], aug[i]
		}
		piv := aug[i][i]
		for j := i; j < 8; j++ {
			aug[i][j] /= piv
		}
		for r := 0; r < 4; r++ {
			if r == i {
				continue
			}
			f := aug[r][i]
			for j := i; j < 8; j++ {
				aug[r][j] -= f * aug[i][j]
			}
		}
	}
	var inv [4][4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			inv[i][j] = aug[i][4+j]
		}
	}
	return inv, true
}

// MaxInt returns the larger of a and b.
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
