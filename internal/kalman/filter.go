// Package kalman implements the 4-state Kalman filter for the SSM
// private equity nowcasting model, including both real-valued and
// complex-step NLL evaluation for gradient computation.
//
// State vector: [η_t, v*_t, m_t, d_t]
//
//	η_t : weekly idiosyncratic log-return
//	v*_t: true (unsmoothed) log-NAV level
//	m_t : reported (smoothed) log-NAV level
//	d_t : distribution intensity state
package kalman

import (
	"math"
	"math/cmplx"

	"ssm_go/internal/mathutil"
)

const log2pi = 1.8378770664093453

// NLLReal computes the Kalman filter negative log-likelihood using the
// 4-state SSM with two observation equations (NAV + comparable return).
//
// Parameters in theta[0..7]:
//
//	0: β_c, 1: ψ_c, 2: log(F), 3: log(σ_nav), 4: log(F_c),
//	5: λ, 6: δ, 7: log(σ_d)
func NLLReal(
	yNav []float64, navMask []bool,
	rc []float64,
	h []float64,
	driftBase []float64,
	theta [8]float64,
) float64 {
	betaC := theta[0]
	psiC := theta[1]
	F := math.Exp(theta[2])
	sigmaNav := math.Exp(theta[3])
	Fc := math.Exp(theta[4])
	lam := theta[5]
	delta := theta[6]
	sigmaD := math.Exp(theta[7])

	T := len(yNav)
	if T == 0 {
		return 1e9
	}

	// Initialise state from mean of observed NAVs
	m0 := 0.0
	cnt := 0
	for i := 0; i < T; i++ {
		if navMask[i] && mathutil.IsFinite(yNav[i]) {
			m0 += yNav[i]
			cnt++
		}
	}
	if cnt > 0 {
		m0 /= float64(cnt)
	}

	_, x1, x2, x3 := 0.0, m0, m0, 0.0
	var P [4][4]float64
	P[0][0], P[1][1], P[2][2], P[3][3] = 1e4, 1e4, 1e4, 1e4

	c := 1.0 - lam
	sigmaNav2 := sigmaNav * sigmaNav
	F2 := F * F
	Fc2 := Fc * Fc
	sigmaD2 := sigmaD * sigmaD

	ll := 0.0
	var AP, Ppt [4][4]float64

	for t := 0; t < T; t++ {
		drift := driftBase[t]
		xp0 := drift
		xp1 := x1 + drift
		xp2 := c*x1 + lam*x2 + c*drift
		xp3 := delta * x3

		qEta := F2 * h[t]
		qD := sigmaD2 * h[t]
		g0, g1, g2 := 1.0, 1.0, c

		// Predicted covariance: A·P·A' + G·Q·G'
		for j := 0; j < 4; j++ {
			AP[0][j] = 0
			AP[1][j] = P[1][j]
			AP[2][j] = c*P[1][j] + lam*P[2][j]
			AP[3][j] = delta * P[3][j]
		}
		for i := 0; i < 4; i++ {
			Ppt[i][0] = 0
			Ppt[i][1] = AP[i][1]
			Ppt[i][2] = c*AP[i][1] + lam*AP[i][2]
			Ppt[i][3] = delta * AP[i][3]
		}
		Ppt[0][0] += qEta * g0 * g0
		Ppt[0][1] += qEta * g0 * g1
		Ppt[0][2] += qEta * g0 * g2
		Ppt[1][0] += qEta * g1 * g0
		Ppt[1][1] += qEta * g1 * g1
		Ppt[1][2] += qEta * g1 * g2
		Ppt[2][0] += qEta * g2 * g0
		Ppt[2][1] += qEta * g2 * g1
		Ppt[2][2] += qEta * g2 * g2
		// g3 = 0, so all cross-terms with g3 vanish
		Ppt[3][3] += qD
		mathutil.Sym4Fast(&Ppt)

		xu0, xu1, xu2, xu3 := xp0, xp1, xp2, xp3
		Pu := Ppt

		// ── NAV observation update ──────────────────────────────────
		if navMask[t] {
			S := Pu[2][2] + sigmaNav2
			if S < 1e-12 {
				S = 1e-12
			}
			v := yNav[t] - xu2
			ll += -0.5 * (log2pi + math.Log(S) + v*v/S)
			K0, K1, K2, K3 := Pu[0][2]/S, Pu[1][2]/S, Pu[2][2]/S, Pu[3][2]/S
			xu0 += K0 * v
			xu1 += K1 * v
			xu2 += K2 * v
			xu3 += K3 * v
			r20, r21, r22, r23 := Pu[2][0], Pu[2][1], Pu[2][2], Pu[2][3]
			Pu[0][0] -= K0 * r20
			Pu[0][1] -= K0 * r21
			Pu[0][2] -= K0 * r22
			Pu[0][3] -= K0 * r23
			Pu[1][0] -= K1 * r20
			Pu[1][1] -= K1 * r21
			Pu[1][2] -= K1 * r22
			Pu[1][3] -= K1 * r23
			Pu[2][0] -= K2 * r20
			Pu[2][1] -= K2 * r21
			Pu[2][2] -= K2 * r22
			Pu[2][3] -= K2 * r23
			Pu[3][0] -= K3 * r20
			Pu[3][1] -= K3 * r21
			Pu[3][2] -= K3 * r22
			Pu[3][3] -= K3 * r23
			mathutil.Sym4Fast(&Pu)
		}

		// ── Comparable-return observation update ────────────────────
		if mathutil.IsFinite(rc[t]) {
			yhat := betaC*xu0 + psiC + xu3
			v := rc[t] - yhat
			R := Fc2 * h[t]
			if R < 1e-12 {
				R = 1e-12
			}
			PH0 := betaC*Pu[0][0] + Pu[0][3]
			PH1 := betaC*Pu[1][0] + Pu[1][3]
			PH2 := betaC*Pu[2][0] + Pu[2][3]
			PH3 := betaC*Pu[3][0] + Pu[3][3]
			S := betaC*PH0 + PH3 + R
			if S < 1e-12 {
				S = 1e-12
			}
			ll += -0.5 * (log2pi + math.Log(S) + v*v/S)
			K0, K1, K2, K3 := PH0/S, PH1/S, PH2/S, PH3/S
			xu0 += K0 * v
			xu1 += K1 * v
			xu2 += K2 * v
			xu3 += K3 * v
			r0 := betaC*Pu[0][0] + Pu[3][0]
			r1 := betaC*Pu[0][1] + Pu[3][1]
			r2 := betaC*Pu[0][2] + Pu[3][2]
			r3 := betaC*Pu[0][3] + Pu[3][3]
			Pu[0][0] -= K0 * r0
			Pu[0][1] -= K0 * r1
			Pu[0][2] -= K0 * r2
			Pu[0][3] -= K0 * r3
			Pu[1][0] -= K1 * r0
			Pu[1][1] -= K1 * r1
			Pu[1][2] -= K1 * r2
			Pu[1][3] -= K1 * r3
			Pu[2][0] -= K2 * r0
			Pu[2][1] -= K2 * r1
			Pu[2][2] -= K2 * r2
			Pu[2][3] -= K2 * r3
			Pu[3][0] -= K3 * r0
			Pu[3][1] -= K3 * r1
			Pu[3][2] -= K3 * r2
			Pu[3][3] -= K3 * r3
			mathutil.Sym4Fast(&Pu)
		}

		_, x1, x2, x3 = xu0, xu1, xu2, xu3
		P = Pu
	}
	return -ll
}

// NLLComplex computes the Kalman NLL using complex arithmetic for
// complex-step gradient evaluation. The interface mirrors NLLReal.
func NLLComplex(
	yNav []float64, navMask []bool,
	rc []float64,
	h []float64,
	driftBase []float64,
	th [8]complex128,
) complex128 {
	betaC := th[0]
	psiC := th[1]
	F := cmplx.Exp(th[2])
	sigmaNav := cmplx.Exp(th[3])
	Fc := cmplx.Exp(th[4])
	lam := th[5]
	delta := th[6]
	sigmaD := cmplx.Exp(th[7])

	T := len(yNav)
	if T == 0 {
		return complex(1e9, 0)
	}
	m0 := 0.0
	cnt := 0
	for i := 0; i < T; i++ {
		if navMask[i] && mathutil.IsFinite(yNav[i]) {
			m0 += yNav[i]
			cnt++
		}
	}
	if cnt > 0 {
		m0 /= float64(cnt)
	}

	x1 := complex(m0, 0)
	x2 := complex(m0, 0)
	x3 := complex(0, 0)
	var P [4][4]complex128
	P[0][0], P[1][1], P[2][2], P[3][3] = 1e4, 1e4, 1e4, 1e4

	c := 1 - lam
	sigmaNav2 := sigmaNav * sigmaNav
	F2 := F * F
	Fc2 := Fc * Fc
	sigmaD2 := sigmaD * sigmaD

	ll := complex(0, 0)
	var AP, Ppt [4][4]complex128

	symC := func(M *[4][4]complex128) {
		for ii := 0; ii < 4; ii++ {
			for jj := ii + 1; jj < 4; jj++ {
				v := 0.5 * ((*M)[ii][jj] + (*M)[jj][ii])
				(*M)[ii][jj] = v
				(*M)[jj][ii] = v
			}
		}
	}

	for t := 0; t < T; t++ {
		drift := complex(driftBase[t], 0)
		xp0 := drift
		xp1 := x1 + drift
		xp2 := c*x1 + lam*x2 + c*drift
		xp3 := delta * x3

		qEta := F2 * complex(h[t], 0)
		qD := sigmaD2 * complex(h[t], 0)
		g0, g1, g2, g3 := complex(1, 0), complex(1, 0), c, complex(0, 0)

		for j := 0; j < 4; j++ {
			AP[0][j] = 0
			AP[1][j] = P[1][j]
			AP[2][j] = c*P[1][j] + lam*P[2][j]
			AP[3][j] = delta * P[3][j]
		}
		for i := 0; i < 4; i++ {
			Ppt[i][0] = 0
			Ppt[i][1] = AP[i][1]
			Ppt[i][2] = c*AP[i][1] + lam*AP[i][2]
			Ppt[i][3] = delta * AP[i][3]
		}
		Ppt[0][0] += qEta * g0 * g0
		Ppt[0][1] += qEta * g0 * g1
		Ppt[0][2] += qEta * g0 * g2
		Ppt[0][3] += qEta * g0 * g3
		Ppt[1][0] += qEta * g1 * g0
		Ppt[1][1] += qEta * g1 * g1
		Ppt[1][2] += qEta * g1 * g2
		Ppt[1][3] += qEta * g1 * g3
		Ppt[2][0] += qEta * g2 * g0
		Ppt[2][1] += qEta * g2 * g1
		Ppt[2][2] += qEta * g2 * g2
		Ppt[2][3] += qEta * g2 * g3
		Ppt[3][0] += qEta * g3 * g0
		Ppt[3][1] += qEta * g3 * g1
		Ppt[3][2] += qEta * g3 * g2
		Ppt[3][3] += qEta * g3 * g3
		Ppt[3][3] += qD
		symC(&Ppt)

		xu0, xu1, xu2, xu3 := xp0, xp1, xp2, xp3
		Pu := Ppt

		if navMask[t] {
			S := Pu[2][2] + sigmaNav2
			if cmplx.Abs(S) < 1e-12 {
				S = complex(1e-12, 0)
			}
			v := complex(yNav[t], 0) - xu2
			ll += -0.5 * (complex(log2pi, 0) + cmplx.Log(S) + v*v/S)
			K0, K1, K2, K3 := Pu[0][2]/S, Pu[1][2]/S, Pu[2][2]/S, Pu[3][2]/S
			xu0 += K0 * v
			xu1 += K1 * v
			xu2 += K2 * v
			xu3 += K3 * v
			r20, r21, r22, r23 := Pu[2][0], Pu[2][1], Pu[2][2], Pu[2][3]
			Pu[0][0] -= K0 * r20
			Pu[0][1] -= K0 * r21
			Pu[0][2] -= K0 * r22
			Pu[0][3] -= K0 * r23
			Pu[1][0] -= K1 * r20
			Pu[1][1] -= K1 * r21
			Pu[1][2] -= K1 * r22
			Pu[1][3] -= K1 * r23
			Pu[2][0] -= K2 * r20
			Pu[2][1] -= K2 * r21
			Pu[2][2] -= K2 * r22
			Pu[2][3] -= K2 * r23
			Pu[3][0] -= K3 * r20
			Pu[3][1] -= K3 * r21
			Pu[3][2] -= K3 * r22
			Pu[3][3] -= K3 * r23
			symC(&Pu)
		}

		if mathutil.IsFinite(rc[t]) {
			yhat := betaC*xu0 + psiC + xu3
			v := complex(rc[t], 0) - yhat
			R := Fc2 * complex(h[t], 0)
			if cmplx.Abs(R) < 1e-12 {
				R = complex(1e-12, 0)
			}
			PH0 := betaC*Pu[0][0] + Pu[0][3]
			PH1 := betaC*Pu[1][0] + Pu[1][3]
			PH2 := betaC*Pu[2][0] + Pu[2][3]
			PH3 := betaC*Pu[3][0] + Pu[3][3]
			S := betaC*PH0 + PH3 + R
			if cmplx.Abs(S) < 1e-12 {
				S = complex(1e-12, 0)
			}
			ll += -0.5 * (complex(log2pi, 0) + cmplx.Log(S) + v*v/S)
			K0, K1, K2, K3 := PH0/S, PH1/S, PH2/S, PH3/S
			xu0 += K0 * v
			xu1 += K1 * v
			xu2 += K2 * v
			xu3 += K3 * v
			r0 := betaC*Pu[0][0] + Pu[3][0]
			r1 := betaC*Pu[0][1] + Pu[3][1]
			r2 := betaC*Pu[0][2] + Pu[3][2]
			r3 := betaC*Pu[0][3] + Pu[3][3]
			Pu[0][0] -= K0 * r0
			Pu[0][1] -= K0 * r1
			Pu[0][2] -= K0 * r2
			Pu[0][3] -= K0 * r3
			Pu[1][0] -= K1 * r0
			Pu[1][1] -= K1 * r1
			Pu[1][2] -= K1 * r2
			Pu[1][3] -= K1 * r3
			Pu[2][0] -= K2 * r0
			Pu[2][1] -= K2 * r1
			Pu[2][2] -= K2 * r2
			Pu[2][3] -= K2 * r3
			Pu[3][0] -= K3 * r0
			Pu[3][1] -= K3 * r1
			Pu[3][2] -= K3 * r2
			Pu[3][3] -= K3 * r3
			symC(&Pu)
		}

		x1, x2, x3 = xu1, xu2, xu3
		P = Pu
	}
	return -ll
}
