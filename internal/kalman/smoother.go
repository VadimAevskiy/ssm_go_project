package kalman

import (
	"math"

	"ssm_go/internal/mathutil"
)

// FilterAndSmooth runs a forward Kalman filter followed by a Rauch-Tung-
// Striebel (RTS) backward smoother to produce smoothed state estimates.
//
// Returns (vSmooth, rSmooth) where:
//   - vSmooth[t] = smoothed latent NAV level v*_t (state index 1)
//   - rSmooth[t] = Δv*_t = vSmooth[t] − vSmooth[t-1]  (NaN for t=0)
func FilterAndSmooth(
	theta [8]float64,
	y []float64, navMask []bool,
	rm []float64, rcAdj []float64, h []float64,
	alphaFixed, betaFixed float64,
) (vSmooth []float64, rSmooth []float64) {

	betaC := theta[0]
	psiC := theta[1]
	F := math.Exp(theta[2])
	sigmaNav := math.Exp(theta[3])
	Fc := math.Exp(theta[4])
	lam := theta[5]
	delta := theta[6]
	sigmaD := math.Exp(theta[7])

	T := len(y)
	hc := mathutil.CleanHt(h)

	// Mean of observed NAVs for state initialisation
	m0 := 0.0
	cnt := 0
	for i := 0; i < T; i++ {
		if navMask[i] && mathutil.IsFinite(y[i]) {
			m0 += y[i]
			cnt++
		}
	}
	if cnt > 0 {
		m0 /= float64(cnt)
	}

	xPrev := [4]float64{0.0, m0, m0, 0.0}
	var PPrev [4][4]float64
	for i := 0; i < 4; i++ {
		PPrev[i][i] = 1e4
	}

	c := 1.0 - lam
	A := [4][4]float64{
		{0, 0, 0, 0},
		{0, 1, 0, 0},
		{0, c, lam, 0},
		{0, 0, 0, delta},
	}
	g := [4]float64{1, 1, c, 0}
	var GGtEta [4][4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			GGtEta[i][j] = g[i] * g[j]
		}
	}
	var GGtEd [4][4]float64
	GGtEd[3][3] = 1

	// Pre-allocate filter storage
	xPred := make([][4]float64, T)
	Ppred := make([][4][4]float64, T)
	xf := make([][4]float64, T)
	Pf := make([][4][4]float64, T)

	F2 := F * F
	Fc2 := Fc * Fc
	sigmaD2 := sigmaD * sigmaD
	sigmaNav2 := sigmaNav * sigmaNav

	// ── Forward pass ────────────────────────────────────────────────────────

	for t := 0; t < T; t++ {
		drift := alphaFixed + betaFixed*rm[t]
		B := [4]float64{drift, drift, (1.0 - lam) * drift, 0.0}
		qEta := F2 * hc[t]
		qD := sigmaD2 * hc[t]
		var Q [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				Q[i][j] = qEta*GGtEta[i][j] + qD*GGtEd[i][j]
			}
		}

		// Predict: x_{t|t-1} = A·x_{t-1} + B
		var xpt [4]float64
		for i := 0; i < 4; i++ {
			s := 0.0
			for j := 0; j < 4; j++ {
				s += A[i][j] * xPrev[j]
			}
			xpt[i] = s + B[i]
		}
		// P_{t|t-1} = A·P_{t-1}·A' + Q
		var tmp [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				s := 0.0
				for k := 0; k < 4; k++ {
					s += A[i][k] * PPrev[k][j]
				}
				tmp[i][j] = s
			}
		}
		var Ppt [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				s := 0.0
				for k := 0; k < 4; k++ {
					s += tmp[i][k] * A[j][k]
				}
				Ppt[i][j] = s + Q[i][j]
			}
		}
		Ppt = mathutil.Sym4(Ppt)

		xut := xpt
		Put := Ppt

		// NAV observation update
		if navMask[t] {
			S := Put[2][2] + sigmaNav2
			if S < 1e-12 {
				S = 1e-12
			}
			v := y[t] - xut[2]
			K := [4]float64{Put[0][2] / S, Put[1][2] / S, Put[2][2] / S, Put[3][2] / S}
			for i := 0; i < 4; i++ {
				xut[i] += K[i] * v
			}
			r20, r21, r22, r23 := Put[2][0], Put[2][1], Put[2][2], Put[2][3]
			for i := 0; i < 4; i++ {
				Put[i][0] -= K[i] * r20
				Put[i][1] -= K[i] * r21
				Put[i][2] -= K[i] * r22
				Put[i][3] -= K[i] * r23
			}
			Put = mathutil.Sym4(Put)
		}

		// Comparable-return observation update
		if mathutil.IsFinite(rcAdj[t]) {
			yhat := betaC*xut[0] + psiC + xut[3]
			R := Fc2 * hc[t]
			if R < 1e-12 {
				R = 1e-12
			}
			PH := [4]float64{
				betaC*Put[0][0] + Put[0][3],
				betaC*Put[1][0] + Put[1][3],
				betaC*Put[2][0] + Put[2][3],
				betaC*Put[3][0] + Put[3][3],
			}
			S := betaC*PH[0] + PH[3] + R
			if S < 1e-12 {
				S = 1e-12
			}
			v := rcAdj[t] - yhat
			K := [4]float64{PH[0] / S, PH[1] / S, PH[2] / S, PH[3] / S}
			for i := 0; i < 4; i++ {
				xut[i] += K[i] * v
			}
			r0 := betaC*Put[0][0] + Put[3][0]
			r1 := betaC*Put[0][1] + Put[3][1]
			r2 := betaC*Put[0][2] + Put[3][2]
			r3 := betaC*Put[0][3] + Put[3][3]
			for i := 0; i < 4; i++ {
				Put[i][0] -= K[i] * r0
				Put[i][1] -= K[i] * r1
				Put[i][2] -= K[i] * r2
				Put[i][3] -= K[i] * r3
			}
			Put = mathutil.Sym4(Put)
		}

		xPred[t] = xpt
		Ppred[t] = Ppt
		xf[t] = xut
		Pf[t] = Put
		xPrev = xut
		PPrev = Put
	}

	// ── RTS backward smoother ───────────────────────────────────────────────

	xs := make([][4]float64, T)
	Ps := make([][4][4]float64, T)
	xs[T-1] = xf[T-1]
	Ps[T-1] = Pf[T-1]

	for t := T - 2; t >= 0; t-- {
		Pp := Ppred[t+1]
		for i := 0; i < 4; i++ {
			Pp[i][i] += 1e-12 // regularise
		}
		invPp, ok := mathutil.Inv4(Pp)
		if !ok {
			xs[t] = xf[t]
			Ps[t] = Pf[t]
			continue
		}
		// J = P_{t|t} · A' · P_{t+1|t}^{-1}
		var M [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				s := 0.0
				for k := 0; k < 4; k++ {
					s += Pf[t][i][k] * A[j][k]
				}
				M[i][j] = s
			}
		}
		var J [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				s := 0.0
				for k := 0; k < 4; k++ {
					s += M[i][k] * invPp[k][j]
				}
				J[i][j] = s
			}
		}
		// x_{t|T} = x_{t|t} + J · (x_{t+1|T} − x_{t+1|t})
		var diff [4]float64
		for i := 0; i < 4; i++ {
			diff[i] = xs[t+1][i] - xPred[t+1][i]
		}
		for i := 0; i < 4; i++ {
			s := 0.0
			for k := 0; k < 4; k++ {
				s += J[i][k] * diff[k]
			}
			xs[t][i] = xf[t][i] + s
		}
		// P_{t|T} = P_{t|t} + J · (P_{t+1|T} − P_{t+1|t}) · J'
		var D [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				D[i][j] = Ps[t+1][i][j] - Ppred[t+1][i][j]
			}
		}
		var JD [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				s := 0.0
				for k := 0; k < 4; k++ {
					s += J[i][k] * D[k][j]
				}
				JD[i][j] = s
			}
		}
		var JDJt [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				s := 0.0
				for k := 0; k < 4; k++ {
					s += JD[i][k] * J[j][k]
				}
				JDJt[i][j] = s
			}
		}
		var Pnew [4][4]float64
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				Pnew[i][j] = Pf[t][i][j] + JDJt[i][j]
			}
		}
		Ps[t] = mathutil.Sym4(Pnew)
	}

	// Extract latent NAV level (state 1) and compute log-return differences
	vSmooth = make([]float64, T)
	for i := 0; i < T; i++ {
		vSmooth[i] = xs[i][1]
	}
	rSmooth = make([]float64, T)
	for i := range rSmooth {
		rSmooth[i] = math.NaN()
	}
	for i := 1; i < T; i++ {
		rSmooth[i] = vSmooth[i] - vSmooth[i-1]
	}
	return vSmooth, rSmooth
}
