// Package pipeline orchestrates the SSM nowcasting workflow:
// data preparation, per-asset MLE estimation, and result collection.
package pipeline

import (
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"ssm_go/internal/csvio"
	"ssm_go/internal/garch"
	"ssm_go/internal/mathutil"
	"ssm_go/internal/timeutil"
)

// ── Data structures ─────────────────────────────────────────────────────────

// MappingRow links a private asset to its public proxy and market index.
type MappingRow struct {
	AssetLabel  string
	ProxyTicker string
	IndexTicker string
	LambdaInit  *float64
}

// BasePanel holds the pre-processed observation data for a single asset.
type BasePanel struct {
	AssetID      string
	Y            []float64            // log-NAV (NaN where unobserved)
	NavMask      []bool               // true where NAV is observed
	NavObsByDate map[time.Time]float64 // raw NAV keyed by date
	Rm           []float64            // market log-returns (cleaned)
	Rc           []float64            // comparable log-returns (cleaned)
	Ht           []float64            // GARCH conditional variance (cleaned)
	BetaBase     float64              // full-sample OLS β
	AlphaBase    float64              // full-sample OLS α
	BetaC2M      float64              // comparable-to-market β
	LambdaInit   *float64             // optional initial λ from mapping
}

// Prepared holds the complete model data ready for per-date computation.
type Prepared struct {
	DailyDatesFull []time.Time
	AllDates       []time.Time
	BasePanels     map[string]*BasePanel
}

// ── Preparation ─────────────────────────────────────────────────────────────

// PrepareFullData reads all input CSVs and builds aligned weekly panels
// for every asset in the mapping table.
func PrepareFullData(baseDir string, debug bool) (*Prepared, error) {
	quarterlyCSV := filepath.Join(baseDir, "inputs", "private_assets_prices_quarterly_wide.csv")
	mappingCSV := filepath.Join(baseDir, "inputs", "etp_and_index_mapping_table.csv")
	proxyDailyCSV := filepath.Join(baseDir, "inputs", "etp_proxy_idio_daily.csv")
	indexDailyCSV := filepath.Join(baseDir, "inputs", "index_prices_daily.csv")

	// ── 1. Read quarterly NAV data ──────────────────────────────────────────
	navHeader, navRows, err := csvio.ReadCSV(quarterlyCSV)
	if err != nil {
		return nil, fmt.Errorf("reading NAV csv: %w", err)
	}
	if len(navHeader) < 2 {
		return nil, fmt.Errorf("NAV csv must have date + assets")
	}
	navAssetCols := navHeader[1:]
	navByAsset := make(map[string]map[time.Time]float64, len(navAssetCols))
	for _, a := range navAssetCols {
		navByAsset[a] = make(map[time.Time]float64)
	}
	navDatesSet := make(map[time.Time]bool)
	for _, row := range navRows {
		if len(row) < len(navHeader) {
			continue
		}
		d, ok := timeutil.ParseDate(row[0])
		if !ok {
			continue
		}
		navDatesSet[d] = true
		for j, a := range navAssetCols {
			v := mathutil.ToFloat(row[1+j])
			if mathutil.IsFinite(v) {
				navByAsset[a][d] = v
			}
		}
	}

	// ── 2. Read mapping table ───────────────────────────────────────────────
	mapHeader, mapRows, err := csvio.ReadCSV(mappingCSV)
	if err != nil {
		return nil, fmt.Errorf("reading mapping csv: %w", err)
	}
	iAsset := csvio.FindColumn(mapHeader, "asset_label")
	iProxy := csvio.FindColumn(mapHeader, "etp_ticker")
	iIdx := csvio.FindColumn(mapHeader, "broad_index_ticker")
	iLambdaInit := csvio.FindColumn(mapHeader, "lambda_init")
	if iAsset < 0 || iProxy < 0 || iIdx < 0 {
		return nil, fmt.Errorf("mapping csv must contain asset_label, etp_ticker, broad_index_ticker")
	}

	navSet := make(map[string]bool, len(navAssetCols))
	for _, a := range navAssetCols {
		navSet[a] = true
	}
	var mappingLocal []MappingRow
	for _, row := range mapRows {
		if len(row) <= mathutil.MaxInt(mathutil.MaxInt(iAsset, iProxy), iIdx) {
			continue
		}
		asset := strings.TrimSpace(row[iAsset])
		if !navSet[asset] {
			continue
		}
		proxy := strings.TrimSpace(row[iProxy])
		idxT := strings.TrimSpace(row[iIdx])
		var lamPtr *float64
		if iLambdaInit >= 0 && iLambdaInit < len(row) {
			v := mathutil.ToFloat(row[iLambdaInit])
			if mathutil.IsFinite(v) {
				tmp := v
				lamPtr = &tmp
			}
		}
		mappingLocal = append(mappingLocal, MappingRow{
			AssetLabel: asset, ProxyTicker: proxy, IndexTicker: idxT, LambdaInit: lamPtr,
		})
	}

	// ── 3. Read proxy daily prices ──────────────────────────────────────────
	proxyByTicker, proxyDatesSet, err := readDailyPrices(proxyDailyCSV)
	if err != nil {
		return nil, fmt.Errorf("reading proxy daily csv: %w", err)
	}

	// ── 4. Read index daily prices ──────────────────────────────────────────
	indexByTicker, indexDatesSet, err := readDailyPrices(indexDailyCSV)
	if err != nil {
		return nil, fmt.Errorf("reading index daily csv: %w", err)
	}

	// ── 5. Build unified daily and weekly grids ─────────────────────────────
	dailySet := make(map[time.Time]bool, len(proxyDatesSet)+len(indexDatesSet))
	for d := range proxyDatesSet {
		dailySet[d] = true
	}
	for d := range indexDatesSet {
		dailySet[d] = true
	}
	dailyDatesFull := make([]time.Time, 0, len(dailySet))
	for d := range dailySet {
		dailyDatesFull = append(dailyDatesFull, d)
	}
	sort.Slice(dailyDatesFull, func(i, j int) bool { return dailyDatesFull[i].Before(dailyDatesFull[j]) })

	weeklyGrid := timeutil.BuildWeeklyFridayGrid(dailyDatesFull)

	proxyWeeklyPrices := csvio.ResampleWeeklyLastOnGrid(weeklyGrid, proxyByTicker)
	indexWeeklyPrices := csvio.ResampleWeeklyLastOnGrid(weeklyGrid, indexByTicker)

	proxyWeeklyRet := make(map[string]map[time.Time]float64, len(proxyWeeklyPrices))
	indexWeeklyRet := make(map[string]map[time.Time]float64, len(indexWeeklyPrices))
	for t, wp := range proxyWeeklyPrices {
		proxyWeeklyRet[t] = csvio.BuildLogRetOnGrid(weeklyGrid, wp)
	}
	for t, wp := range indexWeeklyPrices {
		indexWeeklyRet[t] = csvio.BuildLogRetOnGrid(weeklyGrid, wp)
	}

	// ── 6. Build unified all-dates axis ─────────────────────────────────────
	allSet := make(map[time.Time]bool, len(weeklyGrid)+len(navDatesSet))
	for _, d := range weeklyGrid {
		allSet[d] = true
	}
	for d := range navDatesSet {
		allSet[d] = true
	}
	allDates := make([]time.Time, 0, len(allSet))
	for d := range allSet {
		allDates = append(allDates, d)
	}
	sort.Slice(allDates, func(i, j int) bool { return allDates[i].Before(allDates[j]) })

	buildAligned := func(m map[time.Time]float64) []float64 {
		out := make([]float64, len(allDates))
		for i, d := range allDates {
			if v, ok := m[d]; ok {
				out[i] = v
			} else {
				out[i] = math.NaN()
			}
		}
		return out
	}

	// ── 7. Resolve proxy → index mapping (majority vote) ────────────────────
	proxyToIndex := resolveProxyToIndex(mappingLocal)

	// ── 8. Compute comparable-to-market betas ───────────────────────────────
	betaC2MByProxy := make(map[string]float64)
	for proxyTicker, m := range proxyWeeklyRet {
		rp := mathutil.CleanReturnSeries(buildAligned(m))
		idxTicker := proxyToIndex[proxyTicker]
		idxMap, ok := indexWeeklyRet[idxTicker]
		if idxTicker == "" || !ok {
			betaC2MByProxy[proxyTicker] = 0.0
			continue
		}
		rm := mathutil.CleanReturnSeries(buildAligned(idxMap))
		yy := make([]float64, 0, len(rp))
		xx := make([]float64, 0, len(rp))
		for i := range rp {
			if mathutil.IsFinite(rp[i]) && mathutil.IsFinite(rm[i]) {
				yy = append(yy, rp[i])
				xx = append(xx, rm[i])
			}
		}
		if len(yy) >= 60 {
			_, b := mathutil.OLSBetaAlpha(yy, xx, 2)
			if mathutil.IsFinite(b) {
				betaC2MByProxy[proxyTicker] = b
				continue
			}
		}
		betaC2MByProxy[proxyTicker] = 0.0
	}

	// ── 9. Build per-asset base panels ──────────────────────────────────────
	basePanels := make(map[string]*BasePanel, len(mappingLocal))

	for _, row := range mappingLocal {
		assetID := row.AssetLabel
		proxyTicker := row.ProxyTicker
		idxTicker := row.IndexTicker

		proxyMap, ok1 := proxyWeeklyRet[proxyTicker]
		indexMap, ok2 := indexWeeklyRet[idxTicker]
		if !(ok1 && ok2) {
			continue
		}

		rmRaw := buildAligned(indexMap)
		rcRaw := buildAligned(proxyMap)
		rm := mathutil.CleanReturnSeries(rmRaw)
		rc := mathutil.CleanReturnSeries(rcRaw)

		ht, htErr := garch.ComputeFundHt(rcRaw, rmRaw)
		if htErr != nil {
			if debug {
				fmt.Printf("[DEBUG] h_t error for %s: %v\n", assetID, htErr)
			}
			varProxy := mathutil.SampleVariance(rcRaw)
			if !mathutil.IsFinite(varProxy) || varProxy <= 0 {
				varProxy = 1e-4
			}
			ht = make([]float64, len(rcRaw))
			for i := range ht {
				ht[i] = varProxy
			}
			ht = mathutil.CleanHt(ht)
		}

		betaC2M := betaC2MByProxy[proxyTicker]

		// Full-sample OLS for initial α, β
		maskCnt := 0
		meanRm, meanRc := 0.0, 0.0
		for i := range rcRaw {
			if mathutil.IsFinite(rc[i]) && mathutil.IsFinite(rm[i]) {
				maskCnt++
				meanRm += rm[i]
				meanRc += rc[i]
			}
		}
		betaBase := 1.0
		if maskCnt >= 60 {
			meanRm /= float64(maskCnt)
			meanRc /= float64(maskCnt)
			cov, varRm := 0.0, 0.0
			for i := range rcRaw {
				if mathutil.IsFinite(rc[i]) && mathutil.IsFinite(rm[i]) {
					dr := rm[i] - meanRm
					dc := rc[i] - meanRc
					cov += dc * dr
					varRm += dr * dr
				}
			}
			if varRm > 0 {
				betaBase = cov / varRm
			}
		}
		alphaBase := 0.0
		if maskCnt >= 10 {
			sum := 0.0
			for i := range rcRaw {
				if mathutil.IsFinite(rc[i]) && mathutil.IsFinite(rm[i]) {
					sum += rc[i] - betaBase*rm[i]
				}
			}
			alphaBase = sum / float64(maskCnt)
		}

		// Build NAV observation series
		Y := make([]float64, len(allDates))
		navMask := make([]bool, len(allDates))
		navObsByDate := make(map[time.Time]float64)
		navMap := navByAsset[assetID]
		for i, d := range allDates {
			if v, ok := navMap[d]; ok && mathutil.IsFinite(v) {
				Y[i] = mathutil.SafeLog(v)
				if v > 0 {
					navMask[i] = true
					navObsByDate[d] = v
				}
			} else {
				Y[i] = math.NaN()
			}
		}

		basePanels[assetID] = &BasePanel{
			AssetID: assetID,
			Y: Y, NavMask: navMask, NavObsByDate: navObsByDate,
			Rm: rm, Rc: rc, Ht: ht,
			BetaBase: betaBase, AlphaBase: alphaBase,
			BetaC2M: betaC2M, LambdaInit: row.LambdaInit,
		}
	}

	return &Prepared{
		DailyDatesFull: dailyDatesFull,
		AllDates:       allDates,
		BasePanels:     basePanels,
	}, nil
}

// ── Internal helpers ────────────────────────────────────────────────────────

// readDailyPrices reads a daily price CSV (Date + ticker columns) into maps.
func readDailyPrices(path string) (map[string]map[time.Time]float64, map[time.Time]bool, error) {
	header, rows, err := csvio.ReadCSV(path)
	if err != nil {
		return nil, nil, err
	}
	dateCol := csvio.FindColumn(header, "Date")
	byTicker := make(map[string]map[time.Time]float64)
	datesSet := make(map[time.Time]bool)
	for j, h := range header {
		if j == dateCol {
			continue
		}
		t := strings.TrimSpace(h)
		if t != "" {
			byTicker[t] = make(map[time.Time]float64)
		}
	}
	for _, row := range rows {
		if len(row) != len(header) {
			continue
		}
		d, ok := timeutil.ParseDate(row[dateCol])
		if !ok {
			continue
		}
		datesSet[d] = true
		for j, h := range header {
			if j == dateCol {
				continue
			}
			t := strings.TrimSpace(h)
			v := mathutil.ToFloat(row[j])
			if mathutil.IsFinite(v) {
				byTicker[t][d] = v
			}
		}
	}
	return byTicker, datesSet, nil
}

// resolveProxyToIndex maps each proxy ticker to its most-frequently-paired
// index ticker using majority vote, with first-seen tiebreaking.
func resolveProxyToIndex(mapping []MappingRow) map[string]string {
	counts := make(map[string]map[string]int)
	firstSeen := make(map[string][]string)
	seen := make(map[string]map[string]bool)

	for _, r := range mapping {
		if _, ok := counts[r.ProxyTicker]; !ok {
			counts[r.ProxyTicker] = make(map[string]int)
			firstSeen[r.ProxyTicker] = nil
			seen[r.ProxyTicker] = make(map[string]bool)
		}
		counts[r.ProxyTicker][r.IndexTicker]++
		if !seen[r.ProxyTicker][r.IndexTicker] {
			seen[r.ProxyTicker][r.IndexTicker] = true
			firstSeen[r.ProxyTicker] = append(firstSeen[r.ProxyTicker], r.IndexTicker)
		}
	}

	result := make(map[string]string, len(counts))
	for proxy, cm := range counts {
		bestN := -1
		var cands []string
		for idx, n := range cm {
			if n > bestN {
				bestN = n
				cands = []string{idx}
			} else if n == bestN {
				cands = append(cands, idx)
			}
		}
		if len(cands) == 0 {
			continue
		}
		best := ""
		if len(cands) == 1 {
			best = cands[0]
		} else {
			candSet := make(map[string]bool, len(cands))
			for _, c := range cands {
				candSet[c] = true
			}
			for _, idx := range firstSeen[proxy] {
				if candSet[idx] {
					best = idx
					break
				}
			}
			if best == "" {
				sort.Strings(cands)
				best = cands[0]
			}
		}
		result[proxy] = best
	}
	return result
}
