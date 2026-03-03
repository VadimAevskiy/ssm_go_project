// Package csvio provides CSV reading and writing functions tailored
// to the SSM pipeline's input/output formats.
package csvio

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"ssm_go/internal/mathutil"
	"ssm_go/internal/timeutil"
)

// ReadCSV reads a CSV file and returns (header, dataRows, error).
func ReadCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(bufio.NewReader(f))
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) == 0 {
		return nil, nil, fmt.Errorf("empty csv: %s", path)
	}
	return rows[0], rows[1:], nil
}

// FindColumn returns the column index of the named header, or -1.
func FindColumn(header []string, name string) int {
	for i, h := range header {
		if strings.TrimSpace(h) == name {
			return i
		}
	}
	return -1
}

// ── Resampling ──────────────────────────────────────────────────────────────

// ResampleWeeklyLastOnGrid takes daily prices keyed by ticker → date → price
// and produces weekly prices aligned to the given Friday grid, using the
// last available daily price within each week.
func ResampleWeeklyLastOnGrid(weeklyGrid []time.Time, byTicker map[string]map[time.Time]float64) map[string][]float64 {
	weeklyIndex := make(map[time.Time]int, len(weeklyGrid))
	for i, d := range weeklyGrid {
		weeklyIndex[d] = i
	}
	out := make(map[string][]float64, len(byTicker))
	for ticker, m := range byTicker {
		type kv struct {
			d time.Time
			v float64
		}
		arr := make([]kv, 0, len(m))
		for d, v := range m {
			arr = append(arr, kv{d: d, v: v})
		}
		sort.Slice(arr, func(i, j int) bool { return arr[i].d.Before(arr[j].d) })
		wp := make([]float64, len(weeklyGrid))
		for i := range wp {
			wp[i] = math.NaN()
		}
		for _, it := range arr {
			if !mathutil.IsFinite(it.v) {
				continue
			}
			fr := timeutil.WeekEndingFriday(it.d)
			if j, ok := weeklyIndex[fr]; ok {
				wp[j] = it.v
			}
		}
		out[ticker] = wp
	}
	return out
}

// BuildLogRetOnGrid computes weekly log-returns from weekly price levels.
func BuildLogRetOnGrid(weeklyGrid []time.Time, weeklyPrices []float64) map[time.Time]float64 {
	out := make(map[time.Time]float64, len(weeklyGrid))
	prevLog := math.NaN()
	for i, d := range weeklyGrid {
		p := weeklyPrices[i]
		lp := math.NaN()
		if !math.IsNaN(p) {
			if math.IsInf(p, 1) {
				lp = math.Inf(1)
			} else {
				if p < 1e-12 {
					p = 1e-12
				}
				lp = math.Log(p)
			}
		}
		ret := math.NaN()
		if mathutil.IsFinite(lp) && mathutil.IsFinite(prevLog) {
			ret = lp - prevLog
		}
		out[d] = ret
		prevLog = lp
	}
	return out
}
