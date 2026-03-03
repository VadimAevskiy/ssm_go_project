package csvio

import (
	"encoding/csv"
	"os"
	"strconv"
	"time"

	"ssm_go/internal/mathutil"
)

// WriteReturnsCSV writes a wide-format CSV: date × fund log-returns.
func WriteReturnsCSV(
	path string,
	allDates []time.Time,
	activeAssets []string,
	retByAsset map[string]map[time.Time]float64,
) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	header := make([]string, 0, 1+len(activeAssets))
	header = append(header, "date")
	header = append(header, activeAssets...)
	if err := w.Write(header); err != nil {
		return err
	}

	for _, d := range allDates {
		row := make([]string, 0, 1+len(activeAssets))
		row = append(row, d.Format("2006-01-02"))
		for _, aid := range activeAssets {
			m := retByAsset[aid]
			if v, ok := m[d]; ok && mathutil.IsFinite(v) {
				row = append(row, strconv.FormatFloat(v, 'g', -1, 64))
			} else {
				row = append(row, "")
			}
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}
