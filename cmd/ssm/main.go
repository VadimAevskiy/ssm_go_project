// Command ssm runs the SSM-based private equity NAV nowcasting model
// for a single as-of date, producing a wide-format CSV of weekly
// log-returns for all assets in the mapping table.
//
// Environment variables:
//
//	SSM_ASOF_DATE  – target date (default: 2025-11-14)
//	SSM_WORKERS    – parallelism (default: runtime.NumCPU())
//	SSM_DEBUG      – enable debug logging (any non-empty value)
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	"ssm_go/internal/config"
	"ssm_go/internal/csvio"
	"ssm_go/internal/pipeline"
	"ssm_go/internal/timeutil"
)

func main() {
	t0 := time.Now()

	// ── Configuration ───────────────────────────────────────────────────────
	wd, err := os.Getwd()
	if err != nil {
		panic(fmt.Errorf("cannot determine working directory: %w", err))
	}
	cfg := config.LoadRunConfig(wd)

	workers := cfg.Workers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	runtime.GOMAXPROCS(workers)

	fmt.Printf("SSM single-date run. as_of=%s workers=%d\n", cfg.AsofDate, workers)

	// ── Data preparation ────────────────────────────────────────────────────
	prep, err := pipeline.PrepareFullData(cfg.BaseDir, cfg.Debug)
	if err != nil {
		panic(fmt.Errorf("data preparation failed: %w", err))
	}
	fmt.Printf("Prepared: assets=%d all_dates=%d\n",
		len(prep.BasePanels), len(prep.AllDates))

	// ── As-of date resolution ───────────────────────────────────────────────
	asofIdx, asofDate, err := timeutil.PickAsofIndex(prep.AllDates, cfg.AsofDate)
	if err != nil {
		panic(fmt.Errorf("as-of date error: %w", err))
	}
	fmt.Printf("As-of date: %s (index %d/%d)\n",
		asofDate.Format("2006-01-02"), asofIdx, len(prep.AllDates))

	// ── Deterministic asset ordering ────────────────────────────────────────
	assetIDs := make([]string, 0, len(prep.BasePanels))
	for k := range prep.BasePanels {
		assetIDs = append(assetIDs, k)
	}
	sort.Strings(assetIDs)

	// ── Parallel computation ────────────────────────────────────────────────
	type result struct {
		AssetID string
		Dates   []time.Time
		Returns []float64
	}
	results := make([]result, len(assetIDs))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	// Progress counter (atomic via channel)
	progress := make(chan int, len(assetIDs))

	for idx, assetID := range assetIDs {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, aid string) {
			defer wg.Done()
			defer func() { <-sem }()
			base := prep.BasePanels[aid]
			d, r := pipeline.ComputeSSMReturns(base, prep, asofIdx)
			results[i] = result{AssetID: aid, Dates: d, Returns: r}
			progress <- i
		}(idx, assetID)
	}

	// Monitor progress
	go func() {
		completed := 0
		for range progress {
			completed++
			if completed%20 == 0 || completed == len(assetIDs) {
				fmt.Printf("  completed %d/%d assets\n", completed, len(assetIDs))
			}
		}
	}()

	wg.Wait()
	close(progress)

	// ── Collect output ──────────────────────────────────────────────────────
	dateSet := make(map[time.Time]bool)
	for _, r := range results {
		for _, d := range r.Dates {
			dateSet[d] = true
		}
	}
	allOutDates := make([]time.Time, 0, len(dateSet))
	for d := range dateSet {
		allOutDates = append(allOutDates, d)
	}
	sort.Slice(allOutDates, func(i, j int) bool { return allOutDates[i].Before(allOutDates[j]) })

	retByAsset := make(map[string]map[time.Time]float64)
	activeAssets := make([]string, 0)
	for _, r := range results {
		if len(r.Dates) == 0 {
			continue
		}
		activeAssets = append(activeAssets, r.AssetID)
		m := make(map[time.Time]float64, len(r.Dates))
		for i, d := range r.Dates {
			m[d] = r.Returns[i]
		}
		retByAsset[r.AssetID] = m
	}
	sort.Strings(activeAssets)

	// ── Write output CSV ────────────────────────────────────────────────────
	outputDir := filepath.Join(cfg.BaseDir, "outputs")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		panic(fmt.Errorf("cannot create output directory: %w", err))
	}
	outPath := filepath.Join(outputDir, "ssm_returns_"+asofDate.Format("2006-01-02")+".csv")

	if err := csvio.WriteReturnsCSV(outPath, allOutDates, activeAssets, retByAsset); err != nil {
		panic(fmt.Errorf("writing output CSV: %w", err))
	}

	fmt.Printf("Output: %s (%d dates x %d funds)\n", outPath, len(allOutDates), len(activeAssets))
	fmt.Printf("Runtime: %.1f sec\n", time.Since(t0).Seconds())
}
