// Package timeutil provides date parsing, weekly grid construction,
// and as-of date lookup functions for the SSM pipeline.
package timeutil

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"ssm_go/internal/config"
)

// ParseDate attempts to parse a date string using common layouts.
// Returns (time, true) on success or (zero, false) on failure.
func ParseDate(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}
	layouts := []string{"2006-01-02", "2006-01-02 15:04:05", time.RFC3339}
	for _, ly := range layouts {
		if t, err := time.ParseInLocation(ly, s, time.UTC); err == nil {
			return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC), true
		}
	}
	return time.Time{}, false
}

// weekdayMon0 returns the day-of-week with Monday=0, Sunday=6.
func weekdayMon0(t time.Time) int {
	wd := int(t.Weekday())
	if wd == 0 {
		return 6
	}
	return wd - 1
}

// WeekEndingFriday snaps a date to its week-ending Friday.
func WeekEndingFriday(d time.Time) time.Time {
	wd := weekdayMon0(d)
	shift := 4 - wd
	if shift < 0 {
		shift += 7
	}
	return time.Date(d.Year(), d.Month(), d.Day(), 0, 0, 0, 0, time.UTC).AddDate(0, 0, shift)
}

// BuildWeeklyFridayGrid generates a complete Friday-to-Friday weekly
// grid spanning the first to last date in dailyDates.
func BuildWeeklyFridayGrid(dailyDates []time.Time) []time.Time {
	if len(dailyDates) == 0 {
		return nil
	}
	start := WeekEndingFriday(dailyDates[0])
	end := WeekEndingFriday(dailyDates[len(dailyDates)-1])
	var out []time.Time
	for d := start; !d.After(end); d = d.AddDate(0, 0, 7) {
		out = append(out, d)
	}
	return out
}

// PickAsofIndex finds the position of asofStr within allDates.
// If the exact date is absent, the latest date on or before is used.
func PickAsofIndex(allDates []time.Time, asofStr string) (int, time.Time, error) {
	if len(allDates) == 0 {
		return 0, time.Time{}, errors.New("allDates is empty")
	}
	if strings.TrimSpace(asofStr) == "" {
		i := len(allDates) - 1
		return i, allDates[i], nil
	}
	target, ok := ParseDate(asofStr)
	if !ok {
		return 0, time.Time{}, fmt.Errorf("cannot parse ASOF_DATE: %s", asofStr)
	}
	pos := sort.Search(len(allDates), func(i int) bool { return !allDates[i].Before(target) })
	if pos < len(allDates) && allDates[pos].Equal(target) {
		return pos, allDates[pos], nil
	}
	pos--
	if pos < 0 {
		return 0, time.Time{}, fmt.Errorf("ASOF_DATE=%s before first model date %s", asofStr, allDates[0].Format("2006-01-02"))
	}
	return pos, allDates[pos], nil
}

// CutoffDateFromDaily returns the start of the rolling window relative to asof.
func CutoffDateFromDaily(daily []time.Time, asof time.Time) time.Time {
	if len(daily) == 0 {
		return asof
	}
	i := sort.Search(len(daily), func(k int) bool { return daily[k].After(asof) }) - 1
	if i < 0 {
		return daily[0]
	}
	iCut := i - config.WindowDays + 1
	if iCut < 0 {
		iCut = 0
	}
	return daily[iCut]
}

// StartDateFromDaily returns the calibration start date: nDays back from asof.
func StartDateFromDaily(daily []time.Time, asof time.Time, nDays int) time.Time {
	if len(daily) == 0 {
		return asof
	}
	i := sort.Search(len(daily), func(k int) bool { return daily[k].After(asof) }) - 1
	if i < 0 {
		return daily[0]
	}
	iStart := i - nDays + 1
	if iStart < 0 {
		iStart = 0
	}
	return daily[iStart]
}
