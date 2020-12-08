package xmath

import (
	"math"
	"time"
)

// Set is a set of statistical properties of a set of numbers.
type Set struct {
	count          int
	first, last    float64
	min, max       float64
	mean, dSquared float64
}

// NewSet creates a new Set.
func NewSet() Set {
	return Set{
		min: math.MaxFloat64,
	}
}

// Push adds another element to the set.
func (s *Set) Push(v float64) {
	s.count++
	diff := (v - s.mean) / float64(s.count)
	mean := s.mean + diff
	squaredDiff := (v - mean) * (v - s.mean)
	s.dSquared += squaredDiff
	s.mean = mean

	if s.first == 0.0 {
		s.first = v
	}

	if s.min > v {
		s.min = v
	}

	if s.max < v {
		s.max = v
	}

	s.last = v
}

// Avg returns the average value of the set.
func (s Set) Avg() float64 {
	return s.mean
}

// Diff returns the difference of max and min.
func (s Set) Diff() float64 {
	return s.last - s.first
}

// Variance is the mathematical variance of the set.
func (s Set) Variance() float64 {
	return s.dSquared / float64(s.count)
}

// StDev is the standard deviation of the set.
func (s Set) StDev() float64 {
	return math.Sqrt(s.Variance())
}

// SampleVariance is the sample variance of the set.
func (s Set) SampleVariance() float64 {
	return s.dSquared / float64(s.count-1)
}

// SampleStDev is the sample standard deviation of the set.
func (s Set) SampleStdev() float64 {
	return math.Sqrt(s.SampleVariance())
}

// Stats gathers stats about a set of floats
type Stats struct {
	Iteration int
	Set
}

// NewStats creates a new stats struct.
// It allows to pass already gathered elements.
func NewStats(vv ...float64) *Stats {
	stats := &Stats{
		Iteration: 0,
		Set:       NewSet(),
	}
	for _, v := range vv {
		stats.Push(v)
	}
	return stats
}

// Inc adds another stats element to the set.
func (s *Stats) Inc(v float64) {
	s.Iteration++
	s.Set.Push(v)
}

// Bucket groups together objects with the same Index
// it keeps track of statistical quantities relating to the collection
// by using streaming techniques
type Bucket struct {
	stats Set
	index int64
}

// NewBucket creates a new bucket
func NewBucket(id int64) Bucket {
	return Bucket{
		stats: NewSet(),
		index: id,
	}
}

// Push adds an element to the bucket for the given index.
func (b *Bucket) Push(v float64, index int64) bool {
	if index != b.index {
		return false
	}

	b.stats.Push(v)

	return true
}

// Size returns the number of elements in the bucket.
func (b Bucket) Size() int {
	return b.stats.count
}

// Stats returns the current stats for the bucket.
func (b Bucket) Stats() Set {
	return b.stats
}

// Index returns the bucket index.
func (b Bucket) Index() int64 {
	return b.index
}

// Window is a helper struct allowing to retrieve buckets of stats from a streaming data set.
type Window struct {
	size      int64
	lastIndex int64
	last      *Bucket
	current   Bucket
}

// NewWindow creates a new window of the given window size e.g. the index range for each bucket.
func NewWindow(size int64) *Window {
	return &Window{
		size: size,
	}
}

// Push adds an element to a window at the given index.
// returns if the window closed.
// and the index of the closed window
func (w *Window) Push(index int64, value float64) (int64, bool) {

	ready := false

	lastIndex := w.lastIndex

	if index == 0 {
		// new start ...
		w.lastIndex = index
		w.current = NewBucket(index)
	} else if index >= w.lastIndex+w.size {
		// start a new one
		// but first close the last one
		if w.last != nil {
			panic("last bucket has not been consumed. Cant create a new one!")
		}

		if w.current.Size() > 0 {
			tmpBucket := w.current
			w.last = &tmpBucket
			ready = true
		}

		w.current = NewBucket(index)
		w.lastIndex = index
	}

	w.current.Push(value, w.lastIndex)

	return lastIndex, ready

}

// Current returns the current index the window accumulates data on.
func (w *Window) Current() int64 {
	return w.lastIndex
}

// Next is the next index at which a new bucket will be created
func (w *Window) Next() int64 {
	return w.lastIndex + w.size
}

// Get returns the last complete Bucket.
func (w *Window) Get() Bucket {
	tmpBucket := *w.last
	w.last = nil
	return tmpBucket
}

// TimeWindow is a window indexed by the current time.
type TimeWindow struct {
	c        int
	i        int64
	index    int64
	duration int64
	window   *Window
}

// NewTimeWindow creates a new TimeWindow with the given duration.
func NewTimeWindow(duration time.Duration) *TimeWindow {
	d := int64(duration.Seconds())
	return &TimeWindow{
		duration: d,
		window:   NewWindow(1),
	}
}

// Push adds an element to the time window.
func (tw *TimeWindow) Push(v float64, t time.Time) (*Bucket, bool) {

	index := t.Unix() / tw.duration

	index, closed := tw.window.Push(index, v)

	if closed {
		tw.index = index
		bucket := tw.window.Get()
		return &bucket, true
	}

	return nil, false

}

// Next returns the next timestamp for the coming window.
func (tw *TimeWindow) Next(iterations int64) time.Time {
	nextIndex := tw.index + tw.duration*(iterations+1)
	return time.Unix(nextIndex*int64(time.Second.Seconds()), 0)
}

// TimeWindow is a window indexed by the current time.
type SizeWindow struct {
	i      int64
	window *Window
}

// NewSizeWindow creates a new SizeWindow with the given duration.
func NewSizeWindow(size int) *SizeWindow {
	return &SizeWindow{
		window: NewWindow(int64(size)),
	}
}

// Push adds an element to the time window.
func (sw *SizeWindow) Push(v float64) (*Bucket, bool) {

	iter := sw.i

	sw.window.Push(iter, v)

	if int(sw.window.size) > sw.window.Get().Size() {
		b := sw.window.Get()
		sw.i++
		return &b, true
	}

	return nil, false

}
