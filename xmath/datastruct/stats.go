package datastruct

import (
	"fmt"
	"math"
	"time"
)

// Stats is a set of statistical properties of a set of numbers.
type Stats struct {
	count          int
	sum            float64
	first, last    float64
	min, max       float64
	mean, dSquared float64
}

// NewStats creates a new Stats.
func NewStats() *Stats {
	return &Stats{
		min: math.MaxFloat64,
	}
}

// Push adds another element to the set.
func (s *Stats) Push(v float64) {
	s.count++
	s.sum += v
	diff := (v - s.mean) / float64(s.count)
	mean := s.mean + diff
	squaredDiff := (v - mean) * (v - s.mean)
	s.dSquared += squaredDiff
	s.mean = mean

	if s.count == 1 {
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
func (s Stats) Avg() float64 {
	return s.mean
}

// Avg returns the average value of the set.
func (s Stats) Sum() float64 {
	return s.sum
}

// Count returns the number of elements.
func (s Stats) Count() int {
	return s.count
}

// Diff returns the difference of max and min.
func (s Stats) Diff() float64 {
	return s.last - s.first
}

// Variance is the mathematical variance of the set.
func (s Stats) Variance() float64 {
	return s.dSquared / float64(s.count)
}

// StDev is the standard deviation of the set.
func (s Stats) StDev() float64 {
	return math.Sqrt(s.Variance())
}

// SampleVariance is the sample variance of the set.
func (s Stats) SampleVariance() float64 {
	return s.dSquared / float64(s.count-1)
}

// SampleStDev is the sample standard deviation of the set.
func (s Stats) SampleStDev() float64 {
	return math.Sqrt(s.SampleVariance())
}

// StatsCollector is a collection of Stats variables.
type StatsCollector struct {
	dim   int
	Stats []*Stats
}

// NewStatsCollector creates a new Stats collector.
func NewStatsCollector(dim int) *StatsCollector {
	stats := make([]*Stats, dim)
	for i := 0; i < dim; i++ {
		stats[i] = NewStats()
	}
	return &StatsCollector{
		dim:   dim,
		Stats: stats,
	}
}

// Push pushes each value to the corresponding dimension.
func (sc *StatsCollector) Push(v ...float64) {
	if len(v) != sc.dim {
		panic(fmt.Sprintf("inconsistent dimensions %d vs %d", len(v), sc.dim))
	}
	for i := 0; i < len(sc.Stats); i++ {
		sc.Stats[i].Push(v[i])
	}
}

// Push pushes each value to the corresponding dimension.
func (sc *StatsCollector) Size() int {
	// we expect all stats to have the same size
	return sc.Stats[0].count
}

// Bucket groups together objects with the same Index
// it keeps track of statistical quantities relating to the collection
// by using streaming techniques
type Bucket struct {
	stats *StatsCollector
	time  time.Time
	index int64
}

// NewBucket creates a new bucket
func NewBucket(id int64, dim int) Bucket {
	return Bucket{
		stats: NewStatsCollector(dim),
		index: id,
	}
}

// TODO : clean abstraction (who is owner of time ?)
// Time returns the time of the bucket index.
func (b *Bucket) Time() time.Time {
	return b.time
}

// Push adds an element to the bucket for the given index.
func (b *Bucket) Push(index int64, v ...float64) bool {
	if index != b.index {
		return false
	}
	b.stats.Push(v...)
	return true
}

// Size returns the number of elements in the bucket.
func (b Bucket) Size() int {
	return b.stats.Size()
}

// Stats returns the current Stats for the bucket.
func (b Bucket) Stats() StatsCollector {
	return *b.stats
}

// Index returns the bucket index.
func (b Bucket) Index() int64 {
	return b.index
}

// Window is a helper struct allowing to retrieve buckets of Stats from a streaming data set.
type Window struct {
	size      int64
	lastIndex int64
	last      *Bucket
	current   Bucket
}

// NewWindow creates a new window of the given window size e.g. the index range for each bucket.
func NewWindow(size int64) *Window {
	return &Window{
		size:    size,
		current: NewBucket(0, int(size)),
	}
}

// Push adds an element to a window at the given index.
// returns if the window closed, e.g. if last element initiated a new bucket.
// (Note that based on this logic we ll only know when a window closed only on the initiation of a new one)
// and the index of the closed window.
func (w *Window) Push(index int64, value ...float64) (int64, bool) {

	ready := false

	lastIndex := w.lastIndex

	if index == 0 {
		// new start ...
		w.lastIndex = index
		w.current = NewBucket(index, len(value))
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

		w.current = NewBucket(index, len(value))
		w.lastIndex = index
	}

	w.current.Push(w.lastIndex, value...)

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
// It will return true, if the last addition caused a bucket to close.
func (tw *TimeWindow) Push(t time.Time, v ...float64) (*Bucket, bool) {

	// TODO : provide a inverse hash operation
	index := t.Unix() / tw.duration

	index, closed := tw.window.Push(index, v...)

	if closed {
		tw.index = index
		bucket := tw.window.Get()
		bucket.time = time.Unix(bucket.index*tw.duration, 0)
		return &bucket, true
	}

	return nil, false

}

// Next returns the next timestamp for the coming window.
func (tw *TimeWindow) Next(iterations int64) time.Time {
	nextIndex := tw.index + tw.duration*(iterations+1)
	return time.Unix(nextIndex*int64(time.Second.Seconds()), 0)
}

// HistoryWindow keeps the last x buckets based on the window interval given
type HistoryWindow struct {
	window  *TimeWindow
	buckets *BucketRing
}

// NewHistoryWindow creates a new history window.
func NewHistoryWindow(duration time.Duration, size int) *HistoryWindow {
	return &HistoryWindow{
		window:  NewTimeWindow(duration),
		buckets: NewBucketRing(size),
	}
}

// Push adds an element to the given time index.
// It will return true, if there was a new bucket completed at the last operation
func (h *HistoryWindow) Push(t time.Time, v ...float64) (*Bucket, bool) {
	if bucket, ok := h.window.Push(t, v...); ok {
		h.buckets.Push(bucket)
		return bucket, true
	}
	return nil, false
}

// Get returns the transformed bucket value at the corresponding index.
func (h *HistoryWindow) Get(transform Transform) []interface{} {
	return h.buckets.Get(transform)
}

// SizeWindow is a window indexed by the current time.
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
