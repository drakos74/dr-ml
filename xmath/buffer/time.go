package buffer

import "time"

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
func (tw *TimeWindow) Push(t time.Time, v ...float64) (time.Time, *Bucket, bool) {

	// TODO : provide a inverse hash operation
	index := t.Unix() / tw.duration

	index, closed := tw.window.Push(index, v...)

	if closed {
		tw.index = index
		bucket := tw.window.Get()
		return time.Unix(bucket.index*tw.duration, 0), &bucket, true
	}

	return time.Time{}, nil, false

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
func (h *HistoryWindow) Push(t time.Time, v ...float64) (time.Time, *Bucket, bool) {
	if t, bucket, ok := h.window.Push(t, v...); ok {
		h.buckets.Push(bucket)
		return t, bucket, true
	}
	return time.Time{}, nil, false
}

// Get returns the transformed bucket value at the corresponding index.
func (h *HistoryWindow) Get(transform Transform) []interface{} {
	return h.buckets.Get(transform)
}
