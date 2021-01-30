package datastruct

import (
	"math"
)

// Ring acts like a ring buffer keeping the last x elements
type Ring struct {
	index  int
	count  int
	values []float64
}

// Size returns the number of non-nil elements within the ring.
func (r *Ring) Size() int {
	if r.count == r.index {
		return r.count
	}
	return len(r.values)
}

// NewRing creates a new ring with the given buffer size.
func NewRing(size int) *Ring {
	return &Ring{
		values: make([]float64, size),
	}
}

// Push adds an element to the ring.
func (r *Ring) Push(v float64) {
	r.values[r.index] = v
	r.index = r.next(r.index)
	r.count++
}

func (r *Ring) next(index int) int {
	return (index + 1) % len(r.values)
}

// Get returns an ordered slice of the ring elements
func (r *Ring) Get() []float64 {
	v := make([]float64, len(r.values))
	for i := 0; i < len(r.values); i++ {
		idx := i
		if r.count > len(r.values) {
			idx = r.next(r.index - 1 + i)
		}
		v[i] = r.values[idx]
	}
	return v
}

// Get returns an ordered slice of the ring elements
func (r *Ring) Aggregate(process Func) float64 {
	s := 0.0
	for i := 0; i < len(r.values); i++ {
		s = process(s, r.values[i])
	}
	return s / float64(len(r.values))
}

// Transform is a operation acting on a bucket and returning a float.
// It is used to get the relevant bucket metric, without the need to make repeated iterations.
type Transform func(bucket *Bucket) interface{}

// BucketRing acts like a ring buffer keeping the last x elements
type BucketRing struct {
	index  int
	count  int
	values []*Bucket
}

// Size returns the number of non-nil elements within the ring.
func (r *BucketRing) Size() int {
	if r.count == r.index {
		return r.count
	}
	return len(r.values)
}

// NewBucketRing creates a new ring with the given buffer size.
func NewBucketRing(size int) *BucketRing {
	return &BucketRing{
		values: make([]*Bucket, size),
	}
}

// Push adds an element to the ring.
func (r *BucketRing) Push(v *Bucket) {
	r.values[r.index] = v
	r.index = r.next(r.index)
	r.count++
}

func (r *BucketRing) next(index int) int {
	return (index + 1) % len(r.values)
}

// Get returns an ordered slice of the ring elements
func (r *BucketRing) Get(transform Transform) []interface{} {

	l := len(r.values)
	if r.count < l {
		l = r.count
	}

	v := make([]interface{}, l)
	for i := 0; i < l; i++ {
		idx := i
		if r.count > l {
			idx = r.next(r.index - 1 + i)
		}
		v[i] = transform(r.values[idx])
	}
	return v
}

type Func func(p, v float64) float64

func Sum(s, v float64) float64 {
	return s + v
}

func Pow(p float64) Func {
	return func(s, v float64) float64 {
		return s + math.Pow(v, p)
	}
}
