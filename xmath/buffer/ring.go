package buffer

import (
	"math"

	xmath2 "github.com/drakos74/go-ex-machina/xmath"
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

// VectorRing is a temporary cache of vectors
// it re-uses a slice of vectors (matrix) and keeps track of the starting index.
// In that sense it s effectively a ring.
// A major differentiating factor is the (+1) logic,
// where the last element is handled separately.
type VectorRing struct {
	idx int
	mem xmath2.Matrix
}

func NewVectorRing(n int) *VectorRing {
	return &VectorRing{
		mem: xmath2.Mat(n),
	}
}

func NewSplitVectorRing(n int) *VectorRing {
	return &VectorRing{
		mem: xmath2.Mat(n + 1),
	}
}

// Push adds an element to the window.
func (w *VectorRing) Push(v xmath2.Vector) (xmath2.Matrix, bool) {
	w.mem[w.idx%len(w.mem)] = v
	w.idx++
	if w.isReady() {
		batch := w.batch()
		return batch, true
	}
	return nil, false
}

// isReady returns true if we completed the batch requirements.
func (w *VectorRing) isReady() bool {
	return w.idx >= len(w.mem)
}

// batch returns the current batch.
func (w VectorRing) batch() xmath2.Matrix {
	m := xmath2.Mat(len(w.mem))
	for i := 0; i < len(w.mem); i++ {
		ii := w.next(i)
		m[i] = w.mem[ii]
	}
	return m
}

func (w VectorRing) Copy() VectorRing {
	m := w.batch()
	return VectorRing{
		idx: w.idx,
		mem: m,
	}
}

func (w VectorRing) next(i int) int {
	return (w.idx + i) % len(w.mem)
}

func (w VectorRing) Size() int {
	return len(w.mem)
}
