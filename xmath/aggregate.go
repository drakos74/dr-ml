package xmath

import (
	"fmt"
	"math"
)

type Bucket struct {
	min   float64
	max   float64
	sum   float64
	count int
	aggr  int
}

func NewBucket() Bucket {
	return Bucket{min: math.MaxFloat64, max: -1 * math.MaxFloat64}
}

func NewBucketFromVector(v Vector) Bucket {
	b := NewBucket()
	b.Aggregate(v)
	return b
}

func (b *Bucket) Aggregate(v Vector) {
	for i := range v {
		b.Add(v[i])
	}
}

func (b *Bucket) Add(x float64) {
	if x > b.max {
		b.max = x
	}
	if x < b.min {
		b.min = x
	}
	b.sum += x
	b.count++
}

func (b *Bucket) Merge(bucket Bucket) {
	if b.count != bucket.count {
		panic(fmt.Sprintf("cannot merge different size buckets %v vs %v", b.count, bucket.count))
	}
	if bucket.max > b.max {
		b.max = bucket.max
	}
	if bucket.min < b.min {
		b.min = bucket.min
	}
	b.sum += bucket.sum
	b.aggr += 1

}

type Aggregate struct {
	buckets map[int]Bucket
}

func NewAggregate() Aggregate {
	return Aggregate{buckets: make(map[int]Bucket)}
}

func (a *Aggregate) Add(b Bucket) {
	if bucket, ok := a.buckets[b.count]; ok {
		bucket.Merge(b)
		a.buckets[b.count] = bucket
	} else {
		a.buckets[b.count] = b
	}
}

type Window struct {
	idx int
	mem Matrix
}

func NewWindow(n int) *Window {
	return &Window{
		mem: Mat(n),
	}
}

func (w *Window) Push(v Vector) bool {
	w.mem[w.idx%len(w.mem)] = v
	w.idx++
	return w.idx >= len(w.mem)
}

func (w Window) Batch() Matrix {
	m := Mat(len(w.mem))
	for i := 0; i < len(w.mem); i++ {
		ii := (w.idx + i) % len(w.mem)
		m[i] = w.mem[ii]
	}
	return m
}

func (w Window) Size() int {
	return len(w.mem)
}
