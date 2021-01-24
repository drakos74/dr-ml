package time

import (
	"github.com/drakos74/go-ex-machina/xmath"
)

// Window is a temporary cache of vectors
// it re-uses a slice of vectors (matrix) and keeps track of the starting index.
// In that sense it s effectively a ring.
// A major differentiating factor is the (+1) logic,
// where the last element is handled separately.
type Window struct {
	idx int
	mem xmath.Matrix
}

func NewWindow(n int) *Window {
	return &Window{
		mem: xmath.Mat(n),
	}
}

func NewSplitWindow(n int) *Window {
	return &Window{
		mem: xmath.Mat(n + 1),
	}
}

// Push adds an element to the window.
func (w *Window) Push(v xmath.Vector) (xmath.Matrix, bool) {
	w.mem[w.idx%len(w.mem)] = v
	w.idx++
	if w.isReady() {
		batch := w.batch()
		return batch, true
	}
	return nil, false
}

// isReady returns true if we completed the batch requirements.
func (w *Window) isReady() bool {
	return w.idx >= len(w.mem)
}

// batch returns the current batch.
func (w Window) batch() xmath.Matrix {
	m := xmath.Mat(len(w.mem))
	for i := 0; i < len(w.mem); i++ {
		ii := w.next(i)
		m[i] = w.mem[ii]
	}
	return m
}

func (w Window) Copy() Window {
	m := w.batch()
	return Window{
		idx: w.idx,
		mem: m,
	}
}

func (w Window) next(i int) int {
	return (w.idx + i) % len(w.mem)
}

func (w Window) Size() int {
	return len(w.mem)
}
