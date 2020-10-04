package xmath

func Inp(s Matrix) Matrix {
	return Mat(len(s) - 1).With(s[:len(s)-1]...)
}

func Outp(s Matrix) Matrix {
	return Mat(len(s) - 1).With(s[1:]...)
}

// Window is a temporary cache of vectors
// it re-uses a slice of vectors (matrix) and keeps track of the starting index.
// In that sense it s effectively ring.
// A major differentiating factor is the (+1) logic,
// where the last element is handled separately.
type Window struct {
	idx int
	mem Matrix
}

func NewWindow(n int) *Window {
	return &Window{
		mem: Mat(n),
	}
}

func NewSplitWindow(n int) *Window {
	return &Window{
		mem: Mat(n + 1),
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
		ii := w.next(i)
		m[i] = w.mem[ii]
	}
	return m
}

func (w Window) next(i int) int {
	return (w.idx + i) % len(w.mem)
}

func (w Window) Size() int {
	return len(w.mem)
}
