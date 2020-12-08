package time

import (
	"fmt"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/stretchr/testify/assert"
)

func TestWindow_Add(t *testing.T) {

	size := 10

	w := NewWindow(10)

	for i := 0; i < 125; i++ {
		isReady := w.Push(xmath.Vec(1).With(float64(i)))
		if i < size {
			assert.False(t, isReady)
		} else {
			assert.True(t, isReady)
		}

		if isReady {
			batch := w.Batch()
			// batch size should be consistent
			assert.Equal(t, size, len(batch))
			// first batch element should be 10 elements back
			assert.Equal(t, float64(i-size), batch[0][0])
			// last element should be the last inserted
			assert.Equal(t, float64(i), Outp(batch)[0])
		}
	}
}

func TestWindow_Copy(t *testing.T) {

	// TODO : ...

}

func TestSequence_Get(t *testing.T) {

	const l = 10

	s := xmath.Mat(l)

	for i := 0; i < l; i++ {
		s[i] = xmath.Vec(1).With(float64(i))
	}

	println(fmt.Sprintf("seq = %v", s))
	println(fmt.Sprintf("inp = %v", Inp(s)))
	println(fmt.Sprintf("outp = %v", Outp(s)))

	println(fmt.Sprintf("s = %v", s))

}
