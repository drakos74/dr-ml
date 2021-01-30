package buffer

import (
	"fmt"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/stretchr/testify/assert"
)

func TestNewRing_Push(t *testing.T) {

	ring := NewRing(10)

	for i := 0; i < 102; i++ {
		ring.Push(float64(i))
		if ring.Size() != i+1 {
			t.Fatalf("unexpected size %d vs. %d", ring.Size(), i)
			return
		}
		println(fmt.Sprintf("ring.values = %v", ring.values))
		println(fmt.Sprintf("ring.Get() = %v", ring.Get()))
		println(fmt.Sprintf("ring.Aggregate(Sum) = %v", ring.Aggregate(Sum)))
	}

}

func TestVectorRing_Add(t *testing.T) {

	size := 10

	w := NewVectorRing(10)

	for i := 1; i < 25; i++ {
		batch, isReady := w.Push(xmath.Vec(1).With(float64(i)))
		if i < size {
			assert.False(t, isReady)
		} else {
			assert.True(t, isReady)
		}

		if isReady {
			// batch size should be consistent
			assert.Equal(t, size, len(batch))
			// first batch element should be 10 elements back
			assert.Equal(t, float64(i-size)+1, batch[0][0])
			// last element should be the last inserted
			assert.Equal(t, float64(i), batch[len(batch)-1][0])
		}
	}
}

func TestVectorRing_Copy(t *testing.T) {

	// TODO : ...

}

func TestSequence_Get(t *testing.T) {

	const l = 10

	s := xmath.Mat(l)

	for i := 0; i < l; i++ {
		s[i] = xmath.Vec(1).With(float64(i))
	}

	println(fmt.Sprintf("seq = %v", s))
	println(fmt.Sprintf("s = %v", s))

}
