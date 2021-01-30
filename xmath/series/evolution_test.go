package series

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIncNum(t *testing.T) {

	a := 2.0
	v := a

	w := 4.0

	procedure := NewSequence(&v, IncNum(w, 0), 50)

	j := 0
	for i := 0; i < 100; i++ {

		done := procedure.Next()

		j++
		if j >= 50 {
			assert.True(t, done)
		} else {
			assert.False(t, done)
		}

		evolution := float64(i) * w
		assert.Equal(t, a+evolution, v)
	}

}

func TestIncMul(t *testing.T) {

	a := 2.0
	v := a

	w := 3.0

	procedure := NewSequence(&v, IncMul(w, 0), 3)

	j := 0
	for i := 0; i < 10; i++ {
		done := procedure.Next()

		j++
		if j >= 3 {
			assert.True(t, done)
		} else {
			assert.False(t, done)
		}

		evolution := math.Pow(w, float64(i))
		assert.Equal(t, fmt.Sprintf("%.2f", a*evolution), fmt.Sprintf("%.2f", v))
	}

}

func TestProcedure_Reset(t *testing.T) {

	v := 2.0
	proc := NewSequence(&v, IncMul(2, 0), 5)

	for i := 0; i < 100; i++ {
		reset := proc.Next()
		if reset {
			// assert reset should happen every 5 iterations
			assert.Equal(t, 0, (i+1)%5)
			proc.Reset()
		}
	}

}

func TestEvolution_Run(t *testing.T) {

	a := 1.0
	b := 10.0
	c := 100.0

	ev := NewEvolution(
		NewSequence(&a, IncNum(1.0, 0), 3),
		NewSequence(&b, IncNum(10.0, 0), 3),
		NewSequence(&c, IncNum(100.0, 0), 3),
	)

	var count int

	evolutions := make(map[string]struct{})

	for ev.Next() {
		k := fmt.Sprintf("[%v,%v,%v]", a, b, c)
		// check that every series is a unnique combination
		_, ok := evolutions[k]
		assert.False(t, ok)
		evolutions[k] = struct{}{}
		count++
	}

	assert.Equal(t, 3*3*3, count)

}

func TestRangeSequence(t *testing.T) {

	var value float64

	sequence := RangeSequence(&value, 1, 100, 10, 0)

	for i := 0; i < 100; i++ {
		reset := sequence.Next()
		if reset {
			// assert reset should happen every 10 iterations
			assert.Equal(t, 0, (i+1)%10)
			sequence.Reset()
		}
	}

}

func TestPerturbationSequence(t *testing.T) {

	value := 10.0

	sequence := PerturbationSequence(&value, 1, 10, 0)

	for i := 0; i < 100; i++ {
		reset := sequence.Next()
		if reset {
			// assert reset should happen every 10 iterations
			assert.Equal(t, 0, (i+1)%10)
			sequence.Reset()
		}
	}

}
