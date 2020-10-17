package xmath

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

	procedure := NewProcedure(&v, IncNum(w), 50)

	j := 0
	for i := 0; i < 100; i++ {

		done := procedure.Next()

		j++
		if j >= 50 {
			assert.True(t, done)
		} else {
			assert.False(t, done)
		}

		evolution := float64(i+1) * w
		assert.Equal(t, a+evolution, v)
	}

}

func TestIncMul(t *testing.T) {

	a := 2.0
	v := a

	w := 3.0

	procedure := NewProcedure(&v, IncMul(w), 3)

	j := 0
	for i := 0; i < 10; i++ {
		done := procedure.Next()

		j++
		if j >= 3 {
			assert.True(t, done)
		} else {
			assert.False(t, done)
		}

		evolution := math.Pow(w, float64(i+1))
		assert.Equal(t, fmt.Sprintf("%.2f", a*evolution), fmt.Sprintf("%.2f", v))
	}

}

func TestProcedure_Reset(t *testing.T) {

	v := 2.0
	proc := NewProcedure(&v, IncMul(2), 5)

	for i := 0; i < 100; i++ {

		reset := proc.Next()
		println(fmt.Sprintf("v = %v", v))
		if reset {
			proc.Reset()
		}

	}

}

func TestEvolution_Run(t *testing.T) {

	a := 0.0
	b := 0.0
	c := 0.0

	ev := NewEvolution(
		NewProcedure(&a, IncNum(1.0), 3),
		NewProcedure(&b, IncNum(10.0), 3),
		NewProcedure(&c, IncNum(100.0), 3),
	)

	var count int

	var done bool
	for !done {
		done = ev.Next()
		println(fmt.Sprintf("[%v,%v,%v]", a, b, c))
		count++
	}

	assert.Equal(t, 3*3*3, count)

}
