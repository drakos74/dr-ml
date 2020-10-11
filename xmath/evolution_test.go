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

	procedure := NewProcedure(&v, IncNum(w))

	for i := 0; i < 100; i++ {
		procedure.Next()
		evolution := float64(i+1) * w
		assert.Equal(t, a+evolution, v)
	}

}

func TestIncMul(t *testing.T) {

	a := 2.0
	v := a

	w := 3.0

	procedure := NewProcedure(&v, IncMul(w))

	for i := 0; i < 10; i++ {
		procedure.Next()
		evolution := math.Pow(w, float64(i+1))
		assert.Equal(t, fmt.Sprintf("%.2f", a*evolution), fmt.Sprintf("%.2f", v))
	}

}
