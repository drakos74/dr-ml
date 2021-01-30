package ml

import (
	"fmt"
	"log"
	"math"
	"strconv"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath/algebra"

	"github.com/stretchr/testify/assert"
)

func TestSigmoid_Function(t *testing.T) {

	s := sigmoid{}

	for i := -100; i < 100; i++ {
		x := float64(i) * 0.01
		y := s.F(x)
		log.Println(fmt.Sprintf("x = %v -> y = %v", x, y))
	}
}

func TestSigmoid_Derivative(t *testing.T) {

	s := sigmoid{}

	var x0 float64
	var y0 float64
	for i := -100; i < 100; i++ {
		x := float64(i) * 0.1
		y := s.F(x)
		if x0 != 0 && y0 != 0 {
			// calculate the derivative approximately
			drv := (y - y0) / (x - x0)
			back := s.D(s.F(x))
			assert.True(t, math.Abs(drv-back) < 0.01, fmt.Sprintf("x = %v -> y = %v -> dy/dx = %v, b = %v , err = %v", x, y, drv, back, drv-back))
		}
		x0 = x
		y0 = y
	}
}

func TestTanH_Function(t *testing.T) {

	tan := tanH{}

	for i := -100; i < 100; i++ {
		x := float64(i) * 0.01
		y := tan.F(x)
		log.Println(fmt.Sprintf("x = %v -> y = %v", x, y))
	}
}

func TestTanH_Derivative(t *testing.T) {

	tanh := tanH{}

	var x0 float64
	var y0 float64
	for i := -100; i < 100; i++ {
		x := float64(i) * 0.1
		y := tanh.F(x)
		if x0 != 0 && y0 != 0 {
			// calculate the derivative approximately
			drv := (y - y0) / (x - x0)
			back := tanh.D(y)
			assert.True(t, math.Abs(drv-back) < 0.05, fmt.Sprintf("x = %v -> y = %v -> dy/dx = %v, b = %v , err = %v", x, y, drv, back, drv-back))
		}
		x0 = x
		y0 = y
	}

}

func TestSoftMax_Function(t *testing.T) {

	softmax := SoftMax{}

	for i := 0; i < 100; i++ {
		v := algebra.Rand(0, 10, algebra.Unit)(100, 0)
		w := softmax.F(v)
		assert.Equal(t, "1.00", strconv.FormatFloat(w.Sum(), 'f', 2, 64))
	}
}

// numbers taken from https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
func TestSoftMax_Derivative(t *testing.T) {

	softmax := SoftMax{}

	x0 := algebra.Vec(2).With(1, 2)
	println(fmt.Sprintf("x0 = %v", x0))
	y0 := softmax.F(x0)
	println(fmt.Sprintf("y0 = %v", y0))

	// check that softmax sum is 1.
	assert.True(t, math.Abs(1-y0.Sum()) < 0.01)

	div := softmax.D(y0)
	assert.Equal(t, algebra.Mat(2).With(
		algebra.Vec(2).With(0.19661193, -0.19661193),
		algebra.Vec(2).With(-0.19661193, 0.19661193),
	), div.Op(algebra.Round(8)))

}
