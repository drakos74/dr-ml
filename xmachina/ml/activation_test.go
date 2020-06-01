package ml

import (
	"fmt"
	"log"
	"math"
	"strconv"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath"
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
			back := s.D(x)
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
		v := xmath.Rand(0, 10, xmath.Unit)(100, 0)

		w := softmax.F(v)

		assert.Equal(t, "1.00", strconv.FormatFloat(w.Sum(), 'f', 2, 64))

	}

}

func TestSoftMax_Derivative(t *testing.T) {

	softmax := SoftMax{}

	v := xmath.Vec(3).With(10, 20, 5)

	w := softmax.F(v)

	println(fmt.Sprintf("w = %v", w))

	r := xmath.Vec(3).With(0.3, 0.3, 0.4)

	diff := Diff(r, w)
	println(fmt.Sprintf("diff = %v", diff))

	println(fmt.Sprintf("diff.Sum() = %v", diff.Sum()))

	err := softmax.D(diff)

	println(fmt.Sprintf("err = %v", err))

	println(fmt.Sprintf("errSum = %v", err.T().Sum()))
}
