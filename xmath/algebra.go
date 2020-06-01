package xmath

import (
	"fmt"
	"math"
)

type N interface {
	Op(op Op) N
	Dop(dop Dop) N
}

// pp is the print precision for floats
const pp = 8

// Op is a general mathematical operation from one number to another
type Op func(x float64) float64

func Clip(min, max float64) Op {
	return func(x float64) float64 {
		if x < min {
			return min
		}
		if x > max {
			return max
		}
		return x
	}
}

func Scale(s float64) Op {
	return func(x float64) float64 {
		return x * s
	}
}

func Add(c float64) Op {
	return func(x float64) float64 {
		return x + c
	}
}

var Unit Op = func(x float64) float64 {
	return 1
}

var Sqrt Op = func(x float64) float64 {
	return math.Sqrt(x)
}

var Square Op = func(x float64) float64 {
	return math.Pow(x, 2)
}

// Dop is a general mathematical operation from 2 numbers to another
type Dop func(x, y float64) float64

var Mult Dop = func(x, y float64) float64 {
	return x * y
}

var Div Dop = func(x, y float64) float64 {
	return x / (y + 1e-8)
}

type Vop func(x Vector) Vector

var Unary Vop = func(x Vector) Vector {
	return x
}

// MustHaveSize will check and make sure that the given vector is of the given size
func MustHaveDim(m Matrix, n int) {
	if len(m) != n {
		panic(fmt.Sprintf("matrix must have primary dimenstion '%v' vs '%v'", m, n))
	}
}

// MustHaveSize will check and make sure that the given vector is of the given size
func MustHaveSize(v Vector, n int) {
	if len(v) != n {
		panic(fmt.Sprintf("vector must have size '%v' vs '%v'", v, n))
	}
}

// MustHaveSameSize verifies if the given vectors are of the same size
func MustHaveSameSize(v, w Vector) {
	if len(v) != len(w) {
		panic(fmt.Sprintf("vectors must have the same size '%v' vs '%v'", v, w))
	}
}
