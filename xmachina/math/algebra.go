package math

import (
	"fmt"
)

// Op is a general mathematical operation from one number to another
type Op func(x float64) float64

var Unit Op = func(x float64) float64 {
	return x
}

// Dop is a general mathematical operation from 2 numbers to another
type Dop func(x, y float64) float64

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
