package buffer

import "github.com/drakos74/go-ex-machina/xmath"

// Inp keeps all rows of a matrix except the last.
func Inp(s xmath.Matrix) xmath.Matrix {
	return xmath.Mat(len(s) - 1).With(s[:len(s)-1]...)
}

// Outp keeps all rows of a matrix except for the first.
func Outp(s xmath.Matrix) xmath.Matrix {
	return xmath.Mat(len(s) - 1).With(s[1:]...)
}
