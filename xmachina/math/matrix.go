package math

import (
	"fmt"
	"strings"
)

type Matrix []Vector

// Diag creates a new diagonal Matrix with the given elements in the diagonal
func Diag(v Vector) Matrix {
	m := Mat(len(v))
	for i := range v {
		m[i] = Vec(len(v))
		m[i][i] = v[i]
	}
	return m
}

// Mat creates a newMatrix of the given dimension
func Mat(m int) Matrix {
	mat := make([]Vector, m)
	return mat
}

// T calculates the transpose of a matrix
func (m Matrix) T() Matrix {
	n := Mat(len(m[0]))
	for i := range m {
		for j := range m[i] {
			if n[j] == nil {
				n[j] = Vec(len(m))
			}
			n[j][i] = m[i][j]
		}
	}
	return n
}

// Sum returns a vector that carries the sum of all elements for each row of the Matrix
func (m Matrix) Sum() Vector {
	v := Vec(len(m))
	for i := range m {
		v[i] = m[i].Sum()
	}
	return v
}

// Prod returns the product of the given vector with the matrix
func (m Matrix) Prod(v Vector) Vector {
	w := Vec(len(m))
	for i := range m {
		MustHaveSameSize(m[i], v)
		w[i] = m[i].Dot(v)
	}
	return w
}

// With creates a matrix with the given vector replicated at each row
func (m Matrix) From(v Vector) Matrix {
	for i := range m {
		m[i] = v
	}
	return m
}

// With applies the elements of the given vectors to the corresponding positions in the matrix
func (m Matrix) With(v ...Vector) Matrix {
	for i := range m {
		m[i] = v[i]
	}
	return m
}

// String prints the matrix in an easily readable form
func (m Matrix) String() string {
	builder := strings.Builder{}
	builder.WriteString("\n")
	for i := 0; i < len(m); i++ {
		builder.WriteString("\t")
		builder.WriteString(fmt.Sprintf("[%d]", i))
		builder.WriteString(fmt.Sprintf("%v", m[i]))
		builder.WriteString("\n")
	}
	return builder.String()
}
