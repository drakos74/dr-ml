package math

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

type N uint

type R float64

type Vector []float64

func NewVector(dim int) Vector {
	v := make([]float64, dim)
	return v
}

func (v Vector) From(w []float64) Vector {
	MustHaveSameSize(v, w)
	for i, vv := range w {
		v[i] = vv
	}
	return v
}

func (v Vector) Dot(w Vector) float64 {
	MustHaveSameSize(v, w)
	var p float64
	for i := 0; i < len(v); i++ {
		p += v[i] * w[i]
	}
	return p
}

func (v Vector) Diff(w Vector) Vector {
	return v.Add(w.Mult(-1))
}

func (v Vector) Add(w Vector) Vector {
	MustHaveSameSize(v, w)
	z := NewVector(len(v))
	for i := 0; i < len(v); i++ {
		z[i] = v[i] + w[i]
	}
	return z
}

func (v Vector) Mult(s float64) Vector {
	z := NewVector(len(v))
	for i := 0; i < len(v); i++ {
		z[i] = v[i] * s
	}
	return z
}

func (v Vector) Round() Vector {
	abs := NewVector(len(v))
	for i := 0; i < len(v); i++ {
		abs[i] = math.Round(v[i])
	}
	return abs
}

func (v Vector) Abs() Vector {
	abs := NewVector(len(v))
	for i := 0; i < len(v); i++ {
		abs[i] = math.Abs(v[i])
	}
	return abs
}

func (v Vector) Norm() float64 {
	var sum float64
	for i := 0; i < len(v); i++ {
		sum += math.Pow(v[i], 2)
	}
	return math.Sqrt(sum)
}

type VectorGenerator func(p int) Vector

// Rand generates a vector of the given size with random values between 0 and 1
var Rand VectorGenerator = func(p int) Vector {
	rand.Seed(time.Now().UnixNano())
	w := NewVector(p)
	for i := 0; i < p; i++ {
		w[i] = rand.Float64()
	}
	return w
}

// Const generates a vector of the given size with constant values
var Const = func(v float64) VectorGenerator {
	return func(p int) Vector {
		w := NewVector(p)
		for i := 0; i < p; i++ {
			w[i] = v
		}
		return w
	}
}

type Matrix []Vector

func NewMatrix(m int) Matrix {
	mat := make([]Vector, m)
	return mat
}

func (m Matrix) From(v Vector) Matrix {
	for i := range m {
		m[i] = v
	}
	return m
}

func (m Matrix) With(v ...Vector) Matrix {
	for i := range m {
		m[i] = v[i]
	}
	return m
}

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

type Cube []Matrix

func NewCube(d int) Cube {
	cube := make([]Matrix, d)
	return cube
}

func (c Cube) String() string {
	builder := strings.Builder{}
	builder.WriteString("\n")
	for i := 0; i < len(c); i++ {
		builder.WriteString(fmt.Sprintf("[%d]", i))
		builder.WriteString("\n")
		builder.WriteString(fmt.Sprintf("%v", c[i].String()))
	}
	return builder.String()
}

func MustHaveSameSize(v, w Vector) {
	if len(v) != len(w) {
		panic(fmt.Sprintf("vectors must have the same size '%v' vs '%v'", v, w))
	}
}
