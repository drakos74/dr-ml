package math

import (
	"math"
	"math/rand"
	"time"
)

// TODO : clarify which methods mutate the vector and which not

// Vector is an alias for a one dimensional array
type Vector []float64

// Vec creates a new vector
func Vec(dim int) Vector {
	v := make([]float64, dim)
	return v
}

// With applies the given elements in the corresponding positions of the vector
func (v Vector) With(w ...float64) Vector {
	MustHaveSameSize(v, w)
	for i, vv := range w {
		v[i] = vv
	}
	return v
}

// Prod returns the hadamard product of the given vectors
func (v Vector) Prod(w Vector) Vector {
	MustHaveSameSize(v, w)
	z := Vec(len(v))
	for i := 0; i < len(v); i++ {
		z[i] = v[i] * w[i]
	}
	return z
}

// Dot returns the dot product of the 2 vectors
func (v Vector) Dot(w Vector) float64 {
	MustHaveSameSize(v, w)
	var p float64
	for i := 0; i < len(v); i++ {
		p += v[i] * w[i]
	}
	return p
}

// Add adds 2 vectors
func (v Vector) Add(w Vector) Vector {
	MustHaveSameSize(v, w)
	z := Vec(len(v))
	for i := 0; i < len(v); i++ {
		z[i] = v[i] + w[i]
	}
	return z
}

// Diff returns the difference of the corresponding elements between the given vectors
func (v Vector) Diff(w Vector) Vector {
	return v.Add(w.Mult(-1))
}

// Pow returns a vector with all the elements to the given power
func (v Vector) Pow(p float64) Vector {
	return v.Op(func(x float64) float64 {
		return math.Pow(x, p)
	})
}

// Mult multiplies a vector with a constant number
func (v Vector) Mult(s float64) Vector {
	return v.Op(func(x float64) float64 {
		return x * s
	})
}

// Round rounds all elements of the given vector
func (v Vector) Round() Vector {
	return v.Op(math.Round)
}

// Sum returns the sum of all elements of the vector
func (v Vector) Sum() float64 {
	var sum float64
	for i := 0; i < len(v); i++ {
		sum += v[i]
	}
	return sum
}

// Norm returns the norm of the vector
func (v Vector) Norm() float64 {
	var sum float64
	for i := 0; i < len(v); i++ {
		sum += math.Pow(v[i], 2)
	}
	return math.Sqrt(sum)
}

// Op applies to each of the elements a specific function
func (v Vector) Op(transform Op) Vector {
	w := Vec(len(v))
	for i := range v {
		w[i] = transform(v[i])
	}
	return w
}

// Dop applies to each of the elements a specific function based on the elements index
func (v Vector) Dop(transform Dop, w Vector) Vector {
	z := Vec(len(v))
	for i := range v {
		z[i] = transform(v[i], w[i])
	}
	return z
}

// VectorGenerator is a type alias defining the creation instructions for vectors
type VectorGenerator func(p, index int) Vector

// Def defines the vector at the corresponding index
var Def = func(m ...Vector) VectorGenerator {
	return func(p, index int) Vector {
		MustHaveSize(m[index], p)
		return m[index]
	}
}

// Rand generates a vector of the given size with random values between 0 and 1
var Rand = func() VectorGenerator {
	return func(p, index int) Vector {
		rand.Seed(time.Now().UnixNano())
		w := Vec(p)
		for i := 0; i < p; i++ {
			w[i] = rand.Float64()
		}
		return w
	}
}

// Const generates a vector of the given size with constant values
var Const = func(v float64) VectorGenerator {
	return func(p, index int) Vector {
		w := Vec(p)
		for i := 0; i < p; i++ {
			w[i] = v
		}
		return w
	}
}
