package xmath

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
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

// Dot returns the dot product of the 2 vectors
func (v Vector) Dot(w Vector) float64 {
	MustHaveSameSize(v, w)
	var p float64
	for i := 0; i < len(v); i++ {
		p += v[i] * w[i]
	}
	return p
}

// ProdH returns the product of the given vectors
func (v Vector) Prod(w Vector) Matrix {
	z := Mat(len(v)).Of(len(w))
	for i := 0; i < len(v); i++ {
		for j := 0; j < len(w); j++ {
			z[i][j] = v[i] * w[j]
		}
	}
	return z
}

// ProdH returns the hadamard product of the given vectors
func (v Vector) ProdH(w Vector) Vector {
	MustHaveSameSize(v, w)
	z := Vec(len(v))
	for i := 0; i < len(v); i++ {
		z[i] = v[i] * w[i]
	}
	return z
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
	return v.Dop(func(x, y float64) float64 {
		return x - y
	}, w)
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

// Copy copies the vector into a new one with the same values
// this is for cases where we want to apply mutations, but would like to leave the initial vector intact
func (v Vector) Copy() Vector {
	w := Vec(len(v))
	for i := 0; i < len(v); i++ {
		w[i] = v[i]
	}
	return w
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

// String prints the vector in an easily readable form
func (v Vector) String() string {
	builder := strings.Builder{}
	builder.WriteString(fmt.Sprintf("(%d)", len(v)))
	builder.WriteString("[ ")
	for i := 0; i < len(v); i++ {
		ss := ""
		if v[i] > 0 {
			ss = " "
		}
		builder.WriteString(fmt.Sprintf("%s%s", ss, strconv.FormatFloat(v[i], 'f', pp, 64)))
		if i < len(v)-1 {
			// dont add the comma to the last element
			builder.WriteString(" , ")
		}
	}
	builder.WriteString(" ]")
	return builder.String()
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
var Rand = func(min, max float64, op Op) VectorGenerator {
	rand.Seed(time.Now().UnixNano())
	return func(p, index int) Vector {
		mmin := min / op(float64(p))
		mmax := max / op(float64(p))
		w := Vec(p)
		for i := 0; i < p; i++ {
			w[i] = rand.Float64()*(mmax-mmin) + mmin
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
