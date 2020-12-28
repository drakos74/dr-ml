package xmath

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

type N interface {
	Op(op Op) N
	Dop(dop Dop) N
}

// Check checks if the given number is a valid one.
func Check(v float64) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		panic(fmt.Sprintf("%v is not a valid number", v))
	}
}

// pp is the print precision for floats
const pp = 8

// Op is a general mathematical operation from one number to another
type Op func(x float64) float64

// Round defines a round operation for the amount of digits after the comma, provided.
func Round(digits int) Op {
	factor := math.Pow(10, float64(digits))
	return func(x float64) float64 {
		return math.Round(factor*x) / factor
	}
}

// Clip clips the given number to the corresponding min or max value.
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

// Scale scales the given number according to the scaling factor provided.
func Scale(s float64) Op {
	return func(x float64) float64 {
		return x * s
	}
}

// Add adds the given number to the argument.
func Add(c float64) Op {
	return func(x float64) float64 {
		return x + c
	}
}

// Unit is a predefined operation that always returns 1.
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
		panic(fmt.Sprintf("matrix must have primary dimension '%v' vs '%v'", m, n))
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
		panic(fmt.Sprintf("vectors must have the same size '%v' vs '%v'", len(v), len(w)))
	}
}

// CartesianProduct finds all possible combinations of the given data matrix.
// follows the same logic as https://stackoverflow.com/questions/53244303/all-combinations-in-array-of-arrays
func CartesianProduct(data [][]float64, current int, length int) [][]float64 {
	result := make([][]float64, 0)
	if current == length {
		return result
	}

	subCombinations := CartesianProduct(data, current+1, length)
	size := len(subCombinations)

	for i := 0; i < len(data[current]); i++ {
		if size > 0 {
			for j := 0; j < size; j++ {
				combinations := make([]float64, 0)
				combinations = append(combinations, data[current][i])
				combinations = append(combinations, subCombinations[j]...)
				result = append(result, combinations)
			}
		} else {
			combinations := make([]float64, 0)
			combinations = append(combinations, data[current][i])
			result = append(result, combinations)
		}
	}

	return result
}

// TODO : clarify which methods mutate the vector and which not

// Vector is an alias for a one dimensional array.
type Vector []float64

// Vec creates a new vector.
func Vec(dim int) Vector {
	v := make([]float64, dim)
	return v
}

// Check checks if the elements of the vetor are well defined
func (v Vector) Check() {
	for _, vv := range v {
		Check(vv)
	}
}

// With applies the given elements in the corresponding positions of the vector
func (v Vector) With(w ...float64) Vector {
	MustHaveSameSize(v, w)
	for i, vv := range w {
		v[i] = vv
	}
	return v
}

// Generate generates values for the vector
func (v Vector) Generate(gen VectorGenerator) Vector {
	return gen(len(v), 0)
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

// Prod returns the product of the given vectors
// it returns a matrix
func (v Vector) Prod(w Vector) Matrix {
	z := Mat(len(v)).Of(len(w))
	for i := 0; i < len(v); i++ {
		for j := 0; j < len(w); j++ {
			z[i][j] = v[i] * w[j]
		}
	}
	return z
}

// X returns the hadamard product of the given vectors.
// e.g. pointwise multiplication
func (v Vector) X(w Vector) Vector {
	MustHaveSameSize(v, w)
	z := Vec(len(v))
	for i := 0; i < len(v); i++ {
		z[i] = v[i] * w[i]
	}
	return z
}

// Stack concatenates 2 vectors , producing another with the sum of their lengths.
func (v Vector) Stack(w Vector) Vector {
	x := Vec(len(v) + len(w))
	return x.With(append(v, w...)...)
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
// s is the size of the vector
type VectorGenerator func(s, index int) Vector

// Row defines a vector at the corresponding row index of a matrix
var Row = func(m ...Vector) VectorGenerator {
	return func(s, index int) Vector {
		MustHaveSize(m[index], s)
		return m[index]
	}
}

// Rand generates a vector of the given size with random values between min and max
// op defines a scaling operation for the min and max, based on the size of the vector
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

// ScaledVectorGenerator produces a vector generator scaled by the given factor
type ScaledVectorGenerator func(d float64) VectorGenerator

// RangeSqrt produces a vector generator scaled by the given factor
// and within the range provided
var RangeSqrt = func(min, max float64) ScaledVectorGenerator {
	return func(d float64) VectorGenerator {
		return Rand(min, max, func(x float64) float64 {
			return math.Sqrt(d * x)
		})
	}
}

// Range produces a vector generator scaled by the given factor
// and within the range provided
var Range = func(min, max float64) ScaledVectorGenerator {
	return func(d float64) VectorGenerator {
		return Rand(min, max, func(x float64) float64 {
			return x
		})
	}
}

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
	n := Mat(len(m[0])).Of(len(m))
	for i := range m {
		for j := range m[i] {
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

// Add returns the addition operation on 2 matrices
func (m Matrix) Add(v Matrix) Matrix {
	w := Mat(len(m))
	for i := range m {
		n := Vec(len(m[i]))
		for j := 0; j < len(m[i]); j++ {
			n[j] = m[i][j] + v[i][j]
		}
		w[i] = n
	}
	return w
}

// Dot returns the product of the given matrix with the matrix
func (m Matrix) Dot(v Matrix) Matrix {
	w := Mat(len(m))
	for i := range m {
		for j := 0; j < len(v); j++ {
			MustHaveSameSize(m[i], v[j])
			w[i][j] = m[i].Dot(v[j])
		}
	}
	return w
}

// Prod returns the cross product of the given vector with the matrix
func (m Matrix) Prod(v Vector) Vector {
	w := Vec(len(m))
	for i := range m {
		MustHaveSameSize(m[i], v)
		w[i] = m[i].Dot(v)
	}
	return w
}

// Mult multiplies each element of the matrix with the given factor
func (m Matrix) Mult(s float64) Matrix {
	n := Mat(len(m))
	for i := range m {
		n[i] = m[i].Mult(s)
	}
	return n
}

// Of initialises the rows of the matrix with vectors of the given length
func (m Matrix) Of(n int) Matrix {
	for i := 0; i < len(m); i++ {
		m[i] = Vec(n)
	}
	return m
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

// Generate generates the rows of the matrix using the generator func
func (m Matrix) Generate(p int, gen VectorGenerator) Matrix {
	for i := range m {
		m[i] = gen(p, i)
	}
	return m
}

// Copy copies the matrix into a new one with the same values
// this is for cases where we want to apply mutations, but would like to leave the initial vector intact
func (m Matrix) Copy() Matrix {
	n := Mat(len(m))
	for i := 0; i < len(m); i++ {
		n[i] = m[i].Copy()
	}
	return n
}

// Op applies to each of the elements a specific function
func (m Matrix) Op(transform Op) Matrix {
	n := Mat(len(m))
	for i := range m {
		n[i] = m[i].Op(transform)
	}
	return n
}

// Op applies to each of the elements a specific function
func (m Matrix) Dop(transform Dop, n Matrix) Matrix {
	w := Mat(len(m))
	for i := range m {
		w[i] = m[i].Dop(transform, n[i])
	}
	return w
}

// String prints the matrix in an easily readable form
func (m Matrix) String() string {
	builder := strings.Builder{}
	builder.WriteString(fmt.Sprintf("(%d)", len(m)))
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

func Cub(d int) Cube {
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
