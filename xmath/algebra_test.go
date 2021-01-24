package xmath

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMathSyntax(t *testing.T) {
	v := 1.0
	w := v + 1e-8

	println(fmt.Sprintf("w = %v", w))
}

func TestRoundOp(t *testing.T) {
	type test struct {
		p    int
		x, y float64
	}

	tests := map[string]test{
		"no-decimals":      {p: 0, x: 1.1111111111, y: 1},
		"one-decimal-down": {p: 1, x: 2.222, y: 2.2},
		"one-decimal-up":   {p: 1, x: 6.6666, y: 6.7},
		"more-decimals":    {p: 10, x: 1.1111, y: 1.1111},
		"less-decimals":    {p: 2, x: 1.1111, y: 1.11},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			op := Round(tt.p)
			Y := op(tt.x)
			assert.Equal(t, tt.y, Y)
		})
	}
}

func TestScaleOp(t *testing.T) {
	type test struct {
		f    float64
		x, y float64
	}

	tests := map[string]test{
		"zero-x":     {f: 1, x: 0, y: 0},
		"zero-f":     {f: 0, x: 1, y: 0},
		"one":        {f: 1, x: 1, y: 1},
		"scale-up":   {f: 10, x: 1, y: 10},
		"scale-down": {f: 0.1, x: 10, y: 1},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			op := Scale(tt.f)
			Y := op(tt.x)
			assert.Equal(t, tt.y, Y)
		})
	}
}

func TestSqrtOp(t *testing.T) {
	type test struct {
		x, y float64
	}

	tests := map[string]test{
		"zero":     {x: 0, y: 0},
		"one":      {x: 1, y: 1},
		"scale-up": {x: 100, y: 10},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			Y := Sqrt(tt.x)
			assert.Equal(t, tt.y, Y)
		})
	}
}

func TestSquareOp(t *testing.T) {
	type test struct {
		x, y float64
	}

	tests := map[string]test{
		"zero":     {x: 0, y: 0},
		"one":      {x: 1, y: 1},
		"scale-up": {x: 10, y: 100},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			Y := Square(tt.x)
			assert.Equal(t, tt.y, Y)
		})
	}
}

func TestDiag(t *testing.T) {

	v := Vec(3).With(1, 2, 3)
	m := Diag(v)

	for i := range m {
		for j := range m[i] {
			if i == j {
				assert.Equal(t, v[i], m[i][j])
			} else {
				assert.Equal(t, 0.0, m[i][j])
			}
		}
	}
}

func Test_T(t *testing.T) {

	mat := Mat(4).With(
		Vec(2).With(11, 12),
		Vec(2).With(21, 22),
		Vec(2).With(31, 32),
		Vec(2).With(41, 42),
	)

	println(fmt.Sprintf("mat = %v", mat))

	matT := mat.T()
	println(fmt.Sprintf("matT = %v", matT))

}

var testDiv = func(matrix Matrix) Matrix {
	// we want to make out of this sequence of vectors a 1 d longer vector with the direction
	// TODO : abstract into an xmath operation
	l := len(matrix) - 1
	mat := Mat(l)
	j := 0
	for i := l; i > 0; i-- {
		w := Vec(2)
		diff := matrix[i][0] - matrix[i-1][0]
		if diff > 0 {
			w[0] = 1
		} else if diff < 0 {
			w[1] = 1
		}
		j++
		index := l - j
		mat[index] = w
	}
	return mat
}

func Test_Div(t *testing.T) {

	type test struct {
		matrix Matrix
	}

	tests := map[string]test{
		"all-up": {
			matrix: []Vector{
				{0},
				{1},
				{2},
				{3},
				{4},
			},
		},
		"all-down": {
			matrix: []Vector{
				{5},
				{4},
				{3},
				{2},
				{1},
			},
		},
		"up-and-down": {
			matrix: []Vector{
				{0},
				{2},
				{4},
				{8},
				{10},
				{9},
				{5},
				{1},
				{0},
				{1},
			},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			nat := tt.matrix.T().Vop(Diff, UpOrDown)
			println(fmt.Sprintf("nat = %+v", nat))

			mat := testDiv(tt.matrix)
			println(fmt.Sprintf("mat = %+v", mat))
		})
	}

}
