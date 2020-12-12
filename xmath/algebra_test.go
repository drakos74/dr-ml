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
