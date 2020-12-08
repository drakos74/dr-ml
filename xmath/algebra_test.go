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
