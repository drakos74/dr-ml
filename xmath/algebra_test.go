package xmath

import (
	"fmt"
	"testing"
)

func TestMathSyntax(t *testing.T) {

	v := 1.0
	w := v + 1e-8

	println(fmt.Sprintf("w = %v", w))

}
