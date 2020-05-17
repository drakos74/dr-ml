package ml

import (
	"fmt"
	"log"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSigmoid_Forward(t *testing.T) {

	s := sigmoid{}

	for i := -100; i < 100; i++ {
		x := float64(i) * 0.01
		y := s.Forward(x)
		log.Println(fmt.Sprintf("x = %v -> y = %v", x, y))
	}

}

func TestSigmoid_Back(t *testing.T) {

	s := sigmoid{}

	var x0 float64
	var y0 float64
	for i := -100; i < 100; i++ {
		x := float64(i) * 0.1
		y := s.Forward(x)
		if x0 != 0 && y0 != 0 {
			// calculate the derivative approximately
			drv := (y - y0) / (x - x0)
			back := s.Back(x)
			assert.True(t, math.Abs(drv-back) < 0.01, fmt.Sprintf("x = %v -> y = %v -> dy/dx = %v, b = %v , err = %v", x, y, drv, back, drv-back))
		}
		x0 = x
		y0 = y
	}

}
