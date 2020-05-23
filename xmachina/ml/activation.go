package ml

import (
	"math"

	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
)

type Activation interface {
	F(x float64) float64
	D(x float64) float64
}

var Sigmoid = sigmoid{}

type sigmoid struct {
}

func (s sigmoid) F(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

func (s sigmoid) D(y float64) float64 {
	return y * (1.0 - y)
}

var TanH = tanH{}

type tanH struct {
}

func (t tanH) F(x float64) float64 {
	return math.Tanh(x)
}

func (t tanH) D(y float64) float64 {
	return 1 - math.Pow(y, 2)
}

var ReLU = relu{}

// TODO for RELU
type relu struct {
}

func (r relu) F(x float64) float64 {
	return math.Max(0, x)
}

func (r relu) D(x float64) float64 {
	return 1
}

type Void struct {
}

func (v Void) F(x float64) float64 {
	return x
}

func (v Void) D(x float64) float64 {
	return 1
}

type SoftMax struct {
}

func (sm SoftMax) max(v []float64) float64 {
	var max float64
	for _, x := range v {
		max = math.Max(x, max)
	}
	return max
}

func (sm SoftMax) expSum(v []float64, max float64) float64 {
	var sum float64
	for _, x := range v {
		sum += sm.exp(x, max)
	}
	return sum
}

func (sm SoftMax) exp(x, max float64) float64 {
	return math.Exp(x - max)
}

func (sm SoftMax) F(v xmath.Vector) xmath.Vector {
	softmax := xmath.Vec(len(v))
	max := sm.max(v)
	sum := sm.expSum(v, max)
	for i, x := range v {
		softmax[i] = sm.exp(x, max) / sum
	}
	return softmax
}

func (sm SoftMax) D(s xmath.Vector) xmath.Matrix {

	jacobian := xmath.Diag(s)

	for i := range jacobian {
		for j := range jacobian[i] {
			if i == j {
				jacobian[i][j] = s[i] * (1 - s[i])
			} else {
				jacobian[i][j] = -s[i] * s[j]
			}
		}
	}

	return jacobian
}
