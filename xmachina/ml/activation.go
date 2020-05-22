package ml

import (
	"math"

	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
)

type Activation interface {
	Forward(x float64) float64
	Back(x float64) float64
}

var Sigmoid = sigmoid{}

type sigmoid struct {
}

func (s sigmoid) Forward(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (s sigmoid) Back(x float64) float64 {
	return s.Forward(x) * (1.0 - s.Forward(x))
}

type relu struct {
}

func (r relu) Forward(x float64) float64 {
	return math.Max(0, x)
}

func (r relu) Back(x float64) float64 {
	//dZ = np.array(dA, copy = True)
	//dZ[Z <= 0] = 0;
	//return dZ;
	// TODO
	return 0
}

type Void struct {
}

func (v Void) Forward(x float64) float64 {
	return x
}

func (v Void) Back(x float64) float64 {
	return 1
}

type SoftMax struct {
}

func (sm SoftMax) Max(v []float64) float64 {
	var max float64
	for _, x := range v {
		max = math.Max(x, max)
	}
	return max
}

func (sm SoftMax) ExpSum(v []float64, max float64) float64 {
	var sum float64
	for _, x := range v {
		sum += sm.Exp(x, max)
	}
	return sum
}

func (sm SoftMax) Exp(x, max float64) float64 {
	return math.Exp(x - max)
}

func (sm SoftMax) Forward(v xmath.Vector) xmath.Vector {
	softmax := xmath.Vec(len(v))
	max := sm.Max(v)
	sum := sm.ExpSum(v, max)
	for i, x := range v {
		softmax[i] = sm.Exp(x, max) / sum
	}
	return softmax
}

func (sm SoftMax) Back(s xmath.Vector) xmath.Matrix {

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
