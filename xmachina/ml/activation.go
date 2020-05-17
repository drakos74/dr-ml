package ml

import "math"

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

type Void struct {
}

func (v Void) Forward(x float64) float64 {
	return x
}

func (v Void) Back(x float64) float64 {
	return x
}
