package net

import "github.com/drakos74/go-ex-machina/xmath"

// TODO : unify layer abstraction

type FFLayer interface {
	// F will take the input from the previous layer and generate an input for the next layer
	Forward(v xmath.Vector) xmath.Vector
	// Backward will take the loss from next layer and generate a loss for the previous layer
	Backward(dv xmath.Vector) xmath.Vector
	// Weights returns the current weight matrix for the layer
	Weights() xmath.Matrix
	// Size returns the Size of the layer e.g. number of neurons
	Size() int
}
