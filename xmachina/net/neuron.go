package net

import "github.com/drakos74/go-ex-machina/xmachina/ml"

// Neuron is a minimal computation unit with an activation function.
type Neuron interface {
	ml.Activation
}

type Meta struct {
	Layer int
	Index int
}
