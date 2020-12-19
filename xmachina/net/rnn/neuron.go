package rnn

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/rs/zerolog/log"
)

type state struct {
	x xmath.Vector
	h xmath.Vector
	y xmath.Vector
}

type neuron struct {
	ml.Activation
	net.Meta
	state
}

// NeuronFactory is a factory for construction of a recursive neuronFactory within the context of a recursive layer / network
type NeuronFactory func(p, n int, meta net.Meta) *neuron

// Neuron is the neuronFactory implementation for a recursive neural network.
var Neuron = func(activation ml.Activation) NeuronFactory {
	return func(p, n int, meta net.Meta) *neuron {
		return &neuron{
			Activation: activation,
			state: state{
				x: xmath.Vec(p),
				h: xmath.Vec(n),
				y: xmath.Vec(n),
			},
			Meta: meta,
		}
	}
}

func (rn *neuron) forward(x, h xmath.Vector, weights *Weights) (y, wh xmath.Vector) {
	xmath.MustHaveSameSize(x, rn.x)
	// keep the input state in memory for backpropagation
	rn.x = x
	// apply internal state weights
	wxh := weights.Wxh.Prod(x)
	whh := weights.Whh.Prod(h)
	rn.h = wxh.
		Add(whh).
		Add(weights.Bh)
	// apply activation
	rn.h = rn.h.Op(rn.F)

	// compute output
	xhy := weights.Why.Prod(rn.h)
	rn.y = xhy.Add(weights.By)

	log.Trace().
		Floats64("input", rn.x).
		Floats64("h-in", h).
		Floats64("h-out", rn.h).
		Floats64("output", rn.y).
		Msg("neuronFactory forward")
	return rn.y, rn.h
}

func (rn *neuron) backward(dy, u xmath.Vector, params *Parameters) (h xmath.Vector, dWhy, dWxh, dWhh xmath.Matrix) {

	// we need to trace our  steps back ...
	dWhy = dy.Prod(rn.h)

	// delta
	dh := params.Why.T().Prod(dy)
	dh = dh.Add(u)
	// de-activation
	h = dh.X(rn.h.Op(rn.D))

	dWxh = dh.Prod(rn.x)

	dWhh = dh.Prod(u)

	return h, dWhy, dWxh, dWhh
}
