package lstm

import (
	"fmt"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/rs/zerolog/log"
)

type neuron struct {
	input      net.Neuron
	hidden     net.Neuron
	activation net.Neuron
	output     net.Neuron
	meta       net.Meta
}

// NeuronBuilder holds all neuron properties needed to construct the neuron.
type NeuronBuilder struct {
	x, y, h                        int
	g1, g2                         ml.Activation
	rate                           ml.Learning
	weightGenerator, biasGenerator xmath.VectorGenerator
}

// NewNeuronBuilder creates a new neuron builder.
func NewNeuronBuilder(x, y, h int) *NeuronBuilder {
	return &NeuronBuilder{
		x: x,
		y: y,
		h: h,
	}
}

// WithActivation defines the activation functions for the rnn neuron.
func (nb *NeuronBuilder) WithActivation(g1, g2 ml.Activation) *NeuronBuilder {
	nb.g1 = g1
	nb.g2 = g2
	return nb
}

// WithWeights defines the weight generator functions for the rnn neuron.
func (nb *NeuronBuilder) WithWeights(weightGenerator, biasGenerator xmath.VectorGenerator) *NeuronBuilder {
	nb.weightGenerator = weightGenerator
	nb.biasGenerator = biasGenerator
	return nb
}

// WithRate defines the learning rate for the rnn neuron.
func (nb *NeuronBuilder) WithRate(learning ml.Learning) *NeuronBuilder {
	nb.rate = learning
	return nb
}

// NeuronFactory is a factory for construction of a recursive neuronFactory within the context of a recursive layer / network
type NeuronFactory func(meta net.Meta) *neuron

// Neuron is the neuronFactory implementation for a recursive neural network.
var Neuron = func(builder NeuronBuilder) NeuronFactory {
	wxh := net.NewWeights(builder.x, builder.h, builder.weightGenerator, xmath.VoidVector)
	whh := net.NewWeights(builder.h, builder.h, builder.weightGenerator, xmath.VoidVector)
	why := net.NewWeights(builder.h, builder.h, builder.weightGenerator, builder.biasGenerator)
	wyy := net.NewWeights(builder.h, builder.y, builder.weightGenerator, builder.biasGenerator)
	return func(meta net.Meta) *neuron {
		return &neuron{
			input:      net.NewWeightCell(builder.x, builder.h, *ml.Base().WithRate(&builder.rate), wxh, meta.WithID("input")),
			hidden:     net.NewWeightCell(builder.h, builder.h, *ml.Base().WithRate(&builder.rate), whh, meta.WithID("hidden")),
			activation: net.NewActivationCell(builder.h, builder.h, *ml.Base().WithRate(&builder.rate).WithActivation(builder.g1), why, meta.WithID("activation")),
			output:     net.NewWeightCell(builder.h, builder.y, *ml.Base().WithRate(&builder.rate).WithActivation(builder.g2), wyy, meta.WithID("output")),
			meta:       meta,
		}
	}
}

func (rn *neuron) forward(x, prev_h xmath.Vector) (y, next_h xmath.Vector) {
	// apply internal state weights
	wx := rn.input.Fwd(x)
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("wx", wx).
		Msg("wx")
	wh := rn.hidden.Fwd(prev_h)
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("wh", wh).
		Msg("wh")
	w := wx.Add(wh)
	// apply activation
	next_h = rn.activation.Fwd(w)
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("h", next_h).
		Msg("next")
	// compute output
	y = rn.output.Fwd(next_h)
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("y", y).
		Msg("output")
	return y, next_h
}

func (rn *neuron) backward(dy, dh xmath.Vector) (x, h xmath.Vector) {
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("dy", dy).
		Floats64("dh", dh).
		Msg("error")
	// we need to trace our  steps back ...
	dh = dh.Add(rn.output.Bwd(dy))
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("dh", dh).
		Msg("dh")
	dh.Check()

	// de-activation
	dW := rn.activation.Bwd(dh)
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("dW", dW).
		Msg("dW")
	dW.Check()

	// hidden state
	dWh := rn.hidden.Bwd(dW)
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("dWh", dWh).
		Msg("dWh")
	dWh.Check()

	dWx := rn.input.Bwd(dW)
	log.Trace().
		Str("meta", fmt.Sprintf("%+v", rn.meta)).
		Floats64("dWx", dWx).
		Msg("dWx")
	dWx.Check()

	return dWx, dWh
}
