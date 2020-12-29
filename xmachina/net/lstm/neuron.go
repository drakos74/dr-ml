package lstm

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
)

type neuron struct {
	forget net.Neuron
	input1 net.Neuron
	input2 net.Neuron
	state  net.Neuron
	output net.Neuron
	soft   net.Neuron
	meta   net.Meta
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
	return func(meta net.Meta) *neuron {
		return &neuron{
			forget: net.NewActivationCell(builder.x, builder.h, *ml.Base().
				WithActivation(ml.Sigmoid).
				WithRate(&builder.rate),
				net.NewWeights(builder.h, builder.h, builder.weightGenerator, builder.biasGenerator),
				meta.WithID("forget")),
			input1: net.NewActivationCell(builder.x, builder.h, *ml.Base().
				WithActivation(ml.Sigmoid).
				WithRate(&builder.rate),
				net.NewWeights(builder.h, builder.h, builder.weightGenerator, builder.biasGenerator),
				meta.WithID("input-1")),
			input2: net.NewActivationCell(builder.x, builder.h, *ml.Base().
				WithActivation(ml.TanH).
				WithRate(&builder.rate),
				net.NewWeights(builder.h, builder.h, builder.weightGenerator, builder.biasGenerator),
				meta.WithID("input-2")),
			state: net.NewActivationCell(builder.x, builder.h, *ml.Base().
				WithActivation(ml.TanH).
				WithRate(&builder.rate),
				net.NewWeights(builder.h, builder.h, builder.weightGenerator, builder.biasGenerator),
				meta.WithID("state")),
			output: net.NewActivationCell(builder.x, builder.h, *ml.Base().
				WithActivation(ml.Sigmoid).
				WithRate(&builder.rate),
				net.NewWeights(builder.h, builder.h, builder.weightGenerator, builder.biasGenerator),
				meta.WithID("output")),
			soft: net.NewSoftCell(builder.x, builder.h, meta.WithID("softmax")),
			meta: meta,
		}
	}
}

func (lstm *neuron) forward(x, prev_h, prev_s xmath.Vector) (y, next_h, next_s xmath.Vector) {

	v := x.Stack(prev_h)

	f := lstm.forget.Fwd(v)
	i1 := lstm.input1.Fwd(v)
	i2 := lstm.input2.Fwd(v)
	o := lstm.output.Fwd(v)

	s := f.X(prev_s)
	i := i1.X(i2)
	next_s = s.Add(i)
	c := lstm.state.Fwd(next_s)

	next_h = c.X(o)

	y = lstm.soft.Fwd(next_h)

	return y, next_h, next_s
}

func (lstm *neuron) backward(dy, dh, ds xmath.Vector) (x, h, s xmath.Vector) {

}
