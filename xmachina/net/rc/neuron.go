package rc

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

type Neuron interface {
	Forward(x, prev_h xmath.Vector) (y, next_h xmath.Vector)
	Backward(sdy, dh xmath.Vector) (x, h xmath.Vector)
}

// NeuronBuilder holds all neuron properties needed to construct the neuron.
type NeuronBuilder struct {
	X, Y, H, S                     int
	G                              []ml.Activation
	Rate                           ml.Learning
	WeightGenerator, BiasGenerator xmath.VectorGenerator
	Softmax                        bool
}

// NewNeuronBuilder creates a new neuron builder.
func NewNeuronBuilder(x, y, h int) *NeuronBuilder {
	return &NeuronBuilder{
		X: x,
		Y: y,
		H: h,
	}
}

// WithActivation defines the activation functions for the rnn neuron.
func (nb *NeuronBuilder) WithActivation(g ...ml.Activation) *NeuronBuilder {
	nb.G = g
	return nb
}

// WithWeights defines the weight generator functions for the rnn neuron.
func (nb *NeuronBuilder) WithWeights(weightGenerator, biasGenerator xmath.VectorGenerator) *NeuronBuilder {
	nb.WeightGenerator = weightGenerator
	nb.BiasGenerator = biasGenerator
	return nb
}

// WithRate defines the learning rate for the rnn neuron.
func (nb *NeuronBuilder) WithRate(learning ml.Learning) *NeuronBuilder {
	nb.Rate = learning
	return nb
}

// SoftMax adds an extra softmax operation at the end
func (nb *NeuronBuilder) SoftMax(s int) *NeuronBuilder {
	nb.Softmax = true
	nb.S = s
	return nb
}
