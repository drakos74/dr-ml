package rnn

import (
	"fmt"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmachina/net/rc"
	"github.com/drakos74/go-ex-machina/xmath"
)

type neuron struct {
	input      net.Neuron
	hidden     net.Neuron
	activation net.Neuron
	output     net.Neuron
	soft       net.Neuron
	meta       net.Meta
}

// NeuronFactory is a factory for construction of a recursive neuronFactory within the context of a recursive layer / network
type NeuronFactory func(meta net.Meta) *neuron

// Neuron is the neuronFactory implementation for a recursive neural network.
var Neuron = func(builder rc.NeuronBuilder) NeuronFactory {
	println(fmt.Sprintf("builder = %+v", builder))
	if len(builder.G) != 2 {
		panic(fmt.Sprintf("cannot construct rnn neuron without 2 activation functions: %+v", builder.G))
	}
	wxh := net.NewWeights(builder.X, builder.H, builder.WeightGenerator, xmath.VoidVector)
	whh := net.NewWeights(builder.H, builder.H, builder.WeightGenerator, xmath.VoidVector)
	why := net.NewWeights(builder.H, builder.H, builder.WeightGenerator, builder.BiasGenerator)
	wyy := net.NewWeights(builder.H, builder.Y, builder.WeightGenerator, builder.BiasGenerator)
	softCell := func(meta net.Meta) net.Neuron {
		return net.NoOp(builder.Y, builder.Y, meta)
	}
	if builder.Softmax {
		softCell = func(meta net.Meta) net.Neuron {
			return net.NewSoftCell(builder.Y, builder.S, meta)
		}
	}
	return func(meta net.Meta) *neuron {
		return &neuron{
			input:      net.NewWeightCell(builder.X, builder.H, *ml.Base().WithRate(&builder.Rate), wxh, meta.WithID("input")),
			hidden:     net.NewWeightCell(builder.H, builder.H, *ml.Base().WithRate(&builder.Rate), whh, meta.WithID("hidden")),
			activation: net.NewActivationCell(builder.H, builder.H, *ml.Base().WithRate(&builder.Rate).WithActivation(builder.G[0]), why, meta.WithID("activation")),
			output:     net.NewWeightCell(builder.H, builder.Y, *ml.Base().WithRate(&builder.Rate).WithActivation(builder.G[1]), wyy, meta.WithID("output")),
			soft:       softCell(meta.WithID("soft")),
			meta:       meta,
		}
	}
}

func (n *neuron) forward(x, prev_h xmath.Vector) (y, next_h xmath.Vector) {
	// apply internal state weights
	wx := n.input.Fwd(x)
	wh := n.hidden.Fwd(prev_h)
	w := wx.Add(wh)
	// apply activation
	next_h = n.activation.Fwd(w)
	// compute output
	ty := n.output.Fwd(next_h)
	y = n.soft.Fwd(ty)

	return y, next_h
}

func (n *neuron) backward(sdy, dh xmath.Vector) (x, h xmath.Vector) {
	dy := n.soft.Bwd(sdy)
	// we need to trace our  steps back ...
	dh = dh.Add(n.output.Bwd(dy))
	dh.Check()

	// de-activation
	dW := n.activation.Bwd(dh)
	dW.Check()

	// hidden state
	dWh := n.hidden.Bwd(dW)
	dWh.Check()

	dWx := n.input.Bwd(dW)
	dWx.Check()

	return dWx, dWh
}
