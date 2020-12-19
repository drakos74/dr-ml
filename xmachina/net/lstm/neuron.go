package lstm

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
)

type state struct {
	h xmath.Vector
	s xmath.Vector
}

type cell struct {
	forget     net.MLNeuron
	input      net.MLNeuron
	activation net.MLNeuron
	output     net.MLNeuron
	hidden     net.MLNeuron

	net.Meta
	state
}

// NeuronFactory is a factory for construction of a recursive neuronFactory within the context of a recursive layer / network
type NeuronFactory func(p, n int, meta net.Meta) *neuron

// Neuron is the neuronFactory implementation for a recursive neural network.
var Neuron = func(activation ml.Activation) NeuronFactory {
	return func(p, n int, meta net.Meta) *neuron {
		return nil
	}
}

func (rn *cell) forward(x xmath.Vector, prev state) (s, h xmath.Vector) {
	// stack the input and previous hidden state to compound them together
	v := x.Stack(prev.h)

	// forget gate
	f := rn.forget.Fwd(v)
	// input gate
	i := rn.input.Fwd(v)
	// activation gate
	g := rn.activation.Fwd(v)
	// output gate
	o := rn.output.Fwd(v)
	// state
	s = rn.hidden.Fwd(i.X(g).Add(prev.s.X(f)))
	// hidden state
	h = o.X(s)

	rn.state.h = h
	rn.state.s = s

	return s, h
}

func (rn *cell) backward(y, dNext state) (h xmath.Vector, dWhy, dWxh, dWhh xmath.Matrix) {

	//ds := rn.output.grad(dNext.h).Add(dNext.s)
	//do := rn.hidden.grad(dNext.h)
	//di := rn.activation.grad(ds)
	//dg := rn.input.grad(ds)
	//df := rn.state.s.X(ds)

	return h, dWhy, dWxh, dWhh
}

type neuron struct {
	act     ml.Activation
	weights xmath.Matrix
	bias    xmath.Vector
	output  xmath.Vector
}

func (n *neuron) forward(x xmath.Vector) xmath.Vector {
	w := n.weights.Prod(x).Add(n.bias)
	n.output = w.Op(n.act.F)
	return n.output
}

func (n *neuron) grad(d xmath.Vector) xmath.Vector {
	return n.output.X(d)
}

func (n *neuron) backward(x, y xmath.Vector) xmath.Vector {
	return nil
}
