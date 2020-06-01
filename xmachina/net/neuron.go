package net

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

type meta struct {
	layer int
	index int
}

type memory struct {
	input  xmath.Vector
	output float64
}

type learn struct {
	weights xmath.Vector
	bias    float64
}

type Neuron struct {
	ml.Module
	meta
	memory
	learn
}

func (n *Neuron) forward(v xmath.Vector) float64 {
	xmath.MustHaveSameSize(v, n.input)
	n.input = v
	n.output = n.F(v.Dot(n.weights) + n.bias)
	if n.output >= 1 {
		// TODO : notify of bad weight distribution (?)
	}
	return n.output
}

func (n *Neuron) backward(err float64) xmath.Vector {
	loss := xmath.Vec(len(n.weights))
	grad := n.Module.Grad(err, n.Module.D(n.output))
	for i, inp := range n.input {
		// create the error for the previous layer
		loss[i] = grad * n.weights[i]
		// we are updating the weights while going back as well
		n.weights[i] = n.weights[i] + n.Module.WRate()*grad*inp
		n.bias += grad * n.Module.BRate()
	}
	return loss
}

type xNeuron struct {
	*Neuron
	input   chan xmath.Vector
	output  chan xFloat
	backIn  chan float64
	backOut chan xVector
}

// init initialises the neurons to listen for incoming forward and backward directed data from the parent layer
func (xn *xNeuron) init() *xNeuron {
	go func(xn *xNeuron) {
		for {
			select {
			case v := <-xn.input:
				xn.output <- xFloat{
					value: xn.Neuron.forward(v),
					index: xn.Neuron.meta.index,
				}
			case e := <-xn.backIn:
				xn.backOut <- xVector{
					value: xn.Neuron.backward(e),
					index: xn.Neuron.meta.index,
				}
			}
		}
	}(xn)
	return xn
}

// NeuronFactory is a factory for construction of neuron within the context of a neuron layer / network
type NeuronFactory func(p int, meta meta) *Neuron

var Perceptron = func(module ml.Module, weights xmath.VectorGenerator) NeuronFactory {
	return func(p int, meta meta) *Neuron {
		return &Neuron{
			Module: module,
			memory: memory{
				input: xmath.Vec(p),
			},
			meta: meta,
			learn: learn{
				weights: weights(p, meta.index),
			},
		}
	}
}

type state struct {
	x xmath.Vector
	h xmath.Vector
	y xmath.Vector
}

type rNeuron struct {
	ml.Activation
	meta
	state
}

// RNeuronFactory is a factory for construction of a recursive neuron within the context of a recursive layer / network
type RNeuronFactory func(p, n int, meta meta) *rNeuron

var RNeuron = func(activation ml.Activation) RNeuronFactory {
	return func(p, n int, meta meta) *rNeuron {
		return &rNeuron{
			Activation: activation,
			state: state{
				x: xmath.Vec(p),
				h: xmath.Vec(n),
				y: xmath.Vec(n),
			},
			meta: meta,
		}
	}
}

func (rn *rNeuron) forward(v, w xmath.Vector, weights *Weights) (y, wh xmath.Vector) {
	xmath.MustHaveSameSize(v, rn.x)
	rn.x = v
	rn.h = weights.Wxh.Prod(v).
		Add(weights.Whh.Prod(w)).
		Add(weights.Bh)
	rn.h = rn.h.Op(rn.F)
	rn.y = weights.Why.Prod(rn.h).Add(weights.By)
	return rn.y, rn.h
}

func (rn *rNeuron) backward(dy, u xmath.Vector, params *Parameters) (h xmath.Vector, dWhy, dWxh, dWhh xmath.Matrix) {

	// we need to trace our  steps back ...
	dWhy = dy.Prod(rn.h)

	// delta
	dh := params.Why.T().Prod(dy)
	dh = dh.Add(u)
	// de-activation
	h = dh.ProdH(rn.h.Op(rn.D))

	dWxh = dh.Prod(rn.x)

	dWhh = dh.Prod(u)

	return h, dWhy, dWxh, dWhh
}
