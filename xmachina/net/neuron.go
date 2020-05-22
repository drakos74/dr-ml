package net

import (
	"github.com/drakos74/go-ex-machina/xmachina/math"
	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

type meta struct {
	layer int
	index int
}

type memory struct {
	input     math.Vector
	rawOutput float64
	output    float64
	delta     math.Vector
}

type Neuron struct {
	ml.Module
	meta
	memory
	weights math.Vector
	bias    float64
}

func (n *Neuron) forward(v math.Vector) float64 {
	math.MustHaveSameSize(v, n.input)
	n.input = v
	n.rawOutput = v.Dot(n.weights) + n.bias
	n.output = n.Forward(n.rawOutput)
	if n.output >= 1 {
		// TODO : notify of bad weight distribution (?)
	}
	return n.output
}

// TODO : backward for bias
func (n *Neuron) backward(err float64) math.Vector {
	loss := math.Vec(len(n.weights))
	// calculate the gradient of the output
	// we are passing the initial input
	// that is more work, but is more correct in general,
	// as not all activation functions depend on the output only (?)
	// otherwise we could use the output, but only if the activation derivative method supports it
	// TODO ?
	d := n.Module.Grad(err, n.Module.Back(n.rawOutput))
	for i, inp := range n.input {
		// create the error for the previous layer
		loss[i] = d * n.weights[i]
		// we are updating the weights while going back as well
		f := n.Module.Get() * d * inp
		n.weights[i] = n.weights[i] + f
	}
	return loss
}

type NeuronFactory func(p int, meta meta) *Neuron

var Perceptron = func(module ml.Module, weights math.VectorGenerator) NeuronFactory {
	return func(p int, meta meta) *Neuron {
		return &Neuron{
			Module: module,
			memory: memory{
				input: math.Vec(p),
				delta: math.Vec(p),
			},
			meta:    meta,
			weights: weights(p, meta.index),
		}
	}
}

type xNeuron struct {
	*Neuron
	input   chan math.Vector
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
