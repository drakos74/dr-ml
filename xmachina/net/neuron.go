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
	input  math.Vector
	output float64
	delta  math.Vector
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
	n.output = n.Forward(v.Dot(n.weights) + n.bias)
	if n.output >= 1 {
		// TODO : notify of bad weight distribution
	}
	return n.output
}

// TODO : backward for bias
func (n *Neuron) backward(err float64) math.Vector {
	// build the error vector for the previous layer
	dw := math.NewVector(len(n.weights))
	// calculate the gradient of the output
	d := n.Module.Grad(err, n.Module.Back(n.output))
	for i, inp := range n.input {
		// create the error for the previous layer
		dw[i] = d * n.weights[i]
		// we are updating the weights while going back as well
		n.weights[i] = n.weights[i] + n.Module.Get()*d*inp
	}
	return dw
}

type NeuronFactory func(p int, meta meta) *Neuron

var Perceptron = func(module ml.Module, weights math.VectorGenerator) NeuronFactory {
	return func(p int, meta meta) *Neuron {
		return &Neuron{
			Module: module,
			memory: memory{
				input: math.NewVector(p),
				delta: math.NewVector(p),
			},
			meta:    meta,
			weights: weights(p),
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
