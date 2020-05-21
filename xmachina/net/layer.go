package net

import (
	"github.com/drakos74/go-ex-machina/xmachina/math"
)

type Layer struct {
	neurons []*Neuron
}

func NewLayer(p, n int, factory NeuronFactory, index int) Layer {
	neurons := make([]*Neuron, n)
	for i := 0; i < n; i++ {
		neurons[i] = factory(p, meta{
			index: i,
			layer: index,
		})
	}
	return Layer{neurons: neurons}
}

func (l *Layer) Size() int {
	return len(l.neurons)
}

// TODO : parallelize execution
// forward takes as input the outputs of all the neurons of the previous layer
// it returns the output of all the neurons of the current layer
func (l *Layer) forward(v math.Vector) math.Vector {
	// we are building the output vector for the next layer
	out := math.NewVector(len(l.neurons))
	// each neuron will receive the same vector input from the previous layer outputs
	// it will apply it's weights accordingly
	for i, n := range l.neurons {
		out[i] = n.forward(v)
	}
	return out
}

// TODO : parallelize execution
// backward receives all the errors from the following layer
// it returns the full matrix of partial errors for the previous layer
func (l *Layer) backward(dv math.Matrix) math.Matrix {
	// we are building the error output for the previous layer
	dn := math.NewMatrix(len(l.neurons))

	for i, n := range l.neurons {
		// for each neuron aggregate it's error
		// the partial error should be at it's index in dv
		var err float64
		for _, v := range dv {
			err += v[i]
		}
		// and produce partial error for previous layer
		dn[i] = n.backward(err)
	}
	return dn
}

type xVector struct {
	value math.Vector
	index int
}

type xFloat struct {
	value float64
	index int
}

type xLayer struct {
	neurons []*xNeuron
	out     chan xFloat
	backOut chan xVector
}

func newXLayer(p, n int, factory NeuronFactory, index int) xLayer {
	neurons := make([]*xNeuron, n)

	out := make(chan xFloat, n)
	backout := make(chan xVector, n)

	for i := 0; i < n; i++ {
		n := &xNeuron{
			Neuron: factory(p, meta{
				index: i,
				layer: index,
			}),
			input:   make(chan math.Vector, 1),
			output:  out,
			backIn:  make(chan float64, 1),
			backOut: backout,
		}
		n.init()
		neurons[i] = n
	}
	return xLayer{
		neurons: neurons,
		out:     out,
		backOut: backout,
	}
}

func (xl *xLayer) Size() int {
	return len(xl.neurons)
}

func (xl *xLayer) forward(v math.Vector) math.Vector {

	out := math.NewVector(len(xl.neurons))

	for _, n := range xl.neurons {
		n.input <- v
	}

	c := 0
	for o := range xl.out {
		out[o.index] = o.value
		c++
		if c == len(xl.neurons) {
			break
		}
	}

	return out
}

func (xl *xLayer) backward(dv math.Matrix) math.Matrix {
	// we are building the error output for the previous layer
	dn := math.NewMatrix(len(xl.neurons))

	for i, n := range xl.neurons {
		// for each neuron aggregate it's error
		// the partial error should be at it's index in dv
		var err float64
		for _, v := range dv {
			err += v[i]
		}
		// and produce partial error for previous layer
		n.backIn <- err
	}

	c := 0
	for o := range xl.backOut {
		dn[o.index] = o.value
		c++
		if c == len(xl.neurons) {
			break
		}
	}

	return dn
}
