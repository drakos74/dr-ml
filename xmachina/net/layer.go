package net

import "github.com/drakos74/go-ex-machina/xmachina/math"

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
