package net

import (
	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
)

type info struct {
	init       bool
	inputSize  int
	outputSize int
	iterations int
}

type config struct {
	trace bool
	debug bool
}

type Network struct {
	info
	config
	layers []Layer
}

func New(inputSize, outputSize int) *Network {

	return &Network{
		info: info{
			inputSize:  inputSize,
			outputSize: outputSize,
		},
		layers: make([]Layer, 0),
	}
}

func (n *Network) Debug(debug bool) *Network {
	n.debug = debug
	return n
}

func (n *Network) Add(s int, factory NeuronFactory) *Network {

	ps := n.inputSize

	ls := len(n.layers)
	if ls > 0 {
		// check previous layer size
		ps = n.layers[ls-1].Size()
	}

	n.layers = append(n.layers, NewLayer(ps, s, factory, len(n.layers)))
	return n
}

func (n *Network) forward(input xmath.Vector) xmath.Vector {
	output := xmath.NewVector(len(input)).From(input)
	for _, l := range n.layers {
		output = l.forward(output)
	}
	return output
}

func (n *Network) backward(loss xmath.Vector) {
	layer := n.layers[len(n.layers)-1]
	// for the last layer we rebuild the error matrix
	err := xmath.NewMatrix(layer.Size()).From(loss)
	// for the rest of the layers we just iterate in revere order
	for i := len(n.layers) - 1; i >= 0; i-- {
		err = n.layers[i].backward(err)
	}

}

func (n *Network) Train(input xmath.Vector, output xmath.Vector) (err xmath.Vector, weights xmath.Cube) {

	out := n.forward(input)
	err = output.Diff(out)
	n.backward(err)

	n.iterations++

	if n.trace {
		weights = xmath.NewCube(len(n.layers))
		for i := 0; i < len(n.layers); i++ {
			layer := n.layers[i]
			m := xmath.NewMatrix(layer.Size())
			for j := 0; j < len(layer.neurons); j++ {
				m[j] = layer.neurons[j].weights
			}
			weights[i] = m
		}
	}

	return err, weights

}

func (n *Network) Predict(input xmath.Vector) xmath.Vector {
	return n.forward(input)
}
