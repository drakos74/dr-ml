package net

import (
	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
)

type NN interface {
	Train(input xmath.Vector, output xmath.Vector) (err xmath.Vector, weights xmath.Cube)
}

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

	return err, weights

}

func (n *Network) Predict(input xmath.Vector) xmath.Vector {
	return n.forward(input)
}

type XNetwork struct {
	info
	config
	layers []xLayer
}

func XNew(inputSize, outputSize int) *XNetwork {

	return &XNetwork{
		info: info{
			inputSize:  inputSize,
			outputSize: outputSize,
		},
		layers: make([]xLayer, 0),
	}
}

func (xn *XNetwork) Debug(debug bool) *XNetwork {
	xn.debug = debug
	return xn
}

func (xn *XNetwork) Add(s int, factory NeuronFactory) *XNetwork {

	ps := xn.inputSize

	ls := len(xn.layers)
	if ls > 0 {
		// check previous layer size
		ps = xn.layers[ls-1].Size()
	}

	xn.layers = append(xn.layers, newXLayer(ps, s, factory, len(xn.layers)))
	return xn
}

func (xn *XNetwork) forward(input xmath.Vector) xmath.Vector {
	output := xmath.NewVector(len(input)).From(input)
	for _, l := range xn.layers {
		output = l.forward(output)
	}
	return output
}

func (xn *XNetwork) backward(loss xmath.Vector) {
	layer := xn.layers[len(xn.layers)-1]
	// for the last layer we rebuild the error matrix
	err := xmath.NewMatrix(layer.Size()).From(loss)
	// for the rest of the layers we just iterate in revere order
	for i := len(xn.layers) - 1; i >= 0; i-- {
		err = xn.layers[i].backward(err)
	}

}

func (xn *XNetwork) Train(input xmath.Vector, output xmath.Vector) (err xmath.Vector, weights xmath.Cube) {

	out := xn.forward(input)
	err = output.Diff(out)
	xn.backward(err)

	xn.iterations++

	return err, weights

}

func (xn *XNetwork) Predict(input xmath.Vector) xmath.Vector {
	return xn.forward(input)
}
