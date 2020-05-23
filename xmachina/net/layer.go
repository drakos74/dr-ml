package net

import (
	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

type Layer interface {
	// F will take the input from the previous layer and generate an input for the next layer
	Forward(v xmath.Vector) xmath.Vector
	// Backward will take the loss from next layer and generate a loss for the previous layer
	Backward(dv xmath.Vector) xmath.Vector
	// Weights returns the current weight matrix for the layer
	Weights() xmath.Matrix
	// Size returns the Size of the layer e.g. number of neurons
	Size() int
}

type FFLayer struct {
	pSize   int
	neurons []*Neuron
}

func NewFFLayer(p, n int, factory NeuronFactory, index int) Layer {
	neurons := make([]*Neuron, n)
	for i := 0; i < n; i++ {
		neurons[i] = factory(p, meta{
			index: i,
			layer: index,
		})
	}
	return &FFLayer{pSize: p, neurons: neurons}
}

func (l *FFLayer) Size() int {
	return len(l.neurons)
}

// forward takes as input the outputs of all the neurons of the previous layer
// it returns the output of all the neurons of the current layer
func (l *FFLayer) Forward(v xmath.Vector) xmath.Vector {
	// we are building the output vector for the next layer
	out := xmath.Vec(len(l.neurons))
	// each neuron will receive the same vector input from the previous layer outputs
	// it will apply it's weights accordingly
	//aggr := xmath.NewAggregate()

	for i, n := range l.neurons {
		out[i] = n.forward(v)
		//aggr.Add(xmath.NewBucketFromVector(n.weights))
	}
	//println(fmt.Sprintf("aggr = %v", aggr))
	return out
}

// backward receives all the errors from the following layer
// it returns the full matrix of partial errors for the previous layer
func (l *FFLayer) Backward(err xmath.Vector) xmath.Vector {
	// we are preparing the error output for the previous layer
	dn := xmath.Mat(len(l.neurons))
	for i, n := range l.neurons {
		dn[i] = n.backward(err[i])
	}
	// we need the transpose in order to produce a vector corresponding to the neurons of the previous layer
	return dn.T().Sum()
}

func (l *FFLayer) Weights() xmath.Matrix {
	m := xmath.Mat(l.Size())
	for j := 0; j < len(l.neurons); j++ {
		m[j] = l.neurons[j].weights
	}
	return m
}

type xVector struct {
	value xmath.Vector
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

func newXLayer(p, n int, factory NeuronFactory, index int) *xLayer {
	neurons := make([]*xNeuron, n)

	out := make(chan xFloat, n)
	backout := make(chan xVector, n)

	for i := 0; i < n; i++ {
		n := &xNeuron{
			Neuron: factory(p, meta{
				index: i,
				layer: index,
			}),
			input:   make(chan xmath.Vector, 1),
			output:  out,
			backIn:  make(chan float64, 1),
			backOut: backout,
		}
		n.init()
		neurons[i] = n
	}
	return &xLayer{
		neurons: neurons,
		out:     out,
		backOut: backout,
	}
}

func (xl *xLayer) Size() int {
	return len(xl.neurons)
}

func (xl *xLayer) Forward(v xmath.Vector) xmath.Vector {

	out := xmath.Vec(len(xl.neurons))

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

func (xl *xLayer) Backward(err xmath.Vector) xmath.Vector {
	// we are building the error output for the previous layer
	dn := xmath.Mat(len(xl.neurons))

	for i, n := range xl.neurons {
		// and produce partial error for previous layer
		n.backIn <- err[i]
	}

	c := 0
	for o := range xl.backOut {
		dn[o.index] = o.value
		c++
		if c == len(xl.neurons) {
			break
		}
	}

	return dn.T().Sum()
}

func (xl *xLayer) Weights() xmath.Matrix {
	m := xmath.Mat(xl.Size())
	for j := 0; j < len(xl.neurons); j++ {
		m[j] = xl.neurons[j].weights
	}
	return m
}

type SMLayer struct {
	ml.SoftMax
	size int
	out  xmath.Vector
}

func NewSMLayer(p, index int) *SMLayer {
	return &SMLayer{
		size: p,
		out:  xmath.Vec(p),
	}
}

func (sm *SMLayer) Size() int {
	return sm.size
}

func (sm *SMLayer) Forward(v xmath.Vector) xmath.Vector {
	sm.out = sm.F(v)
	return sm.out
}

func (sm *SMLayer) Backward(err xmath.Vector) xmath.Vector {
	return sm.D(sm.out).Prod(err)
}

func (sm *SMLayer) Weights() xmath.Matrix {
	m := xmath.Mat(sm.Size())
	// there are no weights the way we approached this
	return m
}
