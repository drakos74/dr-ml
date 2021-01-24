package rnn

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmachina/net/rc"
	"github.com/drakos74/go-ex-machina/xmath"
)

// Layer is the recurrent network layer.
type Layer struct {
	clip    net.Clip
	neurons []*neuron
	xDim    int
	hDim    int
	out     xmath.Matrix
}

// Weights returns the layer weights.
func (r *Layer) Size() (n, x, h int) {
	return len(r.neurons), r.xDim, r.hDim
}

// Weights returns the layer weights.
func (r *Layer) Weights() map[net.Meta]net.Weights {
	weights := make(map[net.Meta]net.Weights)
	neuron := r.neurons[0]
	weights[neuron.input.Meta()] = *neuron.input.Weights()
	weights[neuron.hidden.Meta()] = *neuron.input.Weights()
	weights[neuron.activation.Meta()] = *neuron.input.Weights()
	weights[neuron.output.Meta()] = *neuron.input.Weights()
	return weights
}

// New creates a new Vanilla Recurrent layer
func New(builder rc.NeuronBuilder) rc.LayerFactory {
	return func(n int, clipping net.Clip, index int) rc.Layer {
		neurons := make([]*neuron, n)
		factory := Neuron(builder)
		for i := 0; i < n; i++ {
			neuron := factory(net.Meta{
				Index: i,
				Layer: index,
			})
			neurons[i] = neuron
		}
		return &Layer{
			neurons: neurons,
			out:     xmath.Mat(n).Of(builder.X),
			xDim:    builder.X,
			hDim:    builder.H,
			clip:    clipping,
		}
	}
}

// LoadRNNLayer loads a new Recurrent layer based on the given weights
// n : batch size e.g. rnn units
// xDim : size of trainInput/trainOutput vector
// hDim : internal hidden layer size
// factory : neuronFactory factory to be used for the rnn unit
// index : index of layer in the network
// TODO : remove the rate and use it within the ml.BiOp
//func LoadRNNLayer(n, xDim, hDim int, learning ml.Learning, factory NeuronFactory, weights Weights, clipping net.Clip, index int) *Layer {
//	neurons := make([]*neuron, n)
//	for i := 0; i < n; i++ {
//		neuron := factory(xDim, n, hDim, net.Meta{
//			Index: i,
//			Layer: index,
//		})
//		neurons[i] = neuron
//	}
//	return &Layer{
//		Learning: learning,
//		neurons:  neurons,
//		out:      xmath.Mat(n).Of(xDim),
//		// TODO : Allow to define the initial weights from the constructor call
//		//Parameters: initWithWeights(xDim, hDim, weights),
//		xDim: xDim,
//		hDim: hDim,
//		clip: clipping,
//	}
//}

// Forward pushes the input through the layer
// x is the input
// rows of x are the input values at different time instances
// e.g. x[0] , x[1] , x[2] etc ...
func (r *Layer) Forward(x xmath.Matrix) xmath.Matrix {

	n := len(r.neurons)

	// we expect a training set equal to our depth
	xmath.MustHaveDim(x, n)

	// inter-neuronFactory communication parameter
	h := xmath.Vec(r.hDim)

	var y xmath.Vector
	for i := 0; i < n; i++ {
		// calculate output
		y, h = r.neurons[i].forward(x[i], h)
		y.Check()
		// apply layer activation
		r.out[i] = y
	}
	return r.out
}

// Backward handles the backpropagation logic for the layer.
// exp : is the expected output
func (r *Layer) Backward(exp xmath.Matrix) xmath.Matrix {

	h := xmath.Vec(r.hDim)

	for i := len(r.neurons) - 1; i >= 0; i-- {
		xmath.MustHaveSameSize(r.out[i], exp[i])
		dy := ml.Diff(exp[i], r.out[i])
		_, dh := r.neurons[i].backward(dy, h)
		h = dh
	}
	// we just need to clip the first neuron weights, as all neurons have the same weight pointer.
	r.clip.Apply(r.neurons[0].input.Weights())
	r.clip.Apply(r.neurons[0].hidden.Weights())
	r.clip.Apply(r.neurons[0].activation.Weights())
	r.clip.Apply(r.neurons[0].output.Weights())
	return nil
}
