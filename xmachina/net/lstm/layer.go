package lstm

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/rs/zerolog/log"
)

// Layer is the recurrent network layer.
type Layer struct {
	clip    Clip
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

// NewLayer creates a new Recurrent layer
func NewLayer(n int, builder NeuronBuilder, clipping Clip, index int) *Layer {
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
		out:     xmath.Mat(n).Of(builder.x),
		xDim:    builder.x,
		hDim:    builder.h,
		clip:    clipping,
	}
}

// LoadRNNLayer loads a new Recurrent layer based on the given weights
// n : batch size e.g. rnn units
// xDim : size of trainInput/trainOutput vector
// hDim : internal hidden layer size
// factory : neuronFactory factory to be used for the rnn unit
// index : index of layer in the network
// TODO : remove the rate and use it within the ml.Module
//func LoadRNNLayer(n, xDim, hDim int, learning ml.Learning, factory NeuronFactory, weights Weights, clipping Clip, index int) *Layer {
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
		log.Trace().
			Int("x", i).
			Floats64("neuronFactory-out", y).
			Floats64("h", h).
			Floats64("out", r.out[i]).
			Msg("layer forward")
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

	//clip the weights on the positive axis to avoid exploding gradients.
	w := r.clip.W
	wClipOp := xmath.Clip(-1*w, 1*w)
	b := r.clip.B
	bClipOp := xmath.Clip(-1*b, 1*b)
	// we just need to clip the first neuron weights, as all neurons have the same weight pointer.
	clipWeights(r.neurons[0].input.Weights(), wClipOp, bClipOp)
	clipWeights(r.neurons[0].hidden.Weights(), wClipOp, bClipOp)
	clipWeights(r.neurons[0].activation.Weights(), wClipOp, bClipOp)
	clipWeights(r.neurons[0].output.Weights(), wClipOp, bClipOp)
	return nil
}

func clipWeights(weights *net.Weights, wClip, bClip xmath.Op) {
	weights.W = weights.W.Op(wClip)
	weights.B = weights.B.Op(bClip)
}
