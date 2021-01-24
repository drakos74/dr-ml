package lstm

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
	sDim    int
	out     xmath.Matrix
}

// Weights returns the layer weights.
func (l *Layer) Size() (x, h, s int) {
	return l.xDim, l.hDim, l.sDim
}

// Weights returns the layer weights.
func (l *Layer) Weights() map[net.Meta]net.Weights {
	weights := make(map[net.Meta]net.Weights)
	for _, neuron := range l.neurons {
		for _, cell := range neuron.cells {
			weights[cell.Meta()] = *cell.Weights()
		}
	}
	return weights
}

// New creates a new LSTM layer
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
			sDim:    builder.H + 1,
			clip:    clipping,
		}
	}
}

// Forward pushes the input through the layer
// x is the input
// rows of x are the input values at different time instances
// e.g. x[0] , x[1] , x[2] etc ...
func (l *Layer) Forward(x xmath.Matrix) xmath.Matrix {

	n := len(l.neurons)

	// we expect a training set equal to our depth
	xmath.MustHaveDim(x, n)

	// inter-neuronFactory communication parameter
	h := xmath.Vec(l.hDim)
	s := xmath.Vec(l.sDim)

	var y xmath.Vector
	for i := 0; i < n; i++ {
		// calculate output
		y, h, s = l.neurons[i].forward(x[i], h, s)
		y.Check()
		// apply layer activation
		l.out[i] = y
	}
	return l.out
}

// Backward handles the backpropagation logic for the layer.
// exp : is the expected output
func (l *Layer) Backward(exp xmath.Matrix) xmath.Matrix {

	h := xmath.Vec(l.hDim)
	s := xmath.Vec(l.sDim)

	for i := len(l.neurons) - 1; i >= 0; i-- {
		xmath.MustHaveSameSize(l.out[i], exp[i])
		dy := ml.Diff(exp[i], l.out[i])
		_, dh, ds := l.neurons[i].backward(dy, h, s)
		h = dh
		s = ds
	}

	//clip the weights on the positive axis to avoid exploding gradients.
	w := l.clip.W
	wClipOp := xmath.Clip(-1*w, 1*w)
	b := l.clip.B
	bClipOp := xmath.Clip(-1*b, 1*b)
	// we just need to clip the first neuron weights, as all cells have the same weight pointer.
	clipWeights(l.neurons[0], wClipOp, bClipOp)
	return nil
}

func clipWeights(neuron *neuron, wClip, bClip xmath.Op) {
	for _, n := range neuron.cells {
		if n.Weights() != nil {
			n.Weights().W = n.Weights().W.Op(wClip)
			n.Weights().B = n.Weights().B.Op(bClip)
		}
	}
}
