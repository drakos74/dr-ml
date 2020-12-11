package rnn

import (
	"fmt"
	"math"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// Weights are the weights for the recurrent layer.
type Weights struct {
	Wxh xmath.Matrix `json:"wxh"`
	Whh xmath.Matrix `json:"whh"`
	Why xmath.Matrix `json:"why"`
	Bh  xmath.Vector `json:"bh"`
	By  xmath.Vector `json:"by"`
}

// String prints the weights.
func (w Weights) String() string {
	return fmt.Sprintf("\n Wxh = %v \n Whh = %v \n Why = %v \n Bh = %v \n By = %v", w.Wxh, w.Whh, w.Why, w.Bh, w.By)
}

// Deltas are the delta parameters for the weights used during backwards propagation.
type Deltas struct {
	dWxh xmath.Matrix
	mWxh xmath.Matrix

	dWhh xmath.Matrix
	mWhh xmath.Matrix

	dWhy xmath.Matrix
	mWhy xmath.Matrix

	dBh xmath.Vector
	mBh xmath.Vector

	dBy xmath.Vector
	mBy xmath.Vector
}

// reset resets the deltas for the next calculation.
func (d *Deltas) reset(xDim, hDim int) {
	d.dWhh = xmath.Mat(hDim).Of(hDim)
	d.dWhy = xmath.Mat(xDim).Of(hDim)
	d.dWxh = xmath.Mat(hDim).Of(xDim)
	d.dBh = xmath.Vec(hDim)
	d.dBy = xmath.Vec(xDim)
}

// Parameters are the internal parameters of the layer.
type Parameters struct {
	Weights
	Deltas
	reset  func(params *Parameters)
	scaleM func(d, m xmath.Matrix) xmath.Matrix
	scaleV func(d, m xmath.Vector) xmath.Vector
}

// initParameters initialises the layer parameters according to the given dimensions.
// xDim : dimension of input and output vector
// hDim : dimnension of internal state vector
// gen : initial weights generator function
func initParameters(xDim, hDim int, gen xmath.ScaledVectorGenerator) *Parameters {
	dd := float64(hDim)
	return &Parameters{
		Weights: Weights{
			Wxh: xmath.Mat(hDim).Rows(xDim, gen(dd)),
			Whh: xmath.Mat(hDim).Rows(hDim, gen(dd)),
			Why: xmath.Mat(xDim).Rows(hDim, gen(dd)),
			Bh:  xmath.Vec(hDim),
			By:  xmath.Vec(xDim),
		},
		Deltas: Deltas{
			mWxh: xmath.Mat(hDim).Of(xDim),
			mWhh: xmath.Mat(hDim).Of(hDim),
			mWhy: xmath.Mat(xDim).Of(hDim),
			mBh:  xmath.Vec(hDim),
			mBy:  xmath.Vec(xDim),
		},
		reset: func(params *Parameters) {
			params.Deltas.reset(xDim, hDim)
		},
		scaleM: func(dm, m xmath.Matrix) xmath.Matrix {
			m = dm.Op(func(x float64) float64 {
				return math.Pow(x, 2)
			}).Add(m)
			sqrtM := m.Op(func(x float64) float64 {
				return math.Sqrt(x + 1e-8)
			})
			return sqrtM
		},
		scaleV: func(dv, v xmath.Vector) xmath.Vector {
			v = dv.Op(func(x float64) float64 {
				return math.Pow(x, 2)
			}).Add(v)
			sqrtM := v.Op(func(x float64) float64 {
				return math.Sqrt(x + 1e-8)
			})
			return sqrtM
		},
	}
}

// initWithWeights initialises the layer parameters according to the given dimensions and weights.
// xDim : dimension of input and output vector
// hDim : dimension of internal state vector
// weights : the network weights
func initWithWeights(xDim, hDim int, weights Weights) *Parameters {
	return &Parameters{
		Weights: weights,
		Deltas: Deltas{
			mWxh: xmath.Mat(hDim).Of(xDim),
			mWhh: xmath.Mat(hDim).Of(hDim),
			mWhy: xmath.Mat(xDim).Of(hDim),
			mBh:  xmath.Vec(hDim),
			mBy:  xmath.Vec(xDim),
		},
		reset: func(params *Parameters) {
			params.Deltas.reset(xDim, hDim)
		},
		scaleM: func(dm, m xmath.Matrix) xmath.Matrix {
			m = dm.Op(func(x float64) float64 {
				return math.Pow(x, 2)
			}).Add(m)
			sqrtM := m.Op(func(x float64) float64 {
				return math.Sqrt(x + 1e-8)
			})
			return sqrtM
		},
		scaleV: func(dv, v xmath.Vector) xmath.Vector {
			v = dv.Op(func(x float64) float64 {
				return math.Pow(x, 2)
			}).Add(v)
			sqrtM := v.Op(func(x float64) float64 {
				return math.Sqrt(x + 1e-8)
			})
			return sqrtM
		},
	}
}

// update updates the weights based on the deltas.
func (p *Parameters) update(rate ml.Learning) {
	dwxh := p.dWxh.Mult(-1*rate.WRate()).Dop(xmath.Div, p.scaleM(p.dWxh, p.mWxh))
	p.Wxh = p.Wxh.Add(dwxh)

	dwhh := p.dWhh.Mult(-1*rate.WRate()).Dop(xmath.Div, p.scaleM(p.dWhh, p.mWhh))
	p.Whh = p.Whh.Add(dwhh)

	dwhy := p.dWhy.Mult(-1*rate.WRate()).Dop(xmath.Div, p.scaleM(p.dWhy, p.mWhy))
	p.Why = p.Why.Add(dwhy)

	dby := p.dBy.Mult(-1*rate.BRate()).Dop(xmath.Div, p.scaleV(p.dBy, p.mBy))
	p.By = p.By.Add(dby)

	dbh := p.dBh.Mult(-1*rate.BRate()).Dop(xmath.Div, p.scaleV(p.dBh, p.mBh))
	p.Bh = p.Bh.Add(dbh)
}

// Layer is the recurrent network layer.
type Layer struct {
	*Parameters
	ml.Learning
	ml.SoftActivation
	clip    Clip
	neurons []*neuron
	xDim    int
	hDim    int
	out     xmath.Matrix
}

// Weights returns the layer weights.
func (r *Layer) Weights() Weights {
	return r.Parameters.Weights
}

// Size returns the number of recurrent neurons in the layer.
func (r *Layer) Size() int {
	return len(r.neurons)
}

// NewRNNLayer creates a new Recurrent layer
// n : batch size e.g. rnn units
// xDim : size of trainInput/trainOutput vector
// hDim : internal hidden layer size
// factory : neuronFactory factory to be used for the rnn unit
// index : index of layer in the network
// TODO : remove the rate and use it within the ml.Module
func NewRNNLayer(n, xDim, hDim int, learning ml.Learning, factory NeuronFactory, weightGenerator xmath.ScaledVectorGenerator, clipping Clip, index int) *Layer {
	neurons := make([]*neuron, n)
	for i := 0; i < n; i++ {
		neuron := factory(xDim, n, net.Meta{
			Index: i,
			Layer: index,
		})
		neurons[i] = neuron
	}
	return &Layer{
		Learning:       learning,
		SoftActivation: ml.SoftUnary{},
		neurons:        neurons,
		out:            xmath.Mat(n).Of(xDim),
		// TODO : Allow to define the initial weights from the constructor call
		Parameters: initParameters(xDim, hDim, weightGenerator),
		xDim:       xDim,
		hDim:       hDim,
		clip:       clipping,
	}
}

// LoadRNNLayer loads a new Recurrent layer based on the given weights
// n : batch size e.g. rnn units
// xDim : size of trainInput/trainOutput vector
// hDim : internal hidden layer size
// factory : neuronFactory factory to be used for the rnn unit
// index : index of layer in the network
// TODO : remove the rate and use it within the ml.Module
func LoadRNNLayer(n, xDim, hDim int, learning ml.Learning, factory NeuronFactory, weights Weights, clipping Clip, index int) *Layer {
	neurons := make([]*neuron, n)
	for i := 0; i < n; i++ {
		neuron := factory(xDim, n, net.Meta{
			Index: i,
			Layer: index,
		})
		neurons[i] = neuron
	}
	return &Layer{
		Learning:       learning,
		SoftActivation: ml.SoftUnary{},
		neurons:        neurons,
		out:            xmath.Mat(n).Of(xDim),
		// TODO : Allow to define the initial weights from the constructor call
		Parameters: initWithWeights(xDim, hDim, weights),
		xDim:       xDim,
		hDim:       hDim,
		clip:       clipping,
	}
}

// SoftMax returns the softmax function used for activation.
func (r *Layer) SoftMax() *Layer {
	r.SoftActivation = ml.SoftMax{}
	return r
}

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
		y, h = r.neurons[i].forward(x[i], h, &r.Parameters.Weights)
		// apply layer activation
		r.out[i] = r.F(y)
		log.Trace().
			Int("x", i).
			Floats64("neuronFactory-out", y).
			Floats64("h", h).
			Floats64("out", r.out[i]).
			Msg("layer forward")
		lvl := log.Logger.GetLevel()
		if lvl == zerolog.TraceLevel {
			println(fmt.Sprintf("r.Weights() = %v", r.Weights()))
		}
	}
	return r.out
}

// Backward handles the backpropagation logic for the layer.
// exp : is the expected output
func (r *Layer) Backward(exp xmath.Matrix) xmath.Matrix {

	r.Parameters.reset(r.Parameters)

	h := xmath.Vec(r.hDim)

	for i := len(r.neurons) - 1; i >= 0; i-- {
		xmath.MustHaveSameSize(r.out[i], exp[i])

		dy := r.out[i].Diff(exp[i])
		//dy := ml.Pow(exp[i], r.out[i])

		dh, dWhy, dWxh, dWhh := r.neurons[i].backward(dy, h, r.Parameters)
		// backprop into y
		r.dWhy = r.dWhy.Add(dWhy)
		r.dBy = r.dBy.Add(dy)
		// backprop into h
		r.dBh = r.dBh.Add(dh)

		r.dWxh = r.dWxh.Add(dWxh)
		r.dWhh = r.dWhh.Add(dWhh)

		h = dh
	}

	// clip the weights on the positive axis to avoid exploding gradients.
	// clip the weights on the negative axis to avoid vanishing gradients.
	w := r.clip.W
	r.dWhy = r.dWhy.Op(xmath.Clip(-1*w, 1*w))
	r.dWxh = r.dWxh.Op(xmath.Clip(-1*w, 1*w))
	r.dWhh = r.dWhh.Op(xmath.Clip(-1*w, 1*w))

	b := r.clip.B
	r.dBh = r.dBh.Op(xmath.Clip(-1*b, 1*b))
	r.dBy = r.dBy.Op(xmath.Clip(-1*b, 1*b))

	// do the updates
	r.Parameters.update(r.Learning)

	return nil
}
