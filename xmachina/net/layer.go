package net

import (
	"math"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
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

type RLayer interface {
	// F will take the trainInput from the previous layer and generate an trainInput for the next layer
	Forward(v xmath.Matrix) xmath.Matrix
	// Backward will take the loss from next layer and generate a loss for the previous layer
	Backward(dv xmath.Matrix) xmath.Matrix
	// Weights returns the current weight matrix for the layer
	Weights() Parameters
	// Size returns the Size of the layer e.g. number of neurons
	Size() int
}

type Weights struct {
	Wxh xmath.Matrix
	Whh xmath.Matrix
	Why xmath.Matrix
	Bh  xmath.Vector
	By  xmath.Vector
}

type Parameters struct {
	Weights
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

	reset func(params *Parameters)

	scaleM func(d, m xmath.Matrix) xmath.Matrix
	scaleV func(d, m xmath.Vector) xmath.Vector
}

func initParameters(p, d int) *Parameters {
	return &Parameters{
		Weights: Weights{
			Wxh: xmath.Mat(d).Rows(p, xmath.Rand(-1, 1, func(x float64) float64 {
				return math.Sqrt(float64(d) * x)
			})),
			Whh: xmath.Mat(d).Rows(d, xmath.Rand(-1, 1, func(x float64) float64 {
				return math.Sqrt(float64(d) * x)
			})),
			Why: xmath.Mat(p).Rows(d, xmath.Rand(-1, 1, func(x float64) float64 {
				return math.Sqrt(float64(p) * x)
			})),
			Bh: xmath.Vec(d),
			By: xmath.Vec(p),
		},
		mWxh: xmath.Mat(d).Of(p),
		mWhh: xmath.Mat(d).Of(d),
		mWhy: xmath.Mat(p).Of(d),
		mBh:  xmath.Vec(d),
		mBy:  xmath.Vec(p),
		reset: func(params *Parameters) {
			params.dWhh = xmath.Mat(d).Of(d)
			params.dWhy = xmath.Mat(p).Of(d)
			params.dWxh = xmath.Mat(d).Of(p)
			params.dBh = xmath.Vec(d)
			params.dBy = xmath.Vec(p)
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

type RNNLayer struct {
	*Parameters
	ml.Learning
	ml.SoftActivation
	neurons []*rNeuron
	idx     int
	p       int
	d       int
	out     xmath.Matrix
}

func (r *RNNLayer) Weights() Weights {
	return r.Parameters.Weights
}

func (r *RNNLayer) Size() int {
	return len(r.neurons)
}

// NewRNNLayer creates a new Recurrent layer
// s : size of trainInput/trainOutput vector
// n : batch size e.g. rnn units
// d : internal hidden layer size
// factory : neuron factory to be used for the rnn unit
// index : index of layer in the network
// TODO : remove the rate and use it within the ml.Module
func NewRNNLayer(s, n, d int, learning ml.Learning, factory RNeuronFactory, index int) *RNNLayer {
	neurons := make([]*rNeuron, n)
	for i := 0; i < n; i++ {
		neuron := factory(s, n, meta{
			index: i,
			layer: index,
		})
		neurons[i] = neuron
	}
	return &RNNLayer{
		Learning:       learning,
		SoftActivation: ml.SoftUnary{},
		neurons:        neurons,
		out:            xmath.Mat(n).Of(s),
		Parameters:     initParameters(s, d),
		p:              s,
		d:              d,
	}
}

func (r *RNNLayer) SoftMax() *RNNLayer {
	r.SoftActivation = ml.SoftMax{}
	return r
}

func (r *RNNLayer) Forward(v xmath.Matrix) xmath.Matrix {

	// we expect a training set equal to our depth
	xmath.MustHaveDim(v, len(r.neurons))

	h := xmath.Vec(r.d)

	var out xmath.Vector
	for i := 0; i < len(r.neurons); i++ {
		out, h = r.neurons[i].forward(v[i], h, &r.Parameters.Weights)
		r.out[i] = r.F(out)
	}

	// push
	return r.out

}

func (r *RNNLayer) Backward(exp xmath.Matrix) xmath.Matrix {

	r.Parameters.reset(r.Parameters)

	h := xmath.Vec(r.d)

	for i := len(r.neurons) - 1; i >= 0; i-- {
		xmath.MustHaveSameSize(r.out[i], exp[i])

		// TODO : use the soft for backwards properly
		dy := r.out[i].Diff(exp[i])

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

	// TODO : maybe we dont need this, or can do it somehow different
	r.dWhy.Op(xmath.Clip(-5, 5))
	r.dWxh.Op(xmath.Clip(-5, 5))
	r.dWhh.Op(xmath.Clip(-5, 5))

	// do the updates
	r.Parameters.update(r.Learning)

	return nil

}
