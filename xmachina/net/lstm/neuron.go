package lstm

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
)

type cellType string

const (
	stackCell        cellType = "stack-cell"
	forgetNeuron     cellType = "forget-neuron"
	forgetCell       cellType = "forget-cell"
	inputLeftNeuron  cellType = "input-left-neuron"
	inputRightNeuron cellType = "input-right-neuron"
	inputCell        cellType = "input-cell"
	stateNeuron      cellType = "state-neuron"
	outputNeuron     cellType = "output-neuron"
	stateCell        cellType = "state-cell"
	softCell         cellType = "soft-cell"
)

type neuron struct {
	biOps map[cellType]net.BiOp
	cells map[cellType]net.Neuron
	meta  net.Meta
}

// NeuronBuilder holds all neuron properties needed to construct the neuron.
type NeuronBuilder struct {
	n, x, h, s                     int
	rate                           ml.Learning
	weightGenerator, biasGenerator xmath.VectorGenerator
}

// NewNeuronBuilder creates a new neuron builder.
func NewNeuronBuilder(n, x, h, s int) *NeuronBuilder {
	return &NeuronBuilder{
		n: n,
		x: x,
		h: h,
		s: s,
	}
}

// WithWeights defines the weight generator functions for the rnn neuron.
func (nb *NeuronBuilder) WithWeights(weightGenerator, biasGenerator xmath.VectorGenerator) *NeuronBuilder {
	nb.weightGenerator = weightGenerator
	nb.biasGenerator = biasGenerator
	return nb
}

// WithRate defines the learning rate for the rnn neuron.
func (nb *NeuronBuilder) WithRate(learning ml.Learning) *NeuronBuilder {
	nb.rate = learning
	return nb
}

// NeuronFactory is a factory for construction of a recursive neuronFactory within the context of a recursive layer / network
type NeuronFactory func(meta net.Meta) *neuron

// Neuron is the neuronFactory implementation for a recursive neural network.
var Neuron = func(builder NeuronBuilder) NeuronFactory {
	return func(meta net.Meta) *neuron {
		z := builder.x + builder.h
		fw := net.NewWeights(z, z, builder.weightGenerator, builder.biasGenerator)
		ilw := net.NewWeights(z, z, builder.weightGenerator, builder.biasGenerator)
		irw := net.NewWeights(z, z, builder.weightGenerator, builder.biasGenerator)
		sw := net.NewWeights(z, z, builder.weightGenerator, builder.biasGenerator)
		ow := net.NewWeights(z, z, builder.weightGenerator, builder.biasGenerator)
		return &neuron{
			biOps: map[cellType]net.BiOp{
				stackCell:  net.NewStackCell(),
				forgetCell: net.NewMulCell(),
				inputCell:  net.NewMulCell(),
				stateCell:  net.NewMulCell(),
			},
			cells: map[cellType]net.Neuron{
				forgetNeuron: net.NewActivationCell(z, z, *ml.Base().
					WithActivation(ml.Sigmoid).
					WithRate(&builder.rate),
					fw,
					meta.WithID(string(forgetNeuron))),
				inputLeftNeuron: net.NewActivationCell(z, z, *ml.Base().
					WithActivation(ml.Sigmoid).
					WithRate(&builder.rate),
					ilw,
					meta.WithID(string(inputLeftNeuron))),
				inputRightNeuron: net.NewActivationCell(z, z, *ml.Base().
					WithActivation(ml.TanH).
					WithRate(&builder.rate),
					irw,
					meta.WithID(string(inputRightNeuron))),
				stateNeuron: net.NewActivationCell(z, z, *ml.Base().
					WithActivation(ml.TanH).
					WithRate(&builder.rate),
					sw,
					meta.WithID(string(stateNeuron))),
				outputNeuron: net.NewActivationCell(z, z, *ml.Base().
					WithActivation(ml.Sigmoid).
					WithRate(&builder.rate),
					ow,
					meta.WithID(string(outputNeuron))),
				softCell: net.NewSoftCell(z, z, meta.WithID(string(softCell))),
			},
			meta: meta,
		}
	}
}

func (n *neuron) forward(x, prev_h, prev_s xmath.Vector) (y, next_h, next_s xmath.Vector) {

	v := n.biOps[stackCell].Fwd(x, prev_h)

	f := n.cells[forgetNeuron].Fwd(v)
	il := n.cells[inputLeftNeuron].Fwd(v)
	ir := n.cells[inputRightNeuron].Fwd(v)
	o := n.cells[outputNeuron].Fwd(v)

	s := n.biOps[forgetCell].Fwd(f, prev_s)
	i := n.biOps[inputCell].Fwd(il, ir)

	next_s = s.Add(i)
	c := n.cells[stateNeuron].Fwd(next_s)

	next_h = n.biOps[stateCell].Fwd(c, o)

	yh := n.cells[softCell].Fwd(next_h)

	y, next_h = n.biOps[stackCell].Bwd(yh)

	return y, next_h, next_s
}

func (n *neuron) backward(dy, dh, ds xmath.Vector) (x, h, s xmath.Vector) {

	dyh := n.biOps[stackCell].Fwd(dy, dh)

	dwh := n.cells[softCell].Bwd(dyh)
	dwh = dyh.Add(dwh)

	dc, do := n.biOps[stateCell].Bwd(dwh)
	dws := n.cells[stateNeuron].Bwd(dc)
	s = ds.Add(dws)

	di1, di2 := n.biOps[inputCell].Bwd(dws)
	df, dws := n.biOps[forgetCell].Bwd(dws)

	dvo := n.cells[outputNeuron].Bwd(do)
	dvi1 := n.cells[inputLeftNeuron].Bwd(di1)
	dvi2 := n.cells[inputRightNeuron].Bwd(di2)
	dvf := n.cells[forgetNeuron].Bwd(df)

	dv := dvo.Add(dvi1).Add(dvi2).Add(dvf)

	x, h = n.biOps[stackCell].Bwd(dv)

	return x, h, s
}
