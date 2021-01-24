package lstm

import (
	"fmt"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmachina/net/rc"
)

type cellType string

const (
	inputStackCell   cellType = "input-stack-cell"
	outputStackCell  cellType = "output-stack-cell"
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

// NeuronFactory is a factory for construction of a recursive neuronFactory within the context of a recursive layer / network
type NeuronFactory func(meta net.Meta) *neuron

// Neuron is the neuronFactory implementation for a recursive neural network.
var Neuron = func(builder rc.NeuronBuilder) NeuronFactory {
	if len(builder.G) != 3 {
		panic(fmt.Sprintf("cannot construct lstm neuron without 2 activation functions: %+v", builder.G))
	}

	z := builder.X + builder.H
	w := builder.Y + builder.H

	softActivation := func(meta net.Meta) net.Neuron {
		return net.NoOp(z, z, meta.WithID("soft"))
	}
	if builder.Softmax {
		softActivation = func(meta net.Meta) net.Neuron {
			return net.NewSoftCell(builder.Y, builder.S, meta.WithID("soft"))
		}
	}

	return func(meta net.Meta) *neuron {

		fw := net.NewWeights(z, z, builder.WeightGenerator, builder.BiasGenerator)
		ilw := net.NewWeights(z, z, builder.WeightGenerator, builder.BiasGenerator)
		irw := net.NewWeights(z, z, builder.WeightGenerator, builder.BiasGenerator)
		sw := net.NewWeights(z, w, builder.WeightGenerator, builder.BiasGenerator)
		ow := net.NewWeights(z, w, builder.WeightGenerator, builder.BiasGenerator)
		return &neuron{
			biOps: map[cellType]net.BiOp{
				inputStackCell:  net.NewStackCell(builder.X),
				outputStackCell: net.NewStackCell(builder.Y),
				forgetCell:      net.NewMulCell(),
				inputCell:       net.NewMulCell(),
				stateCell:       net.NewMulCell(),
			},
			cells: map[cellType]net.Neuron{
				forgetNeuron: net.NewActivationCell(z, z, *ml.Base().
					WithActivation(builder.G[0]).
					WithRate(&builder.Rate),
					fw,
					meta.WithID(string(forgetNeuron))),
				inputLeftNeuron: net.NewActivationCell(z, z, *ml.Base().
					WithActivation(builder.G[0]).
					WithRate(&builder.Rate),
					ilw,
					meta.WithID(string(inputLeftNeuron))),
				inputRightNeuron: net.NewActivationCell(z, z, *ml.Base().
					WithActivation(builder.G[1]).
					WithRate(&builder.Rate),
					irw,
					meta.WithID(string(inputRightNeuron))),
				stateNeuron: net.NewActivationCell(z, w, *ml.Base().
					WithActivation(builder.G[1]).
					WithRate(&builder.Rate),
					sw,
					meta.WithID(string(stateNeuron))),
				outputNeuron: net.NewActivationCell(z, w, *ml.Base().
					WithActivation(builder.G[2]).
					WithRate(&builder.Rate),
					ow,
					meta.WithID(string(outputNeuron))),
				softCell: softActivation(meta),
			},
			meta: meta,
		}
	}
}

func (n *neuron) forward(x, prev_h, prev_s xmath.Vector) (y, next_h, next_s xmath.Vector) {

	// combine memory and input vectors
	v := n.biOps[inputStackCell].Fwd(x, prev_h)

	f := n.cells[forgetNeuron].Fwd(v)
	il := n.cells[inputLeftNeuron].Fwd(v)
	ir := n.cells[inputRightNeuron].Fwd(v)
	o := n.cells[outputNeuron].Fwd(v)

	s := n.biOps[forgetCell].Fwd(f, prev_s)
	i := n.biOps[inputCell].Fwd(il, ir)

	next_s = s.Add(i)
	c := n.cells[stateNeuron].Fwd(next_s)

	next_h = n.biOps[stateCell].Fwd(c, o)

	// split memory and output vectors
	y, next_h = n.biOps[outputStackCell].Bwd(next_h)

	yh := n.cells[softCell].Fwd(y)

	return yh, next_h, next_s
}

func (n *neuron) backward(dyh, dh, ds xmath.Vector) (x, h, s xmath.Vector) {

	dy := n.cells[softCell].Bwd(dyh)

	// combine memory and output delta vectors
	dwh := n.biOps[outputStackCell].Fwd(dy, dh)

	//dwh = ds.Add(dwh)

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

	// split memory and input delta vectors
	x, h = n.biOps[inputStackCell].Bwd(dv)

	return x, h, s
}
