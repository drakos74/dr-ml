package net

import (
	"fmt"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/rs/zerolog/log"
)

// Neuron is a minimal computation unit with an activation function.
type Neuron interface {
	Meta() Meta
	Fwd(x xmath.Vector) xmath.Vector
	Bwd(x xmath.Vector) xmath.Vector
}

type Meta struct {
	Layer int
	Index int
}

type MLNeuron struct {
	ml.Module
	weights       xmath.Matrix
	bias          xmath.Vector
	meta          Meta
	input, output xmath.Vector
}

func NewMLNeuron(n, m int, module ml.Module, weightGenerator xmath.VectorGenerator, meta Meta) *MLNeuron {
	return &MLNeuron{
		Module:  module,
		weights: xmath.Mat(m).Generate(n, weightGenerator),
		bias:    xmath.Vec(m).Generate(weightGenerator),
		meta:    meta,
		input:   xmath.Vec(n),
		output:  xmath.Vec(m),
	}
}

func (n *MLNeuron) Fwd(v xmath.Vector) xmath.Vector {
	xmath.MustHaveSameSize(v, n.input)
	// keep a copy of the input in memory
	n.input = v
	// combine with the weights
	w := n.weights.Prod(v)
	// add bias
	z := w.Add(n.bias)
	// apply activation
	n.output = z.Op(n.Module.F)
	return n.output
}

func (n *MLNeuron) Bwd(diff xmath.Vector) xmath.Vector {
	log.Trace().Floats64("diff", diff).Str("meta", fmt.Sprintf("%v", n.meta)).Msg("train-diff")
	// find the derivative of the output
	deriv := n.output.Op(n.Module.D)
	log.Trace().Floats64("deriv", deriv).Str("meta", fmt.Sprintf("%v", n.meta)).Msg("de-activation")
	// find the gradient compared to the diff
	grad := deriv.X(diff)
	log.Trace().Floats64("grad", grad).Str("meta", fmt.Sprintf("%v", n.meta)).Msg("gradient-descent")
	// compute loss for previous layer
	loss := n.weights.T().Prod(grad)
	log.Trace().Floats64("loss", loss).Str("meta", fmt.Sprintf("%v", n.meta)).Msg("loss")
	// update weights and bias
	dW := grad.Prod(n.input)
	n.weights = n.weights.Add(dW.Mult(n.Module.WRate()))
	n.bias = n.bias.Add(grad.Mult(n.Module.BRate()))
	// return the loss to the previous layer
	return loss
}

func (n MLNeuron) Meta() Meta {
	return n.meta
}
