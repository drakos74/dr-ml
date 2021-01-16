package main

import (
	"fmt"
	"math"

	"github.com/drakos74/go-ex-machina/xmachina"
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmachina/net/rnn"
	"github.com/drakos74/go-ex-machina/xmath"
)

const (
	base = "base"
)

func NewRNNValueNetwork(bufferSize, hiddenLayerSize int) net.NN {
	hls := float64(hiddenLayerSize)
	builder := rnn.NewNeuronBuilder(1, 1, int(hiddenLayerSize)).
		WithRate(*ml.Rate(0.05)).
		WithWeights(xmath.RangeSqrt(-1, 1)(hls), xmath.RangeSqrt(-1, 1)(hls)).
		WithActivation(ml.TanH, ml.Sigmoid)
	return rnn.New(bufferSize, builder, net.Clip{W: 1, B: 1})
}

func NewRNNProbabilityNetwork(bufferSize, hiddenLayerSize, outputSize int) net.NN {
	hls := float64(hiddenLayerSize)
	builder := rnn.NewNeuronBuilder(1, 1, int(hiddenLayerSize)).
		WithRate(*ml.Rate(0.05)).
		WithWeights(xmath.RangeSqrt(-1, 1)(hls), xmath.RangeSqrt(-1, 1)(hls)).
		WithActivation(ml.TanH, ml.Sigmoid)
	// TODO : with output set up and softmax at the end
	return rnn.New(bufferSize, builder, net.Clip{W: 1, B: 1})
}

type T func(i int) float64

type P func(i int, x float64) float64

func X(f float64) T {
	return func(i int) float64 {
		return f * float64(i)
	}
}

var Sine P = func(_ int, x float64) float64 {
	return math.Sin(x)
}

var SineVar P = func(i int, x float64) float64 {
	return 0.3*math.Sin(x) + 0.3*math.Sin(2*x) + 0.3*math.Sin(5*x)
}

type Capture interface {
	Train(i int, x, y float64) float64
	Predict(i int, x, y float64) float64
}

type OutputCapture struct {
	lastOutput float64
	network    net.NN
}

func (c *OutputCapture) Train(i int, x, y float64) float64 {
	next := xmath.Vec(1)
	c.network.Train(xmath.Vec(1).With(y), next)
	c.lastOutput = next[0]
	return c.lastOutput
}

func (c *OutputCapture) Predict(i int, x, y float64) float64 {
	next := c.network.Predict(xmath.Vec(1).With(y))
	c.lastOutput = next[0]
	return c.lastOutput
}

type EvolutionCapture struct {
	OutputCapture
}

func (c *EvolutionCapture) Predict(i int, x, y float64) float64 {
	next := c.network.Predict(xmath.Vec(1).With(c.lastOutput))
	c.lastOutput = next[0]
	return c.lastOutput
}

type DummyCapture struct {
	OutputCapture
}

func (c *DummyCapture) Train(i int, x, y float64) float64 {
	return c.OutputCapture.Predict(i, x, y)
}

type VoidNetwork struct {
	net.NetworkConfig
}

func (v VoidNetwork) Train(input xmath.Vector, output xmath.Vector) (err xmath.Vector, weights map[net.Meta]net.Weights) {
	for i, inp := range input {
		output[i] = inp
	}
	return xmath.Vec(len(input)), nil
}

func (v VoidNetwork) Predict(input xmath.Vector) xmath.Vector {
	return input
}

func Train(xt T, yp P, graph xmachina.Data, cap map[string]Capture) {

	l := 3000
	for i := 0; i < l; i++ {

		println(fmt.Sprintf("i = %v", i))
		x := xt(i)
		y := yp(i, x)
		if i < l*3/5 {
			for name, c := range cap {
				v := c.Train(i, x, y)
				graph.Add(name, x, v)
			}
		} else {
			for name, c := range cap {
				v := c.Predict(i, x, y)
				graph.Add(name, x, v)
			}
		}

	}

}
