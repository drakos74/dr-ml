package net

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/stretchr/testify/assert"
)

func TestNeuron_SimpleForward(t *testing.T) {

	module := ml.Model()

	neuron := Perceptron(module, xmath.Const(0.5))(2, meta{})

	result := neuron.forward([]float64{1, 0})

	// result should be the sigmoid of 0.5
	assert.Equal(t, ml.Sigmoid.F(0.5), result)

}

func TestNeuron_SimpleForward_NegWeights(t *testing.T) {

	module := ml.Model()

	neuron := Perceptron(module, xmath.Const(-0.5))(2, meta{})

	result := neuron.forward([]float64{-1, 1})

	// result should be the sigmoid of 0.5
	assert.Equal(t, 0.5, result)

}

func TestNeuron_ZeroErrorBackward(t *testing.T) {

	module := ml.Model()

	weights := xmath.Const(0.5)
	neuron := Perceptron(module, weights)(2, meta{})
	neuron.forward([]float64{1, 0})

	v := neuron.backward(0)

	assert.Equal(t, xmath.Const(0.0)(2, 0), v)

	assert.Equal(t, weights(2, 0), neuron.weights)

}

func TestNeuron_GreaterErrorBackward(t *testing.T) {

	module := ml.Model()

	weights := xmath.Const(0.5)
	neuron := Perceptron(module, weights)(2, meta{})

	expected := 1.0
	var r float64
	var err float64
	for i := 0; i < 100; i++ {
		r = neuron.forward([]float64{1, 0.3})
		// take into account the oscillations bu apart from that the error should reduce
		assert.True(t, expected-r < err || err < 0.1)
		err := expected - r
		assert.True(t, err < 1)
		neuron.backward(err)
	}

	// note we ll never reach '1' because we are using sigmoid
	assert.True(t, math.Abs(r-expected) < 0.1)

}

func TestNeuron_GreaterErrorBackward_NegWeights(t *testing.T) {

	module := ml.Model()

	weights := xmath.Const(-0.5)
	neuron := Perceptron(module, weights)(2, meta{})

	expected := 1.0
	var r float64
	var err float64
	for i := 0; i < 100; i++ {
		r = neuron.forward([]float64{-1, -0.3})
		// take into account the oscillations bu apart from that the error should reduce
		assert.True(t, expected-r < err || err < 0.1)
		err := expected - r
		assert.True(t, err < 1)
		neuron.backward(err)
	}

	// note we ll never reach '1' because we are using sigmoid
	assert.True(t, math.Abs(r-expected) < 0.1)

}

func TestNeuron_SmallerErrorBackward(t *testing.T) {

	module := ml.Model().Rate(10, 0.05)

	weights := xmath.Const(0.5)
	neuron := Perceptron(module, weights)(2, meta{})

	expected := 0.1

	var r float64
	var err float64
	for i := 0; i < 100; i++ {
		r = neuron.forward([]float64{0.1, 0.8})
		e := expected - r
		assert.True(t, e < err || err < 0.1)
		err := e
		assert.True(t, err < 1)
		neuron.backward(err)
	}

	assert.True(t, math.Abs(r-expected) < 0.01, fmt.Sprintf("result = %v , expected = %v", r, expected))

}

func TestNeuron_BinaryClassification(t *testing.T) {

	module := ml.Model()

	weights := xmath.Const(0.5)
	neuron := Perceptron(module, weights)(2, meta{})

	inp1 := []float64{0, 1}
	exp1 := 1.0

	inp2 := []float64{1, 0}
	exp2 := 0.0

	iterations := 1000

	var r1 float64
	var r2 float64
	var err float64

	var finishedAt int
	for i := 0; i < iterations; i++ {
		r1 = neuron.forward(inp1)
		// note we emulate the loss function
		neuron.backward(exp1 - r1)

		r2 = neuron.forward(inp2)
		// note we emulate the loss function
		neuron.backward(exp2 - r2)

		e := exp1 - r1 + exp2 - r2
		assert.True(t, e < err || err < 0.1)
		err := e
		assert.True(t, math.Abs(err) < 1)
		if math.Abs(err) < 0.0001 && finishedAt == 0 {
			finishedAt = i
		}
	}

	assert.True(t, finishedAt > 0)
	assert.True(t, finishedAt < iterations)
	assert.True(t, math.Abs(exp1-r1) < 0.05)
	assert.True(t, math.Abs(exp2-r2) < 0.05)

}

// TODO : add negative scenarios [negative result , oscillating]

func TestNeuron_SigmoidActivationLimit(t *testing.T) {

	module := ml.Model()

	neuron := Perceptron(module, xmath.Const(-0.5))(2, meta{})

	expected := 10.0
	var r float64
	var err float64
	for i := 0; i < 100; i++ {
		r = neuron.forward([]float64{-1, -0.3})
		// we are still improving ...
		assert.True(t, expected-r < err || err < 0.1)
		err := expected - r
		// but can never go beyond 1
		assert.True(t, err > 1)
		neuron.backward(err)
	}

	// note we ll never go beyond 1 because we are using sigmoid
	assert.True(t, math.Abs(r-expected) >= 10-1, fmt.Sprintf("result = %v , expected = %v", r, expected))

}

func TestRNeuron_DimensionsInForwardPass(t *testing.T) {

	rneuron := RNeuron(ml.TanH)(3, 5, meta{})

	// trainInput layer size : 3
	// trainOutput layer size : 2
	// layer depth : 5

	xt := xmath.Mat(10).Rows(3, xmath.Rand(-1, 1, math.Sqrt))
	h0 := xmath.Mat(10).Rows(5, xmath.Rand(-1, 1, math.Sqrt))

	params := &Parameters{
		Weights: Weights{
			Whh: xmath.Mat(5).Rows(5, xmath.Rand(-1, 1, math.Sqrt)),
			Wxh: xmath.Mat(5).Rows(3, xmath.Rand(-1, 1, math.Sqrt)),
			Why: xmath.Mat(2).Rows(5, xmath.Rand(-1, 1, math.Sqrt)),
			Bh:  xmath.Rand(-1, 1, math.Sqrt)(5, 0),
			By:  xmath.Rand(-1, 1, math.Sqrt)(2, 0),
		},
	}

	y := xmath.Mat(10)
	wh := xmath.Mat(10)

	for i := 0; i < len(xt); i++ {
		y[i], wh[i] = rneuron.forward(xt[i], h0[i], &params.Weights)
	}

	println(fmt.Sprintf("y = %v", y.T()))
	println(fmt.Sprintf("wh = %v", wh.T()))

}

// TODO : fix this test
func TestRNeuron_Train(t *testing.T) {

	// normal increasing series
	x := xmath.Mat(10).Of(1).
		With(
			xmath.Vec(1).With(1),
			xmath.Vec(1).With(2),
			xmath.Vec(1).With(3),
			xmath.Vec(1).With(4),
			xmath.Vec(1).With(5),
			xmath.Vec(1).With(6),
			xmath.Vec(1).With(7),
			xmath.Vec(1).With(8),
			xmath.Vec(1).With(9),
			xmath.Vec(1).With(10),
		)

	e := xmath.Mat(10).Of(1).
		With(
			xmath.Vec(1).With(2),
			xmath.Vec(1).With(3),
			xmath.Vec(1).With(4),
			xmath.Vec(1).With(5),
			xmath.Vec(1).With(6),
			xmath.Vec(1).With(7),
			xmath.Vec(1).With(8),
			xmath.Vec(1).With(9),
			xmath.Vec(1).With(10),
			xmath.Vec(1).With(11),
		)

	rneuron := RNeuron(ml.TanH)(1, 5, meta{})

	params := &Parameters{
		Weights: Weights{
			Whh: xmath.Mat(5).Rows(5, xmath.Rand(-1, 1, math.Sqrt)), // interlayer
			Wxh: xmath.Mat(5).Rows(1, xmath.Rand(-1, 1, math.Sqrt)), // trainInput
			Why: xmath.Mat(1).Rows(5, xmath.Rand(-1, 1, math.Sqrt)), // trainOutput
			Bh:  xmath.Rand(-0.1, 0.1, math.Sqrt)(5, 0),
			By:  xmath.Rand(-0.1, 0.1, math.Sqrt)(1, 0),
		},
	}

	h := xmath.Vec(5)

	o := xmath.Mat(10)
	wh := xmath.Mat(10)

	err := xmath.Mat(10)

	for i := 0; i < len(x); i++ {
		o[i], h = rneuron.forward(x[i], h, &params.Weights)
		wh[i] = h
		err[i] = ml.CrossEntropy(e[i], o[i])
	}

	// do the backward pass
	for i := len(x) - 1; i >= 0; i-- {
		//rneuron.backward(err[i])
	}

	println(fmt.Sprintf("o = %v", o.T()))
	println(fmt.Sprintf("wh = %v", wh.T()))

}
