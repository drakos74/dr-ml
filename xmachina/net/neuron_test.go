package net

import (
	"fmt"
	"math"
	"testing"

	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
	"github.com/drakos74/go-ex-machina/xmachina/ml"

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
