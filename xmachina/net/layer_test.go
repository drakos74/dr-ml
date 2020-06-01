package net

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/stretchr/testify/assert"
)

// This is a repetition of TestNeuron_BinaryClassification in neuron_test.go
func TestLayer_LeaningProcess(t *testing.T) {

	module := ml.Model().Rate(1, 0.05)

	layer := NewFFLayer(2, 2, Perceptron(module, xmath.Const(0.5)), 0)

	inp1 := []float64{0, 1}
	exp1 := []float64{1, 0}
	//
	inp2 := []float64{1, 0}
	exp2 := []float64{0, 1}

	iterations := 10000

	v1 := xmath.Vec(2)
	v2 := xmath.Vec(2)

	err := math.MaxFloat64
	var finishedAt int
	for i := 0; i < iterations; i++ {

		v1 = layer.Forward(inp1)
		err1 := xmath.Vec(2).With(exp1...).Diff(v1)
		layer.Backward(err1)

		v2 = layer.Forward(inp2)
		err2 := xmath.Vec(2).With(exp2...).Diff(v2)
		layer.Backward(err2)

		loss := err1.Add(err2).Norm()

		assert.True(t, loss <= err || math.Abs(loss-err) < 0.00001, fmt.Sprintf("loss = %v > err = %v", loss, err))
		err = loss
		if err < 0.0001 && finishedAt == 0 {
			finishedAt = i
		}

	}

	for i, r := range exp1 {
		v := v1[i]
		assert.True(t, math.Abs(v-r) < 0.01, fmt.Sprintf("v = %v vs r = %v", v, r))
	}

	for i, r := range exp2 {
		v := v2[i]
		assert.True(t, math.Abs(v-r) < 0.01)
	}

	assert.True(t, finishedAt > 0)
	assert.True(t, finishedAt < iterations)
}

// Note : this test might take a while ...
func TestLayer_RandomLearningProcessScenarios(t *testing.T) {

	for i := 0; i < 10; i++ {

		rand.Seed(time.Now().UnixNano())
		// generate inputs
		inp := xmath.Mat(2).With(xmath.Rand(0, 1, xmath.Unit)(2, 0), xmath.Rand(0, 1, xmath.Unit)(2, 1))
		exp := xmath.Mat(2).With(xmath.Rand(0, 1, xmath.Unit)(2, 0), xmath.Rand(0, 1, xmath.Unit)(2, 0))

		assertTraining(t, inp, exp)

	}

}

func assertTraining(t *testing.T, inp, exp xmath.Matrix) {

	log.Println(fmt.Sprintf("inp = %v", inp))
	log.Println(fmt.Sprintf("exp = %v", exp))

	layer := NewFFLayer(2, 2, Perceptron(ml.Model(), xmath.Const(0.5)), 0)

	v := xmath.Mat(len(inp))

	errThreshold := 0.00001

	sumErr := 0.0
	var finishedAt int
	i := 0
	for {
		loss := xmath.Vec(len(v))
		for j := 0; j < len(v); j++ {
			v[j] = layer.Forward(inp[j])
			err := xmath.Vec(2).With(exp[j]...).Diff(v[j])
			layer.Backward(err)
			loss = loss.Add(err)
		}

		if loss.Norm() < errThreshold && finishedAt == 0 {
			finishedAt = i
			break
		}

		assert.True(t, sumErr-loss.Norm()/loss.Norm() < 0.01, fmt.Sprintf("%v : %v < %v for \n inp = %v exp = %v", i, loss.Norm(), sumErr, inp, exp))
		sumErr = loss.Norm()

		if i%10001 == 0 {
			log.Println(fmt.Sprintf("sumErr at %v = %v", i, sumErr))
		}

		i++
	}

	log.Println(fmt.Sprintf("v = %v", v))

	for i, e := range exp {
		for j, r := range e {
			v := v[i][j]
			lossThreshold := errThreshold * 10
			assert.True(t, math.Abs(v-r) < lossThreshold, fmt.Sprintf("%v >= %v for \n inp = %v exp = %v", math.Abs(v-r), lossThreshold, inp, exp))
		}

	}

	assert.True(t, finishedAt > 0)
}

func TestFFLayer_WithNoLearning(t *testing.T) {

	layer := NewFFLayer(2, 2, Perceptron(ml.Model().Rate(0, 0), xmath.Const(0.5)), 0)

	inp := []float64{0.3, 0.7}

	out := xmath.Vec(2).With(1, 0)

	// do one pass to check the output and error
	output := layer.Forward(inp)
	diff := out.Diff(output)
	err := layer.Backward(diff)

	for i := 0; i < 10; i++ {
		outp := layer.Forward(inp)

		assert.Equal(t, output, outp)

		hErr := layer.Backward(diff)

		assert.Equal(t, err, hErr)

		output = outp

	}

}

// TODO : test softmax layer
func TestSoftMaxLayer(t *testing.T) {

	l := NewSMLayer(10, 0)

	v := xmath.Vec(10).With(0.1, 2, 3, 5, 1, 7, 4, 4, 0.5, 7)

	println(fmt.Sprintf("v = %v", v))

	o := l.Forward(v)

	assert.Equal(t, 1.0, math.Round(o.Sum()))

	println(fmt.Sprintf("o = %v", o))

	e := xmath.Const(0.5)(10, 0)

	er := l.Backward(e.Diff(o))

	println(fmt.Sprintf("er = %v", er))

}

func TestRNNLayer_WithSmallLearningRate(t *testing.T) {

	layer := NewRNNLayer(14, 25, 100, ml.Learn(0.0001), RNeuron(ml.TanH), 0).SoftMax()

	inputs, outputs := prepareRNNTrainSet()

	pLoss := math.MaxFloat64
	for i := 0; i < 100; i++ {
		loss := trainRNN(layer, inputs, outputs)
		// with very small learning rate, we should always have incremental improvement!!!
		assert.True(t, loss < pLoss, fmt.Sprintf("new loss %v should be smaller than previous %v", loss, pLoss))
		pLoss = loss
	}

}

func TestRNNLayer_WithoutLearningRate(t *testing.T) {

	layer := NewRNNLayer(14, 25, 100, ml.Learn(0), RNeuron(ml.TanH), 0).SoftMax()

	inputs, outputs := prepareRNNTrainSet()

	var pLoss float64
	for i := 0; i < 100; i++ {
		loss := trainRNN(layer, inputs, outputs)
		// with 'zero' learning rate, we should see no improvement at all!
		if pLoss > 0 {
			assert.Equal(t, loss, pLoss)
		}
		pLoss = loss
	}

}

func trainRNN(layer *RNNLayer, inputs, outputs xmath.Matrix) float64 {

	out := layer.Forward(inputs)

	var err float64
	// calculate the loss
	for i := 0; i < 25; i++ {

		loss := ml.CrossEntropy(outputs[i], out[i])

		err += loss.Sum()

	}

	layer.Backward(outputs)

	return err
}

func prepareRNNTrainSet() (inputs, outputs xmath.Matrix) {
	inp := []int{0, 1, 2, 3, 4, 5, 1, 6, 4, 1, 4, 7, 8, 9, 9, 7, 10, 4, 7, 1, 11, 12, 13, 0, 1}
	// transform to unique elements
	inputs = xmath.Mat(25)
	for i := range inp {
		input := xmath.Vec(14)
		input[inp[i]] = 1
		inputs[i] = input
	}

	exp := []int{1, 2, 3, 4, 5, 1, 6, 4, 1, 4, 7, 8, 9, 9, 7, 10, 4, 7, 1, 11, 12, 13, 0, 1, 2}
	outputs = xmath.Mat(25)
	for i := range inp {
		output := xmath.Vec(14)
		output[exp[i]] = 1
		outputs[i] = output
	}
	return
}
