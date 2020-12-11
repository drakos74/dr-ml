package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/rs/zerolog"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/stretchr/testify/assert"
)

func TestRNNLayer_ForwardWithPositiveWeights(t *testing.T) {

	zerolog.SetGlobalLevel(zerolog.InfoLevel)

	layer := NewRNNLayer(5, 1, 10, ml.Learn(0.0001), RNeuron(ml.TanH), xmath.RangeSqrt(0, 1), Clip{
		W: 1,
		B: 1,
	}, 0)

	// 2 is the lookback rate e.g. 2 neurons, 2 time instances are tracked
	// each timeinstance
	input := xmath.Mat(5).With(
		xmath.Vec(1).With(0.1),
		xmath.Vec(1).With(0.2),
		xmath.Vec(1).With(0.3),
		xmath.Vec(1).With(0.4),
		xmath.Vec(1).With(0.5),
	)

	output := layer.Forward(input)

	println(fmt.Sprintf("output = %v", output))

	// we dont change any weights, so we would expect that outputs are also increasing in value
	min := 0.0
	for _, out := range output {
		assert.True(t, min < out[0])
		min = out[0]
	}

}

func TestRNNLayer_Backward(t *testing.T) {

	zerolog.SetGlobalLevel(zerolog.InfoLevel)

	layer := NewRNNLayer(5, 1, 10, ml.Learn(0.01), RNeuron(ml.ReLU), xmath.RangeSqrt(0, 1), Clip{
		W: 1,
		B: 1,
	}, 0)

	// 2 is the lookback rate e.g. 2 neurons, 2 time instances are tracked
	// each timeinstance
	input := xmath.Mat(5).With(
		xmath.Vec(1).With(0.1),
		xmath.Vec(1).With(0.2),
		xmath.Vec(1).With(0.3),
		xmath.Vec(1).With(0.4),
		xmath.Vec(1).With(0.5),
	)

	output := xmath.Mat(5).With(
		xmath.Vec(1).With(0.1),
		xmath.Vec(1).With(0.2),
		xmath.Vec(1).With(0.3),
		xmath.Vec(1).With(0.2),
		xmath.Vec(1).With(0.1),
	)

	for i := 0; i < 200; i++ {

		out := layer.Forward(input)

		println(fmt.Sprintf("out = %v", out))

		loss := ml.CompLoss(ml.CrossEntropy)(output, out)
		println(fmt.Sprintf("loss = %v", loss.Sum()))

		oo := layer.Backward(output)

		println(fmt.Sprintf("oo = %v", oo))
	}

	out := layer.Forward(input)

	println(fmt.Sprintf("out = %v", out))

	loss := ml.CompLoss(ml.CrossEntropy)(output, out)
	println(fmt.Sprintf("loss = %v", loss))

}

func TestRNNLayer_WithSmallLearningRate(t *testing.T) {

	layer := NewRNNLayer(25, 14, 100, ml.Learn(0.0001), RNeuron(ml.TanH), xmath.RangeSqrt(-1, 1), Clip{
		W: 1,
		B: 1,
	}, 0).SoftMax()

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

	layer := NewRNNLayer(25, 14, 100, ml.Learn(0), RNeuron(ml.TanH), xmath.RangeSqrt(-1, 1), Clip{
		W: 1,
		B: 1,
	}, 0).SoftMax()

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

func trainRNN(layer *Layer, inputs, outputs xmath.Matrix) float64 {

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
