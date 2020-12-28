package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/stretchr/testify/assert"
)

func TestRNNLayer_Forward(t *testing.T) {

	builder := testNeuronBuilder(1, 1, 100).
		WithWeights(xmath.RangeSqrt(-1, 1)(10), xmath.RangeSqrt(-1, 1)(10))

	layer := NewLayer(5, *builder, Clip{
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

	assert.Equal(t, 5, len(output))

}

func TestRNNLayer_Train(t *testing.T) {

	type test struct {
		n, x, y, h                      int
		rate                            ml.Learning
		weightsGenerator, biasGenerator xmath.VectorGenerator
		input, output                   xmath.Matrix
		threshold                       int
	}

	tests := map[string]test{
		"basic": {
			n:                5,
			x:                1,
			y:                1,
			h:                100,
			rate:             *ml.Learn(1, 1),
			weightsGenerator: xmath.RangeSqrt(-1, 1)(10),
			biasGenerator:    xmath.RangeSqrt(-1, 1)(10),
			input: xmath.Mat(5).With(
				xmath.Vec(1).With(0.1),
				xmath.Vec(1).With(0.2),
				xmath.Vec(1).With(0.3),
				xmath.Vec(1).With(0.4),
				xmath.Vec(1).With(0.5),
			),
			output: xmath.Mat(5).With(
				xmath.Vec(1).With(0.1),
				xmath.Vec(1).With(0.2),
				xmath.Vec(1).With(0.3),
				xmath.Vec(1).With(0.2),
				xmath.Vec(1).With(0.1),
			),
			threshold: 1000,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			builder := testNeuronBuilder(tt.x, tt.y, tt.h).
				WithRate(tt.rate).
				WithWeights(tt.weightsGenerator, tt.biasGenerator)
			trainLayer(t, tt.n, *builder, tt.input, tt.output, tt.threshold)
		})
	}

}

func trainLayer(t *testing.T, n int, builder NeuronBuilder, input, output xmath.Matrix, threshold int) {

	layer := NewLayer(n, builder, Clip{
		W: 1,
		B: 1,
	}, 0)

	for i := 0; i < threshold; i++ {

		out := layer.Forward(input)

		loss := output.Dop(func(x, y float64) float64 {
			return x - y
		}, out)
		println(fmt.Sprintf("loss = %v", loss.Op(math.Abs).Sum().Sum()))

		layer.Backward(output)
	}

	out := layer.Forward(input)

	println(fmt.Sprintf("out = %v", out))
	println(fmt.Sprintf("output = %v", output))

	loss := output.Dop(func(x, y float64) float64 {
		return x - y
	}, out)
	println(fmt.Sprintf("loss = %v", loss.Op(math.Abs).Sum().Sum()))
	// we need to be extra
	assert.True(t, loss.Op(math.Abs).Sum().Sum() < 0.01)
}
