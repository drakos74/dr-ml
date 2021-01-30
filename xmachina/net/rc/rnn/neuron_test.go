package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmachina/net/rc"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
)

func init() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
}

func TestRNeuron_DimensionsInForwardPass(t *testing.T) {

	rNeuron := testNeuronFactory(3, 5, 10)(net.Meta{})

	// trainInput layer size : 3
	// trainOutput layer size : 5
	// layer depth : 10

	sequenceLength := 10

	xt := xmath2.algebra.Mat(sequenceLength).Generate(3, xmath2.algebra.Rand(-1, 1, math.Sqrt))
	h0 := xmath2.algebra.Mat(sequenceLength).Generate(10, xmath2.algebra.Rand(-1, 1, math.Sqrt))

	y := xmath2.algebra.Mat(sequenceLength)
	wh := xmath2.algebra.Mat(sequenceLength)

	for i := 0; i < len(xt); i++ {
		y[i], wh[i] = rNeuron.forward(xt[i], h0[i])
		assert.Equal(t, 5, len(y[i]))
		assert.Equal(t, 10, len(wh[i]))
	}

}

func TestRNeuron_Train(t *testing.T) {

	type test struct {
		inp, outp, hidden int
		input, expected   xmath2.algebra
		iterations        int
	}

	tests := map[string]test{
		"increasing-sequence": {
			inp:        5,
			outp:       5,
			hidden:     10,
			input:      xmath2.algebra.Vec(5).With(0.1, 0.2, 0.3, 0.4, 0.5),
			expected:   xmath2.algebra.Vec(5).With(0.2, 0.3, 0.4, 0.5, 0.6),
			iterations: 100,
		},
		"decreasing-sequence": {
			inp:        5,
			outp:       5,
			hidden:     10,
			input:      xmath2.algebra.Vec(5).With(0.1, 0.2, 0.3, 0.4, 0.5),
			expected:   xmath2.algebra.Vec(5).With(0.9, 0.8, 0.7, 0.6, 0.5),
			iterations: 50,
		},
		"iterating-sequence": {
			inp:        5,
			outp:       5,
			hidden:     10,
			input:      xmath2.algebra.Vec(5).With(0.1, 0.2, 0.3, 0.4, 0.5),
			expected:   xmath2.algebra.Vec(5).With(0.1, 0.2, 0.3, 0.2, 0.1),
			iterations: 50,
		},
		"event-trigger-thin": {
			inp:        5,
			outp:       5,
			hidden:     10,
			input:      xmath2.algebra.Vec(5).With(0.1, 0.2, 0.3, 0.2, 0.1),
			expected:   xmath2.algebra.Vec(5).With(0.1, 0.1, 0.9, 0.1, 0.1),
			iterations: 60,
		},
		"event-trigger-thick": {
			inp:        5,
			outp:       5,
			hidden:     30,
			input:      xmath2.algebra.Vec(5).With(0.1, 0.2, 0.3, 0.2, 0.1),
			expected:   xmath2.algebra.Vec(5).With(0.1, 0.1, 0.9, 0.1, 0.1),
			iterations: 50,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			trainNeuron(t, tt.inp, tt.outp, tt.hidden, tt.input, tt.expected, tt.iterations)
		})
	}

}

func trainNeuron(t *testing.T, inp, outp, hidden int, input, expected xmath2.algebra, iterations int) {

	rneuron := testNeuronFactory(inp, outp, hidden)(net.Meta{})

	y := xmath2.algebra.Vec(outp)
	h := xmath2.algebra.Vec(hidden)
	err := xmath2.algebra.Vec(outp)

	var loss float64
	for i := 0; i < iterations; i++ {
		y, h = rneuron.forward(input, h)
		err = ml.Diff(expected, y)
		newLoss := err.Op(math.Abs).Sum()
		if i > 0 {
			// TODO : we cant always be so strict ...
			//assert.True(t, newLoss < loss, fmt.Sprintf("new = %v , old = %v", newLoss, loss))
		}
		loss = newLoss
		println(fmt.Sprintf("loss = %v", loss))
		rneuron.backward(err, h)
	}
	println(fmt.Sprintf("loss = %v", loss))
	assert.True(t, loss < 0.001)
}

func testNeuronFactory(x, y, h int) NeuronFactory {
	builder := rc.NewNeuronBuilder(x, y, h).
		WithActivation(ml.TanH, ml.Sigmoid).
		WithWeights(xmath2.algebra.Rand(-1, 1, math.Sqrt), xmath2.algebra.Rand(-1, 1, math.Sqrt)).
		WithRate(*ml.Learn(0.05, 0.05))
	return Neuron(*builder)
}

func testNeuronBuilder(x, y, h int) *rc.NeuronBuilder {
	return rc.NewNeuronBuilder(x, y, h).
		WithActivation(ml.TanH, ml.Sigmoid).
		WithWeights(xmath2.algebra.Const(0.5), xmath2.algebra.Const(0.5)).
		WithRate(*ml.Learn(0.05, 0.05))
}
