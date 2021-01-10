package net

import (
	"fmt"
	"math"
	"testing"

	"github.com/rs/zerolog"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/stretchr/testify/assert"
)

func init() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
}

func TestActivationCell_FwdAndBwd(t *testing.T) {

	xDim := 2
	yDim := 3

	factory := NewBuilder().
		WithWeights(xmath.Const(0.5), xmath.Const(0.5)).
		WithModule(ml.Base().WithRate(ml.Learn(1, 1))).
		Factory(NewActivationCell)

	neuron := factory(xDim, yDim, Meta{})

	weights := neuron.Weights()
	vec := xmath.Vec(xDim).With(0.5, 0.5)
	// weights should be a 2x3 matrix
	assert.Equal(t, xmath.Mat(yDim).With(vec, vec, vec), weights.W)
	// bias should be same as the output dimension
	assert.Equal(t, xmath.Vec(yDim).With(0.5, 0.5, 0.5), weights.B)

	// expected output
	y := xmath.Vec(3).With(0.25, 0.5, 0.25)

	// do a forward pass ...
	x := xmath.Vec(2).With(0.9, 0.1)
	ty := neuron.Fwd(x)
	// all dimensions of the output should be the same
	for i := 1; i < len(ty); i++ {
		assert.Equal(t, ty[0], ty[i])
	}

	// find the error
	errEpoch1, lossEpoch1 := trainErr(y, ty)
	// do a backwards pass
	neuron.Bwd(errEpoch1)

	// second pass to check the new weights and performance
	ty = neuron.Fwd(x)
	weights = neuron.Weights()
	// left and right most elements should be equal, because we enforced symmetry on the expected output
	assert.Equal(t, weights.W[0], weights.W[2])
	assert.Equal(t, weights.B[0], weights.B[2])
	assert.Equal(t, ty[0], ty[2])

	// middle element should be larger
	assert.True(t, weights.W[1].Sum() > weights.W[0].Sum())
	assert.True(t, weights.B[1] > weights.B[0])
	assert.True(t, ty[1] > ty[0])

	// train for a econd epoch
	_, lossEpoch2 := trainErr(y, ty)
	// loss should be smaller e.g. we learnt something
	assert.True(t, lossEpoch1 > lossEpoch2)
}

func trainErr(expected, actual xmath.Vector) (xmath.Vector, float64) {
	err := ml.Diff(expected, actual)
	println(fmt.Sprintf("err = %v", err))
	loss := err.Op(math.Abs).Sum()
	println(fmt.Sprintf("loss = %v", loss))
	return err, loss
}

func neuronWeights(neuron Neuron) Weights {
	weights := neuron.Weights()
	println(fmt.Sprintf("weights.W = %v", weights.W))
	println(fmt.Sprintf("weights.B = %v", weights.B))
	return *weights
}

// TestActivationNeuron makes sure the neuron can learn and adjust itself to the input/output pairs.
// It uses a vanilla feed forward neuron
func TestActivationNeuron(t *testing.T) {

	activation := ml.Sigmoid

	type test struct {
		rate      *ml.Learning
		inp       []float64
		outp      []float64
		threshold int
	}

	tests := map[string]test{
		"learn-x": {
			rate:      ml.Learn(0.5, 0.5),
			inp:       []float64{0.5, 0.5},
			outp:      []float64{0.1, 0.9},
			threshold: 300,
		},
		"learn-y": {
			rate:      ml.Learn(0.5, 0.5),
			inp:       []float64{0.5, 0.5},
			outp:      []float64{0.9, 0.1},
			threshold: 300,
		},
		"learn-inv": {
			rate:      ml.Learn(0.5, 0.5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 300,
		},
		"learn-const": {
			rate:      ml.Learn(0.5, 0.5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.1, 0.9},
			threshold: 300,
		},
		"low-rate": {
			rate:      ml.Learn(0.005, 0.005),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 25000,
		},
		"high-rate": {
			rate:      ml.Learn(5, 5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 40,
		},
		"dim-diff-y-high": {
			rate:      ml.Learn(5, 5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{1},
			threshold: 500,
		},
		"dim-diff-y-low": {
			rate:      ml.Learn(5, 5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.01},
			threshold: 500,
		},
		"dim-diff-x-high": {
			rate:      ml.Learn(5, 0),
			inp:       []float64{1},
			outp:      []float64{0.9, 0.1},
			threshold: 500,
		},
		"dim-diff-x-low": {
			rate:      ml.Learn(5, 1),
			inp:       []float64{0.01},
			outp:      []float64{0.9, 0.1},
			threshold: 300,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {

			model := ml.Base().
				WithActivation(activation).
				WithRate(tt.rate)
			neuron := NewActivationCell(len(tt.inp), len(tt.outp), *model, NewWeights(len(tt.inp), len(tt.outp), xmath.Const(0.5), xmath.Const(0.5)), Meta{})
			runTest(t, neuron, tt.inp, tt.outp, tt.threshold)
		})
	}
}

func TestWeightCell_FwdAndBwd(t *testing.T) {

	xDim := 2
	yDim := 3

	factory := NewBuilder().
		WithWeights(xmath.Const(0.5), xmath.Const(0.5)).
		WithModule(ml.Base().WithRate(ml.Learn(1, 1))).
		Factory(NewWeightCell)

	neuron := factory(xDim, yDim, Meta{})

	weights := neuronWeights(neuron)
	vec := xmath.Vec(xDim).With(0.5, 0.5)
	// weights should be a 2x3 matrix
	assert.Equal(t, xmath.Mat(yDim).With(vec, vec, vec), weights.W)
	// bias should be same as the output dimension
	assert.Equal(t, xmath.Vec(yDim).With(0.5, 0.5, 0.5), weights.B)

	// expected output
	y := xmath.Vec(3).With(0.25, 0.5, 0.25)

	// do a forward pass ...
	x := xmath.Vec(2).With(0.9, 0.3)
	ty := neuron.Fwd(x)
	println(fmt.Sprintf("ty = %v", ty))
	// all dimensions of the output should be the same
	for i := 1; i < len(ty); i++ {
		assert.Equal(t, ty[0], ty[i])
	}

	// find the error
	errEpoch1, lossEpoch1 := trainErr(y, ty)
	// do a backwards pass
	neuron.Bwd(errEpoch1)

	// second pass to check the new weights and performance
	ty = neuron.Fwd(x)
	println(fmt.Sprintf("ty = %v", ty))
	weights = neuronWeights(neuron)
	// left and right most elements should be equal, because we enforced symmetry on the expected output
	assert.Equal(t, weights.W[0], weights.W[2])
	assert.Equal(t, ty[0], ty[2])

	// middle element should be larger
	assert.True(t, weights.W[1].Sum() > weights.W[0].Sum())
	assert.True(t, ty[1] > ty[0])

	// train for a econd epoch
	_, lossEpoch2 := trainErr(y, ty)
	// loss should be smaller e.g. we learnt something
	assert.True(t, lossEpoch1 > lossEpoch2)

}

// TestWeightNeuron makes sure the neuron can learn and adjust itself to the input/output pairs.
// It uses a vanilla matrix multiplication neuron cell
func TestWeightNeuron(t *testing.T) {

	type test struct {
		rate      *ml.Learning
		inp       []float64
		outp      []float64
		threshold int
	}

	tests := map[string]test{
		"learn-x": {
			rate:      ml.Rate(0.05),
			inp:       []float64{0.5, 0.5},
			outp:      []float64{0.1, 0.9},
			threshold: 150,
		},
		"learn-y": {
			rate:      ml.Rate(0.5),
			inp:       []float64{0.5, 0.5},
			outp:      []float64{0.9, 0.1},
			threshold: 150,
		},
		"learn-inv": {
			rate:      ml.Rate(0.5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 10,
		},
		"learn-const": {
			rate:      ml.Rate(0.5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.1, 0.9},
			threshold: 10,
		},
		"low-rate": {
			rate:      ml.Rate(0.005),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 1000,
		},
		"high-rate": {
			rate:      ml.Rate(2),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 10,
		},
		"dim-diff-y-high": {
			rate:      ml.Rate(1),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{1},
			threshold: 10,
		},
		"dim-diff-y-low": {
			rate:      ml.Rate(1),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.01},
			threshold: 10,
		},
		"dim-diff-x-high": {
			rate:      ml.Rate(1),
			inp:       []float64{1},
			outp:      []float64{0.9, 0.1},
			threshold: 10,
		},
		"dim-diff-x-low": {
			rate:      ml.Rate(2),
			inp:       []float64{0.01},
			outp:      []float64{0.9, 0.1},
			threshold: 20000,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {

			model := ml.Base().
				WithRate(tt.rate)
			neuron := NewWeightCell(len(tt.inp), len(tt.outp), *model, NewWeights(len(tt.inp), len(tt.outp), xmath.Const(0.5), xmath.VoidVector), Meta{})
			runTest(t, neuron, tt.inp, tt.outp, tt.threshold)
		})
	}
}

// TestSoftNeuron makes sure the neuron can learn and adjust itself to the input/output pairs.
// It uses a soft max based neuron cell
func TestSoftNeuron(t *testing.T) {

	type test struct {
		rate      *ml.Learning
		inp       []float64
		outp      []float64
		threshold int
	}

	tests := map[string]test{
		"learn-x": {
			rate:      ml.Rate(0.05),
			inp:       []float64{0.5, 0.5},
			outp:      []float64{0.1, 0.9},
			threshold: 200,
		},
		"learn-y": {
			rate:      ml.Rate(0.5),
			inp:       []float64{0.5, 0.5},
			outp:      []float64{0.9, 0.1},
			threshold: 150,
		},
		"learn-inv": {
			rate:      ml.Rate(0.5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 20,
		},
		"learn-const": {
			rate:      ml.Rate(0.5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.1, 0.9},
			threshold: 20,
		},
		"low-rate": {
			rate:      ml.Rate(0.005),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 2000,
		},
		"high-rate": {
			rate:      ml.Rate(2),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 10,
		},
		"dim-diff-y-high": {
			rate:      ml.Rate(1),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{1},
			threshold: 10,
		},
		// This would not work, because soft activation on a 1-d output will always produce 1
		//"dim-diff-y-low": {
		//	rate:      ml.Learn(0.01, 0.1),
		//	inp:       []float64{0.1, 0.9},
		//	outp:      []float64{0.01},
		//	threshold: 20,
		//},
		"dim-diff-x-high": {
			rate:      ml.Rate(1),
			inp:       []float64{1},
			outp:      []float64{0.9, 0.1},
			threshold: 10,
		},
		"dim-diff-x-low": {
			rate:      ml.Rate(2),
			inp:       []float64{0.01},
			outp:      []float64{0.9, 0.1},
			threshold: 20000,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			neuron := NewSoftCell(len(tt.inp), len(tt.outp), Meta{})
			runTest(t, neuron, tt.inp, tt.outp, tt.threshold)
		})
	}
}

func runTest(t *testing.T, neuron Neuron, inp, outp xmath.Vector, threshold int) {
	var diff xmath.Vector
	for i := 0; i < threshold; i++ {
		out := neuron.Fwd(inp)
		println(fmt.Sprintf("out = %v", out))
		expOutp := xmath.Vec(len(outp)).With(outp...)
		println(fmt.Sprintf("expOutp = %v", expOutp))
		newDiff := expOutp.Diff(out)
		// compare previous error to current ... it should be decreasing ALWAYS
		// ... for pre-defined networks as in this case in the tests
		sum := diff.Op(math.Abs).Sum()
		newSum := newDiff.Op(math.Abs).Sum()
		score := sum - newSum
		if i > 0 {
			// assert new error is either smaller or bigger for a small amount
			if !assert.True(t, score >= -0.1, fmt.Sprintf("Learning score should be always improving [ prev = %v , current = %v -> %v]", diff.Op(math.Abs).Sum(), newDiff.Op(math.Abs).Sum(), score)) {
				return
			}
		}
		diff = newDiff
		// make the backwards pass now to learn from the forward propagation
		neuron.Bwd(diff)
		println(fmt.Sprintf("loss = %v , diff = %v", diff.Op(math.Abs).Sum(), diff))
	}
	println(fmt.Sprintf("loss = %v , diff = %v", diff.Op(math.Abs).Sum(), diff))
	if !assert.True(t, diff.Op(math.Abs).Sum() < 0.02) {
		return
	}

	// do a second pass to make sure we dont cause exploding gradients
	for i := 0; i < threshold; i++ {
		out := neuron.Fwd(inp)
		expOutp := xmath.Vec(len(outp)).With(outp...)
		newDiff := expOutp.Diff(out)
		diff = newDiff
		neuron.Bwd(diff)
		println(fmt.Sprintf("loss = %v , diff = %v", diff.Op(math.Abs).Sum(), diff))
	}
	println(fmt.Sprintf("loss = %v , diff = %v", diff.Op(math.Abs).Sum(), diff))
	assert.True(t, diff.Op(math.Abs).Sum() < 0.01)
}
