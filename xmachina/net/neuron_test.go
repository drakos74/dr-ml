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
	zerolog.SetGlobalLevel(zerolog.TraceLevel)
}

// TestFFNeuron makes sure the neuron can learn and adjust itself to the input/output pairs.
// It uses a vanilla feed forward neuron
func TestFFNeuron(t *testing.T) {

	activation := ml.Sigmoid

	type test struct {
		rate      ml.Learning
		inp       []float64
		outp      []float64
		threshold int
	}

	tests := map[string]test{
		"learn-x": {
			rate:      ml.Learn(0.5),
			inp:       []float64{0.5, 0.5},
			outp:      []float64{0.1, 0.9},
			threshold: 10,
		},
		"learn-y": {
			rate:      ml.Learn(0.5),
			inp:       []float64{0.5, 0.5},
			outp:      []float64{0.9, 0.1},
			threshold: 10,
		},
		"learn-inv": {
			rate:      ml.Learn(0.5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 80,
		},
		"learn-const": {
			rate:      ml.Learn(0.5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.1, 0.9},
			threshold: 80,
		},
		"low-rate": {
			rate:      ml.Learn(0.005),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 6000,
		},
		"high-rate": {
			rate:      ml.Learn(5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{0.9, 0.1},
			threshold: 6,
		},
		"dim-diff-y": {
			rate:      ml.Learn(5),
			inp:       []float64{0.1, 0.9},
			outp:      []float64{1},
			threshold: 20,
		},
		"dim-diff-x": {
			rate:      ml.Learn(5),
			inp:       []float64{1},
			outp:      []float64{0.9, 0.1},
			threshold: 20,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {

			model := ml.Model().
				WithActivation(activation).
				WithRate(tt.rate)
			neuron := NewMLNeuron(len(tt.inp), len(tt.outp), model, xmath.Const(0.5), Meta{})

			var diff xmath.Vector
			for i := 0; i < tt.threshold; i++ {
				outp := neuron.Fwd(tt.inp)
				expOutp := xmath.Vec(len(tt.outp)).With(tt.outp...)
				newDiff := neuron.Bwd(expOutp.Diff(outp))

				score := diff.Op(math.Abs).Sum() - newDiff.Op(math.Abs).Sum()
				if i > 0 {
					assert.True(t, score >= 0 || score > -0.1, fmt.Sprintf("Learning score should be always improving [ prev = %v , current = %v ]", diff.Op(math.Abs).Sum(), newDiff.Op(math.Abs).Sum()))
				}
				diff = newDiff
			}
			println(fmt.Sprintf("loss = %v , diff = %v", diff.Op(math.Abs).Sum(), diff))
			assert.True(t, diff.Op(math.Abs).Sum() < 0.02)
		})
	}

}
