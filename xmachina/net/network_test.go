package net

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/drakos74/go-ex-machina/xmachina/math"
	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

func TestNetwork_Train_NoActivation(t *testing.T) {

	n := New(2, 1).
		Add(2,
			Perceptron(ml.Model().
				Rate(0.05, 0.05).
				WithActivation(ml.Void{}),
				math.Def(
					math.Vec(2).With(0.11, 0.21),
					math.Vec(2).With(0.12, 0.08),
				))).
		Add(1,
			Perceptron(ml.Model().
				Rate(0.05, 0.05).
				WithActivation(ml.Void{}),
				math.Def(
					math.Vec(2).With(0.14, 0.15),
				)))
	//AddSoftMax().
	n.Trace()

	assertTrain(t, n,
		math.Vec(2).With(2, 3),
		math.Vec(1).With(1),
		// err : 0.3272
		[]string{"0.3272"},
		// neuron : [0,0] -> [0.12,0.23]
		// neuron : [0,1] -> [0.13,0.10]
		// neuron : [1,0] -> [0.17,0.17]
		[][][]string{
			{
				{"0.12", "0.23"},
				{"0.13", "0.10"},
			},
			{
				{"0.17", "0.17"},
			},
		},
	)

}

// same as above , just with a parallelizable network
func TestXNetwork_Train_NoActivation(t *testing.T) {

	n := XNew(2, 1).
		Add(2,
			Perceptron(ml.Model().
				Rate(0.05, 0.05).
				WithActivation(ml.Void{}),
				math.Def(
					math.Vec(2).With(0.11, 0.21),
					math.Vec(2).With(0.12, 0.08),
				))).
		Add(1,
			Perceptron(ml.Model().
				Rate(0.05, 0.05).
				WithActivation(ml.Void{}),
				math.Def(
					math.Vec(2).With(0.14, 0.15),
				)))
	//AddSoftMax().
	n.Trace()

	assertTrain(t, n,
		math.Vec(2).With(2, 3),
		math.Vec(1).With(1),
		// err : 0.3272
		[]string{"0.3272"},
		// neuron : [0,0] -> [0.12,0.23]
		// neuron : [0,1] -> [0.13,0.10]
		// neuron : [1,0] -> [0.17,0.17]
		[][][]string{
			{
				{"0.12", "0.23"},
				{"0.13", "0.10"},
			},
			{
				{"0.17", "0.17"},
			},
		},
	)

}

func assertTrain(t *testing.T, n NN, inp, out math.Vector, expErr []string, expWeights [][][]string) {

	err, weights := n.Train(inp, out)

	for i := range err {
		assert.Equal(t, expErr[i], strconv.FormatFloat(err[i], 'f', 4, 64))
	}

	for i, ww := range weights {
		for j, w := range ww {
			exp := expWeights[i][j]
			for k := range w {
				assert.Equal(t, exp[k], strconv.FormatFloat(w[k], 'f', 2, 64))
			}
		}

	}
}
