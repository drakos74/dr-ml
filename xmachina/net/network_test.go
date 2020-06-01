package net

import (
	"fmt"
	"math"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

func TestNetwork_Train_NoActivation(t *testing.T) {

	n := New(2, 1).
		Add(2,
			Perceptron(ml.Model().
				Rate(0.05, 0.05).
				WithActivation(ml.Void{}),
				xmath.Def(
					xmath.Vec(2).With(0.11, 0.21),
					xmath.Vec(2).With(0.12, 0.08),
				))).
		Add(1,
			Perceptron(ml.Model().
				Rate(0.05, 0.05).
				WithActivation(ml.Void{}),
				xmath.Def(
					xmath.Vec(2).With(0.14, 0.15),
				)))
	//AddSoftMax().
	n.Trace()

	assertTrain(t, n,
		xmath.Vec(2).With(2, 3),
		xmath.Vec(1).With(1),
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
				xmath.Def(
					xmath.Vec(2).With(0.11, 0.21),
					xmath.Vec(2).With(0.12, 0.08),
				))).
		Add(1,
			Perceptron(ml.Model().
				Rate(0.05, 0.05).
				WithActivation(ml.Void{}),
				xmath.Def(
					xmath.Vec(2).With(0.14, 0.15),
				)))
	//AddSoftMax().
	n.Trace()

	assertTrain(t, n,
		xmath.Vec(2).With(2, 3),
		xmath.Vec(1).With(1),
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

func assertTrain(t *testing.T, n NN, inp, out xmath.Vector, expErr []string, expWeights [][][]string) {

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

func Test_RNetworkSineFunc(t *testing.T) {

	network := NewRNetwork(1, 10, 100, 0.01)
	network.Loss(ml.Pow)

	f := 0.1

	for i := 0; i < 100; i++ {

		x := f * float64(i)

		err, _ := network.Train(xmath.Vec(1).With(x), xmath.Vec(1).With(math.Sin(x)))
		println(fmt.Sprintf("err = %v", err.Sum()))

	}

}
