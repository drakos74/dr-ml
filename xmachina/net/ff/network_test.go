package ff

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath/algebra"

	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/stretchr/testify/assert"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

func TestNetwork_Train_NoActivation(t *testing.T) {

	n := New(2, 1).
		Add(2,
			net.NewBuilder().
				WithModule(ml.Base().
					WithRate(ml.Learn(0.05, 0.05)).
					WithActivation(ml.Void{})).
				WithWeights(algebra.Row(
					algebra.Vec(2).With(0.11, 0.21),
					algebra.Vec(2).With(0.12, 0.08),
				), algebra.Row(algebra.Vec(2).With(0.11, 0.21))).
				Factory(net.NewActivationCell)).
		Add(1,
			net.NewBuilder().
				WithModule(ml.Base().
					WithRate(ml.Learn(0.05, 0.05)).
					WithActivation(ml.Void{})).
				WithWeights(
					algebra.Row(algebra.Vec(2).With(0.14, 0.15)),
					algebra.Row(algebra.Vec(1).With(0.15))).
				Factory(net.NewActivationCell))
	n.Trace()

	assertTrain(t, n,
		algebra.Vec(2).With(2, 3),
		algebra.Vec(1).With(1),
		// err : 0.3272
		[]string{"0.6121"},
		// neuron : [0,0] -> [0.12,0.23]
		// neuron : [0,1] -> [0.13,0.10]
		// neuron : [1,0] -> [0.17,0.17]
		map[net.Meta]net.Weights{
			net.Meta{}: {
				W: algebra.Mat(2).With(
					algebra.Vec(2).With(0.11, 0.21),
					algebra.Vec(2).With(0.12, 0.08),
				),
				B: algebra.Vec(2).With(0.11, 0.21),
			},
			net.Meta{Layer: 1}: {
				W: algebra.Mat(1).With(algebra.Vec(2).With(0.15, 0.16)),
				B: algebra.Vec(1).With(0.16),
			},
		},
	)

}

// TODO : fix the xNetwork
// same as above , just with a parallelizable network
func TestXNetwork_Train_NoActivation(t *testing.T) {

	n := XNew(2, 1).
		Add(2,
			Perceptron(ml.Base().
				WithRate(ml.Learn(0.05, 0.05)).
				WithActivation(ml.Void{}),
				algebra.Row(
					algebra.Vec(2).With(0.11, 0.21),
					algebra.Vec(2).With(0.12, 0.08),
				))).
		Add(1,
			Perceptron(ml.Base().
				WithRate(ml.Learn(0.05, 0.05)).
				WithActivation(ml.Void{}),
				algebra.Row(algebra.Vec(2).With(0.14, 0.15))))
	n.Trace()

	assertTrain(
		t,
		n,
		algebra.Vec(2).With(2, 3),
		algebra.Vec(1).With(1),
		// err : 0.3272
		[]string{"0.8090"},
		// neuron : [0,0] -> [0.12,0.23]
		// neuron : [0,1] -> [0.13,0.10]
		// neuron : [1,0] -> [0.17,0.17]
		map[net.Meta]net.Weights{
			net.Meta{}: {
				W: algebra.Mat(2).With(
					algebra.Vec(2).With(0.12, 0.23),
					algebra.Vec(2).With(0.13, 0.10),
				),
				B: algebra.Vec(2).With(0.17, 0.17),
			},
			net.Meta{Layer: 1}: {
				W: algebra.Mat(1).With(algebra.Vec(2).With(0.15, 0.15)),
				B: algebra.Vec(1).With(0.02),
			},
		},
	)

}

func assertTrain(t *testing.T, n net.NN, inp, out algebra.Vector, expErr []string, expWeights map[net.Meta]net.Weights) {

	err, weights := n.Train(inp, out)

	for i := range err {
		assert.Equal(t, expErr[i], strconv.FormatFloat(err[i], 'f', 4, 64))
	}

	println(fmt.Sprintf("weights = %v", weights))

	println(fmt.Sprintf("expWeights = %v", expWeights))
	for i, ww := range weights {
		assert.Equal(t, expWeights[i].W, ww.W.Op(algebra.Round(2)), fmt.Sprintf("%+v", i))
		assert.Equal(t, expWeights[i].B, ww.B.Op(algebra.Round(2)), fmt.Sprintf("%+v", i))
	}
}
