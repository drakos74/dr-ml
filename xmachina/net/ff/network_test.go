package ff

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/rs/zerolog"

	"github.com/drakos74/go-ex-machina/xmachina/net"

	"github.com/stretchr/testify/assert"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

func init() {
	zerolog.SetGlobalLevel(zerolog.TraceLevel)
}

func TestNetwork_Train_NoActivation(t *testing.T) {

	n := New(2, 1).
		Add(2,
			net.NewBuilder().
				WithModule(ml.Base().
					WithRate(ml.Learn(0.05, 0.05)).
					WithActivation(ml.Void{})).
				WithWeights(xmath.Row(
					xmath.Vec(2).With(0.11, 0.21),
					xmath.Vec(2).With(0.12, 0.08),
				), xmath.Row(
					xmath.Vec(2).With(0.11, 0.21),
					xmath.Vec(2).With(0.12, 0.08),
				)).
				Factory(net.NewActivationCell)).
		Add(1,
			net.NewBuilder().
				WithModule(ml.Base().
					WithRate(ml.Learn(0.05, 0.05)).
					WithActivation(ml.Void{})).
				WithWeights(xmath.Row(
					xmath.Vec(2).With(0.14, 0.15),
				), xmath.Row(
					xmath.Vec(1).With(0.15),
				)).
				Factory(net.NewActivationCell))
	//AddSoftMax().
	n.Trace()

	assertTrain(t, n,
		xmath.Vec(2).With(2, 3),
		xmath.Vec(1).With(1),
		// err : 0.3272
		[]string{"0.6121"},
		// neuron : [0,0] -> [0.12,0.23]
		// neuron : [0,1] -> [0.13,0.10]
		// neuron : [1,0] -> [0.17,0.17]
		[]net.Weights{
			{
				W: xmath.Mat(2).With(
					xmath.Vec(2).With(0.12, 0.22),
					xmath.Vec(2).With(0.13, 0.09),
				),
				B: xmath.Vec(2).With(0.17, 0.17),
			},
		},
	)

}

// same as above , just with a parallelizable network
func TestXNetwork_Train_NoActivation(t *testing.T) {

	n := XNew(2, 1).
		Add(2,
			Perceptron(ml.Base().
				WithRate(ml.Learn(0.05, 0.05)).
				WithActivation(ml.Void{}),
				xmath.Row(
					xmath.Vec(2).With(0.11, 0.21),
					xmath.Vec(2).With(0.12, 0.08),
				))).
		Add(1,
			Perceptron(ml.Base().
				WithRate(ml.Learn(0.05, 0.05)).
				WithActivation(ml.Void{}),
				xmath.Row(
					xmath.Vec(2).With(0.14, 0.15),
				)))
	//AddSoftMax().
	n.Trace()

	assertTrain(
		t,
		n,
		xmath.Vec(2).With(2, 3),
		xmath.Vec(1).With(1),
		// err : 0.3272
		[]string{"0.8090"},
		// neuron : [0,0] -> [0.12,0.23]
		// neuron : [0,1] -> [0.13,0.10]
		// neuron : [1,0] -> [0.17,0.17]
		[]net.Weights{
			{
				W: xmath.Mat(2).With(
					xmath.Vec(2).With(0.12, 0.23),
					xmath.Vec(2).With(0.13, 0.10),
				),
				B: xmath.Vec(2).With(0.17, 0.17),
			},
		},
	)

}

func assertTrain(t *testing.T, n net.NN, inp, out xmath.Vector, expErr []string, expWeights []net.Weights) {

	err, weights := n.Train(inp, out)

	for i := range err {
		assert.Equal(t, expErr[i], strconv.FormatFloat(err[i], 'f', 4, 64))
	}

	println(fmt.Sprintf("weights = %v", weights))

	for i, ww := range weights {
		assert.Equal(t, expWeights[i].W, ww.W.Op(xmath.Round(2)))
		assert.Equal(t, expWeights[i].B, ww.B.Op(xmath.Round(2)))
	}
}
