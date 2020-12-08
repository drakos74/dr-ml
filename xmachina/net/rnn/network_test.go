package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

func Test_RNetworkSineFunc(t *testing.T) {

	network := New(10, 1, 100).
		Rate(0.001).
		Activation(ml.TanH).
		Loss(ml.CompLoss(ml.CrossEntropy)).
		InitWeights(xmath.RangeSqrt(0, 1))

	f := 0.1

	for i := 0; i < 100; i++ {

		x := f * float64(i)

		err, _ := network.Train(xmath.Vec(1).With(math.Sin(x)))
		println(fmt.Sprintf("err = %v", err.Sum()))

	}

	next := network.Evolve(50)
	println(fmt.Sprintf("next = %v", next))

}
