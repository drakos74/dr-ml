package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath/algebra"

	"github.com/drakos74/go-ex-machina/xmachina/net/rc"

	"github.com/drakos74/go-ex-machina/xmachina/net"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

func Test_RNNetworkSineFunc(t *testing.T) {

	builder := rc.NewNeuronBuilder(1, 1, 30).
		WithRate(*ml.Rate(0.05)).
		WithWeights(algebra.RangeSqrt(-1, 1)(30), algebra.RangeSqrt(-1, 1)(30)).
		WithActivation(ml.TanH, ml.Sigmoid)

	network := rc.New(20, New(*builder), net.NewClip(0.5, 0.5))
	println(fmt.Sprintf("network = %v", network))
	f := 0.025

	var err algebra.Vector
	for i := 0; i < 1000; i++ {

		x := f * float64(i)

		s := math.Sin(x)
		output := algebra.Vec(1)
		err, _ = network.Train(algebra.Vec(1).With(s), output)
		println(fmt.Sprintf("err = %v", err.Op(math.Abs).Sum()))
	}

	println(fmt.Sprintf("err = %v", err.Op(math.Abs).Sum()))

}
