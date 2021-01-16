package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmachina/net"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

func Test_RNetworkSineFunc(t *testing.T) {

	builder := NewNeuronBuilder(1, 1, 30).
		WithRate(*ml.Rate(0.05)).
		WithWeights(xmath.RangeSqrt(-1, 1)(30), xmath.RangeSqrt(-1, 1)(30)).
		WithActivation(ml.TanH, ml.Sigmoid)

	network := New(20, builder, net.Clip{
		W: 0.5,
		B: 0.5,
	})
	println(fmt.Sprintf("network = %v", network))
	f := 0.025

	var err xmath.Vector
	for i := 0; i < 1000; i++ {

		x := f * float64(i)

		s := math.Sin(x)
		output := xmath.Vec(1)
		err, _ = network.Train(xmath.Vec(1).With(s), output)
		println(fmt.Sprintf("err = %v", err.Op(math.Abs).Sum()))
	}

	println(fmt.Sprintf("err = %v", err.Op(math.Abs).Sum()))

}
