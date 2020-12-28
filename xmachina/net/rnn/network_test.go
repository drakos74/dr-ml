package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

func Test_RNetworkSineFunc(t *testing.T) {

	builder := NewNeuronBuilder(1, 1, 10).
		WithRate(*ml.Rate(0.5)).
		WithWeights(xmath.RangeSqrt(-1, 1)(10), xmath.RangeSqrt(-1, 1)(10)).
		WithActivation(ml.TanH, ml.Sigmoid)

	network := New(10, builder, Clip{
		W: 1,
		B: 1,
	})
	println(fmt.Sprintf("network = %v", network))
	f := 0.5

	var err xmath.Vector
	for i := 0; i < 1000; i++ {

		x := f * float64(i)

		s := math.Sin(x)
		err, _ = network.Train(xmath.Vec(1).With(s))
		println(fmt.Sprintf("err = %v", err.Op(math.Abs).Sum()))
	}

	println(fmt.Sprintf("err = %v", err.Op(math.Abs).Sum()))

}
