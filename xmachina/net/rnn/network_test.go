package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

func Test_RNetworkSineFunc(t *testing.T) {

	network := New(10, 1, 100, 0.01)
	network.Loss(ml.CE)

	f := 0.1

	for i := 0; i < 100; i++ {

		x := f * float64(i)

		err, _ := network.Train(xmath.Vec(1).With(math.Sin(x)))
		println(fmt.Sprintf("err = %v", err.Sum()))

	}

}
