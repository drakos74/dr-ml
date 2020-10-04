package rnn

import (
	"fmt"
	"math"
	"testing"

	"github.com/drakos74/go-ex-machina/xmachina/net"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

func TestRNeuron_DimensionsInForwardPass(t *testing.T) {

	rNeuron := RNeuron(ml.TanH)(3, 5, net.Meta{})

	// trainInput layer size : 3
	// trainOutput layer size : 2
	// layer depth : 5

	xt := xmath.Mat(10).Rows(3, xmath.Rand(-1, 1, math.Sqrt))
	h0 := xmath.Mat(10).Rows(5, xmath.Rand(-1, 1, math.Sqrt))

	params := &Parameters{
		Weights: Weights{
			Whh: xmath.Mat(5).Rows(5, xmath.Rand(-1, 1, math.Sqrt)),
			Wxh: xmath.Mat(5).Rows(3, xmath.Rand(-1, 1, math.Sqrt)),
			Why: xmath.Mat(2).Rows(5, xmath.Rand(-1, 1, math.Sqrt)),
			Bh:  xmath.Rand(-1, 1, math.Sqrt)(5, 0),
			By:  xmath.Rand(-1, 1, math.Sqrt)(2, 0),
		},
	}

	y := xmath.Mat(10)
	wh := xmath.Mat(10)

	for i := 0; i < len(xt); i++ {
		y[i], wh[i] = rNeuron.forward(xt[i], h0[i], &params.Weights)
	}

	println(fmt.Sprintf("y = %v", y.T()))
	println(fmt.Sprintf("wh = %v", wh.T()))

}

// TODO : fix this test
func TestRNeuron_Train(t *testing.T) {

	// normal increasing series
	x := xmath.Mat(10).Of(1).
		With(
			xmath.Vec(1).With(1),
			xmath.Vec(1).With(2),
			xmath.Vec(1).With(3),
			xmath.Vec(1).With(4),
			xmath.Vec(1).With(5),
			xmath.Vec(1).With(6),
			xmath.Vec(1).With(7),
			xmath.Vec(1).With(8),
			xmath.Vec(1).With(9),
			xmath.Vec(1).With(10),
		)

	e := xmath.Mat(10).Of(1).
		With(
			xmath.Vec(1).With(2),
			xmath.Vec(1).With(3),
			xmath.Vec(1).With(4),
			xmath.Vec(1).With(5),
			xmath.Vec(1).With(6),
			xmath.Vec(1).With(7),
			xmath.Vec(1).With(8),
			xmath.Vec(1).With(9),
			xmath.Vec(1).With(10),
			xmath.Vec(1).With(11),
		)

	rneuron := RNeuron(ml.TanH)(1, 5, net.Meta{})

	params := &Parameters{
		Weights: Weights{
			Whh: xmath.Mat(5).Rows(5, xmath.Rand(-1, 1, math.Sqrt)), // interlayer
			Wxh: xmath.Mat(5).Rows(1, xmath.Rand(-1, 1, math.Sqrt)), // trainInput
			Why: xmath.Mat(1).Rows(5, xmath.Rand(-1, 1, math.Sqrt)), // trainOutput
			Bh:  xmath.Rand(-0.1, 0.1, math.Sqrt)(5, 0),
			By:  xmath.Rand(-0.1, 0.1, math.Sqrt)(1, 0),
		},
	}

	h := xmath.Vec(5)

	o := xmath.Mat(10)
	wh := xmath.Mat(10)

	err := xmath.Mat(10)

	for i := 0; i < len(x); i++ {
		o[i], h = rneuron.forward(x[i], h, &params.Weights)
		wh[i] = h
		err[i] = ml.CrossEntropy(e[i], o[i])
	}

	// do the backward pass
	for i := len(x) - 1; i >= 0; i-- {
		//rneuron.backward(err[i])
	}

	println(fmt.Sprintf("o = %v", o.T()))
	println(fmt.Sprintf("wh = %v", wh.T()))

}
