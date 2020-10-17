package main

import (
	"math"

	"github.com/drakos74/go-ex-machina/xmachina/net/rnn"

	"github.com/rs/zerolog"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/drakos74/oremi/graph"
)

const (
	sin   = "sin"
	train = "train"
)

func init() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
}

func main() {
	network := rnn.New(25, 1, 100).
		Rate(0.01).
		Activation(ml.Sigmoid).
		Loss(ml.CompLoss(ml.Pow)).
		//Init(xmath.RangeSqrt(-1, 1))
		Init(xmath.Range(0, 1))

	f := 0.025

	rnn := graph.New("RNN")

	rnn.NewSeries(sin, "x", "y")
	rnn.NewSeries(train, "x", "y")

	xx := make([]float64, 0)
	l := 5000
	for i := 0; i < l; i++ {

		x := f * float64(i)
		y := 10 * math.Sin(x)

		if i < l*4/5 {
			rnn.Add(sin, x, y)
			network.Train(xmath.Vec(1).With(y))
			rnn.Add(train, x, network.TmpOutput[len(network.TmpOutput)-1])
			xx = append(xx, x)
		} else {
			output := network.Predict(xmath.Vec(1).With(y))
			rnn.Add(train, x, output[0])
		}

	}

	// draw the data collection
	graph.Draw("sin", rnn)

}
