package main

import (
	"fmt"
	"math"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/drakos74/oremi/graph"
)

const (
	sin   = "sin"
	train = "train"
)

func main() {
	network := net.NewRNetwork(1, 10, 100, 0.001)
	network.Loss(ml.Pow)

	f := 0.1

	rnn := graph.New("RNN")

	rnn.NewSeries(sin, "x", "y")
	rnn.NewSeries(train, "x", "y")

	xx := make([]float64, 0)
	for i := 0; i < 1000; i++ {

		x := f * float64(i)
		y := math.Sin(x)
		rnn.Add(sin, []string{"x", fmt.Sprintf("sin(%.2f)", x)}, x, y)
		network.Train(xmath.Vec(1).With(x), xmath.Vec(1).With(y))
		rnn.Add(train, []string{"x", fmt.Sprintf("train(%.2f)", x)}, x, network.TmpOutput[0])
		xx = append(xx, x)
	}

	// draw the data collection
	graph.Draw("sin", rnn)

}
