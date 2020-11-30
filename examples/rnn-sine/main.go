package main

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"os"

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

	weights, err := load()

	network := rnn.New(25, 1, 100).
		Rate(0.01).
		Activation(ml.Sigmoid).
		Loss(ml.CompLoss(ml.Pow)).
		InitWeights(xmath.RangeSqrt(-1, 1))
	//InitWeights(xmath.Range(0, 1))

	if err == nil && weights != nil {
		println("init with weights")
		network = rnn.New(25, 1, 100).
			Rate(0.01).
			Activation(ml.Sigmoid).
			Loss(ml.CompLoss(ml.Pow)).
			//InitWeights(xmath.RangeSqrt(-1, 1))
			WithWeights(*weights)
	}

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
			_, weights := network.Train(xmath.Vec(1).With(y))
			rnn.Add(train, x, network.TmpOutput[len(network.TmpOutput)-1])
			xx = append(xx, x)
			save(weights)
		} else {
			output := network.Predict(xmath.Vec(1).With(y))
			rnn.Add(train, x, output[0])
		}

	}

	// draw the data collection
	graph.Draw("sin", rnn)

}

const fileName = "examples/rnn-sine/data/results.json"

func save(weights rnn.Weights) {

	f, err := os.Create(fileName)

	if err != nil {
		panic(err.Error())
	}

	defer f.Close()

	b, err := json.Marshal(weights)
	if err != nil {
		panic(err.Error())
	}

	_, err = f.Write(b)

	if err != nil {
		panic(err.Error())
	}

}

func load() (*rnn.Weights, error) {

	data, err := ioutil.ReadFile(fileName)
	if err != nil {
		return nil, err
	}

	var weights rnn.Weights
	err = json.Unmarshal(data, &weights)
	if err != nil {
		return nil, err
	}

	return &weights, nil
}
