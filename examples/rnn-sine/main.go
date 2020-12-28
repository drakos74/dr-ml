package main

import (
	"fmt"
	"math"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmachina/net/rnn"

	"github.com/rs/zerolog"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
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

	//weights, err := load()

	//weights = nil

	builder := rnn.NewNeuronBuilder(1, 1, 10).
		WithRate(*ml.Rate(1)).
		WithWeights(xmath.RangeSqrt(-1, 1)(10), xmath.RangeSqrt(-1, 1)(10)).
		WithActivation(ml.Sigmoid, ml.Sigmoid)

	network := rnn.New(10, builder, rnn.Clip{0.5, 0.5})
	//InitWeights(xmath.Range(0, 1))

	//if err == nil && weights != nil {
	//	println("init with weights")
	//	network = rnn.New(25, 1, 100).
	//		Rate(0.01).
	//		Activation(ml.Sigmoid).
	//		Loss(ml.CompLoss(ml.Pow)).
	//		//InitWeights(xmath.RangeSqrt(-1, 1))
	//		WithWeights(*weights, rnn.Clip{0, 0})
	//}

	// this will capture almost one full cycle for 25 events
	f := 0.03

	rnn := graph.New("RNN")

	rnn.NewSeries(sin, "x", "y")
	rnn.NewSeries(train, "x", "y")

	xx := make([]float64, 0)
	l := 5000
	for i := 0; i < l; i++ {

		x := f * float64(i)
		y := evolveSine(i, x)

		if i < l*4/5 {
			rnn.Add(sin, x, y)
			loss, _ := network.Train(xmath.Vec(1).With(y))
			println(fmt.Sprintf("loss = %v", loss.Sum()))
			rnn.Add(train, x, network.TmpOutput[len(network.TmpOutput)-1])
			xx = append(xx, x)
			//save(weights)
		} else {
			output := network.Predict(xmath.Vec(1).With(y))
			rnn.Add(train, x, output[0])
		}

	}

	// draw the data collection
	graph.Draw("sin", rnn)

}

func evolveSine(i int, x float64) float64 {
	return math.Sin(x)
}

// evolveSineVar reveals the shortcomings of an RNN,
// not being able to capture variations in period.
func evolveSineVar(i int, x float64) float64 {
	twist := float64(i) / float64(100)
	return 10*math.Sin(x) + twist
}

const fileName = "examples/rnn-sine/data/results.json"

//
//func save(weights rnn.Weights) {
//
//	f, err := os.Create(fileName)
//
//	if err != nil {
//		panic(err.Error())
//	}
//
//	defer f.Close()
//
//	b, err := json.Marshal(weights)
//	if err != nil {
//		panic(err.Error())
//	}
//
//	_, err = f.Write(b)
//
//	if err != nil {
//		panic(err.Error())
//	}
//
//}
//
//func load() (*rnn.Weights, error) {
//
//	data, err := ioutil.ReadFile(fileName)
//	if err != nil {
//		return nil, err
//	}
//
//	var weights rnn.Weights
//	err = json.Unmarshal(data, &weights)
//	if err != nil {
//		return nil, err
//	}
//
//	return &weights, nil
//}
