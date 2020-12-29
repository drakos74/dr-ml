package main

import (
	"fmt"
	"math"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmachina/net/lstm"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/drakos74/oremi/graph"
	"github.com/rs/zerolog"
)

const (
	sin    = "sin"
	train  = "train"
	dummy  = "dummy"
	evolve = "evolve"
)

func init() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
}

// TODO : make a component test out of this
func main() {

	//weights, err := load()

	//weights = nil

	bufferSize := 10
	hiddenLayerSize := 30.0
	builder := lstm.NewNeuronBuilder(bufferSize, 1, int(hiddenLayerSize), int(hiddenLayerSize)+1).
		WithRate(*ml.Rate(0.5)).
		WithWeights(xmath.RangeSqrt(-1, 1)(hiddenLayerSize), xmath.RangeSqrt(-1, 1)(hiddenLayerSize))

	clip := net.Clip{W: 1, B: 1}
	network := lstm.New(bufferSize, builder, clip)
	network_dummy := lstm.New(bufferSize, builder, clip)
	network_evolve := lstm.New(bufferSize, builder, clip)

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
	rnn.NewSeries(dummy, "x", "y")
	rnn.NewSeries(evolve, "x", "y")

	xx := make([]float64, 0)
	l := 3000
	var lastOutput float64
	for i := 0; i < l; i++ {

		x := f * float64(i)
		y := evolveSine(i, x)
		//y := evolveSineVar(i, x)

		if i < l*3/5 {
			rnn.Add(sin, x, y)
			loss, _ := network.Train(xmath.Vec(1).With(y))
			network_evolve.Train(xmath.Vec(1).With(y))
			println(fmt.Sprintf("loss = %v", loss.Sum()))
			rnn.Add(train, x, network.TmpOutput[len(network.TmpOutput)-1])
			rnn.Add(evolve, x, network_evolve.TmpOutput[len(network_evolve.TmpOutput)-1])
			xx = append(xx, x)
			//save(weights)
		} else {
			output := network.Predict(xmath.Vec(1).With(y))
			rnn.Add(train, x, output[0])
			evolveOutput := network_evolve.Predict(xmath.Vec(1).With(lastOutput))
			lastOutput = evolveOutput[0]
			rnn.Add(evolve, x, evolveOutput[0])
		}
		dummyOutput := network_dummy.Predict(xmath.Vec(1).With(y))
		rnn.Add(dummy, x, dummyOutput[0])

		if i%250 == 0 {
			println(fmt.Sprintf("i = %v", i))
		}

	}

	// draw the data collection
	graph.Draw("sin", rnn)

}

func evolveSine(i int, x float64) float64 {
	return math.Abs(math.Sin(x))
}

// evolveSineVar reveals the shortcomings of an RNN,
// not being able to capture variations in period.
func evolveSineVar(i int, x float64) float64 {
	return math.Abs(0.3*math.Sin(x) + 0.3*math.Sin(2*x) + 0.3*math.Sin(5*x))
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
