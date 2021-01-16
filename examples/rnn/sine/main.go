package main

import (
	"github.com/drakos74/go-ex-machina/xmachina"
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

	// init graphs
	//graph := oremi.New("RNN").
	data := xmachina.VoidSet().
		Init([]xmachina.Set{
			{Name: sin, X: "x", Y: "y"},
			{Name: train, X: "x", Y: "y"},
			{Name: dummy, X: "x", Y: "y"},
			{Name: evolve, X: "x", Y: "y"},
		}...)

	// init networks
	cap := map[string]Capture{
		sin:    &OutputCapture{network: &VoidNetwork{}},
		train:  &OutputCapture{network: NewRNNValueNetwork(10, 10)},
		dummy:  &DummyCapture{OutputCapture{network: NewRNNValueNetwork(10, 10)}},
		evolve: &EvolutionCapture{OutputCapture{network: NewRNNValueNetwork(10, 10)}},
	}

	// this will capture almost one full cycle for 25 events
	Train(X(0.03), Sine, data, cap)

	// draw the data collection
	data.Export("sin")

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
