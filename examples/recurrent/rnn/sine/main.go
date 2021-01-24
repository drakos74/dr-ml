package main

import (
	"github.com/drakos74/go-ex-machina/examples/recurrent"
	"github.com/drakos74/go-ex-machina/oremi"
	"github.com/drakos74/go-ex-machina/xmachina"
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmachina/net/rc"
	"github.com/drakos74/go-ex-machina/xmachina/net/rc/rnn"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/rs/zerolog"
)

const (
	sin    = "sin"
	train  = "train"
	dummy  = "dummy"
	evolve = "evolve"
	soft   = "soft"
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
	data := oremi.New("RNN").
		//data := xmachina.VoidSet().
		Init([]xmachina.Set{
			{Name: sin, X: "x", Y: "y"},
			{Name: train, X: "x", Y: "y"},
			{Name: dummy, X: "x", Y: "y"},
			{Name: evolve, X: "x", Y: "y"},
			{Name: soft, X: "x", Y: "y"},
		}...)

	// init networks
	bufferSize := 10
	layerSize := 10
	cap := map[string]recurrent.Capture{
		sin:   &recurrent.OutputCapture{Network: &recurrent.VoidNetwork{}},
		train: &recurrent.OutputCapture{Network: NewRNNValueNetwork(bufferSize, layerSize)},
		//dummy:  &recurrent.DummyCapture{recurrent.OutputCapture{Network: NewRNNValueNetwork(10, 10)}},
		//evolve: &recurrent.EvolutionCapture{recurrent.OutputCapture{Network: NewRNNValueNetwork(10, 10)}},
		soft: &recurrent.SoftCapture{Network: NewRNNProbabilityNetwork(bufferSize, layerSize, 2)},
	}

	// this will capture almost one full cycle for 25 events
	recurrent.Train(5000, recurrent.X(0.03), recurrent.Sine, data, cap)

	// draw the data collection
	data.Export("sin")

}

func NewRNNValueNetwork(bufferSize, hiddenLayerSize int) net.NN {
	hls := float64(hiddenLayerSize)
	builder := rc.NewNeuronBuilder(1, 1, int(hiddenLayerSize)).
		WithRate(*ml.Rate(0.05)).
		WithWeights(xmath.RangeSqrt(-1, 1)(hls), xmath.RangeSqrt(-1, 1)(hls)).
		WithActivation(ml.TanH, ml.Sigmoid)
	return rc.New(bufferSize, rnn.New(*builder), net.NewClip(1, 1))
}

func NewRNNProbabilityNetwork(bufferSize, hiddenLayerSize, outputSize int) net.NN {
	hls := float64(hiddenLayerSize)
	builder := rc.NewNeuronBuilder(1, 2, int(hiddenLayerSize)).
		WithRate(*ml.Rate(0.05)).
		WithWeights(xmath.RangeSqrt(-1, 1)(hls), xmath.RangeSqrt(-1, 1)(hls)).
		WithActivation(ml.TanH, ml.Sigmoid).
		SoftMax(outputSize)
	return rc.New(bufferSize, rnn.New(*builder), net.NewClip(1, 1)).OutputTransform(func(matrix xmath.Matrix) xmath.Matrix {
		// we want to make out of this sequence of vectors a 1 d longer vector with the direction
		return matrix.T().Vop(xmath.Diff, xmath.UpOrDown)[0]
	})
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
