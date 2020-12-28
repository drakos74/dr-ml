package rnn

import (
	"fmt"
	"math"

	"github.com/drakos74/go-ex-machina/xmachina/net"

	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/rs/zerolog/log"

	"github.com/drakos74/go-ex-machina/xmath/time"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

type Network struct {
	net.Config
	*Layer
	*xmath.Stats

	loss                      ml.MLoss
	predictInput, trainOutput *time.Window
	TmpOutput                 xmath.Vector
}

// New creates a new Recurrent network
// Note that the output should be not much outside the range of [-1,+1]
// As an RNN this network is very sensitive on the training parameters, weights, batch size etc ...
func New(n int, builder *NeuronBuilder, clipping Clip) *Network {
	return &Network{
		Layer:        NewLayer(n, *builder, clipping, 0),
		predictInput: time.NewWindow(n),
		trainOutput:  time.NewWindow(n + 1),
		loss: func(expected, output xmath.Matrix) xmath.Vector {
			return expected.Dop(func(x, y float64) float64 {
				return x - y
			}, output).Sum()
		},
		Stats: xmath.NewStats(),
	}
}

// WithWeights initialises the network recurrent layer and generates the starting weights.
//func (net *Network) WithWeights(weights Weights, clip Clip) *Network {
//	if net.Layer != nil {
//		panic("rnn layer already initialised")
//	}
//	//net.Layer = LoadRNNLayer(
//	//	net.n,
//	//	net.xDim,
//	//	net.hDim,
//	//	net.learn,
//	//	Neuron(net.g1, net.g2, net.learn,xmath.Const(0.5),xmath.Const(0.5)),
//	//	weights, clip, 0)
//	return net
//}

// InitWeights initialises the network recurrent layer and generates the starting weights.
//func (net *Network) InitWeights(weightGenerator xmath.ScaledVectorGenerator, clip Clip) *Network {
//	if net.Layer != nil {
//		panic("rnn layer already initialised")
//	}
//
//	net.Layer = NewLayer(
//		net.n,
//		net.xDim,
//		net.hDim,
//		net.learn,
//		net.neuronFactory,
//		weightGenerator, clip, 0)
//	return net
//}

func (net *Network) Train(data xmath.Vector) (err xmath.Vector, weights []net.Weights) {
	// add our trainInput & trainOutput to the batch
	batch, batchIsReady := net.trainOutput.Push(data)
	// be ready for predictions ... from the start
	net.predictInput.Push(data)
	loss := xmath.Vec(len(data))
	net.TmpOutput = xmath.Vec(len(data))
	if batchIsReady {
		// we can actually train now ...
		inp := time.Inp(batch)
		exp := time.Outp(batch)

		// forward pass
		out := net.Forward(inp)

		// keep the last data as the standard data
		net.TmpOutput = out[len(out)-1]

		// add the cross entropy loss for each of the vectors
		loss = net.loss(exp, out).Op(math.Abs)

		// backward pass
		net.Backward(exp)
		// update stats
		net.Inc(loss.Sum())
		// log progress
		if net.Iteration%1000 == 0 {
			log.Info().
				Int("epoch", net.Iteration).
				Float64("err", loss.Sum()).
				Str("mean-err", fmt.Sprintf("%+v", net.Stats)).
				Msg("training iteration")
		}
	}

	if net.HasTraceEnabled() {
		weights = gatherWeights(net.Layer)
	}

	return loss, weights

}

func (net *Network) Predict(input xmath.Vector) xmath.Vector {

	batch, batchIsReady := net.predictInput.Push(input)

	if batchIsReady {
		out := net.Forward(batch)
		return out[len(out)-1]
	}

	return xmath.Vec(len(input))
}

func gatherWeights(layer *Layer) []net.Weights {
	weights := make([]net.Weights, 4)
	weights[0] = *layer.neurons[0].input.Weights()
	weights[0] = *layer.neurons[0].hidden.Weights()
	weights[0] = *layer.neurons[0].activation.Weights()
	weights[0] = *layer.neurons[0].output.Weights()
	return weights
}

// TODO: make it the corresponding one compared to the above ..
// so that we can parse the weights we might want to save
func parseWeights(weights []net.Weights) {

}
