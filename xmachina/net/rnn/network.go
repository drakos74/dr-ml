package rnn

import (
	"fmt"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmath/time"

	"github.com/rs/zerolog/log"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
)

type Network struct {
	*Layer
	*xmath.Stats

	learn         ml.Learning
	activation    ml.SoftActivation
	neuronFactory NeuronFactory

	n, xDim, hDim int

	loss                      ml.MLoss
	predictInput, trainOutput *time.Window
	TmpOutput                 xmath.Vector
}

// New creates a new Recurrent layer
// n : batch size e.g. rnn units
// xDim : size of trainInput/trainOutput vector
// hDim : internal hidden layer size
// rate : learning rate
func New(n, xDim, hDim int) *Network {
	return &Network{
		n:            n,
		xDim:         xDim,
		hDim:         hDim,
		predictInput: time.NewWindow(n),
		trainOutput:  time.NewWindow(n + 1),
		Stats:        xmath.NewStats(),
	}
}

type Clip struct {
	W, B float64
}

// WithWeights initialises the network recurrent layer and generates the starting weights.
func (net *Network) WithWeights(weights Weights, clip Clip) *Network {
	if net.Layer != nil {
		panic("rnn layer already initialised")
	}
	net.Layer = LoadRNNLayer(
		net.n,
		net.xDim,
		net.hDim,
		net.learn,
		net.neuronFactory,
		weights, clip, 0)
	return net
}

// InitWeights initialises the network recurrent layer and generates the starting weights.
func (net *Network) InitWeights(weightGenerator xmath.ScaledVectorGenerator, clip Clip) *Network {
	if net.Layer != nil {
		panic("rnn layer already initialised")
	}
	net.Layer = NewRNNLayer(
		net.n,
		net.xDim,
		net.hDim,
		net.learn,
		net.neuronFactory,
		weightGenerator, clip, 0)
	return net
}

func (net *Network) Rate(rate float64) *Network {
	net.learn = ml.Learn(rate)
	return net
}

func (net *Network) Activation(activation ml.Activation) *Network {
	net.neuronFactory = RNeuron(activation)
	return net
}

func (net *Network) SoftActivation(activation ml.SoftActivation) *Network {
	net.activation = activation
	return net
}

func (net *Network) Loss(loss ml.MLoss) *Network {
	net.loss = loss
	return net
}

func (net *Network) Train(data xmath.Vector) (err xmath.Vector, weights Weights) {
	// add our trainInput & trainOutput to the batch
	batch, batchIsReady := net.trainOutput.Push(data)
	// be ready for predictions ... from the start
	net.predictInput.Push(data)
	loss := xmath.Vec(len(data))
	net.TmpOutput = xmath.Vec(len(data))
	if batchIsReady {
		// we can actually train now ...
		inp := time.Inp(batch)
		outp := time.Outp(batch)

		// forward pass
		out := net.Forward(inp)

		// keep the last data as the standard data
		net.TmpOutput = out[len(out)-1]

		// add the cross entropy loss for each of the vectors
		loss = net.loss(outp, out)

		// backward pass
		net.Backward(outp)
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

	return loss, net.Layer.Weights()

}

func (net *Network) Predict(input xmath.Vector) xmath.Vector {

	batch, batchIsReady := net.predictInput.Push(input)

	if batchIsReady {
		out := net.Forward(batch)
		return out[len(out)-1]
	}

	return xmath.Vec(len(input))
}
