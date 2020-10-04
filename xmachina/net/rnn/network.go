package rnn

import (
	"fmt"

	"github.com/drakos74/go-ex-machina/xmachina/net"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

type Network struct {
	*RNNLayer
	net.Stats
	loss ml.MLoss

	predictInput *xmath.Window
	trainOutput  *xmath.Window
	TmpOutput    xmath.Vector
}

// NewRNNLayer creates a new Recurrent layer
// b : batch size e.g. rnn units
// xDim : size of trainInput/trainOutput vector
// hDim : internal hidden layer size
// rate : learning rate
func New(b, xDim, hDim int, rate float64) *Network {
	return &Network{
		RNNLayer:     NewRNNLayer(b, xDim, hDim, ml.Learn(rate), RNeuron(ml.TanH), xmath.RangeSqrt(-1, 1), 0),
		predictInput: xmath.NewWindow(b),
		trainOutput:  xmath.NewWindow(b + 1),
	}
}

func (net *Network) Loss(loss ml.MLoss) *Network {
	net.loss = loss
	return net
}

func (net *Network) Train(data xmath.Vector) (err xmath.Vector, weights xmath.Cube) {
	// add our trainInput & trainOutput to the batch
	var batchIsReady bool
	batchIsReady = net.trainOutput.Push(data)
	// be ready for predictions ... from the start
	net.predictInput.Push(data)
	loss := xmath.Vec(len(data))
	net.TmpOutput = xmath.Vec(len(data))
	if batchIsReady {
		// we can actually train now ...
		batch := net.trainOutput.Batch()

		inp := xmath.Inp(batch)
		outp := xmath.Outp(batch)

		// forward pass
		out := net.Forward(inp)

		// keep the last data as the standard data
		net.TmpOutput = out[len(out)-1]

		// add the cross entropy loss for each of the vectors
		loss = net.loss(outp, out)

		// backward pass
		net.Backward(outp)
		// update stats
		net.Iteration++
		net.Stats.Add(loss.Sum())
		// log progress
		if net.Iteration%100 == 0 {
			println(fmt.Sprintf("epoch =  = %v , err = %v , mean-err = %v", net.Iteration, loss.Sum(), net.Stats.Bucket))
		}
	}

	return loss, weights

}

func (net *Network) Predict(input xmath.Vector) xmath.Vector {

	batchIsReady := net.predictInput.Push(input)

	if batchIsReady {
		batch := net.predictInput.Batch()
		out := net.Forward(batch)
		return out[len(out)-1]
	}

	return xmath.Vec(len(input))
}
