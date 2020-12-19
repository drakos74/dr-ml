package lstm

import "github.com/drakos74/go-ex-machina/xmachina/ml"

type Network struct {
	loss ml.Loss
}

// New creates a new Recurrent layer
// n : batch size e.g. rnn units
// xDim : size of trainInput/trainOutput vector
// hDim : internal hidden layer size
// rate : learning rate
func New(n, xDim, hDim int) *Network {
	return &Network{
		loss: ml.Pow,
	}
}

type Clip struct {
	W, B float64
}
