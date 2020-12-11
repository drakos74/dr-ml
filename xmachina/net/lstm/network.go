package lstm

type Network struct {
}

// New creates a new Recurrent layer
// n : batch size e.g. rnn units
// xDim : size of trainInput/trainOutput vector
// hDim : internal hidden layer size
// rate : learning rate
func New(n, xDim, hDim int) *Network {
	return &Network{}
}

type Clip struct {
	W, B float64
}
