package rc

import (
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
)

// Layer defines a recurrent layer interface.
type Layer interface {
	Forward(x xmath.Matrix) xmath.Matrix
	Backward(exp xmath.Matrix) xmath.Matrix
	Weights() map[net.Meta]net.Weights
}

// LayerFactory defines the constructor for a recurrent layer.
type LayerFactory func(n int, clipping net.Clip, index int) Layer

func gatherWeights(layer Layer) map[net.Meta]net.Weights {
	return layer.Weights()
}
