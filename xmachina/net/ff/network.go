package ff

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
)

type Network struct {
	net.Info
	net.Config
	loss   ml.Loss
	layers []net.FFLayer
}

func New(inputSize, outputSize int) *Network {
	return &Network{
		Info: net.Info{
			InputSize:  inputSize,
			OutputSize: outputSize,
		},
		layers: make([]net.FFLayer, 0),
		loss:   ml.Diff,
	}
}

func (n *Network) Loss(loss ml.Loss) {
	n.loss = loss
}

func (n *Network) Add(s int, factory NeuronFactory) *Network {

	ps := n.InputSize

	ls := len(n.layers)
	if ls > 0 {
		// check previous layer size
		ps = n.layers[ls-1].Size()
	}

	n.layers = append(n.layers, NewLayer(ps, s, factory, len(n.layers)))
	return n
}

func (n *Network) AddSoftMax() *Network {

	ps := n.InputSize

	ls := len(n.layers)
	if ls > 0 {
		// check previous layer size
		ps = n.layers[ls-1].Size()
	}

	n.layers = append(n.layers, NewSMLayer(ps, len(n.layers)))
	return n
}

func (n *Network) forward(input xmath.Vector) xmath.Vector {
	output := xmath.Vec(len(input)).With(input...)
	for _, l := range n.layers {
		output = l.Forward(output)
	}
	return output
}

func (n *Network) backward(err xmath.Vector) {
	// we go through the layers in reverse order
	for i := len(n.layers) - 1; i >= 0; i-- {
		err = n.layers[i].Backward(err)
	}

}

func (n *Network) Train(input xmath.Vector, expected xmath.Vector) (err xmath.Vector, weights xmath.Cube) {

	out := n.forward(input)

	diff := expected.Diff(out)

	// quadratic error
	err = n.loss(expected, out)
	// cross entropy
	//err = expected.Dop(func(x, y float64) float64 {
	//	return -1 * x * math.Log(y)
	//}, out)

	n.backward(diff)

	n.Iterations++

	if n.HasTraceEnabled() {
		weights = xmath.Cub(len(n.layers))
		for i := 0; i < len(n.layers); i++ {
			layer := n.layers[i]
			m := layer.Weights()
			weights[i] = m
		}
	}

	return err, weights

}

func (n *Network) Predict(input xmath.Vector) xmath.Vector {
	return n.forward(input)
}

type XNetwork struct {
	*Network
}

func XNew(inputSize, outputSize int) *XNetwork {

	network := XNetwork{
		Network: New(inputSize, outputSize),
	}
	return &network

}

func (xn *XNetwork) Add(s int, factory NeuronFactory) *XNetwork {

	ps := xn.InputSize

	ls := len(xn.layers)
	if ls > 0 {
		// check previous layer size
		ps = xn.layers[ls-1].Size()
	}

	xn.layers = append(xn.layers, newXLayer(ps, s, factory, len(xn.layers)))
	return xn
}
