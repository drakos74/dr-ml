package net

import (
	"fmt"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

type NN interface {
	Train(input xmath.Vector, output xmath.Vector) (err xmath.Vector, weights xmath.Cube)
	Predict(input xmath.Vector) xmath.Vector
	Loss(loss ml.Loss)
}

type info struct {
	init       bool
	inputSize  int
	outputSize int
	iterations int
}

type config struct {
	trace bool
	debug bool
}

func (cfg *config) Debug() {
	cfg.debug = true
}

func (cfg *config) Trace() {
	cfg.trace = true
}

type Network struct {
	info
	config
	loss   ml.Loss
	layers []Layer
}

func New(inputSize, outputSize int) *Network {

	return &Network{
		info: info{
			inputSize:  inputSize,
			outputSize: outputSize,
		},
		layers: make([]Layer, 0),
		loss:   ml.Diff,
	}
}

func (n *Network) Loss(loss ml.Loss) {
	n.loss = loss
}

func (n *Network) Add(s int, factory NeuronFactory) *Network {

	ps := n.inputSize

	ls := len(n.layers)
	if ls > 0 {
		// check previous layer size
		ps = n.layers[ls-1].Size()
	}

	n.layers = append(n.layers, NewFFLayer(ps, s, factory, len(n.layers)))
	return n
}

func (n *Network) AddSoftMax() *Network {

	ps := n.inputSize

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
	//println(fmt.Sprintf("n.iter = %v", n.iterations))
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

	n.iterations++

	if n.trace {
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

	ps := xn.inputSize

	ls := len(xn.layers)
	if ls > 0 {
		// check previous layer size
		ps = xn.layers[ls-1].Size()
	}

	xn.layers = append(xn.layers, newXLayer(ps, s, factory, len(xn.layers)))
	return xn
}

type stats struct {
	iteration int
	xmath.Bucket
}

type RNetwork struct {
	*RNNLayer
	stats
	loss         ml.Loss
	predictInput *xmath.Window
	trainInput   *xmath.Window
	trainOutput  *xmath.Window
	TmpOutput    xmath.Vector
}

func NewRNetwork(s, n, d int, rate float64) *RNetwork {
	return &RNetwork{
		RNNLayer:     NewRNNLayer(s, n, d, ml.Learn(rate), RNeuron(ml.TanH), 0),
		predictInput: xmath.NewWindow(n),
		trainInput:   xmath.NewWindow(n),
		trainOutput:  xmath.NewWindow(n),
	}
}

func (net *RNetwork) Loss(loss ml.Loss) {
	net.loss = loss
}

func (net *RNetwork) Train(input xmath.Vector, output xmath.Vector) (err xmath.Vector, weights xmath.Cube) {
	// add our trainInput & trainOutput to the batch
	var batchIsReady bool
	batchIsReady = net.trainInput.Push(input)
	batchIsReady = net.trainOutput.Push(output)

	loss := xmath.Vec(net.trainInput.Size())
	net.TmpOutput = xmath.Vec(len(input))
	if batchIsReady {
		// we can actually train now ...
		inp := net.trainInput.Batch()
		outp := net.trainOutput.Batch()
		out := net.Forward(inp)
		// add the cross entropy loss for each of the vectors
		for i := 0; i < len(outp); i++ {
			loss[i] = net.loss(outp[i], out[i]).Sum()
		}
		net.TmpOutput = out[len(out)-1]
		net.Backward(outp)
		net.iteration++
		net.stats.Add(loss.Sum())

		if net.iteration%100 == 0 {
			println(fmt.Sprintf("epoch =  = %v , err = %v , mean-err = %v", net.iteration, loss.Sum(), net.stats.Bucket))
		}

	}

	return loss, weights

}

func (net *RNetwork) Predict(input xmath.Vector) xmath.Vector {

	batchIsReady := net.predictInput.Push(input)

	if batchIsReady {
		inp := net.predictInput.Batch()
		out := net.Forward(inp)
		return out[len(out)-1]
	}

	return xmath.Vec(len(input))
}
