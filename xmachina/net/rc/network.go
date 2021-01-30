package rc

import (
	"math"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/drakos74/go-ex-machina/xmath/buffer"
)

type Network struct {
	net.Config
	Layer
	*buffer.Stats

	loss                            ml.MLoss
	predictInput, trainOutput       *buffer.VectorRing
	inputTransform, outputTransform func(matrix xmath.Matrix) xmath.Matrix
}

// New creates a new Recurrent network
// Note that the output should be not much outside the range of [-1,+1]
// As an RNN this network is very sensitive on the training parameters, weights, batch size etc ...
func New(n int, layerFactory LayerFactory, clipping net.Clip) *Network {
	return &Network{
		Layer:        layerFactory(n, clipping, 0),
		predictInput: buffer.NewVectorRing(n),
		trainOutput:  buffer.NewVectorRing(n + 1),
		loss: func(expected, output xmath.Matrix) xmath.Vector {
			return expected.Dop(func(x, y float64) float64 {
				return x - y
			}, output).Sum()
		},
		Stats: buffer.NewStats(),
		inputTransform: func(matrix xmath.Matrix) xmath.Matrix {
			return buffer.Inp(matrix)
		},
		outputTransform: func(matrix xmath.Matrix) xmath.Matrix {
			return buffer.Outp(matrix)
		},
	}
}

// OutputTransform defines the transformation on the output sequence for training.
func (net *Network) OutputTransform(transform func(matrix xmath.Matrix) xmath.Matrix) *Network {
	net.outputTransform = transform
	return net
}

func (net *Network) Train(data xmath.Vector, outputData xmath.Vector) (err xmath.Vector, weights map[net.Meta]net.Weights) {
	// add our trainInput & trainOutput to the batch
	batch, batchIsReady := net.trainOutput.Push(data)
	// be ready for predictions ... from the start
	net.predictInput.Push(data)
	loss := xmath.Vec(len(data))
	if batchIsReady {
		// we can actually train now ...
		inp := net.inputTransform(batch)
		exp := net.outputTransform(batch)
		// forward pass
		out := net.Forward(inp)

		// pass the data to the output vector
		for i := 0; i < len(outputData); i++ {
			// keep the last row as output data
			outputData[i] = out[len(out)-1][i]
		}

		// add the cross entropy loss for each of the vectors
		loss = net.loss(exp, out).Op(math.Abs)

		// backward pass
		net.Backward(exp)
		// update buffer
		// TODO:
		//net.Inc(loss.Sum())
		//// log progress
		//if net.Iteration%1000 == 0 {
		//	log.Info().
		//		Int("epoch", net.Iteration).
		//		Float64("err", loss.Sum()).
		//		Str("mean-err", fmt.Sprintf("%+v", net.Stats)).
		//		Msg("training iteration")
		//}
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

// TODO: make it the corresponding one compared to the above ..
// so that we can parse the weights we might want to save
func parseWeights(weights []net.Weights) {

}
