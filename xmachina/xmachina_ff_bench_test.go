package xmachina

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
	"testing"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmachina/net/ff"
	"github.com/drakos74/go-ex-machina/xmath"
	"github.com/stretchr/testify/assert"
)

func TestNetwork_BinaryClassificationInMem_Benchmark(t *testing.T) {

	// build the network
	network := ff.New(2, 1).
		Add(10, net.NewBuilder().
			WithModule(ml.Base().
				WithRate(ml.Learn(0.5, 0.05)).
				WithActivation(ml.Sigmoid)).
			WithWeights(xmath.Rand(-1, 1, xmath.Unit), xmath.Rand(-1, 1, xmath.Unit)).
			Factory(net.NewActivationCell)). // hidden layer
		Add(1, net.NewBuilder().
			WithModule(ml.Base().
				WithRate(ml.Learn(0.5, 0.05)).
				WithActivation(ml.Sigmoid)).
			WithWeights(xmath.Rand(-1, 1, xmath.Unit), xmath.Rand(-1, 1, xmath.Unit)).
			Factory(net.NewActivationCell)) // output layer

	// parse the input data
	b, err := ioutil.ReadFile("test/testdata/bin_class_input.csv")
	assert.NoError(t, err)

	reader := csv.NewReader(bytes.NewBuffer(b))

	records, err := reader.ReadAll()
	assert.NoError(t, err)

	inputSet := xmath.Mat(len(records))
	outputSet := xmath.Mat(len(records))

	for i, record := range records {
		inp := xmath.Vec(len(record) - 1)
		out := xmath.Vec(len(record) - 2)

		for j, value := range record {
			f, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
			if err != nil {
				panic(fmt.Sprintf("cannot Train with non-numeric value %v: %v", value, err))
			}
			if j < 2 {
				inp[j] = f
			} else {
				out[j-2] = f
			}
		}
		inputSet[i] = inp
		outputSet[i] = out
	}

	TrainInMem(Training(0.001, 10000), network, inputSet, outputSet)

	// check trained network performance
	for i, input := range inputSet {
		o := network.Predict(input).Round()
		r := outputSet[i]
		assert.Equal(t, o, r)
	}

}
