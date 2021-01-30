package xmachina

import (
	"bytes"
	"context"
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/drakos74/go-ex-machina/xmath/algebra"

	"github.com/rs/zerolog"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/drakos74/go-ex-machina/xmachina/net/ff"
	"github.com/stretchr/testify/assert"
)

func init() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
}

func TestNetwork_BinaryClassificationSimple(t *testing.T) {

	// build the network
	network := ff.New(2, 1).
		Add(2,
			net.NewBuilder().
				WithModule(ml.Base().
					WithRate(ml.Learn(0.05, 0.05)).
					WithActivation(ml.Sigmoid)).
				WithWeights(algebra.Rand(0, 1, algebra.Unit), algebra.Rand(0, 1, algebra.Unit)).
				Factory(net.NewActivationCell),
		) // output layer

	inputSet := algebra.Mat(2).With([]float64{1, 0}, []float64{0, 1})
	outputSet := algebra.Mat(2).With([]float64{0, 1}, []float64{1, 0})

	TrainInMem(Training(0.001, 10000), network, inputSet, outputSet)

	// check trained network performance

	for i, input := range inputSet {
		o := network.Predict(input).Round()
		r := outputSet[i]
		assert.Equal(t, o, r)
	}

}

func TestNetwork_BinaryClassificationInMem(t *testing.T) {

	// build the network
	network := ff.New(2, 1).
		Add(2, net.NewBuilder().
			WithModule(ml.Base().
				WithRate(ml.Learn(0.05, 0.05)).
				WithActivation(ml.Sigmoid)).
			WithWeights(algebra.Rand(0, 1, algebra.Unit), algebra.Rand(0, 1, algebra.Unit)).
			Factory(net.NewActivationCell)). // hidden layer
		Add(1, net.NewBuilder().
			WithModule(ml.Base().
				WithRate(ml.Learn(0.05, 0.05)).
				WithActivation(ml.Sigmoid)).
			WithWeights(algebra.Rand(0, 1, algebra.Unit), algebra.Rand(0, 1, algebra.Unit)).
			Factory(net.NewActivationCell)) // output layer

	// parse the input data
	b, err := ioutil.ReadFile("test/testdata/bin_class_input.csv")
	assert.NoError(t, err)

	reader := csv.NewReader(bytes.NewBuffer(b))

	records, err := reader.ReadAll()
	assert.NoError(t, err)

	inputSet := algebra.Mat(len(records))
	outputSet := algebra.Mat(len(records))

	for i, record := range records {
		inp := algebra.Vec(len(record) - 1)
		out := algebra.Vec(len(record) - 2)

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

func TestNetwork_BinaryClassificationStream(t *testing.T) {

	start := time.Now().UnixNano()

	// build the network
	network := ff.New(2, 1).
		Add(2, net.NewBuilder().
			WithModule(ml.Base().
				WithRate(ml.Learn(0.05, 0.05)).
				WithActivation(ml.Sigmoid)).
			WithWeights(algebra.Rand(0, 1, algebra.Unit), algebra.Rand(0, 1, algebra.Unit)).
			Factory(net.NewActivationCell)). // hidden layer
		Add(1, net.NewBuilder().
			WithModule(ml.Base().
				WithRate(ml.Learn(0.05, 0.05)).
				WithActivation(ml.Sigmoid)).
			WithWeights(algebra.Rand(0, 1, algebra.Unit), algebra.Rand(0, 1, algebra.Unit)).
			Factory(net.NewActivationCell)) // output layer

	data := make(DataSource)
	defer close(data)

	config := StreamingTraining(Training(0.0001, 100), 1, 1000)
	defer close(config.Epoch)

	ack := make(Ack)

	ctx, cnl := context.WithCancel(context.Background())
	go TrainInStream(ctx, config, network, data, ack)

	inputSet, outputSet, err := ReadFile("test/testdata/bin_class_input.csv", 10, 500, parseBinClassLine, data, config.Epoch, ack)

	if err != nil {
		t.Fail()
	}

	cnl()

	// check trained network performance
	for i, input := range inputSet {
		o := network.Predict(input).Round()
		r := outputSet[i]
		assert.Equal(t, o, r)
	}

	d := time.Now().UnixNano() - start
	println(fmt.Sprintf("d = %v", d))

}

func TestXNetwork_BinaryClassificationStream(t *testing.T) {

	start := time.Now().UnixNano()
	// build the network
	network := ff.XNew(2, 1).
		Add(2, ff.Perceptron(ml.Base(), algebra.Rand(0, 1, algebra.Unit))). // hidden layer
		Add(1, ff.Perceptron(ml.Base(), algebra.Rand(0, 1, algebra.Unit)))  // output layer

	data := make(DataSource)
	defer close(data)

	config := StreamingTraining(Training(0.0001, 100), 1, 1000)
	defer close(config.Epoch)

	ack := make(Ack)

	ctx, cnl := context.WithCancel(context.Background())
	go TrainInStream(ctx, config, network, data, ack)

	inputSet, outputSet, err := ReadFile("test/testdata/bin_class_input.csv", 10, 500, parseBinClassLine, data, config.Epoch, ack)

	if err != nil {
		t.Fail()
	}

	cnl()

	// check trained network performance
	for i, input := range inputSet {
		o := network.Predict(input).Round()
		r := outputSet[i]
		assert.Equal(t, o, r)
	}

	d := time.Now().UnixNano() - start
	println(fmt.Sprintf("d = %v", d))

}

func TestNetwork_BinaryClassification2D_InMem(t *testing.T) {

	// build the network
	network := ff.New(2, 2).
		Add(20,
			net.NewBuilder().
				WithModule(ml.Base().
					WithRate(ml.Learn(5, 0.05)).
					WithActivation(ml.Sigmoid)).
				WithWeights(algebra.Rand(-1, 1, algebra.Unit), algebra.Rand(-1, 1, algebra.Unit)).
				Factory(net.NewActivationCell)).
		Add(2, net.NewBuilder().
			WithModule(ml.Base().
				WithRate(ml.Learn(5, 0.05)).
				WithActivation(ml.Sigmoid)).
			WithWeights(algebra.Rand(-1, 1, algebra.Unit), algebra.Rand(-1, 1, algebra.Unit)).
			Factory(net.NewActivationCell))
	//.AddSoftMax()

	// parse the input data
	b, err := ioutil.ReadFile("test/testdata/bin_class_input.csv")
	assert.NoError(t, err)

	reader := csv.NewReader(bytes.NewBuffer(b))

	records, err := reader.ReadAll()
	assert.NoError(t, err)

	inputSet := algebra.Mat(len(records))
	outputSet := algebra.Mat(len(records))

	for i, record := range records {
		inp := algebra.Vec(2)
		out := algebra.Vec(2)

		for j, value := range record {
			f, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
			if err != nil {
				panic(fmt.Sprintf("cannot Train with non-numeric value %v: %v", value, err))
			}
			if j < 2 {
				inp[j] = f
			} else {
				if f == 0 {
					out[0] = 0.1
					out[1] = 0.9
				} else {
					out[0] = 0.9
					out[1] = 0.1
				}

			}
		}
		inputSet[i] = inp
		outputSet[i] = out
	}

	TrainInMem(Training(0.00001, 10000), network, inputSet, outputSet)

	// check trained network performance

	for i, input := range inputSet {
		o := network.Predict(input)
		r := outputSet[i]
		assert.Equal(t, fmt.Sprintf("%.2f", r[0]), fmt.Sprintf("%.2f", o[0]))
		assert.Equal(t, fmt.Sprintf("%.2f", r[1]), fmt.Sprintf("%.2f", o[1]))
	}

}

func parseBinClassLine(record []string) (inp, out algebra.Vector) {
	inp = algebra.Vec(len(record) - 1)
	out = algebra.Vec(len(record) - 2)

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
	return inp, out
}
