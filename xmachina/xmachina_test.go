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

	"github.com/drakos74/go-ex-machina/xmachina/net"

	"github.com/drakos74/go-ex-machina/xmachina/math"
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/stretchr/testify/assert"
)

func TestNetwork_BinaryClassificationSimple(t *testing.T) {

	// build the network
	network := net.New(2, 1).
		Add(2, net.Perceptron(ml.New(), math.Rand())) // output layer

	inputSet := math.Mat(2).With([]float64{1, 0}, []float64{0, 1})
	outputSet := math.Mat(2).With([]float64{0, 1}, []float64{1, 0})

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
	network := net.New(2, 1).
		Add(2, net.Perceptron(ml.New(), math.Rand())). // hidden layer
		Add(1, net.Perceptron(ml.New(), math.Rand()))  // output layer

	// parse the input data
	b, err := ioutil.ReadFile("test/testdata/bin_class_input.csv")
	assert.NoError(t, err)

	reader := csv.NewReader(bytes.NewBuffer(b))

	records, err := reader.ReadAll()
	assert.NoError(t, err)

	inputSet := math.Mat(len(records))
	outputSet := math.Mat(len(records))

	for i, record := range records {
		inp := math.Vec(len(record) - 1)
		out := math.Vec(len(record) - 2)

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

	TrainInMem(Training(0.0001, 10000), network, inputSet, outputSet)

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
	network := net.New(2, 1).
		Add(2, net.Perceptron(ml.New(), math.Rand())). // hidden layer
		Add(1, net.Perceptron(ml.New(), math.Rand()))  // output layer

	data := make(Data)
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
	network := net.XNew(2, 1).
		Add(2, net.Perceptron(ml.New(), math.Rand())). // hidden layer
		Add(1, net.Perceptron(ml.New(), math.Rand()))  // output layer

	data := make(Data)
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

func testNetwork_BinaryClassificationSoftmaxInMem(t *testing.T) {

	// build the network
	network := net.New(2, 2).
		Add(2, net.Perceptron(ml.New().Rate(0.05), math.Rand())).
		Add(2, net.Perceptron(ml.New().Rate(0.05), math.Rand())).
		AddSoftMax() // output layer

	// parse the input data
	b, err := ioutil.ReadFile("test/testdata/bin_class_input.csv")
	assert.NoError(t, err)

	reader := csv.NewReader(bytes.NewBuffer(b))

	records, err := reader.ReadAll()
	assert.NoError(t, err)

	inputSet := math.Mat(len(records))
	outputSet := math.Mat(len(records))

	for i, record := range records {
		inp := math.Vec(2)
		out := math.Vec(2)

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

	println(fmt.Sprintf("outputSet = %v", outputSet))
	println(fmt.Sprintf("inputSet = %v", inputSet))

	TrainInMem(Training(0.0001, 10000), network, inputSet, outputSet)

	// check trained network performance

	for i, input := range inputSet {
		o := network.Predict(input)
		r := outputSet[i]
		assert.Equal(t, r, o)
	}

}

func parseBinClassLine(record []string) (inp, out math.Vector) {
	inp = math.Vec(len(record) - 1)
	out = math.Vec(len(record) - 2)

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
