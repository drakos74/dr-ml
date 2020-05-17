package xmachina

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/drakos74/go-ex-machina/xmachina/net"

	"github.com/drakos74/go-ex-machina/xmachina/math"
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/stretchr/testify/assert"
)

func TestNetwork_BinaryClassificationSimple(t *testing.T) {

	// build the network
	network := net.New(2, 1).Debug(true).
		Add(2, net.Perceptron(ml.New(), math.Rand)) // output layer

	inputSet := math.NewMatrix(2).With([]float64{1, 0}, []float64{0, 1})
	outputSet := math.NewMatrix(2).With([]float64{0, 1}, []float64{1, 0})

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
	network := net.New(2, 1).Debug(true).
		Add(2, net.Perceptron(ml.New(), math.Rand)). // hidden layer
		Add(1, net.Perceptron(ml.New(), math.Rand))  // output layer

	// parse the input data
	b, err := ioutil.ReadFile("test/testdata/bin_class_input.csv")
	assert.NoError(t, err)

	reader := csv.NewReader(bytes.NewBuffer(b))

	records, err := reader.ReadAll()
	assert.NoError(t, err)

	inputSet := math.NewMatrix(len(records))
	outputSet := math.NewMatrix(len(records))

	for i, record := range records {
		inp := math.NewVector(len(record) - 1)
		out := math.NewVector(len(record) - 2)

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

	// build the network
	network := net.New(2, 1).Debug(true).
		Add(2, net.Perceptron(ml.New(), math.Rand)). // hidden layer
		Add(1, net.Perceptron(ml.New(), math.Rand))  // output layer

	data := make(Data)
	defer close(data)

	config := StreamingTraining(Training(0.0001, 100), 1, 1000)
	defer close(config.epoch)

	ack := make(Ack)

	ctx, cnl := context.WithCancel(context.Background())
	go TrainInStream(ctx, config, network, data, ack)

	inputSet, outputSet, err := ReadFile("test/testdata/bin_class_input.csv", 10, 500, parseBinClassLine, data, config.epoch, ack)

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

}

func parseBinClassLine(record []string) (inp, out math.Vector) {
	inp = math.NewVector(len(record) - 1)
	out = math.NewVector(len(record) - 2)

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

func TestNetwork_Mnist(t *testing.T) {

	// build the network
	network := net.New(784, 10).Debug(true).
		Add(784, net.Perceptron(ml.New(), math.Rand)). // hidden layer
		Add(200, net.Perceptron(ml.New(), math.Rand)). // hidden layer
		Add(10, net.Perceptron(ml.New(), math.Rand))   // output layer

	data := make(Data)

	config := StreamingTraining(Training(0.1, 1), 10, 1000)

	ack := make(Ack)

	ctx, cnl := context.WithCancel(context.Background())
	go TrainInStream(ctx, config, network, data, ack)

	_, _, err := ReadFile("test/testdata/mnist/mnist_train.csv", 0, 5, parseMnistLine, data, config.epoch, ack)

	if err != nil {
		t.Fail()
	}

	cnl()

	// score the network

	checkFile, _ := os.Open("test/testdata/mnist/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	rTest := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := rTest.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, 784)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}
		outputs := network.Predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < len(outputs); i++ {
			if outputs[i] > highest {
				best = i
				highest = outputs[i]
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}

	log.Println(fmt.Sprintf("score = %v", score))

}

func parseMnistLine(record []string) (inp, out math.Vector) {

	inp = math.NewVector(784)
	for i := range inp {
		x, _ := strconv.ParseFloat(record[i+1], 64)
		inp[i] = (x / 255.0 * 0.99) + 0.01
	}

	out = make([]float64, 10)
	for i := range out {
		out[i] = 0.1
	}
	x, _ := strconv.Atoi(record[0])
	out[x] = 0.9

	return inp, out
}
