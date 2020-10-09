package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/drakos74/go-ex-machina/xmachina/net/ff"

	"github.com/drakos74/go-ex-machina/xmachina"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

func main() {

	start := time.Now()

	// optional profiling block
	//f, err := os.Create("examples/mnist-sigmoid/profile")
	//if err != nil {
	//	log.Fatal(err)
	//}
	//pprof.StartCPUProfile(f)
	//defer pprof.StopCPUProfile()

	// build the network

	// sigmoid
	//network := net.XNew(784, 10).
	//	Add(200, net.Perceptron(ml.Model().Rate(0.1, 0), xmath.Rand(-1, 1, math.Sqrt))).
	//	Add(10, net.Perceptron(ml.Model().Rate(0.1, 0), xmath.Rand(-1, 1, math.Sqrt)))
	// tanh with softmax
	network := ff.XNew(784, 10).
		Add(200, ff.Perceptron(ml.Model().Rate(0.1, 0).WithActivation(ml.TanH), xmath.Rand(-1, 1, math.Sqrt))).
		Add(10, ff.Perceptron(ml.Model().Rate(0.1, 0).WithActivation(ml.TanH), xmath.Rand(-1, 1, math.Sqrt))).
		AddSoftMax()
	// ReLU
	//network := ff.XNew(784, 10).
	//	Add(200, ff.Perceptron(ml.Model().Rate(0.1, 0).WithActivation(ml.ReLU), xmath.Rand(-1, 1, math.Sqrt))).
	//	//note : we should not use ReLU in the output layer
	//	Add(10, ff.Perceptron(ml.Model().Rate(0.1, 0).WithActivation(ml.TanH), xmath.Rand(-1, 1, math.Sqrt))).
	//	AddSoftMax()

	data := make(xmachina.Data)

	config := xmachina.StreamingTraining(xmachina.Training(0.1, 1), 10, 1000)

	ack := make(xmachina.Ack)

	ctx, cnl := context.WithCancel(context.Background())
	go xmachina.TrainInStream(ctx, config, network, data, ack)

	_, _, err := xmachina.ReadFile("examples/mnist/mnist_train.csv", 0, 5, parseMnistLine, data, config.Epoch, ack)

	if err != nil {
		log.Fatalf("could not read file: %v", err)
	}

	cnl()

	println(fmt.Sprintf("train duration = %v", time.Since(start)))

	// score the network

	checkFile, _ := os.Open("examples/mnist/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	rTest := csv.NewReader(bufio.NewReader(checkFile))
	total := 0
	for {
		record, err := rTest.Read()
		if err == io.EOF {
			break
		}
		total++
		inputs, _ := parseMnistLine(record)
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

	log.Println(fmt.Sprintf("score = %v", float64(score)/float64(total)))

}

func parseMnistLine(record []string) (inp, out xmath.Vector) {

	inp = xmath.Vec(784)
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
