package net

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"testing"
	"time"

	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/stretchr/testify/assert"
)

// This is a repetition of TestNeuron_BinaryClassification in neuron_test.go
func TestLayer_Propagation(t *testing.T) {

	module := ml.New().Rate(10)

	layer := NewLayer(2, 2, Perceptron(module, xmath.Const(0.5)), 0)

	inp1 := []float64{0, 1}
	exp1 := []float64{1, 0}
	//
	inp2 := []float64{1, 0}
	exp2 := []float64{0, 1}

	iterations := 1000

	v1 := xmath.NewVector(2)
	v2 := xmath.NewVector(2)

	err := math.MaxFloat64
	var finishedAt int
	for i := 0; i < iterations; i++ {
		v1 = layer.forward(inp1)
		err1 := xmath.NewVector(2).From(exp1).Diff(v1)
		layer.backward(xmath.NewMatrix(2).From(err1))

		v2 = layer.forward(inp2)
		err2 := xmath.NewVector(2).From(exp2).Diff(v2)
		layer.backward(xmath.NewMatrix(2).From(err2))

		loss := err1.Add(err2).Norm()

		assert.True(t, loss < err)
		err = loss
		if err < 0.0001 && finishedAt == 0 {
			finishedAt = i
		}

	}

	for i, r := range exp1 {
		v := v1[i]
		assert.True(t, math.Abs(v-r) < 0.01)
	}

	for i, r := range exp2 {
		v := v2[i]
		assert.True(t, math.Abs(v-r) < 0.01)
	}

	assert.True(t, finishedAt > 0)
	assert.True(t, finishedAt < iterations)
}

// Note : this test might take a while ...
func TestLayer_RandomPropagationScenarios(t *testing.T) {

	for i := 0; i < 10; i++ {

		rand.Seed(time.Now().UnixNano())
		// generate inputs
		inp := xmath.NewMatrix(2).With(xmath.Rand(2), xmath.Rand(2))
		exp := xmath.NewMatrix(2).With(xmath.Rand(2), xmath.Rand(2))

		assertTraining(t, inp, exp)

	}

}

func assertTraining(t *testing.T, inp, exp xmath.Matrix) {

	log.Println(fmt.Sprintf("inp = %v", inp))
	log.Println(fmt.Sprintf("exp = %v", exp))

	layer := NewLayer(2, 2, Perceptron(ml.New(), xmath.Const(0.5)), 0)

	v := xmath.NewMatrix(len(inp))

	errThreshold := 0.001

	sumErr := 0.0
	var finishedAt int
	i := 0
	for {
		loss := xmath.NewVector(len(v))
		for j := 0; j < len(v); j++ {
			v[j] = layer.forward(inp[j])
			err := xmath.NewVector(2).From(exp[j]).Diff(v[j])
			layer.backward(xmath.NewMatrix(2).From(err))
			loss = loss.Add(err)
		}

		if loss.Norm() < errThreshold && finishedAt == 0 {
			finishedAt = i
			break
		}

		assert.True(t, sumErr-loss.Norm()/loss.Norm() < 0.01, fmt.Sprintf("%v : %v < %v for \n inp = %v exp = %v", i, loss.Norm(), sumErr, inp, exp))
		sumErr = loss.Norm()

		if i%10001 == 0 {
			log.Println(fmt.Sprintf("sumErr at %v = %v", i, sumErr))
		}

		i++
	}

	log.Println(fmt.Sprintf("v = %v", v))

	for i, e := range exp {
		for j, r := range e {
			v := v[i][j]
			lossThreshold := errThreshold * 10
			assert.True(t, math.Abs(v-r) < lossThreshold, fmt.Sprintf("%v >= %v for \n inp = %v exp = %v", math.Abs(v-r), lossThreshold, inp, exp))
		}

	}

	assert.True(t, finishedAt > 0)
}
