package ff

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/drakos74/go-ex-machina/xmachina/net"
	"github.com/rs/zerolog"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/stretchr/testify/assert"
)

func init() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
}

// This is a repetition of TestNeuron_BinaryClassification in neuron_test.go
func TestLayer_LeaningProcess(t *testing.T) {

	module := ml.Base().WithRate(ml.Learn(0.5, 0))

	layer := NewLayer(2, 2,
		net.NewBuilder().
			WithModule(module).
			WithWeights(xmath.Const(0.5), xmath.Const(0.5)).
			Factory(net.NewActivationCell), 0)

	inp1 := []float64{0, 1}
	exp1 := []float64{1, 0}
	//
	inp2 := []float64{1, 0}
	exp2 := []float64{0, 1}

	iterations := 10000

	v1 := xmath.Vec(2)
	v2 := xmath.Vec(2)

	err := math.MaxFloat64
	var finishedAt int
	for i := 0; i < iterations; i++ {

		v1 = layer.Forward(inp1)
		err1 := xmath.Vec(2).With(exp1...).Diff(v1)
		layer.Backward(err1)

		v2 = layer.Forward(inp2)
		err2 := xmath.Vec(2).With(exp2...).Diff(v2)
		layer.Backward(err2)

		loss := err1.Add(err2).Norm()

		// check that performance improves
		assert.True(t, loss <= err || math.Abs(loss-err) < 0.001, fmt.Sprintf("i:%d loss = %v > err = %v", i, loss, err))
		err = loss
		if err < 0.0001 && finishedAt == 0 {
			finishedAt = i
		}

	}

	for i, r := range exp1 {
		v := v1[i]
		assert.True(t, math.Abs(v-r) < 0.1, fmt.Sprintf("v = %v vs r = %v", v, r))
	}

	for i, r := range exp2 {
		v := v2[i]
		assert.True(t, math.Abs(v-r) < 0.1)
	}

	assert.True(t, finishedAt > 0)
	assert.True(t, finishedAt < iterations)
}

// Note : this test might take a while ... because we might hit a 'bad' initial weight randomisation
// but ... it should always eventually complete
func TestLayer_RandomLearningProcessScenarios(t *testing.T) {

	for i := 0; i < 10; i++ {

		rand.Seed(time.Now().UnixNano())
		// generate inputs
		inp := xmath.Mat(2).With(xmath.Rand(0, 1, xmath.Unit)(2, 0), xmath.Rand(0, 1, xmath.Unit)(2, 1))
		exp := xmath.Mat(2).With(xmath.Rand(0, 1, xmath.Unit)(2, 0), xmath.Rand(0, 1, xmath.Unit)(2, 0))

		assertTraining(t, inp, exp)

	}

}

func assertTraining(t *testing.T, inp, exp xmath.Matrix) {

	layer := NewLayer(2, 2,
		net.NewBuilder().
			WithModule(ml.Base()).
			WithWeights(xmath.Const(0.5), xmath.Const(0.5)).
			Factory(net.NewActivationCell), 0)

	v := xmath.Mat(len(inp))

	errThreshold := 0.0001

	sumErr := 0.0
	var finishedAt int
	i := 0
	for {
		loss := xmath.Vec(len(v))
		for j := 0; j < len(v); j++ {
			v[j] = layer.Forward(inp[j])
			err := xmath.Vec(2).With(exp[j]...).Diff(v[j])
			layer.Backward(err)
			loss = loss.Add(err)
		}

		if loss.Norm() < errThreshold && finishedAt == 0 {
			finishedAt = i
			break
		}

		sumErr = loss.Norm()

		if i%10001 == 0 {
			log.Println(fmt.Sprintf("sumErr at %v = %v", i, sumErr))
		}

		i++
	}

	for i, e := range exp {
		for j, r := range e {
			v := v[i][j]
			lossThreshold := errThreshold * 100
			assert.True(t, math.Abs(v-r) < lossThreshold, fmt.Sprintf("i:%d %v >= %v for \n inp = %v exp = %v", i, math.Abs(v-r), lossThreshold, inp, exp))
		}

	}

	assert.True(t, finishedAt > 0)
}

func TestFFLayer_WithNoLearning(t *testing.T) {

	layer := NewLayer(2, 2,
		net.NewBuilder().
			WithModule(ml.Base().WithRate(ml.Learn(0, 0))).
			WithWeights(xmath.Const(0.5), xmath.Const(0.5)).
			Factory(net.NewActivationCell), 0)

	inp := []float64{0.3, 0.7}

	out := xmath.Vec(2).With(1, 0)

	// do one pass to check the output and error
	output := layer.Forward(inp)
	diff := out.Diff(output)
	err := layer.Backward(diff)

	for i := 0; i < 10; i++ {
		outp := layer.Forward(inp)

		assert.Equal(t, output, outp)

		hErr := layer.Backward(diff)

		assert.Equal(t, err, hErr)

		output = outp

	}
}
