package xmachina

import (
	"context"
	"fmt"
	"log"
	"math"

	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
	"github.com/drakos74/go-ex-machina/xmachina/net"
)

type Pair struct {
	input  xmath.Vector
	output xmath.Vector
}

type Ack chan error

type Data chan Pair

type Epoch chan int

type InMemTraining struct {
	lossThreshold    float64
	epochs           int
	epochLogInterval int
	debug            bool
}

func Training(threshold float64, epochLogInterval int) InMemTraining {
	return InMemTraining{
		lossThreshold:    threshold,
		epochLogInterval: epochLogInterval,
		debug:            true,
	}
}

func (t *InMemTraining) init() InMemTraining {
	if t.epochs == 0 && t.lossThreshold == 0 {
		panic("cannot train network without epochs or loss threshold")
	}

	if t.epochLogInterval == 0 {
		t.epochLogInterval = math.MaxInt16
	}

	if t.epochs == 0 && t.lossThreshold > 0 {
		// if there are no epochs we will train until we converge ... or until a predefined limit
		t.epochs = math.MaxInt32
	}
	return *t
}

type InStreamTraining struct {
	InMemTraining
	Epoch              Epoch
	inputCountInterval int
	outputSize         int
}

func StreamingTraining(cfg InMemTraining, outputSize int, inputLogInterval int) InStreamTraining {
	return InStreamTraining{
		InMemTraining:      cfg,
		Epoch:              make(Epoch),
		outputSize:         outputSize,
		inputCountInterval: inputLogInterval,
	}
}

func (cfg *InStreamTraining) init() {
	cfg.InMemTraining = cfg.InMemTraining.init()
	if cfg.outputSize == 0 {
		panic("cannot init streaming training without an predefined output size")
	}
}

func TrainInMem(config InMemTraining, network net.NN, inputSet xmath.Matrix, outputSet xmath.Matrix) {

	config = config.init()

	loss := math.MaxFloat64

	for epoch := 0; epoch < config.epochs; epoch++ {
		sumErr := xmath.Vec(len(outputSet[0]))
		var finalWeights xmath.Cube
		for i, input := range inputSet {
			err, weights := network.Train(input, outputSet[i])
			sumErr = sumErr.Add(err)
			finalWeights = weights
		}

		// log the iteration performance for monitoring
		if config.debug && epoch%config.epochLogInterval == 0 {
			score := loss - sumErr.Norm()
			log.Println(fmt.Sprintf("Epoch = %v , error = %v , learningScore = %v , weights = %v ", epoch, sumErr.Norm(), score, finalWeights))
		}

		loss = sumErr.Norm()

		if sumErr.Norm() < config.lossThreshold {
			log.Println(fmt.Sprintf("Epoch = %v ,error => %v < %v , weights = %v ", epoch, sumErr.Norm(), config.lossThreshold, finalWeights))
			return
		}

	}
}

func TrainInStream(ctx context.Context, config InStreamTraining, network net.NN, data Data, ack Ack) {

	defer close(ack)

	config.init()

	sumErr := xmath.Vec(config.outputSize)
	e := 0
	score := 0.0
	i := 0
	var finalWeights xmath.Cube

	loss := math.MaxFloat64

	for {
		select {
		case <-ctx.Done():
			log.Println(fmt.Sprintf("Epoch = %v , error = %v , learningScore = %v , weights = %v. Stopped!", e, sumErr.Norm(), score, finalWeights))
			return
		case pair := <-data:
			i++
			err, weights := network.Train(pair.input, pair.output)
			sumErr = sumErr.Add(err.Op(math.Abs))
			finalWeights = weights
			if config.debug && i%config.inputCountInterval == 0 {
				log.Println(fmt.Sprintf("e = %v , i = %v , err = %v", e, i, sumErr.Norm()))
			}
		case <-config.Epoch:
			e++
			// log the iteration performance for monitoring
			if config.debug && e%config.epochLogInterval == 0 {
				score = loss - sumErr.Norm()
				log.Println(fmt.Sprintf("Epoch = %v , error = %v , learningScore = %v , weights = %v ... ", e, sumErr.Norm(), score, finalWeights))
			}

			loss = sumErr.Norm()

			err := fmt.Errorf(fmt.Sprintf("Epoch = %v ,error => %v < %v , weights = %v.", e, sumErr.Norm(), config.lossThreshold, finalWeights))

			if sumErr.Norm() < config.lossThreshold {
				err = nil
			}

			ack <- err

			// reset the error
			sumErr = xmath.Vec(config.outputSize)
			i = 0
		}
	}

}
