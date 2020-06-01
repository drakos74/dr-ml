package xmachina

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"sync/atomic"

	"github.com/drakos74/go-ex-machina/xmath"
)

func ReadFile(filename string, lines int, iterations int, parse func(record []string) (inp, out xmath.Vector), data Data, epoch Epoch, ack Ack) (inputSet, outputSet xmath.Matrix, readErr error) {

	// TODO: fix this ugliness with the line parameter
	if lines > 0 {
		inputSet = xmath.Mat(lines)
		outputSet = xmath.Mat(lines)
	}

	for i := 0; i < iterations; i++ {

		var c int32

		testFile, err := os.Open(filename)
		if err != nil {
			return inputSet, outputSet, err
		}
		rTrain := csv.NewReader(bufio.NewReader(testFile))

		for {

			record, err := rTrain.Read()

			if err == io.EOF {
				break
			}

			inp, out := parse(record)

			if lines > 0 {
				inputSet[c] = inp
				outputSet[c] = out
			}

			atomic.AddInt32(&c, 1)

			data <- Pair{
				input:  inp,
				output: out,
			}

		}

		readErr = testFile.Close()
		if readErr != nil {
			return inputSet, outputSet, fmt.Errorf("could not close file: %w", readErr)
		}

		epoch <- i

		// wait for the acknowledgement
		err = <-ack
		log.Println(fmt.Sprintf("err = %v", err))
		if err == nil {
			break
		}

	}

	return inputSet, outputSet, nil

}
