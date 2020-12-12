package ml

import (
	"fmt"
	"math"

	"github.com/drakos74/go-ex-machina/xmath"
)

type Module interface {
	Activation
	Learning
	Descent
}

type LearningModule struct {
	Activation
	Learning
	Descent
}

func Model() *LearningModule {
	return &LearningModule{
		Activation: Sigmoid,
		Learning: &LearningRate{
			wrate: 1,
			brate: 0,
		},
		Descent: GradientDescent{},
	}
}

func (ml *LearningModule) Rate(wRate, bRate float64) *LearningModule {
	ml.Learning = &LearningRate{wrate: wRate, brate: bRate}
	return ml
}

func (ml *LearningModule) WithActivation(activation Activation) *LearningModule {
	ml.Activation = activation
	return ml
}

func (ml *LearningModule) WithRate(rate Learning) *LearningModule {
	ml.Learning = rate
	return ml
}

func (ml *LearningModule) WithDescent(descent Descent) *LearningModule {
	ml.Descent = descent
	return ml
}

var NoML = LearningModule{
	Activation: Void{},
	Learning:   Zero{},
	Descent:    Zero{},
}

type Loss func(expected, output xmath.Vector) xmath.Vector

type MLoss func(expected, output xmath.Matrix) xmath.Vector

var Diff Loss = func(expected, output xmath.Vector) xmath.Vector {
	return expected.Diff(output)
}

var Pow Loss = func(expected, output xmath.Vector) xmath.Vector {
	return expected.Diff(output).Pow(2).Mult(0.5)
}

var CrossEntropy Loss = func(expected, output xmath.Vector) xmath.Vector {
	xmath.MustHaveSameSize(expected, output)
	return expected.Dop(func(x, y float64) float64 {
		if y == 1 {
			panic(fmt.Sprintf("cross entropy calculation threshold breached for output %v", y))
		}
		return -1 * x * math.Log(y)
	}, output)
}

func CompLoss(mloss Loss) MLoss {
	return func(expected, output xmath.Matrix) xmath.Vector {
		size := len(expected)
		xmath.MustHaveDim(expected, size)
		xmath.MustHaveDim(output, size)
		loss := xmath.Vec(len(expected))
		for i := 0; i < size; i++ {
			xmath.MustHaveSameSize(expected[i], output[i])
			entropy := mloss(expected[i], output[i])
			loss[i] = entropy.Sum() / float64(size)
		}
		return loss
	}
}
