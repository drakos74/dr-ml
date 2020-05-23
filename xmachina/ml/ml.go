package ml

import (
	xmath "github.com/drakos74/go-ex-machina/xmachina/math"
	"math"
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
		Learning: ConstantRate{
			wrate: 1,
			brate: 0,
		},
		Descent: GradientDescent{},
	}
}

func (ml *LearningModule) Rate(wrate, brate float64) *LearningModule {
	ml.Learning = ConstantRate{wrate: wrate, brate: brate}
	return ml
}

func (lm *LearningModule) WithActivation(activation Activation) *LearningModule {
	lm.Activation = activation
	return lm
}

func (lm *LearningModule) WithRate(rate Learning) *LearningModule {
	lm.Learning = rate
	return lm
}

func (lm *LearningModule) WithDescent(descent Descent) *LearningModule {
	lm.Descent = descent
	return lm
}

var NoML = LearningModule{
	Activation: Void{},
	Learning:   Zero{},
	Descent:    Zero{},
}

type Loss func(expected, output xmath.Vector) xmath.Vector

var Diff Loss = func(expected, output xmath.Vector) xmath.Vector {
	return expected.Diff(output)
}

var Pow Loss = func(expected, output xmath.Vector) xmath.Vector {
	return expected.Diff(output).Pow(2).Mult(0.5)
}

var CrossEntropy Loss = func(expected, output xmath.Vector) xmath.Vector {
	return expected.Dop(func(x, y float64) float64 {
		return -1 * x * math.Log(y)
	}, output)
}
