package ml

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

func New() *LearningModule {
	return &LearningModule{
		Activation: Sigmoid,
		Learning: ConstantRate{
			rate: 1,
		},
		Descent: GradientDescent{},
	}
}

func (ml *LearningModule) Rate(rate float64) *LearningModule {
	ml.Learning = ConstantRate{rate: rate}
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
