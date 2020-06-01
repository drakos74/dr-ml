package ml

type Op func(x float64) float64

type Learning interface {
	WRate() float64
	BRate() float64
}

func (z Zero) WRate() float64 {
	return 0
}

func (z Zero) BRate() float64 {
	return 0
}

type LearningRate struct {
	wrate float64
	brate float64
}

func Learn(rate float64) Learning {
	return LearningRate{wrate: rate, brate: rate}
}

func (c LearningRate) WRate() float64 {
	return c.wrate
}

func (c LearningRate) BRate() float64 {
	return c.brate
}
