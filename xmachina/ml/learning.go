package ml

type Op func(x float64) float64

type Learning interface {
	Get() float64
}

func (z Zero) Get() float64 {
	return 0
}

type ConstantRate struct {
	rate float64
}

func (c ConstantRate) Get() float64 {
	return c.rate
}
