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

type ConstantRate struct {
	wrate float64
	brate float64
}

func (c ConstantRate) WRate() float64 {
	return c.wrate
}

func (c ConstantRate) BRate() float64 {
	return c.brate
}
