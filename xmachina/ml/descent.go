package ml

type Descent interface {
	Grad(err, deriv float64) float64
}

type GradientDescent struct {
}

func (g GradientDescent) Grad(err, deriv float64) float64 {
	return err * deriv
}

type Zero struct {
}

func (z Zero) Grad(err, deriv float64) float64 {
	return 0
}
