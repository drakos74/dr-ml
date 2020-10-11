package xmath

type Procedure struct {
	value *float64
	Transform
}

func NewProcedure(value *float64, transform Transform) *Procedure {
	return &Procedure{value: value, Transform: transform}
}

func (p *Procedure) Next() {
	newValue := p.Transform(*p.value)
	*p.value = newValue
}

type Transform func(v float64) float64

func IncNum(w float64) Transform {
	return func(v float64) float64 {
		return v + w
	}
}

func IncMul(w float64) Transform {
	return func(v float64) float64 {
		return v * w
	}
}
