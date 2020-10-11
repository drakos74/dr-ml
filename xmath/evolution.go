package xmath

type Procedure struct {
	initialValue float64
	value        *float64
	limit        int
	count        int
	Transform
}

func NewProcedure(value *float64, transform Transform, limit int) *Procedure {
	return &Procedure{
		initialValue: *value,
		value:        value,
		limit:        limit,
		Transform:    transform,
	}
}

func (p *Procedure) Next() bool {
	p.count++
	newValue := p.Transform(*p.value)
	*p.value = newValue
	return p.count >= p.limit
}

func (p *Procedure) Reset() {
	*p.value = p.initialValue
	p.count = 0
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
