package xmath

type Evolution struct {
	i, j         int
	combinations [][]float64
	procedures   []*Procedure
}

func NewEvolution(procedures ...*Procedure) *Evolution {

	// build the parameter space
	parameters := make([][]float64, len(procedures))

	for i, proc := range procedures {
		parameters[i] = proc.Run()
	}

	combinations := CartesianProduct(parameters, 0, len(parameters))

	return &Evolution{
		combinations: combinations,
		procedures:   procedures,
	}

}

func (e *Evolution) Next() bool {
	for i, value := range e.combinations[e.i] {
		e.procedures[i].set(value)
	}
	e.i++
	return e.i >= len(e.combinations)
}

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
	p.set(newValue)
	return p.count >= p.limit
}

func (p *Procedure) set(newValue float64) {
	*p.value = newValue
}

func (p *Procedure) Reset() {
	*p.value = p.initialValue
	p.count = 0
}

func (p *Procedure) Run() []float64 {
	values := make([]float64, 0)
	var done bool
	for !done {
		done = p.Next()
		values = append(values, *p.value)
		if done {
			p.Reset()
		}
	}
	return values
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
