package xmath

type Evolution struct {
	i            int
	combinations [][]float64
	procedures   []*Sequence
}

func NewEvolution(procedures ...*Sequence) *Evolution {

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

// Limit returns the amount of iterations defined in this evolution
func (e *Evolution) Limit() int {
	return len(e.combinations)
}

// Current returns the current state of the evolution
func (e *Evolution) Current() int {
	return e.i
}

// Next updates the procedures with the next value
// it returns true if there was an update and false if there is nothing more to evolve.
func (e *Evolution) Next() bool {
	if e.i >= len(e.combinations) {
		return false
	}
	for i, value := range e.combinations[e.i] {
		e.procedures[i].set(value)
	}
	e.i++
	return true
}

type Sequence struct {
	initialValue float64
	value        *float64
	limit        int
	count        int
	Transform
}

func PerturbationSequence(value *float64, step float64, limit, rounding int) *Sequence {
	transform := IncNum(step, rounding)
	initialValue := *value - (step * float64(limit) / 2)
	s := &Sequence{
		initialValue: initialValue,
		value:        value,
		limit:        limit,
		Transform:    transform,
	}
	s.set(initialValue)
	return s
}

func RangeSequence(value *float64, start, end float64, limit, rounding int) *Sequence {
	step := (end - start) / float64(limit)
	transform := IncNum(step, rounding)

	s := &Sequence{
		initialValue: start,
		value:        value,
		limit:        limit,
		Transform:    transform,
	}
	s.set(start)
	return s
}

func NewSequence(value *float64, transform Transform, limit int) *Sequence {
	return &Sequence{
		initialValue: *value,
		value:        value,
		limit:        limit,
		Transform:    transform,
	}
}

func (p *Sequence) Next() bool {
	p.count++
	if p.count > 1 {
		newValue := p.Transform(*p.value)
		p.set(newValue)
	}
	return p.count >= p.limit
}

func (p *Sequence) set(newValue float64) {
	*p.value = newValue
}

func (p *Sequence) Reset() {
	*p.value = p.initialValue
	p.count = 0
}

func (p *Sequence) Run() []float64 {
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

func IncNum(w float64, rounding int) Transform {
	return func(v float64) float64 {
		return Round(rounding)(v + w)
	}
}

func IncMul(w float64, rounding int) Transform {
	return func(v float64) float64 {
		return Round(rounding)(v * w)
	}
}
