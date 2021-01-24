package recurrent

import (
	"fmt"
	"math"
	"strings"

	"github.com/drakos74/go-ex-machina/xmath"

	"github.com/drakos74/go-ex-machina/xmachina"
	"github.com/drakos74/go-ex-machina/xmachina/net"
)

const (
	base = "base"
)

type T func(i int) float64

type P func(i int, x float64) float64

func X(f float64) T {
	return func(i int) float64 {
		return f * float64(i)
	}
}

var Sine P = func(_ int, x float64) float64 {
	return math.Sin(x)
}

var SineVar P = func(i int, x float64) float64 {
	return 0.3*math.Sin(x) + 0.3*math.Sin(2*x) + 0.3*math.Sin(3*x)
}

type Capture interface {
	Train(i int, x, y float64) float64
	Predict(i int, x, y float64) float64
}

type OutputCapture struct {
	lastOutput float64
	Network    net.NN
}

func (c *OutputCapture) Train(i int, x, y float64) float64 {
	next := xmath.Vec(1)
	c.Network.Train(xmath.Vec(1).With(y), next)
	c.lastOutput = next[0]
	return c.lastOutput
}

func (c *OutputCapture) Predict(i int, x, y float64) float64 {
	next := c.Network.Predict(xmath.Vec(1).With(y))
	c.lastOutput = next[0]
	return c.lastOutput
}

type EvolutionCapture struct {
	OutputCapture
}

func (c *EvolutionCapture) Predict(i int, x, y float64) float64 {
	next := c.Network.Predict(xmath.Vec(1).With(c.lastOutput))
	c.lastOutput = next[0]
	return c.lastOutput
}

type OnceCapture struct {
	OutputCapture
}

func (c *OnceCapture) Train(i int, x, y float64) float64 {
	c.OutputCapture.Train(i, x, y)
	if c.lastOutput == 0 {
		return y
	}
	return 0
}

func (c *OnceCapture) Predict(i int, x, y float64) float64 {
	return 0
}

type SoftCapture struct {
	lastOutput float64
	Network    net.NN
}

func eval(vector xmath.Vector) float64 {
	var out float64
	threshold := 0.05
	if vector[0]-vector[1] > threshold {
		out = 1
	} else if vector[1]-vector[0] > threshold {
		out = -1
	}
	return out
}

func (s SoftCapture) Train(i int, x, y float64) float64 {
	next := xmath.Vec(2)
	s.Network.Train(xmath.Vec(1).With(y), next)
	s.lastOutput = eval(next)
	return s.lastOutput
}

func (s SoftCapture) Predict(i int, x, y float64) float64 {
	next := s.Network.Predict(xmath.Vec(1).With(y))
	s.lastOutput = eval(next)
	return s.lastOutput
}

type DummyCapture struct {
	OutputCapture
}

func (c *DummyCapture) Train(i int, x, y float64) float64 {
	return c.OutputCapture.Predict(i, x, y)
}

type VoidNetwork struct {
	net.NetworkConfig
}

func (v VoidNetwork) Train(input xmath.Vector, output xmath.Vector) (err xmath.Vector, weights map[net.Meta]net.Weights) {
	for i, inp := range input {
		output[i] = inp
	}
	return xmath.Vec(len(input)), nil
}

func (v VoidNetwork) Predict(input xmath.Vector) xmath.Vector {
	return input
}

func Train(l int, xt T, yp P, graph xmachina.Data, cap map[string]Capture) {

	order := make([]string, len(cap))
	// order the caps
	for name, _ := range cap {
		order = append(order, name)
	}

	for i := 0; i < l; i++ {

		x := xt(i)
		y := yp(i, x)
		report := strings.Builder{}
		if i < l*4/5 {
			report.WriteString(fmt.Sprintf("train %d |", i))
			for _, name := range order {
				if c, ok := cap[name]; ok {
					v := c.Train(i, x, y)
					report.WriteString(fmt.Sprintf("%s:v = %+v |", name, v))
					graph.Add(name, x, v)
				}
			}
		} else {
			report.WriteString(fmt.Sprintf("predict %d |", i))
			for _, name := range order {
				if c, ok := cap[name]; ok {
					v := c.Predict(i, x, y)
					report.WriteString(fmt.Sprintf("%s:p = %+v |", name, v))
					graph.Add(name, x, v)
				}
			}
		}
		println(fmt.Sprintf("report.String = %+v", report.String()))

	}

}
