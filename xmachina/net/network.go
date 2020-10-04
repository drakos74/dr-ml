package net

import (
	"github.com/drakos74/go-ex-machina/xmachina/ml"
	"github.com/drakos74/go-ex-machina/xmath"
)

type NN interface {
	Train(input xmath.Vector, output xmath.Vector) (err xmath.Vector, weights xmath.Cube)
	Predict(input xmath.Vector) xmath.Vector
	Loss(loss ml.Loss)
}

type Info struct {
	Init       bool
	InputSize  int
	OutputSize int
	Iterations int
}

type Config struct {
	trace bool
	debug bool
}

func (cfg *Config) Debug() {
	cfg.debug = true
}

func (cfg *Config) Trace() {
	cfg.trace = true
}

func (cfg *Config) HasDebugEnabled() bool {
	return cfg.debug
}

func (cfg *Config) HasTraceEnabled() bool {
	return cfg.trace
}

type Stats struct {
	Iteration int
	xmath.Bucket
}
