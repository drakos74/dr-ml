// Package Oremi implements a data visualisation/exporting logic by usig oremi project.
// It encapsulates all dependencies regarding oremi , so that we can run the project without this dependency.
package oremi

import (
	"github.com/drakos74/go-ex-machina/xmachina"
	"github.com/drakos74/oremi/graph"
)

type DataSet struct {
	name       string
	collection *graph.RawCollection
}

func New(name string) *DataSet {
	return &DataSet{
		name: name,
	}
}

func (d *DataSet) Init(sets ...xmachina.Set) xmachina.Data {
	plot := graph.New(d.name)
	for _, set := range sets {
		plot.NewSeries(set.Name, set.X, set.Y)
	}
	d.collection = plot
	return d
}

func (d *DataSet) Add(name string, v ...float64) {
	d.collection.Add(name, v...)
}

func (d *DataSet) Export(index string) error {
	graph.Draw(index, d.collection)
	return nil
}
