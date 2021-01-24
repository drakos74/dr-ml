package xmachina

type Data interface {
	Init(sets ...Set) Data
	Add(name string, v ...float64)
	Export(index string) error
}

type Set struct {
	Name string
	X, Y string
}

type NoSet struct {
}

func VoidSet() *NoSet {
	return &NoSet{}
}

func (n *NoSet) Init(sets ...Set) Data {
	return n
}

func (n *NoSet) Add(name string, v ...float64) {
	// nothing to do
}

func (n *NoSet) Export(index string) error {
	return nil
}
