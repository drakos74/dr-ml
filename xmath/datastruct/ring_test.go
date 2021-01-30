package datastruct

import (
	"fmt"
	"testing"
)

func TestNewRing_Push(t *testing.T) {

	ring := NewRing(10)

	for i := 0; i < 102; i++ {
		ring.Push(float64(i))
		if ring.Size() != i+1 {
			t.Fatalf("unexpected size %d vs. %d", ring.Size(), i)
			return
		}
		println(fmt.Sprintf("ring.values = %v", ring.values))
		println(fmt.Sprintf("ring.Get() = %v", ring.Get()))
		println(fmt.Sprintf("ring.Aggregate(Sum) = %v", ring.Aggregate(Sum)))
	}

}
