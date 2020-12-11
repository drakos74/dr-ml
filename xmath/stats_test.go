package xmath

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestBucket_Push(t *testing.T) {

	b := Bucket{}

	rand.Seed(time.Now().UnixNano())

	var sum float64
	var squaredSum float64
	values := make([]float64, 100)
	for i := 0; i < 100; i++ {
		f := rand.Float64()
		b.Push(f, 1)
		sum += f
		values[i] = f
		squaredSum += math.Pow(f, 2)
	}

	mean := sum / 100

	assert.True(t, mean > 0)

	assert.Equal(t, strconv.FormatFloat(mean, 'f', 5, 64), strconv.FormatFloat(b.stats.mean, 'f', 5, 64))

	var sqrtSum float64
	for _, v := range values {
		sqrtSum += math.Pow(mean-v, 2)
	}

	assert.Equal(t, strconv.FormatFloat(sqrtSum, 'f', 5, 64), strconv.FormatFloat(b.stats.dSquared, 'f', 5, 64))
}

func TestWindow_Push(t *testing.T) {

	w := NewWindow(5)

	sum := 0
	for i := 0; i < 100; i++ {
		index := int64(i)
		v := i * 10
		_, ready := w.Push(index, float64(v))
		// compute avg manually

		if ready {
			bucket := w.Get()
			avg := bucket.Stats().mean

			assert.Equal(t, 5, bucket.Size())
			assert.Equal(t, float64(sum)/5, avg)

			// last value was added to the next bucket
			sum = v
		} else {
			// increase the sum manually
			sum += v
		}
	}

}

func TestTimeWindow_Push(t *testing.T) {

	w := NewTimeWindow(time.Second)

	c := 0

	// run for 5 seconds ...
	ticker := time.NewTicker(5 * time.Second)
	go func() {
		i := int64(0)

		for {
			now := time.Now()
			v := float64(now.Unix())
			//println(fmt.Sprintf("v = %v", v))
			if b, ok := w.Push(v, now); ok {
				println(fmt.Sprintf("b = %v", b))
				c++
				next := w.Next(1)
				println(fmt.Sprintf("next = %v", next))
				println(fmt.Sprintf("now = %v", now))
			}
			i++
		}
	}()

	<-ticker.C

	println(fmt.Sprintf("w = %+v", w))
	assert.Equal(t, 5, c)

}
