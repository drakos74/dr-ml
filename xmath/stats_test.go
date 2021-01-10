package xmath

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
	"time"
)

func TestBucket_Push(t *testing.T) {

	var idx int64 = 1
	b := Bucket{
		index: idx,
	}

	rand.Seed(time.Now().UnixNano())

	var sum float64
	var squaredSum float64
	values := make([]float64, 100)
	for i := 0; i < 100; i++ {
		f := rand.Float64()
		b.Push(f, idx)
		sum += f
		values[i] = f
		squaredSum += math.Pow(f, 2)
	}

	mean := sum / 100

	assert.True(t, mean > 0)

	// check that the mean is calculated correctly
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

	var c int
	count := make(chan int)

	// run for 5 seconds ...
	go func() {
		i := int64(0)
		tick := time.NewTicker(5 * time.Second)
		for {
			select {
			case <-tick.C:
				close(count)
				return
			default:
				now := time.Now()
				v := float64(now.Unix())
				//println(fmt.Sprintf("v = %v", v))
				if b, ok := w.Push(v, now); ok {
					assert.True(t, b.stats.count > 0)
					count <- 1
					next := w.Next(1)
					assert.Equal(t, now.Add(time.Second).Unix(), next.Unix())
				}
				i++
			}
		}
	}()

	for i := range count {
		c += i
	}

	assert.Equal(t, 5, c)

}
