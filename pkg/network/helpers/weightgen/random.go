package weightgen

import "math/rand"

func RandomWithin(min, max float64) func()float64 {
	return func() float64{
		return min + rand.Float64()*(max-min)
	}
}
