package activation

import "math"

type sigmoid struct {
}

func Sigmoid() Activation {
	return &sigmoid{}
}

func (s *sigmoid) Actual(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}
func (s *sigmoid) Derivative(sum, actual float64) float64 {
	return actual * (1 - actual)
}
