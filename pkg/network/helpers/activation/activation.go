package activation

type Activation interface {
	Actual(sum float64) float64
	Derivative(sum, actualResult float64) float64
}
