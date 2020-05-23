package processing

type Processing interface {
	Process([]float64) []float64
}
