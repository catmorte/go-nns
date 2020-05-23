package processing

type softMax struct {
}

func SoftMax() Processing {
	return &softMax{}
}

func (s *softMax) Process(in []float64) []float64 {
	output := make([]float64, len(in))
	sum := float64(0)
	for _, item := range in {
		sum += item
	}
	for i, item := range in {
		output[i] = item / sum
	}
	return output
}
