package ffnn

func (net *Network) Work(input []float64) []float64 {
	for _, l := range net.Layers {
		input = l.calculate(input)
	}
	return input
}

func (l *Layer) calculate(input []float64) []float64 {
	neuronsAmounts := len(l.neurons)
	output := make([]float64, neuronsAmounts)
	for i, n := range l.neurons {
		output[i] = n.calculate(input)
	}
	return output
}

func (n *Neuron) calculate(input []float64) float64 {
	sum := n.weightShift
	for i, weight := range n.weights {
		sum += weight * input[i]
	}
	n.sum = sum
	n.activationResult = n.activation.Actual(sum)
	return n.activationResult
}
