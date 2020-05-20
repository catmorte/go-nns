package ffnn

func (net *Network) Work(input []float64) []float64 {
	for _, l := range net.Layers {
		input = l.calculate(input)
	}
	return input
}

func (l *Layer) calculate(input []float64) []float64 {
	nLength := len(l.neurons)
	output := make([]float64, nLength)
	for i, n := range l.neurons {
		output[i] = n.calculate(input)
	}
	return output
}

func (n *Neuron) calculate(input []float64) float64 {
	sum := n.shiftNeuronWeight
	for i, weight := range n.inputWeights {
		sum += weight * input[i]
	}
	n.currentStepSum = sum
	n.currentStepActivationResult = n.activation.Actual(sum)
	return n.currentStepActivationResult
}
