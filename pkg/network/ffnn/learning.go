package ffnn

func (net *Network) Train(input []float64, output []float64) {
	errs := make([]float64, len(net.Layers[len(net.Layers)-1].neurons))
	realOutput := net.Work(input)
	for i, o := range realOutput {
		errs[i] = output[i] - o
	}
	net.propagate(errs)
	net.update(input)
}

func (net *Network) update(in []float64) {
	for _, l := range net.Layers {
		in = l.update(in)
	}
}

func (net *Network) propagate(errs []float64) {
	outputLayer := net.Layers[len(net.Layers)-1]
	for i, err := range errs {
		outputLayer.neurons[i].currentStepErr = err
	}
	for i := len(net.Layers) - 2; i >= 0; i-- {
		net.Layers[i].propagate(net.Layers[i+1])
	}
}

func (l *Layer) propagate(prevL *Layer) {
	for i, n := range l.neurons {
		currentStepErr := 0.0
		for _, prevN := range prevL.neurons {
			currentStepErr += prevN.inputWeights[i] * prevN.currentStepErr
		}
		n.currentStepErr = currentStepErr
	}
}

func (l *Layer) update(in []float64) []float64 {
	out := make([]float64, len(l.neurons))
	for i, n := range l.neurons {
		out[i] = n.update(in, l.learnSpeed)
	}

	return out
}

func (n *Neuron) update(in []float64, learnSpeed float64) float64 {
	for i, weight := range n.inputWeights {
		n.inputWeights[i] = weight + n.currentStepErr*n.activation.Derivative(n.currentStepSum, n.currentStepActivationResult)*in[i]*learnSpeed
	}
	n.shiftNeuronWeight = n.shiftNeuronWeight + n.currentStepErr*n.activation.Derivative(n.currentStepSum, n.currentStepActivationResult)*learnSpeed
	return n.currentStepActivationResult
}
