package ffnn

import (
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
)

type (
	Network struct {
		inputSize int
		Layers    []*Layer
	}

	Layer struct {
		neurons    []*Neuron
		learnSpeed float64
	}

	Neuron struct {
		activation activation.Activation

		inputWeights      []float64 // length depends on Network input length(if it's first Layer) or from size of previous Layer
		shiftNeuronWeight float64

		currentStepSum              float64
		currentStepErr              float64
		currentStepActivationResult float64
	}
)
