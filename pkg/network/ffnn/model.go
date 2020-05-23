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

		weights     []float64 // length depends on Network input length(if it's first Layer) or from size of previous Layer
		weightShift float64

		sum              float64
		err              float64
		activationResult float64
	}
)
