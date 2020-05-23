package ffnn_ch

import (
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"github.com/catmorte/go-nns/pkg/network/helpers/weightgen"
)

type LayerBuilder func(prevLayersAxonsConstructors []AxonsConstructor) (curLayerAxons []AxonsConstructor)

func LayerConstructor(layerSize int, activation activation.Activation, weightGen weightgen.WeightGen) LayerBuilder {
	return func(prevLayersAxonsConstructors []AxonsConstructor) []AxonsConstructor {
		return aliveLayer(prevLayersAxonsConstructors, layerSize, activation, weightGen)
	}
}

func aliveLayer(
	prevLayersAxonsConstructors []AxonsConstructor,
	curLayerSize int,
	activation activation.Activation,
	weightGen weightgen.WeightGen) (curLayerAxons []AxonsConstructor) {

	prevLayerAxonsLength := len(prevLayersAxonsConstructors)
	curLayerDendrites := make([][]chan float64, prevLayerAxonsLength)
	curLayerAxons = make([]AxonsConstructor, curLayerSize)

	for i := 0; i < prevLayerAxonsLength; i++ {
		curLayerDendrites[i] = prevLayersAxonsConstructors[i](curLayerSize)
	}
	for i := 0; i < curLayerSize; i++ {
		dendrites := make([]chan float64, prevLayerAxonsLength)
		for j := 0; j < prevLayerAxonsLength; j++ {
			dendrites[j] = curLayerDendrites[j][i]
		}
		curLayerAxons[i] = aliveNeuron(dendrites, activation, weightGen)
	}
	return
}
