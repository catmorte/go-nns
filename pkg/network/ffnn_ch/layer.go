package ffnn_ch

import (
	"context"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"github.com/catmorte/go-nns/pkg/network/helpers/weightgen"
)

func aliveLayer(ctx context.Context,
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
		curLayerAxons[i] = aliveNeuron(ctx, dendrites, activation, weightGen)
	}
	return
}

type LayerBuilder func(ctx context.Context, prevLayersAxonsConstructors []AxonsConstructor) (curLayerAxons []AxonsConstructor)

func LayerConstructor(layerSize int, activation activation.Activation, weightGen weightgen.WeightGen) LayerBuilder {
	return func(ctx context.Context, prevLayersAxonsConstructors []AxonsConstructor) []AxonsConstructor {
		return aliveLayer(ctx, prevLayersAxonsConstructors, layerSize, activation, weightGen)
	}
}
