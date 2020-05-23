package ffnn_ch

import (
	"context"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"github.com/catmorte/go-nns/pkg/network/helpers/weightgen"
)

func aliveNeuron(ctx context.Context, dendrites []chan float64, activation activation.Activation, weightGen weightgen.WeightGen) <-chan float64 {
	dendritesAmount := len(dendrites)
	soma := make(chan float64, dendritesAmount)
	axon := make(chan float64)
	weights := make([]float64, dendritesAmount)
	for i := range dendrites {
		weights[i] = weightGen()
		go func(dendrite <-chan float64, dendriteIndex int) {
			for {
				select {
				case <-ctx.Done():
					return
				case val := <-dendrite:
					soma <- weights[dendriteIndex] * val
				}
			}
		}(dendrites[i], i)
	}
	go func() {
		wShift := weightGen()
		for {
			sum := wShift
			for i := 0; i < dendritesAmount; i++ {
				select {
				case <-ctx.Done():
					return
				case val := <-soma:
					sum = sum + val
					axon <- activation.Actual(sum)
				}
			}
		}
	}()
	return axon
}

func aliveLayer(ctx context.Context, inputs []<-chan float64, layerSize int, activation activation.Activation, weightGen weightgen.WeightGen) (neuronsOutputs []<-chan float64) {
	inputsLength := len(inputs)
	neuronsInputs := make([][]chan float64, layerSize)
	neuronsOutputs = make([]<-chan float64, layerSize)

	for i := 0; i < layerSize; i++ {
		neuronInputs := make([]chan float64, inputsLength)
		for i := 0; i < inputsLength; i++ {
			neuronInputs[i] = make(chan float64)
		}
		neuronsInputs[i] = neuronInputs
		neuronsOutputs[i] = aliveNeuron(ctx, neuronInputs, activation, weightGen)
	}

	for i := 0; i < inputsLength; i++ {
		go func(input <-chan float64, inputIndex int) {
			for {
				select {
				case <-ctx.Done():
					return
				case val := <-input:
					for k := 0; k < layerSize; k++ {
						neuronsInputs[k][inputIndex] <- val
					}
				}
			}
		}(inputs[i], i)
	}
	return
}

type LayerBuilder func(context.Context, []<-chan float64) []<-chan float64

func LayerConstructor(layerSize int, activation activation.Activation, weightGen weightgen.WeightGen) LayerBuilder {
	return func(ctx context.Context, inputs []<-chan float64) []<-chan float64 {
		return aliveLayer(ctx, inputs, layerSize, activation, weightGen)
	}
}

func AliveNetwork(ctx context.Context, inputs []<-chan float64, layerMakers ...LayerBuilder) (outputs []<-chan float64) {
	for _, layerMaker := range layerMakers {
		outputs = layerMaker(ctx, inputs)
		inputs = outputs
	}
	return
}
