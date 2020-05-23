package ffnn_ch

import (
	"context"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"github.com/catmorte/go-nns/pkg/network/helpers/weightgen"
)

type AxonsConstructor func(int) []chan float64

func aliveNeuron(ctx context.Context,
	dendrites []chan float64,
	activation activation.Activation,
	weightGen weightgen.WeightGen) AxonsConstructor {
	dendritesAmount := len(dendrites)
	soma := make(chan float64, dendritesAmount)

	initialized := make(chan interface{})

	var broadcastAxon chan float64
	var axons []chan float64

	for i := range dendrites {
		go func(dendrite <-chan float64, dendriteIndex int) {
			w := weightGen()
			for {
				select {
				case <-ctx.Done():
					return
				case val := <-dendrite:
					soma <- w * val
				}
			}
		}(dendrites[i], i)
	}

	go func() {
		<-initialized
		wShift := weightGen()
		for {
			sum := wShift
			for i := 0; i < dendritesAmount; i++ {
				select {
				case <-ctx.Done():
					return
				case val := <-soma:
					sum = sum + val
				}
			}
			activationResult := activation.Actual(sum)
			broadcastAxon <- activationResult
		}
	}()

	return func(outputSize int) []chan float64 {
		broadcastAxon = make(chan float64, outputSize)
		axons = broadcastTo(ctx, broadcastAxon, outputSize)
		close(initialized)
		return axons
	}
}

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

func AliveNetwork(ctx context.Context, inputs []chan float64, layerMakers ...LayerBuilder) (outputs []chan float64) {
	inputsLength := len(inputs)
	inputToAxonsConstructors := make([]AxonsConstructor, inputsLength)

	for i := 0; i < inputsLength; i++ {
		inputToAxonsConstructors[i] = func(broadcastAxon chan float64) AxonsConstructor {
			return func(outputSize int) []chan float64 {
				return broadcastTo(ctx, broadcastAxon, outputSize)
			}
		}(inputs[i])
	}
	for _, layerMaker := range layerMakers {
		inputToAxonsConstructors = layerMaker(ctx, inputToAxonsConstructors)
	}

	axonsLength := len(inputToAxonsConstructors)
	outputs = make([]chan float64, axonsLength)
	for i := 0; i < axonsLength; i++ {
		outputs[i] = inputToAxonsConstructors[i](1)[0]
	}
	return
}

func broadcastTo(ctx context.Context, broadcastAxon <-chan float64, axonsAmount int) (axons []chan float64) {
	axons = make([]chan float64, axonsAmount)
	for i := 0; i < axonsAmount; i++ {
		axons[i] = make(chan float64)
	}
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case val := <-broadcastAxon:
				for i := 0; i < axonsAmount; i++ {
					axons[i] <- val
				}
			}
		}
	}()
	return
}
