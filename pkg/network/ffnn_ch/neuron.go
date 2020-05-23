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
				val := <-dendrite
				soma <- w * val
			}
		}(dendrites[i], i)
	}

	go func() {
		<-initialized
		wShift := weightGen()
		for {
			sum := wShift
			for i := 0; i < dendritesAmount; i++ {
				val := <-soma
				sum = sum + val
			}
			activationResult := activation.Actual(sum)
			broadcastAxon <- activationResult
		}
	}()

	return func(outputSize int) []chan float64 {
		broadcastAxon = make(chan float64, outputSize)
		axons = BroadcastTo(broadcastAxon, outputSize)
		close(initialized)
		return axons
	}
}
