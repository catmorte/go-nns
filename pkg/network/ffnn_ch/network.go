package ffnn_ch

import "context"

func AliveNetwork(ctx context.Context, inputs []chan float64, layerMakers ...LayerBuilder) (outputs []chan float64) {
	inputsLength := len(inputs)
	inputToAxonsConstructors := make([]AxonsConstructor, inputsLength)

	for i := 0; i < inputsLength; i++ {
		inputToAxonsConstructors[i] = func(broadcastAxon chan float64) AxonsConstructor {
			return func(outputSize int) []chan float64 {
				return BroadcastTo(broadcastAxon, outputSize)
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
