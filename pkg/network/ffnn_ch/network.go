package ffnn_ch

func AliveNetwork(inputs []chan float64, layerMakers ...LayerBuilder) (outputs []chan float64) {
	inputsLength := len(inputs)
	inputToAxonsConstructors := make([]AxonsConstructor, inputsLength)
	for i := 0; i < inputsLength; i++ {
		inputToAxonsConstructors[i] = channelToAxonsConstructor(inputs[i])
	}
	for _, layerMaker := range layerMakers {
		inputToAxonsConstructors = layerMaker(inputToAxonsConstructors)
	}
	axonsLength := len(inputToAxonsConstructors)
	outputs = make([]chan float64, axonsLength)
	for i := 0; i < axonsLength; i++ {
		outputs[i] = inputToAxonsConstructors[i](1)[0]
	}
	return
}

func channelToAxonsConstructor(broadcastAxon chan float64) AxonsConstructor {
	return func(outputSize int) []chan float64 {
		return BroadcastTo(broadcastAxon, outputSize)
	}
}
