package ffnn

func CreateNetwork(inputSize int) *Network {
	return &Network{inputSize: inputSize, Layers: []*Layer{}}
}
