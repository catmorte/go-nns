package ffnn

import (
	"encoding/json"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"io"
)

func (net *Network) AddLayer(layoutamount int, weightInitFunc func() float64, activation activation.Activation, learnSpeed float64) {
	inputSize := net.inputSize
	layersAmount := len(net.Layers)
	if layersAmount > 0 {
		inputSize = len(net.Layers[layersAmount-1].neurons)
	}
	net.Layers = append(net.Layers, newLayer(inputSize, layoutamount, weightInitFunc, activation, learnSpeed))
}

func newLayer(inputsAmount int, layoutSize int, weightInitFunc func() float64, activation activation.Activation, learnSpeed float64) *Layer {
	neurons := make([]*Neuron, layoutSize)
	for i := 0; i < layoutSize; i++ {
		neurons[i] = newNeuron(inputsAmount, weightInitFunc, activation)
	}
	return &Layer{neurons: neurons, learnSpeed: learnSpeed}
}

func newNeuron(inputAmount int, weightsGen func() float64, activation activation.Activation) *Neuron {
	inputWeights := make([]float64, inputAmount)
	for i := 0; i < len(inputWeights); i++ {
		inputWeights[i] = weightsGen()
	}
	return &Neuron{weights: inputWeights, weightShift: weightsGen(), activation: activation}
}

func (net *Network) Export(writer io.Writer) error {
	bytes, err := json.Marshal(net)
	if err != nil {
		return err
	}
	_, err = writer.Write(bytes)
	return err
}

func (net *Network) Import(reader io.Reader) error {
	var bytes []byte
	_, err := reader.Read(bytes)
	if err != nil {
		return err
	}
	return json.Unmarshal(bytes, net)
}

func (l *Layer) ChangeLearnSpeed(learnSpeed float64) {
	l.learnSpeed = learnSpeed
}

func (l *Layer) ChangeActivation(activation activation.Activation) {
	for _, neuron := range l.neurons {
		neuron.activation = activation
	}
}

func (l *Network) ChangeActivation(activation activation.Activation) {
	for _, layer := range l.Layers {
		layer.ChangeActivation(activation)
	}
}

func (l *Network) ChangeLearnSpeed(learnSpeed float64) {
	for _, layer := range l.Layers {
		layer.ChangeLearnSpeed(learnSpeed)
	}
}
