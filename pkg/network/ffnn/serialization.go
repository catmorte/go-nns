package ffnn

import (
	"encoding/json"
)

type (
	JSONNetwork struct {
		Layers []JSONLayer `marshal_unmarshal_json:"layers"`
	}
	JSONLayer struct {
		Neurons []JSONNeuron `marshal_unmarshal_json:"neurons"`
	}
	JSONNeuron struct {
		ShiftWeight float64   `marshal_unmarshal_json:"shift_weight"`
		Weights     []float64 `marshal_unmarshal_json:"weights"`
	}
)

func (net *Network) MarshalJSON() ([]byte, error) {
	jsonNet := JSONNetwork{Layers: make([]JSONLayer, len(net.Layers))}
	for li, layer := range net.Layers {
		jsonNet.Layers[li] = JSONLayer{Neurons: make([]JSONNeuron, len(layer.neurons))}
		for ni, neuron := range layer.neurons {
			jsonNet.Layers[li].Neurons[ni] = JSONNeuron{ShiftWeight: neuron.shiftNeuronWeight, Weights: neuron.inputWeights}
		}
	}
	return json.Marshal(jsonNet)
}

func (net *Network) UnmarshalJSON(b []byte) error {
	jsonNetwork := &JSONNetwork{}
	err := json.Unmarshal(b, jsonNetwork)
	if err != nil {
		return err
	}
	net.Layers = make([]*Layer, len(jsonNetwork.Layers))
	for li, layer := range jsonNetwork.Layers {
		net.Layers[li] = &Layer{neurons: make([]*Neuron, len(layer.Neurons))}
		for ni, neuron := range layer.Neurons {
			net.Layers[li].neurons[ni] = &Neuron{inputWeights: neuron.Weights, shiftNeuronWeight: neuron.ShiftWeight}
		}
	}
	return nil
}
