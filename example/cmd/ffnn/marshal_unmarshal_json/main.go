package main

import (
	"encoding/json"
	"fmt"
	"github.com/catmorte/go-nns/pkg/network/ffnn"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"github.com/catmorte/go-nns/pkg/network/helpers/weightgen"
	"math"
)

func main() {
	weightGen := weightgen.RandomWithin(-0.5, 0.5)
	activationCalc := activation.Sigmoid()
	net := ffnn.CreateNetwork(2)
	net.AddLayer(2, weightGen, activationCalc, 0.01)
	net.AddLayer(2, weightGen, activationCalc, 0.01)
	net.AddLayer(1, weightGen, activationCalc, 0.01)

	for i := 0; i < 100000; i++ {
		net.Train([]float64{1, 1}, []float64{0})
		net.Train([]float64{1, 0}, []float64{1})
		net.Train([]float64{0, 1}, []float64{1})
		net.Train([]float64{0, 0}, []float64{0})
	}

	var bytes []byte
	var err error
	if bytes, err = json.Marshal(net); err != nil {
		panic(err)
	}

	fmt.Println(string(bytes))

	newNet := &ffnn.Network{}
	if err = json.Unmarshal(bytes, newNet); err != nil {
		panic(err)
	}

	newNet.ChangeActivation(activationCalc)

	fmt.Println(math.Round(newNet.Work([]float64{1, 1})[0]))
	fmt.Println(math.Round(newNet.Work([]float64{1, 0})[0]))
	fmt.Println(math.Round(newNet.Work([]float64{0, 1})[0]))
	fmt.Println(math.Round(newNet.Work([]float64{0, 0})[0]))
}
