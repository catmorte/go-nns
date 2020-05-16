package main

import (
	"fmt"
	"github.com/catmorte/go-nns/pkg/network/ffnn"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"github.com/catmorte/go-nns/pkg/network/helpers/weightgen"
)

func main() {
	weightGen := weightgen.RandomWithin(-0.5, 0.5)
	activationCalc := activation.Sigmoid()
	net := ffnn.CreateNetwork(2)
	net.AddLayer(2, weightGen, activationCalc, 0.01)
	net.AddLayer(2, weightGen, activationCalc, 0.01)
	net.AddLayer(4, weightGen, activationCalc, 0.01)

	for i := 0; i < 200000; i++ {
		net.Learn([]float64{1, 1}, []float64{1, 0, 0, 0})
		net.Learn([]float64{1, 0}, []float64{0, 1, 0, 0})
		net.Learn([]float64{0, 1}, []float64{0, 0, 1, 0})
		net.Learn([]float64{0, 0}, []float64{0, 0, 0, 1})
	}

	fmt.Println(getMaxIndexAndVal(net.Calculate([]float64{1, 1})))
	fmt.Println(getMaxIndexAndVal(net.Calculate([]float64{1, 0})))
	fmt.Println(getMaxIndexAndVal(net.Calculate([]float64{0, 1})))
	fmt.Println(getMaxIndexAndVal(net.Calculate([]float64{0, 0})))
}

func getMaxIndexAndVal(values []float64) (int, float64){
	index := 0
	max := values[index]
	for i, value := range values {
		if max < value {
			index = i
			max = value
		}
	}
	return index, max
}
