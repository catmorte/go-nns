package main

import (
	"fmt"
	"github.com/catmorte/go-nns/pkg/network/ffnn"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"github.com/catmorte/go-nns/pkg/network/helpers/weightgen"
	"math"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	weightGen := weightgen.RandomWithin(-0.5, 0.5)
	activationCalc := activation.Sigmoid()
	net := ffnn.CreateNetwork(2)
	net.AddLayer(2, weightGen, activationCalc, 0.01)
	net.AddLayer(2, weightGen, activationCalc, 0.01)
	net.AddLayer(1, weightGen, activationCalc, 0.01)

	for i := 0; i < 400000; i++ {
		net.Train([]float64{1, 1}, []float64{0})
		net.Train([]float64{1, 0}, []float64{1})
		net.Train([]float64{0, 1}, []float64{1})
		net.Train([]float64{0, 0}, []float64{0})
	}

	fmt.Println(math.Round(net.Work([]float64{1, 1})[0]))
	fmt.Println(math.Round(net.Work([]float64{1, 0})[0]))
	fmt.Println(math.Round(net.Work([]float64{0, 1})[0]))
	fmt.Println(math.Round(net.Work([]float64{0, 0})[0]))
}
