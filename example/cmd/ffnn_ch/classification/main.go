package main

import (
	"context"
	"fmt"
	"github.com/catmorte/go-nns/pkg/network/ffnn_ch"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"github.com/catmorte/go-nns/pkg/network/helpers/weightgen"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	ctx := context.Background()
	inputs, feed := ffnn_ch.BuildInputs(2)
	outputSignals := ffnn_ch.AliveNetwork(ctx, inputs,
		ffnn_ch.LayerConstructor(2, activation.Sigmoid(), weightgen.RandomWithin(-0.5, 0.5)),
		ffnn_ch.LayerConstructor(2, activation.Sigmoid(), weightgen.RandomWithin(-0.5, 0.5)),
		ffnn_ch.LayerConstructor(4, activation.Sigmoid(), weightgen.RandomWithin(-0.5, 0.5)),
	)

	feed([]float64{1, 1})
	fmt.Println(ffnn_ch.WaitForAnswer(outputSignals))
	feed([]float64{1, 0})
	fmt.Println(ffnn_ch.WaitForAnswer(outputSignals))
	feed([]float64{0, 1})
	fmt.Println(ffnn_ch.WaitForAnswer(outputSignals))
	feed([]float64{0, 0})
	fmt.Println(ffnn_ch.WaitForAnswer(outputSignals))
}
