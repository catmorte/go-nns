package ffnn_ch

import (
	"context"
	"sync"
)

func BuildInputs(ctx context.Context, size int) ([]chan float64, func([]float64)) {
	inputs := make([]chan float64, size)
	inputsWithDirection := make([]chan float64, size)
	for i := 0; i < size; i++ {
		channel := make(chan float64)
		inputs[i] = channel
		inputsWithDirection[i] = channel
	}
	return inputsWithDirection, func(food []float64) {
		syncSignals := &sync.WaitGroup{}
		syncSignals.Add(size)
		for i := 0; i < size; i++ {
			go func(index int) {
				defer syncSignals.Done()
				select {
				case inputs[index] <- food[index]:
				case <-ctx.Done():
					return
				}
			}(i)
		}
		syncSignals.Wait()
	}
}

func WaitForAnswer(ctx context.Context, outputSignals []chan float64) []float64 {
	outputSize := len(outputSignals)
	answer := make([]float64, outputSize)
	syncSignals := &sync.WaitGroup{}
	syncSignals.Add(outputSize)
	for i, outputSignal := range outputSignals {
		go func(signal <-chan float64, signalIndex int) {
			defer syncSignals.Done()
			select {
			case <-ctx.Done():
			case answer[signalIndex] = <-signal:
			}
		}(outputSignal, i)
	}
	syncSignals.Wait()
	return answer
}
