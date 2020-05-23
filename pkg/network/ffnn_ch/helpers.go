package ffnn_ch

import (
	"sync"
)

func BuildInputs(size int) ([]chan float64, func([]float64)) {
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
				inputs[index] <- food[index]
			}(i)
		}
		syncSignals.Wait()
	}
}

func WaitForAnswer(outputSignals []chan float64) []float64 {
	outputSize := len(outputSignals)
	answer := make([]float64, outputSize)
	syncSignals := &sync.WaitGroup{}
	syncSignals.Add(outputSize)
	for i, outputSignal := range outputSignals {
		go func(signal <-chan float64, signalIndex int) {
			defer syncSignals.Done()
			answer[signalIndex] = <-signal
		}(outputSignal, i)
	}
	syncSignals.Wait()
	return answer
}

func BroadcastTo(broadcastAxon <-chan float64, axonsAmount int) (axons []chan float64) {
	axons = make([]chan float64, axonsAmount)
	for i := 0; i < axonsAmount; i++ {
		axons[i] = make(chan float64)
	}
	go func() {
		for {
			val := <-broadcastAxon
			for i := 0; i < axonsAmount; i++ {
				axons[i] <- val
			}
		}
	}()
	return
}
