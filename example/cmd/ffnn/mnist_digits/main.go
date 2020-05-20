package main

import (
	"github.com/catmorte/go-nns/pkg/network/ffnn"
	"github.com/catmorte/go-nns/pkg/network/helpers/activation"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"time"
)

var pngRegex = regexp.MustCompile(`(\d).png`)
var net *ffnn.Network

func init() {
	activationCalc := activation.Sigmoid()
	net = &ffnn.Network{}
	jsonFile, err := os.Open(os.Args[1])
	checkError(err)
	byteValue, err := ioutil.ReadAll(jsonFile)
	checkError(err)
	err = net.UnmarshalJSON(byteValue)
	checkError(err)
	net.ChangeActivation(activationCalc)
	net.ChangeLearnSpeed(0.001)
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	image.RegisterFormat("png", "png", png.Decode, png.DecodeConfig)
	imagePath := os.Args[2]
	expectedDigit, err := strconv.Atoi(pngRegex.FindStringSubmatch(imagePath)[1])
	checkError(err)

	var digitArray = make([]float64, 10)
	digitArray[expectedDigit] = 1

	output := net.Work(imageAsArray(imagePath))
	actualDigit, _ := getMaxIndexAndVal(output)
	if expectedDigit == actualDigit {
		log.Printf("Found %d", actualDigit)
	} else {
		log.Printf("Expected %d but was %d", expectedDigit, actualDigit)
	}
}

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func imageAsArray(path string) (pixels []float64) {
	file, err := os.Open(path)
	checkError(err)
	defer file.Close()
	img, _, err := image.Decode(file)
	checkError(err)
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			gray, _, _, _ := color.GrayModel.Convert(img.At(x, y)).RGBA()
			pixels = append(pixels, float64(gray))
		}
	}
	return
}

func getMaxIndexAndVal(values []float64) (int, float64) {
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
