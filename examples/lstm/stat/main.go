package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/drakos74/oremi/graph"
)

const (
	index = "airline-passengers"
)

func main() {

	// Open the file
	csvfile, err := os.Open("examples/lstm/data/airline-passengers.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)

	var title []string = nil
	plot := graph.New("LSTM")

	// Iterate through the records
	i := 0
	for {
		// Read each record from csv
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		if title == nil {
			title = record
			plot.NewSeries(index, title[0], title[1])
			continue
		}
		fmt.Printf("%s: %s %s: %s\n", title[0], record[0], title[1], record[1])

		_, err = time.Parse("2006-01", record[0])
		if err != nil {
			panic(fmt.Errorf("could not parse %s as date", record[0]))
		}
		y, err := strconv.ParseFloat(record[1], 64)
		plot.Add(index, float64(i), y)
		i++
	}

	graph.Draw("airline-pasengers", plot)
}
