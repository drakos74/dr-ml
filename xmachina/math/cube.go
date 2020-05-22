package math

import (
	"fmt"
	"strings"
)

type Cube []Matrix

func Cub(d int) Cube {
	cube := make([]Matrix, d)
	return cube
}

func (c Cube) String() string {
	builder := strings.Builder{}
	builder.WriteString("\n")
	for i := 0; i < len(c); i++ {
		builder.WriteString(fmt.Sprintf("[%d]", i))
		builder.WriteString("\n")
		builder.WriteString(fmt.Sprintf("%v", c[i].String()))
	}
	return builder.String()
}
