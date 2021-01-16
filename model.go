package catboost

/*
#cgo linux LDFLAGS: -L. -lcatboostmodel
#cgo darwin LDFLAGS: -L. -lcatboostmodel
#include <stdlib.h>
#include <stdbool.h>
#include <model_calcer_wrapper.h>

static char** makeCharArray(int size) {
	return calloc(sizeof(char*), size);
}

static void setArrayString(char **a, char *s, int n) {
	a[n] = s;
}

static void freeCharArray(char **a, int size) {
	int i;
	for (i = 0; i < size; i++) free(a[i]);
	free(a);
}

static bool CMP(
    char* calcer,
    size_t docCount,
    char* floatFeatures, size_t floatFeaturesSize,
    char* catFeatures, size_t catFeaturesSize,
    char* result, size_t resultSize)
{
	return CalcModelPrediction(
		(ModelCalcerHandle*)(void*)calcer,
		docCount,
		(const float**)(void*)floatFeatures, floatFeaturesSize,
		(const char***)(void*)catFeatures, catFeaturesSize,
		(double*)(void*)result, resultSize
	);
}

static bool CMPS(
    char* calcer,
    char* floatFeatures, size_t floatFeaturesSize,
    char* catFeatures, size_t catFeaturesSize,
    char* result, size_t resultSize)
{
	return CalcModelPredictionSingle(
		(ModelCalcerHandle*)(void*)calcer,
		(const float*)(void*)floatFeatures, floatFeaturesSize,
		(const char**)(void*)catFeatures, catFeaturesSize,
		(double*)(void*)result, resultSize
	);
}
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

func getError() error {
	messageC := C.GetErrorString()
	message := C.GoString(messageC)
	return fmt.Errorf(message)
}

func makeCStringArrayPointer(array []string) **C.char {
	cargs := C.makeCharArray(C.int(len(array)))
	for i, s := range array {
		C.setArrayString(cargs, C.CString(s), C.int(i))
	}
	return cargs
}

// Model is a wrapper over ModelCalcerHandler
type Model struct {
	handler unsafe.Pointer
}

// GetFloatFeaturesCount returns a number of float features used for training
func (model *Model) GetFloatFeaturesCount() int {
	return int(C.GetFloatFeaturesCount(model.handler))
}

// GetCatFeaturesCount returns a number of categorical features used for training
func (model *Model) GetCatFeaturesCount() int {
	return int(C.GetCatFeaturesCount(model.handler))
}

// Close deletes model handler
func (model *Model) Close() error {
	C.ModelCalcerDelete(model.handler)
	return nil
}

// Loads model from file
func New(filename string) (*Model, error) {
	handler := C.ModelCalcerCreate()
	if !C.LoadFullModelFromFile(handler, C.CString(filename)) {
		return nil, getError()
	}

	return &Model{handler: handler}, nil
}

// CalcModelPrediction returns raw predictions for specified data points
func (model *Model) CalcModelPrediction(floats [][]float32, cats [][]string) ([]float64, error) {
	nFloats := len(floats)
	nCats := len(cats)

	nSamples := 0

	floatLength := 0
	catLength := 0

	if nFloats > 0 {
		floatLength = len(floats[0])
		nSamples = nFloats
	}
	if nCats > 0 {
		catLength = len(cats[0])
		if nCats < nSamples {
			nSamples = nCats
		}
	}

	if nSamples == 0 || floatLength+catLength == 0 {
		return nil, fmt.Errorf("empty samples")
	}

	results := make([]float64, nSamples)

	floatsBuf := make([]float32, nSamples*floatLength) // to prevent moving by GC
	floatsC := make([]*C.float, nSamples)
	if floatLength > 0 {
		for i, x := range floats {
			if i >= nSamples {
				break
			}

			data := floatsBuf[i*floatLength : (i+1)*floatLength]
			copy(data, x)
			floatsC[i] = (*C.float)(&data[0])
		}
	}
	floatsCUPtr := unsafe.Pointer(&floatsC[0])

	catsC := make([]**C.char, nSamples)
	for i, x := range cats {
		if i >= nSamples {
			break
		}

		pointer := makeCStringArrayPointer(x)
		defer C.freeCharArray(pointer, C.int(len(x))) // (!) yes, defer here
		catsC[i] = pointer
	}

	if !C.CMP(
		(*C.char)(model.handler),
		C.size_t(nSamples),
		(*C.char)(floatsCUPtr),
		C.size_t(floatLength),
		(*C.char)(unsafe.Pointer(&catsC[0])),
		C.size_t(catLength),
		(*C.char)(unsafe.Pointer(&results[0])),
		C.size_t(nSamples),
	) {
		return nil, getError()
	}

	return results, nil
}

func (model *Model) CalcModelPredictionSingle(floats []float32, cats []string) (float64, error) {
	floatLength := len(floats)
	catLength := len(cats)

	if floatLength+catLength == 0 {
		return 0, fmt.Errorf("empty sample")
	}

	catsPtr := makeCStringArrayPointer(cats)
	defer C.freeCharArray(catsPtr, C.int(len(cats)))

	var floatsUPtr unsafe.Pointer
	if floatLength > 0 {
		floatsUPtr = unsafe.Pointer(&floats[0])
	}

	var result float64

	if !C.CMPS(
		(*C.char)(model.handler),
		(*C.char)(floatsUPtr),
		C.size_t(floatLength),
		(*C.char)(unsafe.Pointer(catsPtr)),
		C.size_t(catLength),
		(*C.char)(unsafe.Pointer(&result)),
		1,
	) {
		return 0, getError()
	}

	return result, nil
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (model *Model) CalcModelPredictionProba(floats [][]float32, cats [][]string) ([]float64, error) {
	results, err := model.CalcModelPrediction(floats, cats)

	for i, v := range results {
		results[i] = sigmoid(v)
	}

	return results, err
}

func (model *Model) CalcModelPredictionSingleProba(floats []float32, cats []string) (float64, error) {
	raw, err := model.CalcModelPredictionSingle(floats, cats)
	return sigmoid(raw), err
}
