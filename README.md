# CatBoost Wrapper for Go
Simple wrapper of [CatBoost C library](https://tech.yandex.com/catboost/doc/dg/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper-docpage/) based on https://github.com/ma3axaka/catboost-go

## Installation
CatBoost library is assumed to be installed (https://catboost.ai/docs/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper.html):
```sh
git clone https://github.com/catboost/catboost.git
cd catboost
ya make -r catboost/libs/model_interface
export CATBOOST_DIR=$(pwd)/catboost/libs/model_interface
export C_INCLUDE_PATH=$CATBOOST_DIR:$C_INCLUDE_PATH
export LIBRARY_PATH=$CATBOOST_DIR:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CATBOOST_DIR:$LD_LIBRARY_PATH
```
The other way is to put compiled library files and include files to `.` or `/usr/local/lib` + `/usr/local/include`.
```
go get -u github.com/tostrivetoseek/catboost-go
```
