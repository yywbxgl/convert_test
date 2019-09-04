#ifndef __DATA_FORMAT_H__
#define __DATA_FORMAT_H__
#include "npy_array.h"
void featuremap_npy_to_nvdla(char *mem, tensor_int8_t *pt);
void weight_npy_to_nvdla(char *mem, tensor_int8_t *pt);
#endif
