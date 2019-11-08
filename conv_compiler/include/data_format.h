#ifndef __DATA_FORMAT_H__
#define __DATA_FORMAT_H__
#include "npy_array.h"
int featuremap_npy_to_nvdla(tensor_int8_t *pt, char *mem);
int weight_npy_to_nvdla(tensor_int8_t *pt, char *mem);
#endif
