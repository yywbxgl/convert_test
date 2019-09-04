#ifndef __NPY_ARRAY_H__
#define __NPY_ARRAY_H__
typedef struct {
	int *shape;
	int shape_order;
	char *data;
} tensor_int8_t; 
int read_numpy(const char *npy_file, tensor_int8_t *pt);
#endif
