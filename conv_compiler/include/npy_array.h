#ifndef __NPY_ARRAY_H__
#define __NPY_ARRAY_H__
typedef struct {
	int *shape;
	int shape_order;
	char *data;
} tensor_int8_t; 
int numpy_read(const char *npy_file, tensor_int8_t *pt);
void tensor_free(tensor_int8_t *pt);
#endif
