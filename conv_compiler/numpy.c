#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npy_array.h"

int numpy_read(const char *npy_file, tensor_int8_t *pt)
{
	FILE *f;
	char s[1024];
	char *p;
	int i;
	size_t data_size;

	f = fopen(npy_file, "rb");
	if(f == NULL) {
		return 1;
	}
	if(fscanf(f, "%*[^{]{%[^}]}%*[^\n]\n", s) != 1) {
		return 2;
	}
	if((p = strstr(s, "\'shape\'")) == NULL) {
		return 2;
	}
	if((p = strstr(p, "(")) == NULL) {
		return 2;
	}
	p++;

	/* 找到了形状字符串的位置 */
	pt->shape = NULL;
	pt->shape_order = 0;
	while(1) {
		int n;
		int dimension;
		if(sscanf(p, "%d%n", &dimension, &n) != 1) {
			break;
		}
		pt->shape_order++;
		pt->shape = realloc(pt->shape, pt->shape_order * sizeof(int));
		if(pt->shape == NULL) {
			return 3;
		}
		pt->shape[pt->shape_order-1] = dimension;
		p += n;
		while(*p != '\0' && *p != ',' && *p != ')') {
			p++;
		}
		if(*p != ',') {
			break;
		}
		p++;
	}
	data_size = 1;
	for(i=0;i<pt->shape_order;i++)
		data_size *= pt->shape[i];
	if((pt->data = malloc(data_size)) == NULL) {
		return 3;
	}
	if(fread(pt->data, 1, data_size, f) < data_size) {
		return 4;
	}

	fclose(f);
	return 0;
}

void tensor_free(tensor_int8_t *pt)
{
	free(pt->shape);
	pt->shape = NULL;
	free(pt->data);
	pt->data = NULL;
}

