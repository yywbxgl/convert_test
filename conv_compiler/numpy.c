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

int numpy_write(const char *npy_file, const tensor_int8_t *pt)
{
	FILE *f;
	char s[40];
	char *p;
	int i, size, n;
	
	if((f = fopen(npy_file, "wb")) == NULL) {
		fprintf(stderr, "%s:%d write failed\n", __FILE__, __LINE__);
		return -1;
	}
	s[0] = 0x93;
	(void)fwrite(s, 1, 1, f);
	fprintf(f, "NUMPY");
	s[0] = 1;
	s[1] = 0;
	s[2] = 'v';
	s[3] = 0;
	(void)fwrite(s, 1, 4, f);


	p = s;
	for(i=0;i<pt->shape_order;i++) {
		if(i==0) {
			sprintf(p, "(%d%n", pt->shape[i], &n);
		} else {
			sprintf(p, ",%d%n", pt->shape[i], &n);
		}
		p += n;
	}
	sprintf(p, ")");

	fprintf(f, "{\'descr\':\'|i1\',\'fortran_order\':False,\'shape\':%s,}", s);
	size = 127-ftell(f);
	s[0] = ' ';
	for(i=0;i<size;i++) {
		(void)fwrite(s, 1, 1, f);
	}
	s[0] = '\n';
	(void)fwrite(s, 1, 1, f);
	
	size = 1;
	for(i=0;i<pt->shape_order;i++) {
		size *= pt->shape[i];
	}
	(void)fwrite(pt->data, 1, size, f);
	
	fclose(f);
	return 0;
}
