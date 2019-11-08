#include "data_format.h"
#include <string.h>

int featuremap_npy_to_nvdla(tensor_int8_t *pt, char *mem)
{
	int offset;
	int CC,HH,WW;
	int c,h,w;
	CC = pt->shape[0];
	HH = pt->shape[1];
	WW = pt->shape[2];
	c = h = w = 0;
	offset = 0;
#ifdef GET_VALUE
#undef GET_VALUE
#endif
#define GET_VALUE(c,h,w) (c<CC ? pt->data[c*HH*WW+h*WW+w] : (char)0)
	while(1) {
		mem[offset++] = GET_VALUE(c,h,w);
		if((c+1)%32!=0) {
			c = c+1;
		} else if(w<WW-1) {
			c = c-c%32;
			w = w+1;
		} else if(h<HH-1) {
			c = c-c%32;
			h = h+1;
			w = 0;
		} else if(c<CC-1) {
			c = c+1;
			h = 0;
			w = 0;
		} else {
			break;
		}
	}
	return offset;
}

int weight_npy_to_nvdla(tensor_int8_t *pt, char *mem)
{
	int offset;
	int KK,CC,HH,WW;
	int k,c,h,w;
	int kernel_group = 32;
	int chanel_group = 64;
	KK = pt->shape[0];
	CC = pt->shape[1];
	HH = pt->shape[2];
	WW = pt->shape[3];
	k = c = h = w = 0;
	offset = 0;
#ifdef GET_VALUE
#undef GET_VALUE
#endif
#define GET_VALUE(k,c,h,w) pt->data[k*CC*HH*WW+c*HH*WW+h*WW+w]
	while(1) {
		mem[offset++] = GET_VALUE(k,c,h,w);
		if((c+1)%chanel_group!=0 && c<CC-1) {
			c = c+1;
		} else if((k+1)%kernel_group!=0 && k<KK-1) {
			k = k+1;
			c = c-c%chanel_group;
		} else if(w<WW-1) {
			k = k-k%kernel_group;
			c = c-c%chanel_group;
			w = w+1;
		} else if(h<HH-1) {
			k = k-k%kernel_group;
			c = c-c%chanel_group;
			h = h+1;
			w = 0;
		} else if(c<CC-1) {
			k = k-k%kernel_group;
			c = c+1;
			h = 0;
			w = 0;
		} else if(k<KK-1) {
			k = k+1;
			c = 0;
			h = 0;
			w = 0;
		} else {
			break;
		}
	}
	if(offset%128 != 0) {
		int pad = 128 - offset%128;
		memset(&mem[offset], 0, pad);
		offset += pad;
	}
	return offset;
}
