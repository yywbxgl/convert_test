#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cal_bank.h"
int main(int argc, char **argv)
{
	conv_arg_t conv_arg;
	conv_block_t *blk;
	int i, cnt;

	if(argc == 1) {
		printf("%s %s\n", argv[0], "conv_stride=4 conv_dilation=1 conv_pad=0 datain_shape=3,227,227 dataout_shape=96,55,55 weight_shape=96,3,11,11");
		return 0;
	}	

	for(i=1;i<argc;i++) {
		char name[100];
		char value[100];
		if(sscanf(argv[i], "%[^=]=%s", name, value) != 2) {
			printf("%s:%d ERROR!",__FILE__,__LINE__);
			return 1;
		}
		if(strcmp("conv_stride",name)==0)conv_arg.conv_stride=atoi(value);
		else if(strcmp("conv_dilation",name)==0)conv_arg.conv_dilation=atoi(value);
		else if(strcmp("conv_pad",name)==0)conv_arg.conv_pad=atoi(value);
		else {
			int *a;
			int ret, j, n;
			char *s;
			char tmp[4];
			if(strcmp("datain_shape",name)==0) {
				a = conv_arg.datain_shape;
			} else if(strcmp("dataout_shape",name)==0) {
				a = conv_arg.dataout_shape;
			} else if(strcmp("weight_shape",name)==0) {
				a = conv_arg.weight_shape;
			}
			s = value;
			j = 0;
			while(1) {
				ret = sscanf(s, "%d%[,]%n", &a[j], tmp, &n);
				if(ret < 2)
					break;
				s += n;
				j++;
			}
		}
	}
	printf("%s:%d\n",__FILE__,__LINE__);

	cnt = split_conv(&conv_arg, &blk);
	for(i=0;i<cnt;i++) {
		printf("[%d]\n", i);
		printf("datain_height_start : %d\n", blk[i].datain_height_start);
		printf("datain_height : %d\n", blk[i].datain_height);
		printf("dataout_height_start : %d\n", blk[i].dataout_height_start);
		printf("dataout_height : %d\n", blk[i].dataout_height);
	}

	free(blk);
	return 0;
}
