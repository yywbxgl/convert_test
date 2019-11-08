#include "cal_bank.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv)
{
	int shape[4];
	int banks,channel,width;

	if(argc == 4) {
		banks = atoi(argv[1]);
		channel = atoi(argv[2]);
		width = atoi(argv[3]);
		printf("%d\n", max_height_in_banks(banks,channel,width));
	} else if(argc == 5) {
		shape[0] = atoi(argv[2]);
		shape[1] = atoi(argv[3]);
		shape[2] = atoi(argv[4]);
		printf("%d\n", banks_featuremap_cal(shape));
	} else if(argc == 6) {
		shape[0] = atoi(argv[2]);
		shape[1] = atoi(argv[3]);
		shape[2] = atoi(argv[4]);
		shape[3] = atoi(argv[5]);
		printf("%d\n", banks_weight_cal(shape));
	}
	return 0;
}
