#include "cal_bank.h"
int banks_featuremap_cal(const int *shape)
{
	int c_groups, entries, feature_size;
	
	c_groups = (shape[0]+31)/32;
	entries = (c_groups/2)*shape[2] + (c_groups%2)*((shape[2]+1)/2);
	feature_size = (entries*shape[1]+511)/512*512*64;
	return (feature_size + BANK_SIZE - 1)/BANK_SIZE;
}

int banks_weight_cal(const int *shape)
{
	int weight_size;

	weight_size = shape[0] * shape[1] * shape[2] * shape[3];
	//weight_size = (weight_Size + 511)/512*512;
	return (weight_size + BANK_SIZE - 1)/BANK_SIZE;
}

int max_height_in_banks(int banks, int channel, int width)
{
	int shape[3];
	int height;
	int c_groups, entries;

	c_groups = (channel+31)/32;
	entries = (c_groups/2)*width+ (c_groups%2)*((width+1)/2);
	height = banks*BANK_SIZE/64/ entries;

	shape[0] = channel;
	shape[1] = height;
	shape[2] = width;

	if(banks_featuremap_cal(shape) <= banks) {
		while(1) {
			shape[1]++;
			if(banks_featuremap_cal(shape) > banks) {
				height = shape[1] - 1;
				break;
			}
		}
	} else {
		while(1) {
			shape[1]--;
			if(banks_featuremap_cal(shape) <= banks) {
				height = shape[1];
				break;
			}
		}
	}
	return height;
}
