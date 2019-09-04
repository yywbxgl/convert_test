#ifndef __CAL_BANK_H__
#define __CAL_BANK_H__

#define BANK_SIZE (32 * 1024)
int banks_featuremap_cal(int *shape);
int banks_weight_cal(int *shape);
int max_height_in_banks(int banks, int channel, int width);

#endif
