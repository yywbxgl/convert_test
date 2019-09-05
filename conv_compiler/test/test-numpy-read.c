#include <stdio.h>
#include <stdlib.h>
#include <npy_array.h>
int main(int argc, char **argv)
{
	tensor_int8_t t;
	int i;
	int size;

	if(numpy_read(argv[1], &t)) {
		printf("%s:%d:ERROR\n", __FILE__, __LINE__);
	}
	size = 1;
	for(i=0;i<t.shape_order;i++) {
		printf("%d ", t.shape[i]);
		size *= t.shape[i];
	}
	printf("\n");

	for(i=0;i<size;i++) {
		printf("%hhd ", t.data[i]);
	}
	printf("\n");
	tensor_free(&t);
	return 0;
}
