//2018.3.15
//By Luo jiuheng

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//
__global__ int calculateLines()
{
	FILE * fp = NULL;
	int c, lc = 0;
	int line = 0;
	fp = fopen("Coefficient.txt", "r");
	while ((c = fgetc(fp)) != EOF)
	{
		if (c == '\n') line++;
		lc = c;
	}
	fclose(fp);
	if (lc != '\n') line++;

	return line;
}


//do the rotation of all lines
__global__ void rotation()
{


}


//find the suitable value in every blocks
__global__ void findMinMax(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



//find the last answer of these lines
__device__ void findAll()
{

}

//parfix all information to all blocks

__global__ void fuck(int *c, const int *a, const int *b)
{
	int f, u, c, k;

}

//main function
int main()
{
    
}
