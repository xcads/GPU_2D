
#include "cuda.h"
#include "host_defines.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Header.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <float.h>

using namespace std;
// PAIR structure
// VERTEX structure
struct Vertex {
	double x, y;
};

struct Pair {
	int index;
	int index1, index2;
	Line line1, line2;
	Vertex point;
	bool pruneFlag;
};

// Object Function
struct Objfunc {
	// xd = c1x + c2y
	double c1, c2;
};



#define FILENAME        Coefficient.txt
#define PI				3.14159265358979323846264338327950288419716939937510       
#define RANDOM_SEED     7
#define RANDOM_PARA     2000
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}



vector<struct Line> originalConstraints;
struct Vertex Solution;
int randomSeed = RANDOM_SEED;
//
//#define FILENAME        Coefficient.txt
//#define PI				3.14159265358979323846264338327950288419716939937510       
//#define RANDOM_SEED     7
//#define RANDOM_PARA     2000
//#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
//
//
//#define HANDLE_NULL( a ) {if (a == NULL) { \
//                            printf( "Host memory failed in %s at line %d\n", \
//                                    __FILE__, __LINE__ ); \
//                            exit( EXIT_FAILURE );}}
//int randomSeed = RANDOM_SEED;
// LINE structure : Constraints
//typedef struct Line {
//	// a1x + a2y >= b
//	double a1, a2, b;
//	double slope;
//	bool pruneFlag;
//
//	int index;
//};

//// Object Function
//struct Objfunc {
//	// xd = c1x + c2y
//	double c1, c2;
//};

// VERTEX structure
//struct Vertex {
//	double x, y;
//};
//
//// PAIR structure
//struct Pair {
//	int index;
//	int index1, index2;
//	Line line1, line2;
//	Vertex point;
//	bool pruneFlag;
//};


 //vector<struct Line> originalConstraints;
 //struct Vertex Solution;
 //int randomSeed = RANDOM_SEED;

bool Intersection(struct Line *l1, struct Line *l2, struct Vertex *v1)
{
	if (fabs(l1->a1 * l2->a2 - l2->a1 * l1->a2) < DBL_EPSILON)
	{
		v1 = NULL;
		return false;
	}
	v1->x = -(l1->b * l2->a2 - l2->b * l1->a2) / (l1->a1 * l2->a2 - l2->a1 * l1->a2);
	v1->y = (l1->b * l2->a1 - l2->b * l1->a1) / (l1->a2 * l2->a1 - l1->a1 * l2->a2);
	return true;
}

//void Slope(struct Line *l)
//{
//	if (fabs(l->a2 - 0.0) < DBL_EPSILON)
//	{
//		if ((l->a1 > 0 && l->a2 < 0) || (l->a1 < 0 && l->a2 > 0))
//		{
//			l->slope = DBL_MAX;
//		}
//		else if ((l->a1 < 0 && l->a2 < 0) || (l->a1 > 0 && l->a2 > 0))
//		{
//			l->slope = -DBL_MAX;
//		}
//		else
//		{
//			l->slope = -l->a1 / l->a2;
//		}
//		return;
//	}
//	l->slope = -l->a1 / l->a2;
//	return;
//}

// Slope line
__device__ void Slope_d(struct Line *l)
{
	if (fabs(l->a2 - 0.0) < DBL_EPSILON)
	{
		if ((l->a1 > 0 && l->a2 < 0) || (l->a1 < 0 && l->a2 > 0))
		{
			l->slope = DBL_MAX;
		}
		else if ((l->a1 < 0 && l->a2 < 0) || (l->a1 > 0 && l->a2 > 0))
		{
			l->slope = -DBL_MAX;
		}
		else
		{
			l->slope = -l->a1 / l->a2;
		}
		return;
	}
	l->slope = -l->a1 / l->a2;
	return;
}

// Compare
//int cmp(const void *a, const void *b)
//{
//	struct Line *aa = (struct Line *)a;
//	struct Line *bb = (struct Line *)b;
//	return ((aa->slope > bb->slope) ? 1 : -1);
//}

// Rotation_d
__global__ void kRotation(struct Line oConstraints[], struct Line lines[], struct Objfunc *object, int *index, int *numG, int *numH)
{
	double thetaArc;

	if (object->c2 == 0 && object->c1 > 0) {
		thetaArc = -PI / 2;
	}
	else if (object->c2 == 0 && object->c1 < 0) {
		thetaArc = PI / 2;
	}
	else {
		thetaArc = atan(-object->c1 / object->c2);
	}

	int i;
	double a1Temp, a2Temp, bTemp;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (offset < (*index)) {
		a1Temp = oConstraints[offset].a1;
		a2Temp = oConstraints[offset].a2;
		bTemp = oConstraints[offset].b;

		lines[offset].a1 = cos(thetaArc) * a1Temp + sin(thetaArc) * a2Temp;
		lines[offset].a2 = cos(thetaArc) * a2Temp - sin(thetaArc) * a1Temp;
		lines[offset].b = bTemp;
		lines[offset].index = offset;

		if (lines[offset].a2 > 0) {
			atomicAdd(numG, 1);
		}
		else if (lines[offset].a2 < 0) {
			atomicAdd(numH, 1);
		}
		else {
			return;
		}

		Slope_d(&lines[offset]);
		lines[offset].pruneFlag = true;
	}

	__syncthreads();
	__threadfence();

}

// Separation - O(n)
bool Separation(struct Line I1[], struct Line I2[], struct Line lines[], int numG, int numH)
{
	int index = numG + numH;
	int i, g = 0, h = 0;
	for (i = 0; i < index; i++) {
		if (lines[i].a2 > 0) {
			I1[g].a1 = -lines[i].a1 / lines[i].a2;
			I1[g].a2 = 1;
			I1[g].b = lines[i].b / lines[i].a2;
			Slope(&I1[g]);
			I1[g].slope = -I1[g].slope;
			I1[g].pruneFlag = true;
			I1[g].index = lines[i].index;

			g++;
		}
		else if (lines[i].a2 < 0) {
			I2[h].a1 = -lines[i].a1 / lines[i].a2;
			I2[h].a2 = 1;
			I2[h].b = lines[i].b / lines[i].a2;
			Slope(&I2[h]);
			I2[h].slope = -I2[h].slope;
			I2[h].pruneFlag = true;
			I2[h].index = lines[i].index;

			h++;
		}
		else {
			printf("%d %lf\n", i, lines[i].a2);
			return false;
		}
	}
	return true;
}

// Make pairs
bool MakePairs(struct Line I1[], struct Line I2[],
	struct Pair pairsG[], struct Pair pairsH[],
	int numG, int numH, int *index,
	double leftBound, double rightBound)
{
	int g, gtemp;
	(*index) = 0;
	for (g = 0; g < numG; g += 1) {
		// drop
		if (I1[g].pruneFlag == false) {
			continue;
		}
		for (gtemp = g + 1; gtemp < numG; gtemp++) {
			if (I1[gtemp].pruneFlag == true) {
				break;
			}
		}
		if (gtemp == numG) break;

		if (fabs(I1[g].slope - I1[gtemp].slope) < DBL_EPSILON) {
			if (I1[g].b > I1[gtemp].b) {
				I1[gtemp].pruneFlag = false;
				g = g - 1;
			}
			else {
				I1[g].pruneFlag = false;
				g = gtemp - 1;
			}

			continue;
		}
		struct Vertex *p = (struct Vertex *)malloc(sizeof(struct Vertex));
		Intersection(&I1[g], &I1[gtemp], p);
		if (p->x < leftBound || p->x > rightBound) {
			if (p->x < leftBound && (I1[g].slope > I1[gtemp].slope)) {
				I1[gtemp].pruneFlag = false;
				g = g - 1;
			}
			else if (p->x < leftBound && (I1[g].slope < I1[gtemp].slope)) {
				I1[g].pruneFlag = false;
				g = gtemp - 1;
			}
			else if (p->x > rightBound && (I1[g].slope < I1[gtemp].slope)) {
				I1[gtemp].pruneFlag = false;
				g = g - 1;
			}
			else if (p->x > rightBound && (I1[g].slope > I1[gtemp].slope)) {
				I1[g].pruneFlag = false;
				g = gtemp - 1;
			}
			continue;
		}
		pairsG[(*index)].index = (*index);
		pairsG[(*index)].line1 = I1[g];
		pairsG[(*index)].index1 = g;
		pairsG[(*index)].line2 = I1[gtemp];
		pairsG[(*index)].index2 = gtemp;
		pairsG[(*index)].point.x = p->x; pairsG[(*index)].point.y = p->y;

		(*index)++;
		g++;
	}

	return true;
}

// sg, Sg, sh, Sh
struct Vertex *TestingLine(struct Pair pairsG[], struct Pair pairsH[],
	struct Line I1[], struct Line I2[],
	int numG, int numH, int numDot,
	double *leftBound, double *rightBound)
{

	int index = (numDot == 0) ? 0 : (getRandomInt(&randomSeed, numDot));

	//printf("%d %d\n", index, numDot);


	if (numDot == 0) {
		int onlyOne = 0;
		bool isFeasible = false;
		struct Vertex *vSln = (struct Vertex *)malloc(sizeof(struct Vertex));
		vSln->y = -FLT_MAX;
		for (onlyOne = 0; onlyOne < numG; onlyOne++) {
			if (I1[onlyOne].pruneFlag == true) {
				isFeasible = true;
				break;
			}
		}
		if (isFeasible == true && numH != 0) {
			struct Vertex *vTemp = (struct Vertex *)malloc(sizeof(struct Vertex));
			for (int i = 0; i < numH; i++) {
				Intersection(&(I1[onlyOne]), &(I2[i]), vTemp);
				if (vSln->y < vTemp->y) {
					vSln->x = vTemp->x;
					vSln->y = vTemp->y;
				}
			}
			printf("sln: %lf %lf\n", vSln->x, vSln->y);
			return vSln;
		}
		else {
			/*
			for (int i = 0; i < numG; i++) {
			cout << "pruneFlag: " << I1[i].pruneFlag << endl;
			}*/
			cout << "No solution!\n";
			exit(0);
		}
	}

	//int index = round ? 1 : 0;
	double xPrimeG = pairsG[index].point.x;   // x' - xPrime
	double yPrimeG = pairsG[index].point.y;
	double yPrimeH;

	//cout << xPrimeG << '\n';

	// struct Line *sg = (&pairsG[index].line1.a1 < &pairsG[index].line2.a1) ? &pairsG[index].line1 : &pairsG[index].line2;
	// struct Line *Sg = (&pairsG[index].line1.a1 >= &pairsG[index].line2.a1) ? &pairsG[index].line1 : &pairsG[index].line2;
	struct Line *sg = NULL;
	struct Line *Sg = NULL;
	struct Line *sh = NULL;
	struct Line *Sh = NULL;
	// struct Line *sh = (&pairsH[index].line1.a1 < &pairsH[index].line2.a1) ? &pairsH[index].line1 : &pairsH[index].line2;
	// struct Line *Sh = (&pairsH[index].line1.a1 < &pairsH[index].line2.a1) ? &pairsH[index].line1 : &pairsH[index].line2;

	vector<int> linesG;
	vector<int> linesH;

	// Finding g(x') and H(x')
	for (int i = 0; i < numG; i++) {
		if (I1[i].pruneFlag == true) {
			if ((fabs(yPrimeG - (I1[i].a1 * xPrimeG + I1[i].b)) >DBL_EPSILON && yPrimeG < (I1[i].a1 * xPrimeG + I1[i].b)) || (sg == NULL || Sg == NULL)) {
				//printf("xPrime yPrime ???: %lf %lf %lf\n", xPrimeG, yPrimeG, (I1[i].a1 * xPrimeG + I1[i].b));



				yPrimeG = I1[i].a1 * xPrimeG + I1[i].b;
				sg = &I1[i];
				Sg = &I1[i];
			}
		}
	}
	for (int i = 0; i < numH; i++) {
		if (I2[i].pruneFlag == true) {
			if (sh == NULL || Sh == NULL) {
				sh = &I2[i];
				Sh = &I2[i];
				yPrimeH = I2[i].a1 * xPrimeG + I2[i].b;
			}
			else if (fabs(yPrimeH - (I2[i].a1 * xPrimeG + I2[i].b)) > DBL_EPSILON && yPrimeH > (I2[i].a1 * xPrimeG + I2[i].b)) {
				yPrimeH = I2[i].a1 * xPrimeG + I2[i].b;
				sh = &I2[i];
				Sh = &I2[i];
			}
		}
	}
	if (numH == 0) {
		yPrimeH = yPrimeG + 1000.0;
	}

	// Finding sg - min g(x') && Finding Sg - max g(x')
	/*
	struct Line *sg = &pairsG[0].line1;
	struct Line *Sg = &pairsG[0].line1;
	struct Line *sh = &pairsH[0].line1;
	struct Line *Sh = &pairsH[0].line1;
	*/
	for (int i = 0; i < numG; i++) {
		double currentLineValueG = I1[i].a1 * xPrimeG + I1[i].b;
		if (I1[i].pruneFlag == false || fabs(currentLineValueG - yPrimeG) >= DBL_EPSILON) {
			continue;
		}

		if (I1[i].a1 < sg->a1) {
			sg = &I1[i];
		}
		if (I1[i].a1 > Sg->a1) {
			Sg = &I1[i];
		}
	}
	// Finding sh - min h(x') && Finding Sh - max h(x')
	for (int i = 0; i < numH; i++) {
		double currentValueH = I2[i].a1 * xPrimeG + I2[i].b;
		if (I2[i].pruneFlag == false || fabs(currentValueH - yPrimeH) >= DBL_EPSILON) {
			continue;
		}

		if (I2[i].a1 < sh->a1) {
			sh = &I2[i];
		}
		if (I2[i].a1 > Sh->a1) {
			Sh = &I2[i];
		}
	}

	// Is feasible
	if (fabs(yPrimeG - yPrimeH) < DBL_EPSILON) {
		if (sg->a1 > 0 && sg->a1 >= Sh->a1) {
			// x* < x'
			if (sh != Sh) {
				sh->pruneFlag = false;
			}
			if (sg != Sg) {
				Sg->pruneFlag = false;
			}
			*rightBound = xPrimeG;
			//cout << "cccccccccc\n";
			return NULL;
		}
		else if (Sg->a1 < 0 && Sg->a1 <= sh->a1) {
			// x* > x'
			if (sh != Sh) {
				Sh->pruneFlag = false;
			}
			if (sg != Sg) {
				sg->pruneFlag = false;
			}
			*leftBound = xPrimeG;

			return NULL;
		}
		else {
			// x* = x'
			Solution.x = xPrimeG;
			Solution.y = yPrimeG;

			return &(Solution);
		}
	}
	else if (yPrimeG > yPrimeH) {   // infeasible
		if (sg->a1 > Sh->a1) {
			// x* < x'
			if (sh != Sh) {
				sh->pruneFlag = false;
			}
			if (sg != Sg) {
				Sg->pruneFlag = false;
			}

			else {
				if (pairsG[index].line1.a1 < pairsG[index].line2.a1) {
					I1[pairsG[index].index2].pruneFlag = false;
				}
				else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {
					I1[pairsG[index].index1].pruneFlag = false;
				}
			}
			*rightBound = xPrimeG;

			return NULL;
		}
		else if (Sg->a1 < sh->a1) {
			// x* > x'
			if (sh != Sh) {
				Sh->pruneFlag = false;
			}
			if (sg != Sg) {
				sg->pruneFlag = false;
			}

			else {
				if (pairsG[index].line1.a1 < pairsG[index].line2.a1) {
					I1[pairsG[index].index1].pruneFlag = false;
				}
				else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {

					I1[pairsG[index].index2].pruneFlag = false;
				}
			}
			*leftBound = xPrimeG;

			return NULL;
		}
		else if ((sg->a1 - Sh->a1) <= 0 && 0 <= (Sg->a1 - sh->a1)) {
			// no feasible
			printf("No feasible solution!\n");
			exit(0);
			return NULL;
		}
	}
	else if (yPrimeG < yPrimeH) {   // feasible
		if (sg->a1 > 0) {
			// x* < x'
			if (sg != Sg) {
				Sg->pruneFlag = false;
			}
			else {
				if (pairsG[index].line1.a1 < pairsG[index].line2.a1) {
					//pairsG[index].line2.pruneFlag = false;
					I1[pairsG[index].index2].pruneFlag = false;
				}
				else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {
					//pairsG[index].line1.pruneFlag = false;
					I1[pairsG[index].index1].pruneFlag = false;
				}
			}
			*rightBound = xPrimeG;
			//cout << "eeeeeeeeeeeeeeeee\n";
			return NULL;
		}
		else if (Sg->a1 < 0) {
			// x* > x'
			if (sg != Sg) {
				sg->pruneFlag = false;
			}
			else {
				if (pairsG[index].line1.a1 < pairsG[index].line2.a1) {
					//pairsG[index].line1.pruneFlag = false;
					I1[pairsG[index].index1].pruneFlag = false;
				}
				else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {

					I1[pairsG[index].index2].pruneFlag = false;
				}
			}
			*leftBound = xPrimeG;

			return NULL;
		}
		else if (sg->a1 <= 0 && 0 <= Sg->a1) {
			// x* = x'
			Solution.x = xPrimeG;
			Solution.y = yPrimeG;
			//cout << "hhhhhhhhhhhhhh\n";
			return &(Solution);
		}
	}
	return NULL;
}


///////////////////////////////////////////////////////////////////////////////////
static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}


bool Rotation_d(struct Line lines[], struct Objfunc object, int index, int *numG, int *numH)
{
	bool ret;

	// Original Constraints
	struct Line *dev_oConstraints;
	unsigned int size = index * sizeof(struct Line);

	HANDLE_ERROR(cudaMalloc((void**)&dev_oConstraints, size));

	// Lines after Rotation_d
	struct Line *dev_lines;

	HANDLE_ERROR(cudaMalloc((void**)&dev_lines, size));

	// Objective function
	struct Objfunc *dev_object;

	HANDLE_ERROR(cudaMalloc((void**)&dev_object, sizeof(struct Objfunc)));

	// Numbers of lines
	int *dev_index;

	HANDLE_ERROR(cudaMalloc((void**)&dev_index, sizeof(int)));

	// Num of G lines
	int *dev_numG;

	HANDLE_ERROR(cudaMalloc((void**)&dev_numG, sizeof(int)));

	// Num of H lines
	int *dev_numH;

	HANDLE_ERROR(cudaMalloc((void**)&dev_numH, sizeof(int)));

	// Space distribution
	unsigned int DIM = 1 + sqrt(index) / 16;

	dim3 blocks(DIM, DIM);
	dim3 threads(16, 16);

	(*numG) = (*numH) = 0;

	//float time_elapsed = 0;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);    
	//cudaEventCreate(&stop);

	//cudaEventRecord(start, 0);  

	//cudaEventRecord(stop, 0);    

	//cudaEventSynchronize(start);   
	//cudaEventSynchronize(stop);    
	//cudaEventElapsedTime(&time_elapsed, start, stop);    




	// Copy from CPU to GPU
	HANDLE_ERROR(cudaMemcpy(dev_oConstraints, &originalConstraints[0], size, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(dev_object, &object, sizeof(struct Objfunc), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(dev_index, &index, sizeof(int), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(dev_numG, numG, sizeof(int), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(dev_numH, numH, sizeof(int), cudaMemcpyHostToDevice));

	// Kernel function <<<blocks, threads>>>
	kRotation << <blocks, threads >> >(dev_oConstraints, dev_lines, dev_object, dev_index, dev_numG, dev_numH);

	// Copy from GPU to CPU
	HANDLE_ERROR(cudaMemcpy(numG, dev_numG, sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaMemcpy(numH, dev_numH, sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaMemcpy(lines, dev_lines, size, cudaMemcpyDeviceToHost));
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//printf("%f(ms)\n", time_elapsed);

	//printf("%d %d\n", (*numG), (*numH));

	if ((*numH) + (*numG) != index) {
		ret = false;
	}
	else {
		ret = true;
	}

	return ret;
}


void LinearProgramming(void)
{
	//float time_elapsed = 0;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	//cudaEventRecord(start, 0);

	//cudaEventRecord(stop, 0);

	//cudaEventSynchronize(start);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&time_elapsed, start, stop);
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int indexRecord = 0;
	int numGRecord;
	int numHRecord;
	int indexPair;
	double leftBound, rightBound;
	double aTemp, bTemp, cTemp;
	bool judge = false;
	struct Objfunc object;

	//int round = 0;
	FILE* fp;

	fp = fopen("Coefficient.txt", "r");

	while (1) {
		fscanf_s(fp, "%lf%lf%lf", &aTemp, &bTemp, &cTemp);
		if (aTemp == 0.0 && bTemp == 0.0 && cTemp == 0.0) {
			break;
		}
		struct Line lineTemp;
		lineTemp.a1 = aTemp;
		lineTemp.a2 = bTemp;
		lineTemp.b = cTemp;
		originalConstraints.push_back(lineTemp);
		indexRecord++;
	}
	fscanf_s(fp, "%lf%lf", &object.c1, &object.c2);
	fscanf_s(fp, "%lf%lf", &leftBound, &rightBound);

	//cout << "lalala\n";

	struct Line *lines = (struct Line *)malloc(indexRecord * sizeof(struct Line));
	struct Line *I1 = (struct Line *)malloc(indexRecord * sizeof(struct Line));
	struct Line *I2 = (struct Line *)malloc(indexRecord * sizeof(struct Line));
	struct Pair *pairG = (struct Pair *)malloc(indexRecord * sizeof(struct Pair));
	struct Pair *pairH = (struct Pair *)malloc(indexRecord * sizeof(struct Pair));
	struct Vertex *sln = NULL;

	judge = Rotation_d(lines, object, indexRecord, &numGRecord, &numHRecord);
	if (judge == false) {
		printf("Fatal Error at LinearProgramming() - Rotation_d()!\n");
		exit(-1);
	}

	judge = Separation(I1, I2, lines, numGRecord, numHRecord);
	if (judge == false) {
		printf("Fatal Error at LinearProgramming() - Segmentation()!\n");
		exit(-1);
	}

	//cout << numGRecord << " " << numHRecord << '\n';
	/*
	for (int i = 0; i < numGRecord; I++) {
	printf("")
	}
	*/

	while (1) {
		judge = MakePairs(I1, I2, pairG, pairH, numGRecord, numHRecord, &indexPair, leftBound, rightBound);
		if (judge == false) {
			printf("Fatal Error at LinearProgramming() - MakePairs()!\n");
			exit(-1);
		}

		sln = TestingLine(pairG, pairH, I1, I2, numGRecord, numHRecord, indexPair, &leftBound, &rightBound);
		//cout << leftBound << " " << rightBound << '\n';
		if (sln != NULL) {
			break;
		}
	}
	//printf("sln: %lf %lf\n", sln->x, sln->y);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << elapsedTime << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(fp);
	return;
}


int main()
{
	LinearProgramming();

	return 0;
}

