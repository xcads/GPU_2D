#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <limits.h>
#include <iostream>

#define RANDOM_SEED  10
#define RANDOM_PARA  2000
#define RANDOM_LEFT  -1000
#define RANDOM_RIGHT 1000

#define DATA_NUM 10000

#define PI 3.141592653589793238462343383279502884197169399375
using namespace std;
struct Point
{
	double x, y;
};

struct Line {
	double a1, a2, b;
	double slope;

	bool pruneFlag;
	int index;
};

struct AnswerXY
{
	int index1;
	struct Line line1;
	int index2;
	struct Line line2;

	struct Point interv;

	struct Line objfunc;
};

typedef int BOOL;
typedef struct Point Point;
typedef struct Line Line;
typedef struct AnswerXY AnswerXY;

struct AnswerXY answer;

struct Line dataset[DATA_NUM];

int getRandomSeed(int *randomSeed)
{
	(*randomSeed) += RANDOM_PARA;
	if ((*randomSeed == 0)) {
		(*randomSeed) = (int)time(NULL) % RANDOM_SEED + RANDOM_SEED*RANDOM_SEED;
	}
	return ((*randomSeed)*(int)time(NULL));
}

double getRandomDouble(int *randomSeed, double leftBound, double rightBound)
{
	srand(getRandomSeed(randomSeed));
	return(leftBound + (rightBound - leftBound)*rand() / (RAND_MAX));
}

int getRandomInt(int *randomSeed, int Bound)
{
	srand(getRandomSeed(randomSeed));
	return (rand() % Bound);
}

bool intersection(struct Line *l1, struct Line *l2, struct Point *v)
{
	if (fabs(l1->a1*l2->a2 - l2->a1*l1->a2) < DBL_EPSILON)
	{
		v = NULL;
		return false;
	}
	v->x = (l1->a2*l2->b - l2->a2*l1->b) / (l1->a1*l2->a2 - l2->a1*l1->a2);
	v->y = (l2->b*l1->a1 - l1->b*l2->a1) / (l1->a1*l2->a2 - l2->a1*l1->a2);
	return true;
}

bool Slope(struct Line *l)
{
	if (fabs(l->a2 - 0.0) < DBL_EPSILON)
	{
		if (l->a1 > 0 && l->a2 != 0) {
			l->slope = DBL_MAX;
		}
		else if (l->a1 < 0 && l->a2 != 0) {
			l->slope = -DBL_MAX;
		}
		return false;
	}
	else
		l->slope = -l->a1 / l->a2;
	return true;
}

BOOL generate(struct Line *l1, struct Line *l2, struct Point *v, int *randomSeed)
{
	l1->a1 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	l1->a2 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	l1->b = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	Slope(l1);

	l2->a1 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	l2->a2 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	l2->b = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	Slope(l2);

	while (fabs(l1->slope - l2->slope)<DBL_EPSILON)
	{
		l2->a1 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
		l2->a2 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
		l2->b = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
		Slope(l2);
	}
	intersection(l1, l2, v);
	if (l1->a2 > 0)
	{

		answer.line1.a1 = l1->a1;
		answer.line1.a2 = l1->a2;
		answer.line1.b = l1->b;
		answer.line1.slope = l1->slope;

	}
	else {
		answer.line1.a1 = -l1->a1;
		answer.line1.a2 = -l1->a2;
		answer.line1.b = -l1->b;
		answer.line1.slope = l1->slope;
	}
	if (l2->a2 > 0)
	{

		answer.line2.a1 = l2->a1;
		answer.line2.a2 = l2->a2;
		answer.line2.b = l2->b;
		answer.line2.slope = l2->slope;

	}
	else {
		answer.line2.a1 = -l2->a1;
		answer.line2.a2 = -l2->a2;
		answer.line2.b = -l2->b;
		answer.line2.slope = l2->slope;
	}
	answer.interv.x = v->x;
	answer.interv.y = v->y;

	answer.index1 = getRandomInt(randomSeed, DATA_NUM);
	answer.index2 = getRandomInt(randomSeed, DATA_NUM);

	while (answer.index1 == answer.index2)
	{
		answer.index2 = getRandomInt(randomSeed, DATA_NUM);
	}

	answer.objfunc.a1 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	answer.objfunc.a2 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	answer.objfunc.b = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	Slope(&(answer.objfunc));

	double maxSlope = std::max(answer.line1.slope, answer.line2.slope);
	double minSlope = std::min(answer.line1.slope, answer.line2.slope);

	while (!((fabs(answer.objfunc.slope - answer.line1.slope) > DBL_EPSILON&&fabs(answer.objfunc.slope - answer.line2.slope) > DBL_EPSILON &&
		(answer.objfunc.slope) > minSlope && (answer.objfunc.slope) < maxSlope)))
	{
		answer.objfunc.a1 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
		answer.objfunc.a2 = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
		answer.objfunc.b = getRandomDouble(randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
	}
	if (answer.objfunc.a2 < 0)
	{
		answer.objfunc.a1 = -answer.objfunc.a1;
		answer.objfunc.a2 = -answer.objfunc.a2;
		answer.objfunc.b = -answer.objfunc.b;
	}
	return true;
}

bool Rotation(struct AnswerXY object, double *x, double *y, struct Line *l1, struct Line *l2)
{
	double theArc, theDec;
	if (object.objfunc.a2 == 0 && object.objfunc.a1 > 0) {
		theArc = -PI / 2;
		theDec = 90;
	}
	else if (object.objfunc.a2 == 0 && object.objfunc.a1 < 0)
	{
		theArc = PI / 2;
		theDec = 90;
	}
	else {
		theArc = atan(-object.objfunc.a1 / object.objfunc.a2);
		theDec = atan(-object.objfunc.a1 / object.objfunc.a2) * 180 / PI;
	}

	(*x) = cos(theArc)*object.interv.x + sin(theArc)*object.interv.y;
	(*y) = cos(theArc)*object.interv.y - sin(theArc)*object.interv.x;

	l1->a1 = cos(theArc)*object.line1.a1 + sin(theArc)*object.line1.a2;
	l1->a2 = cos(theArc)*object.line1.a2 - sin(theArc)*object.line1.a1;
	l1->b = object.line1.b;

	l2->a1 = cos(theArc)*object.line2.a1 + sin(theArc)*object.line2.a2;
	l2->a2 = cos(theArc)*object.line2.a2 - sin(theArc)*object.line2.a1;
	l2->b = object.line2.b;

	return true;
}
//
//int main()
//{
//
//	struct Line *l1 = (struct Line *)malloc(sizeof(struct Line));
//	struct Line *l2 = (struct Line *)malloc(sizeof(struct Line));
//	struct Point *v = (struct Point *)malloc(sizeof(struct Point));
//
//	int randomSeed = RANDOM_SEED;
//
//	generate(l1, l2, v, &randomSeed);
//	std::cout << answer.line1.a1 << " " << answer.line1.a2 << " " << answer.line1.b << " " << answer.line1.slope << '\n';
//	std::cout << answer.line2.a1 << " " << answer.line2.a2 << " " << answer.line2.b << " " << answer.line2.slope << '\n';
//	std::cout << v->x << " " << v->y << '\n';
//	std::cout << answer.index1 << " " << answer.index2 << '\n';
//	std::cout << answer.objfunc.a1 << " " << answer.objfunc.a2 << " " << answer.objfunc.b << " " << answer.objfunc.slope << '\n';
//
//	free(l1);
//	free(l2);
//	free(v);
//
//	for (int i = 0; i < DATA_NUM; i++) {
//		if (i == answer.index1) {
//			dataset[i].a1 = answer.line1.a1;
//			dataset[i].a2 = answer.line1.a2;
//			dataset[i].b = answer.line1.b;
//			dataset[i].slope = answer.line1.slope;
//			continue;
//		}
//		if (i == answer.index2) {
//			dataset[i].a1 = answer.line2.a1;
//			dataset[i].a2 = answer.line2.a2;
//			dataset[i].b = answer.line2.b;
//			dataset[i].slope = answer.line2.slope;
//			continue;
//		}
//		dataset[i].a1 = getRandomDouble(&randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
//		dataset[i].a2 = getRandomDouble(&randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
//		dataset[i].b = getRandomDouble(&randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
//		Slope(&dataset[i]);
//
//		while (!(((dataset[i].a2 > 0) && (dataset[i].a1*answer.interv.x + dataset[i].a2*answer.interv.y > dataset[i].b)) || ((dataset[i].a2 < 0) && (dataset[i].a1*
//			answer.interv.x + dataset[i].a2*answer.interv.y > dataset[i].b))) || (fabs(dataset[i].a1*answer.interv.x + dataset[i].a2*answer.interv.y - dataset[i].b) < DBL_EPSILON)) {
//			dataset[i].a1 = getRandomDouble(&randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
//			dataset[i].a2 = getRandomDouble(&randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
//			dataset[i].b = getRandomDouble(&randomSeed, RANDOM_LEFT, RANDOM_RIGHT);
//			Slope(&dataset[i]);
//
//		}
//		for (int j = 0; j < 200000; j++) {
//			;
//		}
//	}
//
//	std::cout << "\n dddd\n\n";
//
//	std::cout << dataset[0].a1 << " " << dataset[0].a2 << " " << dataset[0].b << " " << dataset[0].slope << '\n';
//	std::cout << dataset[1].a1 << " " << dataset[1].a2 << " " << dataset[1].b << " " << dataset[1].slope << '\n';
//	std::cout << dataset[DATA_NUM - 2].a1 << " " << dataset[DATA_NUM - 2].a2 << " " << dataset[DATA_NUM - 2].b << " " << dataset[DATA_NUM - 2].slope << '\n';
//	std::cout << dataset[DATA_NUM - 1].a1 << " " << dataset[DATA_NUM - 1].a2 << " " << dataset[DATA_NUM - 1].b << " " << dataset[DATA_NUM - 1].slope << '\n';
//
//	double rx, ry;
//	struct Line *ll1 = (struct Line *)malloc(sizeof(struct Line));
//	struct Line *ll2 = (struct Line *)malloc(sizeof(struct Line));
//	struct Point *vv = (struct Point *)malloc(sizeof(struct Point));
//
//	Rotation(answer, &rx, &ry, ll1, ll2);
//	intersection(ll1, ll2, vv);
//
//	FILE *fpt;
//	fopen_s(&fpt, "Coefficient.txt", "w");
//	for (int i = 0; i < DATA_NUM; i++) {
//		fprintf_s(fpt, "%lf\t%lf\t%lf\n", dataset[i].a1, dataset[i].a2, dataset[i].b);
//	}
//	fprintf_s(fpt, "0 0 0\n");
//	fprintf_s(fpt, "%lf\t%lf\n", answer.objfunc.a1, answer.objfunc.a2);
//	fprintf_s(fpt, "-10000 10000\n\n");
//	fprintf_s(fpt, "%lf  %lf\n ", rx, ry);
//	fprintf_s(fpt, "%lf  %lf\n ", answer.line1.a1, answer.line2.a2);
//	fprintf_s(fpt, "%lf  %lf\n ", vv->x, vv->y);
//	fclose(fpt);
//
//	return 0;
//}