/*
 * MLP.h
 *
 *  Created on: 2017年11月3日
 *      Author: yfji
 */

#ifndef SRC_MLP_H_
#define SRC_MLP_H_
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "util.h"
using namespace std;
using namespace cv;

#define TRAIN			0
#define TEST			1
#define FINETUNE	2

class MLP {
public:
	MLP(int mode=TRAIN, int batchIfTrain=32);
	MLP(vector<int>& num, int mode=TRAIN, int batchIfTrain=32);
	virtual ~MLP();

public:
	void train(int iters=1e6);
	void setLayerNumber(vector<int>&);
	void setModeAndBatch(int mode=TRAIN, int batchIfTrain=32);
	void predict(Mat& image, int& pred, double& prob);
	void saveParams(const char* filename);
	void loadParams(const char* filename);

public:
	int step;
	int display;
	double gamma;
	double lr;
	double reg;

public:
	int* nLayers;
	int mode;
	int minibatch;
	int numLayer;
	double* output;
	double* batchSamples;
	int* batchLabels;
	vector<double*> wLayers;
	vector<double*> bLayers;
	vector<double*> gWLayers;
	vector<double*> gBLayers;
	vector<double*> yLayers;
	vector<double*> dLayers;

	vector<string> sampleFiles;
	vector<int> labelMat;
	vector<int> ind;

	int curIndex;
	int numSamples;
	const int pool_h=10;
	const int pool_w=8;
	const int resz_h=20;
	const int resz_w=15;
	string sample_file_lst;

private:
	void shuffle();
	double forward();
	void backward();
	void reset(double std=1e-2);
	void initDefault();
	void alloc();
	void release();
	void preprocess(Mat& image);
	void getFeature(Mat& image, int batchIndex);
	void getFeatureX(Mat& image, int batchIndex);
	void loadSamplePathAndLabels();

};

#endif /* SRC_MLP_H_ */
