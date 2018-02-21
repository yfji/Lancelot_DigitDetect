/*
 * DigitDetector.h
 *
 *  Created on: 2017��?10��?25��?
 *      Author: yfji
 */

#ifndef SRC_DIGITDETECTOR_H_
#define SRC_DIGITDETECTOR_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "lancelot/imageTrans.h"
#include "lancelot/CharacterRecognition.h"
#include "util.h"
using namespace std;
using namespace cv;

class DigitDetector {
public:
	DigitDetector();
	virtual ~DigitDetector();

private:
	inline float gamma(float x){
		return x>0.04045 ? pow((x + 0.055f) / 1.055f, 2.4f) : x / 12.92;
	}
	void RGBToLab(unsigned char * rgbImg, float * labImg);
	void sharpenOneChannel(Mat& image, Mat& sharpen);
public:
	void computeSaliencyFT(Mat& image, Mat& binaryMap);
	void binarize(Mat& image, Mat& binaryMap);
	void binarizeHSV(const Mat& image, Mat& binaryMap);
	void sharpen(Mat& image, Mat& sharpenMap);	
	vector<Rect> getCandidateBndBoxes(Mat& binaryMap);
	Rect points2Rect(vector<Point2f>& poly, Size sz);
	void filterCandidates(vector<vector<Point2f> >& polyCanditates, Size sz);

	vector<Point2f> findRectFromContour(CvSeq* pContour);
};

#endif /* SRC_DIGITDETECTOR_H_ */
