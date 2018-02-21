/*
 * main.cpp
 *
 *  Created on: 2017��?10��?25��?
 *      Author: yfji
 */

#include "DigitDetector.h"
#include "TwoLayerNNFaster.h"
#include <fstream>
using namespace std;

cv::Mat biMap,not_biMap, dummyMap;
int k_dilate=9;
int k_erode=9;
Mat dilate_kernel=cv::getStructuringElement(cv::MORPH_RECT, Size(k_dilate,k_dilate), Point(-1,-1));
Mat erode_kernel=cv::getStructuringElement(cv::MORPH_RECT, Size(k_erode,k_erode), Point(-1,-1));

vector<std::tuple<int, double, cv::Rect> > findPrintDigitAreasWarp(TwoLayerNNFaster& nn, DigitDetector& detector, Mat& frame ){
	detector.binarize(frame, biMap);
	cv::bitwise_not(biMap, not_biMap);
	imshow("bi", biMap);
	dummyMap=not_biMap.clone();
	IplImage ipl=dummyMap;
	CvMemStorage* pStorage=cvCreateMemStorage(0);
	CvSeq * pContour=NULL;
	cvFindContours(&ipl, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	vector<std::tuple<int,double, cv::Rect> > res;
	vector<Point2f> _rect={Point2f(0,0),Point2f(0,120),Point2f(100,120),Point2f(100,0)};
	Mat charRegion;
	bool hasRegion=false;
	for(;pContour;pContour=pContour->h_next){
		CvRect bbox=cvBoundingRect(pContour,0);
		if(bbox.width<30 || bbox.height<30)
			continue;
		if(bbox.width>bbox.height || bbox.height>1.6*bbox.width)
			continue;
		const vector<Point2f>& corners=detector.findRectFromContour(pContour);
		
		Mat m = getPerspectiveTransform(corners, _rect);
		
		warpPerspective(biMap, charRegion, m, Size(100,120));

		cv::threshold(charRegion, charRegion, 50, 255, cv::THRESH_OTSU);
		cv::dilate(charRegion, charRegion, dilate_kernel);
		int pred=-1;
		double prob=0.0;
		//res.push_back(std::make_tuple(pred, prob, Rect(r)));
		
		nn.predict(charRegion, pred, prob);
		if(pred>=0 and pred<10 and prob>0.7){
			res.push_back(std::make_tuple(pred, prob, Rect(bbox)));
			hasRegion=true;
		}	
	}
	cvReleaseMemStorage(&pStorage);
	if(hasRegion)
		imshow("warp", charRegion);
	return res;
}

vector<std::tuple<int, double, cv::Rect> > findPrintDigitAreasByContour(TwoLayerNNFaster& nn, DigitDetector& detector, Mat& frame){
	detector.binarize(frame,biMap);
	imshow("bi", biMap);
	cv::bitwise_not(biMap, not_biMap);
	vector<Rect> digits=detector.getCandidateBndBoxes(not_biMap);
	vector<std::tuple<int, double, cv::Rect> > res;
	int pred;
	double prob;
	for(size_t i=0;i<digits.size();++i){
		Rect& digit=digits[i];
		Mat roi=biMap(digit);
		nn.predict(roi, pred, prob);
		if(pred>=0 and pred<10 and prob>0.9){
			res.push_back(std::make_tuple(pred, prob, digit));
		}
	}
	return res;
}

vector<std::tuple<int, double, cv::Rect>> findPrintDigitAreasByRect(TwoLayerNNFaster& nn, DigitDetector& detector, Mat& frame){
	detector.binarize(frame,biMap);
	imshow("bi", biMap);
	cv::bitwise_not(biMap, not_biMap);
	//vector<Rect> digits=detector.getCandidateBndBoxes(not_biMap);
	//vector<Rect> digits=detector.detectAndWarpRectangles(biMap);
	vector<vector<Point2f>> PolyCandidates;
	vector<Rect> rects;
	detectRectangles(biMap, PolyCandidates);
	//detector.filterCandidates(PolyCandidates, biMap.size());
	vector<std::tuple<int,double, cv::Rect> > res;
	int _w=100,_h=120;
	int pad=3;
	vector<Point2f> _rect={Point2f(0,0),Point2f(0,_h), \
			Point2f(_w,_h),Point2f(_w,0)};

	bool hasDigit=false;

	Mat charRegion, roi;
	
	for(size_t i=0;i<PolyCandidates.size();++i){
		vector<Point2f> & poly=PolyCandidates[i];

		Rect r=detector.points2Rect(poly, biMap.size());
		if(r.width>r.height || r.height>1.6*r.width)
			continue;
		
		Mat m = getPerspectiveTransform(poly, _rect);
		
		warpPerspective(biMap, charRegion, m, Size(_w,_h));
		cv::threshold(charRegion, charRegion, 50, 255, cv::THRESH_OTSU);
		roi=charRegion(Rect(pad,pad,_w-2*pad,_h-2*pad));
		uchar* ptr=roi.data;
		int topBlack=0;
		bool blackFrame=false;
		for(int i=0;i<7;++i){
			if(((int)ptr[i*_w+_w/2])==0)
				++topBlack;
		}
		if(topBlack<5)
			continue;
		//cv::dilate(roi, roi, dilate_kernel);
		int pred=-1;
		double prob=0.0;
		//res.push_back(std::make_tuple(pred, prob, Rect(r)));
		//hasDigit=true;
		
		nn.predict(roi, pred, prob);
		if(pred>=0 and pred<10 and prob>0.7){
			res.push_back(std::make_tuple(pred, prob, Rect(r)));
			hasDigit=true;
		}
	}
	if(hasDigit)
		imshow("warp", roi);
	return res;
}

vector<std::tuple<int, double, cv::Rect>> findPrintDigitAreasNoGrid(TwoLayerNNFaster& nn, DigitDetector& detector, Mat& frame){
	detector.binarize(frame,biMap);
	imshow("bi", biMap);
	cv::bitwise_not(biMap, not_biMap);
	vector<Rect> digits=detector.getCandidateBndBoxes(not_biMap);
	vector<std::tuple<int, double, cv::Rect>> res;
	for(size_t i=0;i<digits.size();++i){
		Rect& r=digits[i];
		Mat roi=biMap(r);
		int pred;
		double prob;
		nn.predict(roi, pred, prob);
		if(pred>=0 and pred<10 and prob>0.9){
			res.push_back(std::make_tuple(pred, prob, Rect(r)));
		}
	}
	return res;
}

vector<std::tuple<int, double, cv::Rect>> findLedDigitAreas(TwoLayerNNFaster& nn, DigitDetector& detector, Mat& frame){
	//detector.binarizeHSV(frame,biMap);
	Mat img=frame.clone();
	detector.computeSaliencyFT(img,biMap);
	cv::dilate(biMap, biMap, dilate_kernel);
	cv::erode(biMap, biMap, erode_kernel);
	imshow("bi", biMap);
	cv::bitwise_not(biMap, not_biMap);
	vector<Rect> digits=detector.getCandidateBndBoxes(biMap);
	vector<std::tuple<int, double, cv::Rect> > res;
	for(size_t i=0;i<digits.size();++i){
		Rect& r=digits[i];
		Mat roi=not_biMap(r);
		if(roi.rows>2*roi.cols){
			cv::resize(roi, roi, Size(96,150), INTER_CUBIC);
		}
		int pred;
		double prob;
		nn.predict(roi, pred, prob);
		if(pred>=0 and pred<10 and prob>0.9){
			res.push_back(std::make_tuple(pred, prob, Rect(r)));
		}
	}
	return res;
}

//find the digit using white pixels
//recognize the digit using black pixels
int main(int argc, char** argv){
	VideoCapture cap;
	stringstream ss;
	ss<<"/home/jyf/Workspace/C++/UAV2017/Debug/videos/"<<argv[1];
	cap.open(ss.str());
	DigitDetector detector;
	TwoLayerNNFaster nn_print(TEST);
	TwoLayerNNFaster nn_led(TEST);
	nn_print.loadParams("./nn_params_hist_iter_12000_180.txt");
	nn_led.loadParams("./nn_params_hist_iter_12000_180_no_box.txt");
	int fps=cap.get(CV_CAP_PROP_FPS);
	cout<<"fps: "<<fps<<endl;
	bool ok;
	char key=0;
	Mat frame;//biMap, not_biMap;
	while(1){
		ok=cap.read(frame);
		if(not ok)
			break;
		//cap>>frame;
		//cv::resize(frame, frame, Size(frame.cols/2, frame.rows/2), cv::INTER_CUBIC);
		//detector.getCandidateBndBoxes(not_biMap) for printed digits with black pixels
		//detector.getCandidateBndBoxes(biMap) for led digits with white pixels
		//predict(biMap(rec), pred, prob) for printed digits with black pixels
		//predict(not_biMap(rec), pred, prob) for led digits with white pixels
		//vector<std::tuple<int,double, cv::Rect> >&& res=findLedDigitAreas(nn_led, detector, frame);
		vector<std::tuple<int,double, cv::Rect> >&& res=findPrintDigitAreasByRect(nn_print, detector, frame);
		for(size_t i=0;i<res.size();++i){
			Rect& r=std::get<2>(res[i]);
			stringstream ss;
			ss<<std::get<0>(res[i]);//<<": "<<std::get<1>(res[i]);
			cv::rectangle(frame, r, Scalar(0,0,255), 1);
			cv::putText(frame, ss.str(), Point(r.x, r.y-5), \
						cv::FONT_ITALIC, 0.8, Scalar(0,0,255), 1);
		}
		cv::imshow("frame", frame);
		key=waitKey(20);
		if(key==27)
			break;
		else if(key==' '){
			imwrite("./screenshot.jpg", frame);
		}
	}
	cap.release();
}


