/*
 * DigitDetector.cpp
 *
 *  Created on: 2017��??10��??25��??
 *      Author: yfji
 */

#include "DigitDetector.h"
#include <omp.h>

DigitDetector::DigitDetector() {
	// TODO Auto-generated constructor stub
}

DigitDetector::~DigitDetector() {
	// TODO Auto-generated destructor stub
}

void DigitDetector::RGBToLab(unsigned char * rgbImg, float * labImg)
{
	float B = gamma(rgbImg[0] / 255.0f);
	float G = gamma(rgbImg[1] / 255.0f);
	float R = gamma(rgbImg[2] / 255.0f);
	float X = 0.412453*R + 0.357580*G + 0.180423*B;
	float Y = 0.212671*R + 0.715160*G + 0.072169*B;
	float Z = 0.019334*R + 0.119193*G + 0.950227*B;

	X /= 0.95047;
	Y /= 1.0;
	Z /= 1.08883;

	float FX = X > 0.008856f ? pow(X, 1.0f / 3.0f) : (7.787f * X + 0.137931f);
	float FY = Y > 0.008856f ? pow(Y, 1.0f / 3.0f) : (7.787f * Y + 0.137931f);
	float FZ = Z > 0.008856f ? pow(Z, 1.0f / 3.0f) : (7.787f * Z + 0.137931f);
	labImg[0] = Y > 0.008856f ? (116.0f * FY - 16.0f) : (903.3f * Y);
	labImg[1] = 500.f * (FX - FY);
	labImg[2] = 200.f * (FY - FZ);
}
void DigitDetector::computeSaliencyFT(Mat& image, Mat& binaryMap){
	assert(image.channels() == 3);
	int scale=3;
	resize(image, image, Size(image.cols/scale,image.rows/scale), cv::INTER_CUBIC);
	image+=cv::Scalar(0,0,50);
	Mat saliencyMap(image.rows, image.cols, CV_32FC1);
	Mat lab, labf;
	int h = image.rows, w = image.cols;
	labf.create(Size(w, h), CV_32FC3);
	uchar* fSrc = image.data;
	float* fLab = (float*)labf.data;
	float* fDst = (float*)saliencyMap.data;

	int stride = w * 3;
	/*
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < stride; j += 3) {
			RGBToLab(fSrc + i*stride + j, fLab + i*stride + j);
		}
	}
	float MeanL = 0, MeanA = 0, MeanB = 0;
	for (int i = 0; i < h; ++i) {
		int index = i*stride;
		for (int x = 0; x < w; ++x) {
			MeanL += fLab[index];
			MeanA += fLab[index + 1];
			MeanB += fLab[index + 2];
			index += 3;
		}
	}
*/
	float MeanL=0, MeanA=0, MeanB=0;
	for(int i=0;i<h;++i){
		int index=i*stride;
		for(int j=0;j<stride;j+=3){
			RGBToLab(fSrc + i*stride + j, fLab + i*stride + j);
			MeanL+=fLab[index];
			MeanA+=fLab[index+1];
			MeanB+=fLab[index+2];
			index+=3;
		}
	}
	MeanL /= (w * h);
	MeanA /= (w * h);
	MeanB /= (w * h);
	//GaussianBlur(labf, labf, Size(5, 5), 1);
	for (int Y = 0; Y < h; Y++)
	{
		int Index = Y * stride;
		int CurIndex = Y * w;
		for (int X = 0; X < w; X++)
		{
			fDst[CurIndex++] = (MeanL - fLab[Index]) *  \
				(MeanL - fLab[Index]) + (MeanA - fLab[Index + 1]) *  \
				(MeanA - fLab[Index + 1]) + (MeanB - fLab[Index + 2]) *  \
				(MeanB - fLab[Index + 2]);
			Index += 3;
		}
	}
	normalize(saliencyMap, saliencyMap, 0, 1, NORM_MINMAX);
	saliencyMap.convertTo(saliencyMap, CV_8UC1, 255);
	cv::threshold(saliencyMap, binaryMap, 50, 255, cv::THRESH_OTSU);

	resize(binaryMap, binaryMap, Size(w*scale,h*scale), INTER_CUBIC);
	cv::threshold(binaryMap, binaryMap, 50, 255, cv::THRESH_OTSU);
	//Mat erode_kernel=cv::getStructuringElement(cv::MORPH_RECT, Size(1,1), Point(-1,-1));
	//cv::erode(binaryMap, binaryMap, erode_kernel);
	//cv::bitwise_not(binaryMap, binaryMap);
}

void DigitDetector::sharpenOneChannel(Mat& image, Mat& sharpen){
	assert(image.channels()==1);
	float sharpen_data[9]={0,-2,0,-2,4,-2,0,-2,0};
	Mat sharpen_kernel(3,3,CV_32FC1,sharpen_data);
	Mat srcf(image.rows,image.cols, CV_32FC1);
	Mat sharpenf;
	image.convertTo(srcf,CV_32FC1, 1.0/255,0.0);
	cv::filter2D(srcf, sharpenf, -1, sharpen_kernel, Point(-1,-1));
	normalize(sharpenf, sharpenf, 0,1,NORM_MINMAX);
	sharpenf.convertTo(sharpen, CV_8UC1, 255.0,0.0);
}

void DigitDetector::sharpen(Mat& image, Mat& sharpenMat){
	if(image.channels()==1){
		sharpenOneChannel(image, sharpenMat);
	}
	else{
		vector<Mat> chns(3);
		vector<Mat> s_chns(3);
		cv::split(image, chns);
		for(int i=0;i<3;++i){
			sharpenOneChannel(chns[i], s_chns[i]);
		}
		cv::merge(s_chns, sharpenMat);
	}
}

void DigitDetector::binarize(Mat& image, Mat& binaryMap){
	Mat gray=image.clone();
	if(image.channels()==3){
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	}
	Mat kernel=cv::getStructuringElement(cv::MORPH_RECT, Size(3,3), Point(-1,-1));
	cv::erode(gray, binaryMap, kernel);
	threshold(binaryMap, binaryMap, 50, 255, cv::THRESH_OTSU);
	//threshold(binaryMap, binaryMap, 120, 255, cv::THRESH_BINARY);
	//cv::bitwise_not(binaryMap, binaryMap);
}
void DigitDetector::binarizeHSV(const Mat& image, Mat& binaryMap){
	assert(image.channels()==3);
	Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	vector<Mat> planes(3);
	cv::split(hsvImage, planes);
	imshow("s",planes[1]);
	cv::threshold(planes[1],binaryMap,50,255,cv::THRESH_OTSU);
}

vector<Rect> DigitDetector::getCandidateBndBoxes(Mat& binaryMap){
	vector<Rect> digitRects;
	IplImage ipl=binaryMap;
	CvMemStorage* pStorage=cvCreateMemStorage(0);
	CvSeq * pContour=NULL;
	cvFindContours(&ipl, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for(;pContour;pContour=pContour->h_next){
		//double inner_min_area=1e4;
		CvRect bbox=cvBoundingRect(pContour,0);

		if(bbox.height>binaryMap.rows-10 or bbox.width>binaryMap.cols-10){
			cvSeqRemove(pContour, 0);
			continue;
		}
		if(bbox.width>bbox.height or 1.0*bbox.height>bbox.width*4){
			cvSeqRemove(pContour, 0);
			continue;
		}
		//double real_area=fabs(cvContourArea(pContour));
		int area=bbox.width*bbox.height;
		if(area<2500){
			cvSeqRemove(pContour, 0);
			continue;
		}
		digitRects.push_back(Rect(bbox));
		cvSeqRemove(pContour, 0);
	}
	cvReleaseMemStorage(&pStorage);
	nms(digitRects);
	return digitRects;
}

Rect DigitDetector::points2Rect(vector<Point2f>& poly, Size sz){
	int ltx=10000,lty=10000,rbx=-10000,rby=-10000;
	for(int i=0;i<4;++i){
		int x=(int)poly[i].x;
		int y=(int)poly[i].y;
		if(x>rbx)
			rbx=x;
		if(y>rby)
			rby=y;
		if(x<ltx)
			ltx=x;
		if(y<lty)
			lty=y;
	}
	ltx=max(0,ltx);
	lty=max(0,lty);
	rbx=min(rbx,sz.width);
	rby=min(rby,sz.height);
	Rect r(ltx,lty,rbx-ltx,rby-lty);
	return r;
}

void DigitDetector::filterCandidates(vector<vector<Point2f> >& polyCandidates, Size sz){
	vector<int> deleteIdx;
	vector<char> flags(polyCandidates.size());
	vector<Rect> rectCandidates;
	for(size_t i=0;i<polyCandidates.size();++i){
		vector<Point2f>& poly=polyCandidates[i];
		Rect r=points2Rect(poly, sz);
		rectCandidates.push_back(r);
	}
	int areai,areaj,nms_area,nms_rbx,nms_rby,nms_ltx,nms_lty, nms_w, nms_h;
	for(size_t i=0;i<rectCandidates.size();++i){
		Rect& reci=rectCandidates[i];
		areai=reci.width*reci.height;
		for(size_t j=i+1;j<rectCandidates.size();++j){
			//if(i==j)	continue;
			Rect& recj=rectCandidates[j];
			areaj=recj.width*recj.height;
			nms_rbx=min(recj.x+recj.width-1, reci.x+reci.width-1);
			nms_rby=min(recj.y+recj.height-1, reci.y+reci.height-1);
			nms_ltx=max(recj.x,reci.x);
			nms_lty=max(recj.y,reci.y);
			nms_w=max(nms_rbx-nms_ltx,0);
			nms_h=max(nms_rby-nms_lty,0);
			nms_area=nms_w*nms_h;
			float ratio2i=1.0*nms_area/areai;
			float ratio2j=1.0*nms_area/areaj;
			if(ratio2i>0.8 and ratio2j>0.8){
				if(ratio2i>ratio2j and not flags[i]){
					deleteIdx.push_back(i);
					flags[i]=1;
				}
				else if(ratio2i<ratio2j and not flags[j]){
					deleteIdx.push_back(j);
					flags[j]=1;
				}
			}
			else if(ratio2i>0.8 and not flags[i]){
				deleteIdx.push_back(i);
				flags[i]=1;
			}
			else if(ratio2j>0.8 and not flags[j]){
				deleteIdx.push_back(j);
				flags[j]=1;
			}
		}
	}
	if(deleteIdx.size()>0){
		auto beginner=polyCandidates.begin();
		sort(deleteIdx.begin(), deleteIdx.end(), Compare);
		int k=0;
		for(size_t i=0;i<deleteIdx.size();++i){
			k=deleteIdx[i]-i;
			polyCandidates.erase(beginner+k);
		}
	}
}

vector<Point2f> DigitDetector::findRectFromContour(CvSeq* pContour){
	Point2f points[4];
	CvBox2D box_outer = cvMinAreaRect2(pContour);
	//cvBoxPoints(box_outer, points);

	vector<Point2f> corners(4);
	RotatedRect rotatedRect(box_outer);
	rotatedRect.points(points);
	int inds[4]={1,0,3,2};
	for(int i=0;i<4;++i){
		corners[i]=Point2f(points[inds[i]].x,points[inds[i]].y);
	}
	return corners;
}
/*
vector<Point2f> DigitDetector::findRectFromContour(CvSeq* pContour){
	int pty[4];
	int ptx[4];
	int ymax=-100,s_ymax=-100,ymin=10000,s_ymin=10000;
	int max_index=0,s_max_index=0,min_index=0,s_min_index=0;
	int x,y;
	vector<Point2f> corners(4);
	for(int p=0;p<pContour->total;++p){
		CvPoint* pt=(CvPoint*)cvGetSeqElem(pContour, p);
		x=pt->x;
		y=pt->y;
		if(y>ymax){
			s_ymax=ymax;
			ymax=y;
			s_max_index=max_index;
			max_index=p;
		}
		else if(y<=ymax && y>s_ymax){
			s_ymax=y;
			s_max_index=p;
		}
		if(y<ymin){
			s_ymin=ymin;
			ymin=y;
			s_min_index=min_index;
			min_index=p;
		}
		else if(y>=ymin && y<s_ymin){
			s_ymin=y;
			s_min_index=p;
		}
	}
	int ymax_x=((CvPoint*)cvGetSeqElem(pContour, max_index))->x;
	int s_ymax_x=((CvPoint*)cvGetSeqElem(pContour, s_max_index))->x;
	int ymin_x=((CvPoint*)cvGetSeqElem(pContour, min_index))->x;
	int s_ymin_x=((CvPoint*)cvGetSeqElem(pContour, s_min_index))->x;
	if(ymax_x<s_ymax_x){
		corners[0]=Point2f(1.0*ymax_x,1.0*ymax);
		corners[3]=Point2f(1.0*s_ymax_x,1.0*s_ymax);
	}
	else{	//ymax_x>s_ymax_x
		corners[0]=Point2f(1.0*s_ymax_x,1.0*s_ymax);
		corners[3]=Point2f(1.0*ymax_x,1.0*ymax);
	}
	if(ymin_x<s_ymin_x){
		corners[1]=Point2f(1.0*ymin_x,1.0*ymin);
		corners[2]=Point2f(1.0*s_ymin_x,1.0*s_ymin);
	}
	else{	//ymin_x>s_ymin_x
		corners[1]=Point2f(1.0*s_ymin_x,1.0*s_ymin);
		corners[2]=Point2f(1.0*ymin_x,1.0*ymin);
	}
	return corners;
}
*/
