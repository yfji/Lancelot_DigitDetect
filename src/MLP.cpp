/*
 * MLP.cpp
 *
 *  Created on: 2017年11月3日
 *      Author: yfji
 */

#include "MLP.h"
#include <cblas.h>

MLP::MLP(int mode, int batchIfTrain) {
	initDefault();
	setModeAndBatch(mode, batchIfTrain);
}
MLP::MLP(vector<int>& num, int _mode, int batchIfTrain){
	initDefault();
	setModeAndBatch(_mode, batchIfTrain);
	setLayerNumber(num);
	alloc();
	if(mode==TRAIN or mode==FINETUNE){
		reset();
		loadSamplePathAndLabels();
		shuffle();
	}
	cout<<"finish init"<<endl;
}
MLP::~MLP() {
	release();
}

void MLP::shuffle(){
	for(size_t i=0;i<ind.size();++i){
		int swap_ind=rand()%(ind.size());
		swap(ind[i],ind[swap_ind]);
	}
}

void MLP::setLayerNumber(vector<int>& num){
	numLayer=num.size();
	cout<<"Layer number: "<<numLayer<<endl;
	nLayers=new int[numLayer];
	for(int i=0;i<numLayer;++i)
		nLayers[i]=num[i];
	wLayers.resize(numLayer-1);
	bLayers.resize(numLayer);
	yLayers.resize(numLayer);
	if(mode!=TEST){
		gWLayers.resize(numLayer-1);
		gBLayers.resize(numLayer);
		dLayers.resize(numLayer);
	}
}

void MLP::setModeAndBatch(int _mode, int batchIfTrain){
	mode=_mode;
	if(mode==TRAIN or mode==FINETUNE){
		minibatch=batchIfTrain;
	}
	else
		minibatch=1;
}

void MLP::initDefault(){
	numLayer=-1;
	mode=TRAIN;
	minibatch=32;
	curIndex=0;
	step=2;
	display=10;
	gamma=1.0;
	lr=0.1;
	reg=1e-4;
	sample_file_lst="./samples_all_voc.txt";
}
void MLP::reset(double std){
	std::mt19937 gen(time(0));
	std::normal_distribution<double> n(0,1);
	int i,j;
	for(i=0;i<numLayer;++i){
		if(i<numLayer-1)
			for(j=0;j<nLayers[i]*nLayers[i+1];++j)
				wLayers[i][j]=std*n(gen);
		if(i>=1)
			for(j=0;j<nLayers[i];++j)
				bLayers[i][j]=0.0;
	}
}
void MLP::alloc(){
	//batchSamples=new double[minibatch*nLayers[0]];
	cout<<"batch size: "<<minibatch<<endl;
	bLayers[0]=NULL;
	if(mode!=TEST){
		dLayers[0]=NULL;
		gBLayers[0]=NULL;
	}
	for(int i=0;i<numLayer;++i){
		if(i<numLayer-1){
			wLayers[i]=new double[nLayers[i]*nLayers[i+1]];
			if(mode!=TEST)
				gWLayers[i]=new double[nLayers[i]*nLayers[i+1]];
		}
		yLayers[i]=new double[minibatch*nLayers[i]];
		if(i>=1){
			bLayers[i]=new double[nLayers[i]];
			if(mode!=TEST){
				gBLayers[i]=new double[nLayers[i]];
				dLayers[i]=new double[minibatch*nLayers[i]];
			}
		}
	}
	output=new double[minibatch*nLayers[numLayer-1]];
	if(mode!=TEST)
		batchLabels=new int[minibatch];
}

void MLP::release(){
	//delete batchSamples;
	if(mode!=TEST)
		delete batchLabels;
	delete output;
	for(int i=0;i<numLayer;++i){
		if(i<numLayer-1){
			delete wLayers[i];
			if(mode!=TEST){
				delete gWLayers[i];
			}
		}
		delete yLayers[i];
		if(i>=1){
			delete bLayers[i];
			if(mode!=TEST){
				delete dLayers[i];
				delete gBLayers[i];
			}
		}
	}
}

void MLP::getFeatureX(Mat& image, int batchIndex){
	assert(pool_h*pool_w==nLayers[0]);
	Mat digit;
	if(image.channels()==3)
		cv::cvtColor(image, digit, cv::COLOR_BGR2GRAY);
	else
		image.copyTo(digit);
	cv::threshold(digit, digit, 50,255, cv::THRESH_OTSU);
	uchar* digitdata=digit.data;
	int h=digit.rows;
	int w=digit.cols;
	int max_cnt=-100;
	double bin_size_h=static_cast<double>(h)/(pool_h);
	double bin_size_w=static_cast<double>(w)/(pool_w);

	int offset=batchIndex*nLayers[0];
	for(int ph=0;ph<pool_h;++ph){
		int start_h=std::max<int>(0,round(ph*bin_size_h));
		int end_h=std::min<int>(h,round((ph+1)*bin_size_h));
		for(int pw=0;pw<pool_w;++pw){
			int start_w=std::max<int>(0,round(pw*bin_size_w));
			int end_w=std::min<int>(w,round((pw+1)*bin_size_w));
			int cnt=0;
			for(int y=start_h;y<end_h;++y){
				for(int x=start_w;x<end_w;++x){
					if(digitdata[y*w+x]==0)	++cnt;
				}
			}
			if(cnt>max_cnt)	max_cnt=cnt;
			yLayers[0][offset+(ph*pool_w+pw)]=static_cast<double>(cnt);
		}
	}
	for(int i=0;i<nLayers[0]; ++i)	yLayers[0][offset+i]/=(1.0*max_cnt);
}

void MLP::loadSamplePathAndLabels(){
	ifstream in;
	in.open(sample_file_lst.c_str(), ios::in);
	string line;
	int cnt=0;
	cout<<"loading samples"<<endl;
	ifstream item_in;
	while(!in.eof()){
		std::getline(in, line);
		if(line.length()<5)	continue;
		item_in.open(line.c_str());
		if(not item_in){
			cout<<"file "<<line<<" not exists"<<endl;
			continue;
		}
		item_in.close();
		int i=line.find_last_of('/')+1;
		int index=int(line[i])-48;
		if(line[i+1]!='_'){
			char num[2]={line[i],line[i+1]};
			index=atoi(num);
		}
		sampleFiles.push_back(line);
		labelMat.push_back(index);
		//if(index==10)
		//	cout<<line<<": "<<index<<endl;
		++cnt;
	}
	numSamples=cnt;
	for(int i=0;i<numSamples;++i){
		ind.push_back(i);
	}
	cout<<numSamples<<" samples are loaded"<<endl;
}
double MLP::forward(){
	int offset=0;
	double sum_loss=0.0;
	/*
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, minibatch, nLayers[1], nLayers[0], \
			1.0, batchSamples, nLayers[0], wLayers[0], nLayers[1], 0.0, yLayers[1], nLayers[1]);
	//yLayer2
	for(int i=0;i<minibatch*nLayers[1];++i){
		if(yLayers[1][i]<0.0)
			yLayers[1][i]=0.0;
	}
	*/
	for(int i=0;i<numLayer-1;++i){
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, minibatch, nLayers[i+1], nLayers[i], \
			1.0, yLayers[i], nLayers[i], wLayers[i], nLayers[i+1], 0.0, yLayers[i+1], nLayers[i+1]);
		for(int j=0;j<minibatch*nLayers[i+1];++j){
			//yLayers[i+1][j]+=bLayers[i+1][j%nLayers[i+1]];
			if(yLayers[i+1][j]<0.0)
				yLayers[i+1][j]=0.0;
		}
	}
	offset=0;
	for(int i=0;i<minibatch;++i){
		double max_out=-1.0*1e3;
		double sum_out=0.0;
		for(int j=0;j<nLayers[numLayer-1];++j){
			if(yLayers[numLayer-1][j+offset]>max_out)
				max_out=yLayers[numLayer-1][j+offset];
		}
		for(int j=0;j<nLayers[numLayer-1];++j){
			output[j+offset]=std::exp(yLayers[numLayer-1][j+offset]-max_out);
			sum_out+=output[j+offset];
		}
		for(int j=0;j<nLayers[numLayer-1];++j){
			output[j+offset]/=sum_out;
		}
		sum_loss += -1.0*std::log(output[offset+batchLabels[i]]);
		output[offset+batchLabels[i]]-=1.0;
		for(int j=0;j<nLayers[numLayer-1];++j){
			dLayers[numLayer-1][j+offset]= output[j+offset]/minibatch;
		}
		offset+=nLayers[numLayer-1];
	}
	for(int i=numLayer-1;i>=1;--i){
		/*
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nLayers[1], nLayers[2], minibatch, \
			1.0, yLayer1, nLayers[1], dLayer2, nLayers[2], 0.0, gW2, nLayers[2]);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, minibatch, nLayers[1], nLayers[2],\
			1.0, dLayer2, nLayers[2], W2, nLayers[2], 0.0, dLayer1, nLayers[1]);
		 */
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nLayers[i-1], nLayers[i], minibatch, \
					1.0, yLayers[i-1], nLayers[i-1], dLayers[i], nLayers[i], 0.0, gWLayers[i-1], nLayers[i]);
		//dLayer1=dLayer2*W2T: [N*H]=[N*O].[O*H]
		if(i>=2){
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, minibatch, nLayers[i-1], nLayers[i],\
					1.0, dLayers[i], nLayers[i], wLayers[i-1], nLayers[i], 0.0, dLayers[i-1], nLayers[i-1]);
			for(int j=0;j<minibatch*nLayers[i-1];++j){
				if(yLayers[i-1][j]<1e-9)
					dLayers[i-1][j]=0.0;
			}
		}
		for(int j=0;j<nLayers[i];++j){
			gBLayers[i][j]=0.0;
			for(int k=0;k<minibatch;++k){
				gBLayers[i][j]+=dLayers[i][j+k*nLayers[i]];
			}
		}
	}
	sum_loss/=minibatch;
	return sum_loss;
}

void MLP::backward(){
	for(int i=0;i<numLayer;++i){
		if(i<numLayer-1){
			for(int j=0;j<nLayers[i]*nLayers[i+1];++j)
				wLayers[i][j]-=(lr*gWLayers[i][j]+reg*wLayers[i][j]);
		}
		if(i>=1){
			for(int j=0;j<nLayers[i];++j)
				bLayers[i][j]-=lr*gBLayers[i][j];
		}
	}
}
void MLP::train(int iters){
	int iter=0;
	int step_duration=0;
	if(step>100)
		step_duration=step;
	else
		step_duration=std::max(1,iters/step);
	cout<<"start train"<<endl;
	while(iter<iters){
		Rect rect;
		for(int k=curIndex;k<curIndex+min(minibatch,numSamples-curIndex);++k){
			Mat image=imread(sampleFiles[ind[k]]);
			//if(labelMat[ind[k]]!=10)
			//	rect=findDigitBndBox(image);
			//else
			rect=Rect(0,0,image.cols,image.rows);
			Mat digit=image(rect);
			getFeatureX(digit, k-curIndex);
			//getFeature(digit, k-curIndex);
			batchLabels[k-curIndex]=labelMat[ind[k]];
		}
		double loss=forward();
		backward();
		if(iter%display==0 or iter==iters-1)
			cout<<"iteration: "<<iter<<", loss: "<<loss<<endl;
		curIndex+=minibatch;
		if(curIndex>=numSamples){
			shuffle();
			curIndex=0;
		}
		++iter;
		if(iter%step_duration==0){
			lr*=gamma;
		}
	}
	cout<<endl<<"finished"<<endl;
}
void MLP::predict(Mat& image, int& pred, double& prob){
	int batch=1;
	getFeatureX(image, 0);
	for(int i=0;i<numLayer-1;++i){
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, nLayers[i+1], nLayers[i], \
			1.0, yLayers[i], nLayers[i], wLayers[i], nLayers[i+1], 0.0, yLayers[i+1], nLayers[i+1]);
		for(int j=0;j<minibatch*nLayers[i+1];++j){
			//yLayers[i+1][j]+=bLayers[i+1][j%nLayers[i+1]];
			if(yLayers[i+1][j]<0.0)
				yLayers[i+1][j]=0.0;
		}
	}
	double max_out=-1.0*1e3;
	double sum_out=0.0;
	for(int j=0;j<nLayers[numLayer-1];++j){
		if(yLayers[numLayer-1][j]>max_out){
			max_out=yLayers[numLayer-1][j];
			pred=j;
		}
	}
	for(int j=0;j<nLayers[numLayer-1];++j){
		output[j]=std::exp(yLayers[numLayer-1][j]-max_out);
		sum_out+=output[j];
	}
	for(int j=0;j<nLayers[numLayer-1];++j){
		output[j]/=sum_out;
	}
}
void MLP::saveParams(const char* filename){
	ofstream out;
	out.open(filename, ios::out);
	for(int i=0;i<numLayer;++i){
		out<<nLayers[i];
		if(i<numLayer-1)
			out<<' ';
		else
			out<<endl;
	}
	for(int i=0;i<numLayer-1;++i){
		out<<"params"<<endl;
		for(int j=0;j<nLayers[i]*nLayers[i+1];++j){
			out<<wLayers[i][j];
			if((j+1)%nLayers[i+1]==0)	out<<endl;
			else 	out<<' ';
		}
	}
	for(int i=1;i<numLayer;++i){
		out<<"bias"<<endl;
		for(int j=0;j<nLayers[i];++j){
			out<<bLayers[i][j];
			if(j<nLayers[i]-1)	out<<' ';
			else 	out<<endl;
		}
	}
}
void MLP::loadParams(const char* filename){
	cout<<"loading params..."<<endl;
	ifstream in;
	in.open(filename, ios::in);
	if(!in){
		cerr<<"no parameter file found, reset randomly"<<endl;
		reset();
	}
	string line;
	int layerIndexParams=-1;
	int layerIndexBias=0;
	int paramRow=0;

	int n;
	getline(in,line);
	stringstream ss(line);
	vector<int> layers;
	while(not ss.eof()){
		ss>>n;
		layers.push_back(n);
	}
	setLayerNumber(layers);
	alloc();
	while(!in.eof()){
		getline(in, line);
		if(line.length()<=1)	continue;
		if(line=="params"){
			++layerIndexParams;
			paramRow=0;
		}
		else if(line=="bias"){
			++layerIndexBias;
		}
		else if(layerIndexBias==0){
			stringstream ss(line);
			double param;
			for(int i=0;i<nLayers[layerIndexParams+1];++i){
				ss>>param;
				wLayers[layerIndexParams][paramRow*nLayers[layerIndexParams+1]+i]=param;
			}
			++paramRow;
		}
		else if(layerIndexBias>=1){
			stringstream ss(line);
			double param;
			for(int i=0;i<nLayers[layerIndexBias];++i){
				ss>>param;
				bLayers[layerIndexBias][i]=param;
			}
		}
	}
	cout<<"params loaded"<<endl;
}
