//============================================================================
// Name        : cSPS.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cstdio>
#include <omp.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <string>
#include <sys/time.h>
#include "generic.h"
#include "slic.h"
#include "immintrin.h"

#define THREADS 12
//#define USE_AVX2
//#define M_TIME

//Optimal SLIC Params
#define REGION 60 //60
#define REGULARIZATION 200 //200
#define MIN_REGION 800 //800

using namespace std;
using namespace cv;

typedef uint16_t uint16;

bool isEmptyModel(cv::Mat& M){

	if (M.rows == 3) {
		float a = M.at<float>(0,0);
		float b = M.at<float>(1,0);
		float c = M.at<float>(2,0);
		if (a == 0 && b == 0 && c == 0){
			return true;
		}
		else{
			return false;
		}
	}
	else{
		float a = M.at<float>(0,0);
		float b = M.at<float>(1,0);
		if (a == 0 && b == 0){
			return true;
		}
		else{
			return false;
		}
	}
}

bool isEmptyModel(std::vector<cv::Vec3f>& bestM){
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			if (bestM[i].val[j] != 0.0){
				return false;
			}
		}
	}
	return true;
}


float* CV_32F_2float(cv::Mat& CV_32FMap ){
	float* buffer = new float[CV_32FMap.rows * CV_32FMap.cols];

	unsigned int w = CV_32FMap.cols;
	for (int i = 0; i < CV_32FMap.rows; i++){
		for (int j = 0; j < CV_32FMap.cols; j++){
			buffer[i*w + j] = CV_32FMap.at<float>(i,j);
		}
	}

	return buffer;
}
int buildModel_ylinear(std::vector<Point2f>& pair1, std::vector<Point2f>& pair2, cv::Mat& M){

	float y1 = pair1[0].y;
	float y2 = pair2[0].y;

	if ((y1 - y2) < FLT_EPSILON){
		return 0;
	}

	float d1, d2;
	d1 = (pair1[0].y + pair1[1].y)/2.0; //y avg pair1
	d2 = (pair2[0].y + pair2[1].y)/2.0; //y avg pair2

	float m = (d1-d2)/(y1-y2);
	M.at<float>(0,0) = m;
	M.at<float>(1,0) = d1 - m*y1;

	return 1;
}

// check whether machine is little endian
int littleendian()
{
	int intval = 1;
	uchar *uval = (uchar *)&intval;
	return uval[0] == 1;
}
// write pfm image (from LIB-ELAS Middlebury 2014 example
// 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
void WriteFilePFM(float *data, int width, int height, const char* filename, float scalefactor=1/255.0)
{
	// Open the file
	FILE *stream = fopen(filename, "wb");
	if (stream == 0) {
		fprintf(stderr, "WriteFilePFM: could not open %s\n", filename);
		exit(1);
	}

	// sign of scalefact indicates endianness, see pfms specs
	if (littleendian())
		scalefactor = -scalefactor;

	// write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
	fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);

	int n = width;
	// write rows -- pfm stores rows in inverse order!
	for (int y = height-1; y >= 0; y--) {
		float* ptr = data + y * width;
		// change invalid pixels (which seem to be represented as -10) to INF
		for (int x = 0; x < width; x++) {
			if (ptr[x] < 0)
				ptr[x] = INFINITY;
		}
		if ((int)fwrite(ptr, sizeof(float), n, stream) != n) {
			fprintf(stderr, "WriteFilePFM: problem writing data\n");
			exit(1);
		}
	}

	// close file
	fclose(stream);
}

void findInlierFeaturePairs_ylinear(cv::Mat& M, std::vector<std::vector<Point2f> >& leftPts_rightPts,
		std::vector<std::vector<Point2f> >& inliersTmp, float t){
	//for each pair in leftPts_rightPts, check if residual < t
	std::vector<Point2f> tmp;
	float y_left, y_right, y_target, y_model;
	for (unsigned int i = 0; i < leftPts_rightPts.size(); i++){
		y_left = leftPts_rightPts[i][0].y;
		y_right = leftPts_rightPts[i][1].y;

		y_target = (y_left + y_right)/2.0;
		y_model = (M.at<float>(0,0)*y_left) + M.at<float>(1,0);

		if (fabs(y_target - y_model) <= t){ //within error tolerance
			std::vector<Point2f> tmp;
			tmp.push_back(leftPts_rightPts[i][0]);
			tmp.push_back(leftPts_rightPts[i][1]);
			inliersTmp.push_back(tmp);
		}
	}
}

void RANSACFitTransform_ylinear( std::vector<std::vector<Point2f> >& leftPts_rightPts,
		std::vector<std::vector<Point2f> >& inliers, cv::Mat& returnModel, unsigned int N, float t, bool isAdaptive){

	unsigned int maxTrials = 1000;
	unsigned int maxDataTrials = 1000;
	unsigned int numPoints = leftPts_rightPts.size();

	float p = 0.99;

	cv::Mat bestM = cv::Mat::zeros(2,1, CV_32F);
	cv::Mat M = cv::Mat::zeros(2,1, CV_32F);

	unsigned int trialCount = 0;
	unsigned int dataTrialCount = 0;
	unsigned int ninliers = 0;
	float bestScore = 0.0;
	float fracinilers = 0.0;
	float pNoOutliers = 0.0;

	int degenerate = 0; //default
	std::vector<Point2f> pair1;
	std::vector<Point2f> pair2;

	if (isAdaptive){
		N = 1; //dummy initialization for adaptive termination RANSAC
	}

	while (N > trialCount){

		degenerate = 0; //default- singular.
		dataTrialCount = 1;

		while (degenerate == 0){//
			std::random_shuffle(leftPts_rightPts.begin(), leftPts_rightPts.end());
			pair1 = leftPts_rightPts[0];
			pair2 = leftPts_rightPts[1];

			//return 1 if non-singular, OK model. return 0 if singular (degenerate model)
			degenerate = buildModel_ylinear(pair1, pair2, M);

			dataTrialCount++;
			if ( dataTrialCount > maxDataTrials){
				//cout << "Unable to select a non-degenerate data set." << endl;
				return;
			}
		}

		//now we know that M contains some type of non-degenerate model,
		std::vector<std::vector<Point2f> > inliersTmp;
		findInlierFeaturePairs_ylinear(M, leftPts_rightPts, inliersTmp, t);
		ninliers = inliersTmp.size();

		if (ninliers > bestScore){
			bestScore = ninliers;
			inliers = inliersTmp;
			bestM = M;

			//if adaptive termination RANSAC, update estimate of N
			if (isAdaptive){
				fracinilers = float(ninliers)/float(numPoints);
				pNoOutliers = 1 - pow(fracinilers,3); //3 for three points selected
				pNoOutliers = max(FLT_EPSILON, pNoOutliers);
				pNoOutliers = min(1-FLT_EPSILON, pNoOutliers);
				N = (unsigned int) round(log(1-p)/log(pNoOutliers));
			}
		}

		trialCount++;
		if (trialCount > maxTrials){
			break;
		}
	}

	if (!isEmptyModel(bestM)){ //found a solution
		returnModel = bestM;
	}
	else{
		//cout << "RANSAC was unable to find a useful solution." << endl;
	}
}

void compute_y_offset_ylinear(cv::Mat& returnKeypoints, cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& left_warped, cv::Mat& right_warped,
		 double w, double h, cv::Mat& H_L, cv::Mat& H_R){

	//Detect features with SURF (upright)
	int minHessian = 600;

	Ptr<xfeatures2d::SURF> detector=xfeatures2d::SURF::create(minHessian, 4, 2, true, true);
	std::vector<KeyPoint> keypoints_left, keypoints_right;

	detector->detect( leftImageG, keypoints_left);
	detector->detect( rightImageG, keypoints_right);

	//calculate descriptors (feature vectors)
	Ptr<xfeatures2d::SURF> extractor = xfeatures2d::SURF::create();
	cv::Mat desc_left, desc_right;

	extractor->compute( leftImageG, keypoints_left, desc_left);
	extractor->compute( rightImageG, keypoints_right, desc_right);

	//compute pairwise y-distances mask to limit possible matches to +/- 6 pixels
	float y_tol = 6.0;
	cv::Mat yDimMask = cv::Mat::zeros(keypoints_left.size(), keypoints_right.size(), CV_8U);
#pragma omp parallel
	{
	float l_val, r_val;
		#pragma omp for
		for (unsigned int l = 0; l < keypoints_left.size(); l++){
			for (unsigned int r = 0; r < keypoints_right.size(); r++){
				l_val = keypoints_left[l].pt.y;
				r_val = keypoints_right[r].pt.y;
				if (fabs(l_val-r_val) <= y_tol){
					yDimMask.at<uchar>(l,r) = 1;
				}
			}
		}

	}
	//match keypoints
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match( desc_left, desc_right, matches, yDimMask);

	//Quick calculation of max and min distances between keypoints
	double max_dist = 0; double min_dist = 100;
	double dist;
	for (unsigned int i = 0; i < keypoints_left.size(); i++){
		dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}


	//-- Localize the object
	std::vector<Point2f> leftPts;
	std::vector<Point2f> rightPts;

	std::vector<char> good_matches;
	float kpd_thresh = 0.1;
	for(unsigned int i = 0; i < matches.size(); i++ ){
		//threshold mask for visualization of matching keypoints
		if( matches[i].distance < (kpd_thresh*max_dist) ){
			good_matches.push_back(1);
			leftPts.push_back( keypoints_left[ matches[i].queryIdx ].pt );
			rightPts.push_back( keypoints_right[ matches[i].trainIdx ].pt );
		}
		else{
			good_matches.push_back(0); //0 FOR  MASKING
		}
	}

	//RANSAC for inliers from feature pairs
	std::vector<std::vector<Point2f> > leftPts_rightPts;
	for (unsigned int i = 0; i < leftPts.size(); i++){
		std::vector<Point2f> tmp;
		tmp.push_back(leftPts[i]);
		tmp.push_back(rightPts[i]);
		leftPts_rightPts.push_back(tmp);
		tmp.clear();
	}

	std::vector<std::vector<Point2f> > inliers;
	bool isAdaptive = true;
	int N = 1000;
	float t = 0.5;
	cv::Mat Model;

	RANSACFitTransform_ylinear(leftPts_rightPts, inliers, Model, N, t, isAdaptive );

	//build matrices for SVD solver to refit to all inliers
	Mat A_l = cv::Mat::ones(inliers.size(), 2, CV_32F);
	Mat A_r = cv::Mat::ones(inliers.size(), 2, CV_32F);
	Mat b_l = cv::Mat::ones(inliers.size(), 1, CV_32F);
	Mat b_r = cv::Mat::ones(inliers.size(), 1, CV_32F);
	Mat x_l, x_r;
	float y_avg, y_left, y_right;
	for (unsigned int i = 0; i < inliers.size(); i++){
		y_left = inliers[i][0].y;
		y_right = inliers[i][1].y;
		y_avg = (y_left + y_right)/2.0;

		A_l.at<float>(i,0) = inliers[i][0].y;
		b_l.at<float>(i,0) = y_avg;

		A_r.at<float>(i,0) = inliers[i][1].y;
		b_r.at<float>(i,0) = y_avg;
	}
	solve(A_l, b_l, x_l, DECOMP_SVD);
	solve(A_r, b_r, x_r, DECOMP_SVD);

	//build homography matrices
	Mat H_l = cv::Mat::zeros(3,3, CV_32F);
	Mat H_r = cv::Mat::zeros(3,3, CV_32F);
	H_l.at<float>(0,0) = 1;
	H_r.at<float>(0,0) = 1;
	H_l.at<float>(2,2) = 1;
	H_r.at<float>(2,2) = 1;

	H_l.at<float>(1,1) = x_l.at<float>(0,0);
	H_l.at<float>(1,2) = x_l.at<float>(1,0);

	H_r.at<float>(1,1) = x_r.at<float>(0,0);
	H_r.at<float>(1,2) = x_r.at<float>(1,0);


	H_L = H_l;
	H_R = H_r;

	Size dsize = Size(w, h);
	warpPerspective(leftImageG, left_warped, H_l, dsize, INTER_LINEAR , BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP
	warpPerspective(rightImageG, right_warped, H_r, dsize, INTER_LINEAR , BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP

	//store keypoints for use in plane fitting later
	Mat tmpL1 = cv::Mat::ones(3, 1, CV_32F);
	Mat modelL1;
	returnKeypoints = cv::Mat::zeros(A_l.rows, 3, CV_32F);
	float x_right;
	float disp = 0.0;
	for (int i = 0; i < A_l.rows; i++){ //for all inlier points
		tmpL1.at<float>(0,0) = inliers[i][0].x; // keypoint left image, x coordinate
		tmpL1.at<float>(1,0) = A_l.at<float>(i,0); // keypoint left image, y coordinate original

		x_right = inliers[i][1].x; // keypoint right image, x coordinate
		disp = tmpL1.at<float>(0,0) - x_right; // d = x_left - x_right

		modelL1 = H_l*tmpL1; // use homography to compute new y location of point, 3 rows x 1 col

		returnKeypoints.at<float>(i,0) = modelL1.at<float>(0,0); //x orig
		returnKeypoints.at<float>(i,1) = modelL1.at<float>(1,0); //y after model
		returnKeypoints.at<float>(i,2) = disp; //disparity from keypoints
	}


}


int getMaxDisparityFromFile(const char* filename){
	ifstream inFile(filename);
	if (!inFile) {
		cerr << "File " << filename << " is not found." << endl;
		exit(0);
	}

	string line;
	for (int i = 0; i < 7; i++){
		getline(inFile, line);
	}
	string sub;
	sub = line.substr(6,3);
	char* cstr = new char [sub.length()+1];
	std::strcpy (cstr, sub.c_str());

	int maxDisparity = atoi(cstr);
	return maxDisparity;
}

int getMaxNumberofSegments(cv::Mat& mat, vl_uint32* segmentation){
	int numSegments = 0;
	int tmp = 0;
	for (int i = 0; i < mat.rows; i++){
		for (int j = 0; j < mat.cols; j++){
			tmp = (int) segmentation[j + mat.cols*i];
			if (tmp > numSegments){
				numSegments = tmp;
			}
		}
	}
	numSegments++; //indexing at 0 - adds one to max segment IDX
	return numSegments;
}

void segmentImage(cv::Mat& mat, vl_uint32* segmentation, int r, float reg, int min_r){
	// Convert image to one-dimensional array.

	float* image = new float[mat.rows*mat.cols*mat.channels()];
#pragma omp parallel
	{
#pragma omp for
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			Vec3b intensity = mat.at<Vec3b>(i, j);
			uchar blue = intensity.val[0];
			uchar green = intensity.val[1];
			uchar red = intensity.val[2];
			image[j + mat.cols*i + mat.cols*mat.rows*0] = float(blue);
			image[j + mat.cols*i + mat.cols*mat.rows*1] = float(green);
			image[j + mat.cols*i + mat.cols*mat.rows*2] = float(red);
		}
	}
	}

	// The algorithm will store the final segmentation in a one-dimensional array.

	vl_size height = mat.rows;
	vl_size width = mat.cols;
	vl_size channels = mat.channels();

	// The region size defines the number of superpixels obtained.
	// Regularization describes a trade-off between the color term and the
	// spatial term.
	vl_size region = r;
	float regularization = reg;
	vl_size minRegion = min_r;

	vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);

	delete[] image;
}

void segmentation_to_segmentMap(vl_uint32* segmentation, cv::Mat& segmentMap_CV_16U){

	int h = segmentMap_CV_16U.rows;
	int w = segmentMap_CV_16U.cols;
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < h; i++) { //rows
			for (int j = 0; j < w; j++) { //cols
				segmentMap_CV_16U.at<unsigned short>(i,j) = segmentation[i*w + j];
			}
		}
	}
}

void warpSegmentMap(cv::Mat& segmentMap_CV_16U, cv::Mat& segmentMap_CV_16U_warped, cv::Mat& H_l){

	int h = segmentMap_CV_16U.rows;
	int w = segmentMap_CV_16U.cols;

	Size dsize = Size(w, h);
	warpPerspective(segmentMap_CV_16U, segmentMap_CV_16U_warped, H_l, dsize, INTER_NEAREST, BORDER_REPLICATE, 0);
}

void arrayIntToBinaryFile(int *m, int size, std::ofstream& oF){
	if(!oF){
		//std::cout << "File I/O Error";
		throw 1;
	}

	int cflt;
	for(int j = 0; j < size; j++){
		cflt = m[j];
		oF.write( reinterpret_cast<char*>( &cflt ), sizeof cflt );
	}
}

void writeBinaryIntArray( int *array, char arrayFileName[], int size){

	ofstream fout;
	fout.open(arrayFileName, ios::binary);
	if (fout == NULL) {
		//cout << "File I/O error" << endl;
	}
	arrayIntToBinaryFile(array, size, fout);
	fout.close();
}

void saveSegmentsToFile(cv::Mat& mat, vl_uint32* segmentation, char binSaveName[]){
	int* segmentation_int = new int[mat.rows*mat.cols];

	for (int i = 0; i < mat.rows; i++){
		for (int j = 0; j < mat.cols; j++){
			segmentation_int[j + mat.cols*i]  = (int) segmentation[j + mat.cols*i];
		}
	}

	writeBinaryIntArray(segmentation_int, binSaveName, mat.rows*mat.cols);

	delete[] segmentation_int;
}

void groupPixelsBySegments(std::vector<std::vector<cv::Vec3f> >& segmentsAll, cv::Mat& leftImage,
		cv::Mat& segmentMap_CV_16U_warped, double px, double py){

	int segmentID;
	float x = 0.0;
	float y = 0.0;
	for (int i = 0; i < leftImage.rows; i++){
		for (int j = 0; j < leftImage.cols; j++){
			y = float(i) - float(py);
			x = float(j) - float(px);
			segmentID = (float) segmentMap_CV_16U_warped.at<unsigned short>(i,j);
			Vec3f tmp = Vec3f(y, x, 0.0); // row, col, disparity (disparity is unknown for this function)
			segmentsAll[segmentID].push_back(tmp); //store pixel IDX at segment vector
		}
	}
}

void randomizePixelsInSegments(std::vector<std::vector<cv::Vec3f> >& segmentsNZ, unsigned int randomInit){
	std::srand(randomInit);
#pragma omp parallel
	{
		#pragma omp for
		for (unsigned int i = 0; i < segmentsNZ.size(); i++){
			if (segmentsNZ[i].size() > 2){ //skip if too few points to fit a plane
				std::random_shuffle ( segmentsNZ[i].begin(), segmentsNZ[i].end() );
			}
		}
	}
}


void getVectorSubset(std::vector<std::vector<cv::Vec3f> >& segmentsNZ, std::vector<std::vector<cv::Vec3f> >& segmentsNZRand, float randPercent){
	//unsigned int lastIDX = 0;
#pragma omp parallel
	{
#pragma omp for
		for (unsigned int i = 0; i < segmentsNZ.size(); i++){ //for each segment
			if (segmentsNZ[i].size() > 2){
				unsigned int lastIDX = randPercent*segmentsNZ[i].size();
				for (unsigned int j = 0; j < lastIDX; j++){ //for each pixel in each segment
					Vec3f tmp = Vec3f(segmentsNZ[i][j]);
					segmentsNZRand[i].push_back(tmp);
				}
			}
		}

	}
}

#ifdef USE_AVX2

void NCCprecompute256(uint16* vec,  double& A, double& C, int vecsize,int orvecsize){
	A = 0.0;
	double B =0.0;

	for(int i=0; i<vecsize; i+=16){

		__m256i sum = _mm256_set1_epi16(0);
		__m256i sumsq = _mm256_set1_epi32(0);
		__m256i v = _mm256_loadu_si256((__m256i const *) &vec[i]); // Load 256 aligned bit to register
		__m256i sqs = _mm256_madd_epi16(v,v); // Multiply and pairwise add


		sum = _mm256_add_epi16(v,sum); // Pass values to sum
		sumsq = _mm256_add_epi32(sqs,sumsq);// Pass valuew to sumsq

		for(int j=0; j<3; j++){
			sum = _mm256_hadd_epi16(sum,sum); //After 3 iterations the sum of the first 8 numbers is located at positions 0:7 and the rest contain the sum from 8:15
		}

		for(int j=0; j<2; j++){
			sumsq = _mm256_hadd_epi32(sumsq,sumsq); //After 2 iterations the sum of squares is located at 0:3 and 4:7 positions
		}

		//Accumulate variables and repeat the computations for the full vectors

		A +=(double)_mm256_extract_epi16(sum,0)+(double)_mm256_extract_epi16(sum,8);
		B += (double)_mm256_extract_epi32(sumsq,0)+(double)_mm256_extract_epi32(sumsq,4);
	}

	C =sqrt(orvecsize*B-(A)*(A));

}


float computencc256(uint16* lvec, uint16* rvec, int vecsize,int orvecsize,double AL, double AR, double CL, double CR){

	double D =0;

	for(int i=0; i<vecsize; i+=16){
		__m256i scalarp  = _mm256_set1_epi32(0);;
		__m256i lv = _mm256_loadu_si256((__m256i const*) &lvec[i]);
		__m256i rv = _mm256_loadu_si256((__m256i const*) &rvec[i]);
		__m256i partialscalarp = _mm256_madd_epi16(lv,rv);
		scalarp = _mm256_add_epi16(partialscalarp,scalarp);
		scalarp = _mm256_hadd_epi32(scalarp,scalarp);
		scalarp = _mm256_hadd_epi32(scalarp,scalarp);

		D += (double)_mm256_extract_epi32(scalarp,0)+(double)_mm256_extract_epi32(scalarp,4);
	}


	return (orvecsize*D - AL*AR)/( CL*CR );

}

#else

void NCCprecompute(uint16* vec,  double& A, double& C, int vecsize,int orvecsize){
	A = 0.0;
	double B =0.0;
	for(int i=0; i<vecsize; i+=8){
		__m128i sum = _mm_set1_epi16(0);
		__m128i sumsq = _mm_set1_epi32(0);
		__m128i v = _mm_load_si128((__m128i const*) &vec[i]);
		__m128i sqs = _mm_madd_epi16(v,v);

		sum = _mm_add_epi16(v,sum);
		sumsq = _mm_add_epi32(sqs,sumsq);
		for(int j=0; j<3; j++){
			sum = _mm_hadd_epi16(sum,sum);
		}


		for(int j=0; j<2; j++){
			sumsq = _mm_hadd_epi32(sumsq,sumsq);
		}

		A +=(double)_mm_extract_epi16(sum,0);
		B += (double)_mm_extract_epi32(sumsq,0);
	}


	C =sqrt(orvecsize*B-(A)*(A));

}

float computencc(uint16* lvec, uint16* rvec, int vecsize,int orvecsize,double AL, double AR, double CL, double CR){

	double D =0;

	for(int i=0; i<vecsize; i+=8){
		__m128i scalarp  = _mm_set1_epi32(0);
		__m128i lv = _mm_load_si128((__m128i const*) &lvec[i]);
		__m128i rv = _mm_load_si128((__m128i const*) &rvec[i]);
		__m128i partialscalarp = _mm_madd_epi16(lv,rv);
		scalarp = _mm_add_epi32(partialscalarp,scalarp);
		scalarp = _mm_hadd_epi32(scalarp,scalarp);
		scalarp = _mm_hadd_epi32(scalarp,scalarp);
		D += (double)_mm_extract_epi32(scalarp,0);
	}


	return (orvecsize*D - AL*AR)/( CL*CR );

}

#endif


void compteNCCDisparityMMX(std::vector<cv::Vec3f>& segmentsNZRand, cv::Mat& leftImageG, cv::Mat& rightImageG,
								double px, double py, unsigned int winSize, int maxDisparity ){
	int winSize2 = winSize/2,vecsize = winSize*winSize;
	if(vecsize%16 > 0)
		vecsize += 16-vecsize%16;
	int h = rightImageG.rows;
	int w = rightImageG.cols;

#pragma omp parallel
			{

	 uint16* lvec = (uint16*)calloc(vecsize,sizeof(uint16));
#pragma omp for
	for (unsigned int i = 0; i < segmentsNZRand.size(); i++){

		float tmp_y = segmentsNZRand[i].val[0]; //y
		float tmp_x = segmentsNZRand[i].val[1]; //x

		int r = (int) round(tmp_y + float(py)); //y
		int c = (int) round(tmp_x + float(px)); //x


		if (r >= winSize2 && c >= winSize2 && r < h - winSize2 && c < w - winSize2){


		for (int k = r - winSize2; k <= r + winSize2; k++){
			for (int j = c - winSize2; j <= c + winSize2; j++){
				lvec[(k-(r - winSize2))*(winSize) + (j - (c - winSize2))] = (uint16)leftImageG.at<uchar>(k,j);

			}
		}


			double AL=0,CL=0;
#ifdef USE_AVX2
			NCCprecompute256(lvec,AL,CL,vecsize,winSize*winSize);
#else
			NCCprecompute(lvec,AL,CL,vecsize,winSize*winSize);
#endif


				uint16* rvec = (uint16*)calloc(vecsize,sizeof(uint16));
				double best_Score= -RAND_MAX;

				int start =0;
				int end = min(maxDisparity,c - winSize2);


				for (int d_int = start; d_int <= end; d_int++){


					for (int k = r - winSize2; k <= r + winSize2; k++){
						for (int j = c - d_int - winSize2; j <= c - d_int + winSize2; j++){
							rvec[(k-(r - winSize2))*(winSize) + (j - (c - d_int - winSize2))] = (uint16)rightImageG.at<uchar>(k,j);

						}
					}


					double AR=0,CR=0;
#ifdef USE_AVX2
					NCCprecompute256(rvec,AR,CR,vecsize,winSize*winSize);
#else
					NCCprecompute(rvec,AR,CR,vecsize,winSize*winSize);
#endif

#ifdef USE_AVX2
					double score =computencc256(lvec, rvec, vecsize,winSize*winSize, AL,  AR,  CL,  CR);
#else
					double score =computencc(lvec, rvec, vecsize,winSize*winSize, AL,  AR,  CL,  CR);
#endif
					if(score > best_Score){
						best_Score = score;
						segmentsNZRand[i].val[2] = (float)abs(d_int);
					}
				}
				delete [] rvec;

			}


		}



	delete [] lvec;

	}


}

void getNCCDisparity(std::vector<cv::Vec3f>& segmentsNZRand, cv::Mat& leftImageG, cv::Mat& rightImageG,
		cv::Mat& leftPatch, cv::Mat& rightPatch, double px, double py, unsigned int winSize, int maxDisparity){

	int winSize2 = (int) floor(float(winSize)/2.0);
	int h = rightImageG.rows;
	int w = rightImageG.cols;

	float tmp_x, tmp_y;
	int r, c;

	int bestMatchSoFar;
	float NCCScore;
	float bestNCCScore;

	for (unsigned int i = 0; i < segmentsNZRand.size(); i++){ //for each pixel in superpixel
		tmp_y = segmentsNZRand[i].val[0]; //y
		tmp_x = segmentsNZRand[i].val[1]; //x

		r = (int) round(tmp_y + float(py)); //y
		c = (int) round(tmp_x + float(px)); //x

		bestMatchSoFar = 0;
		NCCScore = 0.0;
		bestNCCScore = -1*FLT_MAX;

		if (r >= winSize2 && c >= winSize2 && r < h - winSize2 && c < w - winSize2){
			//get left patch

			unsigned int ctr = 0;

			//get image grayscale values left patch
			for (int i = r - winSize2; i <= r + winSize2; i++){
				for (int j = c - winSize2; j <= c + winSize2; j++){
					leftPatch.at<float>(ctr, 0) = float(leftImageG.at<uchar>(i,j));
					ctr++;
				}
			}

			Scalar leftStd, rightStd, leftMean, rightMean;
			meanStdDev(leftPatch, leftMean, leftStd);
			if (leftStd.val[0] != 0){

				leftPatch = leftPatch - leftMean.val[0]; //offset vector by mean
				leftPatch = (1/leftStd.val[0])*(leftPatch); //divide vector by Stdev

				for (int d_int = 0; d_int <= min(maxDisparity,c - winSize2); d_int++){
					//across full disparity search space

					//get image grayscale values right patch for disparity offset
					unsigned int ctr2 = 0;

					//get image grayscale values
					for (int i = r - winSize2; i <= r + winSize2; i++){
						for (int j = c - d_int - winSize2; j <= c - d_int + winSize2; j++){
							rightPatch.at<float>(ctr2, 0) = float(rightImageG.at<uchar>(i,j));
							ctr2++;
						}
					}

					float tmp;
					meanStdDev(rightPatch, rightMean, rightStd);
					if (rightStd.val[0] != 0){
						(rightPatch) = rightPatch - rightMean.val[0];
						(rightPatch) = (1/rightStd.val[0])*rightPatch;
						tmp = float((leftPatch).dot(rightPatch));
						tmp = tmp/(float(winSize)*float(winSize));
						NCCScore = tmp;
					}
					else{
						NCCScore = 0.0;// FLT_MAX;
					}

					if (NCCScore > bestNCCScore){ //if we found a better match, replace disparity
						bestMatchSoFar = d_int;
						bestNCCScore = NCCScore;
					}
				}
			}
		}
		segmentsNZRand[i].val[2] = float(bestMatchSoFar);
	}
}

void getNCCDisparities(std::vector<std::vector<cv::Vec3f> >& segmentsNZRand, cv::Mat& leftImageG, cv::Mat& rightImageG,
		double px, double py, unsigned int winSize, int maxDisparity){

	cv::Mat leftPatch =  cv::Mat::zeros(winSize*winSize, 1, CV_32F); //winSize
	cv::Mat rightPatch =  cv::Mat::zeros(winSize*winSize, 1, CV_32F); //winSize

	for (unsigned int i = 0; i < segmentsNZRand.size(); i++){ //for each superpixel
		if (segmentsNZRand.size() > 2){ //if we have enough points to fit a plane...
			//getNCCDisparity(segmentsNZRand[i], leftImageG, rightImageG, leftPatch, rightPatch, px, py, winSize, maxDisparity);
			compteNCCDisparityMMX(segmentsNZRand[i], leftImageG, rightImageG, px, py, winSize, maxDisparity);
		}
	}
}

void removeZeroPoints(std::vector<std::vector<cv::Vec3f> >& segmentsRand, std::vector<std::vector<cv::Vec3f> >& segmentsNZRand){

	for (unsigned int i = 0; i < segmentsRand.size(); i++){
		for (unsigned int j = 0; j < segmentsRand[i].size(); j++){
			if (segmentsRand[i][j].val[2] != 0){
				segmentsNZRand[i].push_back(segmentsRand[i][j]);
			}
		}
	}

}

void buildEmptyModel(std::vector<cv::Vec3f>& bestM){

	Vec3f tmp0 = Vec3f(0.0, 0.0, 0.0);
	Vec3f tmp1 = Vec3f(0.0, 0.0, 0.0);
	Vec3f tmp2 = Vec3f(0.0, 0.0, 0.0);
	bestM.push_back(tmp0);
	bestM.push_back(tmp1);
	bestM.push_back(tmp2);

}

bool pointsAreColinear(std::vector<cv::Vec3f>& bestM){
	Vec3f vdiff1 = bestM[1] - bestM[0];
	Vec3f vdiff2 = bestM[2] - bestM[1];
	float n = norm(vdiff1.cross(vdiff2));
	if (n < FLT_EPSILON){
		return true;
	}
	return false;
}


bool disparityGradientTest(std::vector<cv::Vec3f>& M){
	bool failTest = true;

	Vec3f vdiff1 = M[1] - M[0];
	Vec3f vdiff2 = M[2] - M[1];
	Vec3f crossv = vdiff1.cross(vdiff2);
	float n = norm(crossv);
	crossv = (1/n)*crossv; //unit normal vector of plane

	float B = crossv.val[0]; //first element is x coeff now
	float C = crossv.val[2];

	float neg_b_over_c = (-1*B)/C;
	if (neg_b_over_c > 0 ){ //positive slant: B and C have opp signs
		if (neg_b_over_c <= 1){ //AND doesn't violate disparity gradient test for left cam
			failTest = false;
		}
		//else failTest = true; // based on left cam disparity gradient test
	}
	else{ //negative slant : B and C have same signs
		float b_over_cpb = (B/(C+B));
		if (b_over_cpb <= 1){ //AND doesn't violate disparity gradient test for right cam
			failTest = false;
		}
		//else failTest = true; //based on right cam disparity gradient test
	}
	return failTest;
}


void findInliers(std::vector<cv::Vec3f>& M, std::vector<cv::Vec3f>& segmentNZRand, std::vector<cv::Vec3f>& inlierListTmp, float t){
	//Get normalized plane surface normal vector
	Vec3f vdiff1 = M[1] - M[0];
	Vec3f vdiff2 = M[2] - M[1];
	Vec3f crossv = vdiff1.cross(vdiff2);
	float n = norm(crossv);
	crossv = (1/n)*crossv; //unit normal vector of plane

	//for each point in segmentNZRand, compute distance from point to plane
	//for equations, see http://mathworld.wolfram.com/Point-PlaneDistance.html
	float d = 0.0;
	for (unsigned int i = 0; i < segmentNZRand.size(); i++){
		d = fabs(crossv.dot(segmentNZRand[i] - M[0]));
		if (d < t){
			inlierListTmp.push_back(segmentNZRand[i]);
		}
	}
}


void RANSACFitPlane(std::vector<cv::Vec3f>& segmentPlPoints, std::vector<cv::Vec3f>& inlierList,
		std::vector<cv::Vec3f>& segmentNZRand, unsigned int N, float t, bool isAdaptive){

	unsigned int maxTrials = 1000;
	unsigned int maxDataTrials = 1000;
	unsigned int numPoints = segmentNZRand.size();

	float p = 0.99;

	std::vector<cv::Vec3f> bestM;
	buildEmptyModel(bestM);
	std::vector<cv::Vec3f> M;
	buildEmptyModel(M);

	unsigned int trialCount = 0;
	unsigned int dataTrialCount = 0;
	unsigned int ninliers = 0;
	float bestScore = 0.0;
	float fracinilers = 0.0;
	float pNoOutliers = 0.0;

	bool degenerate = true;
	bool failDispGradTest = true;

	if (isAdaptive){
		N = 1; //dummy initialization for adaptive termination RANSAC
	}

	while (N > trialCount){
		degenerate = true;
		failDispGradTest = true;
		dataTrialCount = 1;
		while (degenerate || failDispGradTest ){//

			//randomly sample 3 points from segmentNZRand: shuffle points, choose first 3
			std::random_shuffle(segmentNZRand.begin(), segmentNZRand.end());
			M[0] = segmentNZRand[0];
			M[1] = segmentNZRand[1];
			M[2] = segmentNZRand[2];

			//test that 3 points are not a degenerate configuration
			degenerate = pointsAreColinear(M);

			//test that 3 points do not form a plane that violates disparity gradient limit
			failDispGradTest = disparityGradientTest(M);

			dataTrialCount++;
			if ( dataTrialCount > maxDataTrials){
				//cout << "Unable to select a non-degenerate data set." << endl;
				return;
			}
		}
		//now we know that M contains some type of non-degenerate model,

		std::vector<cv::Vec3f> inlierListTmp;
		findInliers(M, segmentNZRand, inlierListTmp, t);
		ninliers = inlierListTmp.size();

		if (ninliers > bestScore){
			bestScore = ninliers;
			inlierList = inlierListTmp;
			bestM = M;

			//if adaptive termination RANSAC, update estimate of N
			if (isAdaptive){
				fracinilers = float(ninliers)/float(numPoints);
				pNoOutliers = 1 - pow(fracinilers,3); //3 for three points selected
				pNoOutliers = max(FLT_EPSILON, pNoOutliers);
				pNoOutliers = min(1-FLT_EPSILON, pNoOutliers);
				N = (unsigned int) round(log(1-p)/log(pNoOutliers));
			}
		}

		trialCount++;
		if (trialCount > maxTrials){
			break;
		}
	}

	if (!isEmptyModel(bestM)){ //found a solution
		segmentPlPoints = bestM;
	}
	else{
		//cout << "RANSAC was unable to find a useful solution." << endl;
	}
}

void RANSACFitPlanes(std::vector<std::vector<cv::Vec3f> >& segmentsPlPoints, std::vector<std::vector<cv::Vec3f> >& inliers,
		std::vector<std::vector<cv::Vec3f> >& segmentsNZRand, unsigned int N, float t, bool isAdaptive){
	//cout << "Starting plane fitting." << endl;
	for (unsigned int i = 0; i < segmentsNZRand.size(); i++){ //for each superpixel
		if (segmentsNZRand[i].size() > 2){ //need minimum of 3 points to fit plane
			RANSACFitPlane( segmentsPlPoints[i], inliers[i], segmentsNZRand[i], N, t, isAdaptive); //fit plane
		}
		else {
			//segmentsPlPoints[i] will be vector of length 0
		}
	}
}

bool disparityGradientTest_plFit(cv::Vec4f& pl ){
	bool failTest = true;

	float B = pl.val[1]; //second element is x coeff now
	float C = pl.val[2];

	float neg_b_over_c = (-1*B)/C;
	if (neg_b_over_c > 0 ){ //positive slant: B and C have opp signs
		if (neg_b_over_c <= 1){ //AND doesn't violate disparity gradient test for left cam
			failTest = false;
		}
		//else failTest = true; // based on left cam disparity gradient test
	}
	else{ //negative slant : B and C have same signs
		float b_over_cpb = (B/(C+B));
		if (b_over_cpb <= 1){ //AND doesn't violate disparity gradient test for right cam
			failTest = false;
		}
		//else failTest = true; //based on right cam disparity gradient test
	}
	return failTest;
}

void refitPlaneInliers(std::vector<cv::Vec4f>& segmentsRefitPl, std::vector<cv::Vec3f>& inliers ){

	unsigned int npts = inliers.size();

	if (npts < 3){
		//cout << "Too few points to fit a plane." << endl;
		return;
	}
	else {
		cv::Mat A, X; //Matrix equation AX = 0;
		if(npts == 3){
			A = cv::Mat::ones(4, 4, CV_32F);
			for (unsigned int i = 0; i < 3; i++){
				A.at<float>(i,0) = inliers[i].val[0];
				A.at<float>(i,1) = inliers[i].val[1];
				A.at<float>(i,2) = inliers[i].val[2];
			}
			//pad with zeros
			A.at<float>(3,0) = 0;
			A.at<float>(3,1) = 0;
			A.at<float>(3,2) = 0;
			A.at<float>(3,3) = 0;
		}
		else{ //npts > 3, over-specified
			A = cv::Mat::ones(npts, 4, CV_32F);
			for (unsigned int i = 0; i < inliers.size(); i++){
				A.at<float>(i,0) = float(inliers[i].val[0]);
				A.at<float>(i,1) = float(inliers[i].val[1]);
				A.at<float>(i,2) = float(inliers[i].val[2]);
			}
		}
		//SVD of A.
		X = cv::Mat::zeros(4, 1, CV_32F);
		cv::Mat w, u, vt, v;
		//last row of w is smallest singular value.
		//transpose vt and get last column of v
		SVD::compute(A, w, u, vt);
		transpose(vt, v);

		Vec4f tmp = Vec4f(v.at<float>(0,3), v.at<float>(1,3), v.at<float>(2,3), v.at<float>(3,3));

		bool failTest = disparityGradientTest_plFit(tmp);
		if (!failTest){ //passed test
			segmentsRefitPl.push_back(tmp);
		}
		else{
			//cout << "fails disparity gradient test after plane fit" << endl;
		}
	}
}

void refitPlanesInliers(std::vector<std::vector<cv::Vec4f> >& segmentsRefitPl, std::vector<std::vector<cv::Vec3f> >& segmentsPlPoints, std::vector<std::vector<cv::Vec3f> >& inliers){
	//cout << "Starting plane re-fitting." << endl;

	for (unsigned int i = 0; i < inliers.size(); i++){
		if (segmentsPlPoints[i].size() > 2){ //we found a model
			refitPlaneInliers(segmentsRefitPl[i], inliers[i]);
		}
		else{
			//segmentsRefitPl[i] will be vector of length 0
		}
	}


	//cout << "Ended plane re-fitting." << endl;
}

void adjustSegment2PlaneFit(std::vector<cv::Vec4f>& segmentsRefitPl,
		std::vector<cv::Vec3f>& segmentsAll,
		std::vector<cv::Vec3f>& segmentsAllFinal ){

	float A = segmentsRefitPl[0].val[0];
	float B = segmentsRefitPl[0].val[1];
	float C = segmentsRefitPl[0].val[2];
	float D = segmentsRefitPl[0].val[3];

	float tmp_x, tmp_y, tmp_d = 0.0;
	for (unsigned int i = 0; i < segmentsAll.size(); i++){ //for each pixel in superpixel
		tmp_y = segmentsAll[i].val[0]; //y
		tmp_x = segmentsAll[i].val[1]; //x
		tmp_d = ((-1*A)/C)*tmp_y - (B/C)*tmp_x - (D/C); //refit to plane

		Vec3f tmp = Vec3f(tmp_y, tmp_x, tmp_d);
		segmentsAllFinal.push_back(tmp);
	}
}

void planeAdjustedSegments2DispMap(std::vector<std::vector<cv::Vec3f> >& segmentsAllFinal, cv::Mat& dispMapRefit, double px, double py){


#pragma omp parallel
	{

#pragma omp for
		for (unsigned int k = 0; k < segmentsAllFinal.size(); k++){ //for all superpixels
			if (segmentsAllFinal[k].size() > 0){ //if we found a model for the superpixel
				for (unsigned int p = 0; p < segmentsAllFinal[k].size(); p++){ //for all pixels in superpixel
					int r = (int) round(segmentsAllFinal[k][p].val[0] + float(py)); //y
					int c = (int) round(segmentsAllFinal[k][p].val[1] + float(px)); //x

					//if within bounds of image
					if (r > 0 && r < dispMapRefit.rows && c > 0 && c < dispMapRefit.cols){
						float d = segmentsAllFinal[k][p].val[2]; //d - NEED TO ROUND LATER

						if (dispMapRefit.at<float>(r,c) == 0){ //first time visiting pixel
							dispMapRefit.at<float>(r,c) = d; //can be in front of or behind camera
						}
						else if (dispMapRefit.at<float>(r,c) < 0){ //previously behind camera
							if (d > 0){
								dispMapRefit.at<float>(r,c) = d; //now in front of camera
							}
						}
						else{ // dispMapRefit.at<float>(r,c) > 0
							if (d > 0 && d > dispMapRefit.at<float>(r,c)){ //in front of camera, but closer to camera
								dispMapRefit.at<float>(r,c) = d;
							}
						}
					}
				}
			}
		}

	}
}

void adjustSegments2PlaneFit(std::vector<std::vector<cv::Vec4f> >& segmentsRefitPl,
		std::vector<std::vector<cv::Vec3f> >& segmentsAll,
		std::vector<std::vector<cv::Vec3f> >& segmentsAllFinal){

	for (unsigned int i = 0; i < segmentsAll.size(); i++){ //for each superpixel
		if (segmentsRefitPl[i].size() > 0){ //if we found a model before
			adjustSegment2PlaneFit(segmentsRefitPl[i], segmentsAll[i], segmentsAllFinal[i]);
		}
	}
}

void unwarpResultDisparity(cv::Mat& dispMapProp, cv::Mat& dispMapProp_unwarped,  cv::Mat& H_l, double w, double h){

	Size dsize = Size(w, h);
	warpPerspective(dispMapProp, dispMapProp_unwarped, H_l, dsize, INTER_LINEAR + WARP_INVERSE_MAP, BORDER_CONSTANT, 0);

}

float getNCCScoreFPMMX(cv::Mat& leftImageG, cv::Mat& rightImageG,
		int r, int c, float d, unsigned int winSize){
	int winSize2 = winSize/2,vecsize = winSize*winSize;
	if(vecsize%16 > 0)
		vecsize += 16-vecsize%16;
	int h = rightImageG.rows;
	int w = rightImageG.cols;
	int d_int = int(d);

	 uint16* lvec = (uint16*)calloc(vecsize,sizeof(uint16));
	 uint16* rvec = (uint16*)calloc(vecsize,sizeof(uint16));



		if (r >= winSize2 + 1 && c >= winSize2 && r < h - winSize2 - 1 && c < w - winSize2
				&& c - d_int >= winSize2 && c - d_int + winSize2 < w ){



		for (int i = r - winSize2; i <= r + winSize2; i++){
			for (int j = c - winSize2; j <= c + winSize2; j++){
				lvec[(i-(r - winSize2))*(winSize) + (j - (c - winSize2))] = (uint16)leftImageG.at<uchar>(i,j);
				rvec[(i-(r - winSize2))*(winSize) + (j - (c - winSize2))] = (uint16)rightImageG.at<uchar>(i,j-d_int);
			}
		}


			double AL=0,CL=0;
#ifdef USE_AVX2
			NCCprecompute256(lvec,AL,CL,vecsize,winSize*winSize);
#else
			NCCprecompute(lvec,AL,CL,vecsize,winSize*winSize);
#endif
			double AR=0,CR=0;
#ifdef USE_AVX2
			NCCprecompute256(rvec,AR,CR,vecsize,winSize*winSize);
#else
			NCCprecompute(rvec,AR,CR,vecsize,winSize*winSize);
#endif

#ifdef USE_AVX2
			float score = computencc256(lvec, rvec, vecsize,winSize*winSize, AL,  AR,  CL,  CR);
#else
			float score = computencc(lvec, rvec, vecsize,winSize*winSize, AL,  AR,  CL,  CR);
#endif

			delete [] lvec;
			delete [] rvec;
			return score;
		}
		return 0;


}

//fronto-parallel window assumption
float getNCCScoreFP(cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& leftPatch, cv::Mat& rightPatch,
		int r, int c, float d, unsigned int winSize){
	float NCCScore = 0.0;
	int winSize2 = (int) floor(float(winSize)/2.0);

	int h = rightImageG.rows;
	int w = rightImageG.cols;
	int d_int = int(d);

	if (r >= winSize2 + 1 && c >= winSize2 && r < h - winSize2 - 1 && c < w - winSize2
			&& c - d_int >= winSize2 && c - d_int + winSize2 < w ){ //&& d_int != 0
		//if within bounds of image including filter edges
		//if within bounds of image for right image also  (c - d)

		unsigned int ctr = 0;
		unsigned int ctr2 = 0;

		//get image grayscale values
		for (int i = r - winSize2; i <= r + winSize2; i++){
			for (int j = c - winSize2; j <= c + winSize2; j++){
				leftPatch.at<float>(ctr, 0) = float(leftImageG.at<uchar>(i,j));
				ctr++;
			}
		}

		//get mean and stdev values
		float tmp;
		Scalar leftStd, rightStd, leftMean, rightMean;
		meanStdDev(leftPatch, leftMean, leftStd);

		if (leftStd.val[0] != 0 ){
			for (int i = r - winSize2; i <= r + winSize2; i++){
				for (int j = c - d - winSize2; j <= c - d + winSize2; j++){
					(rightPatch).at<float>(ctr2, 0) = float(rightImageG.at<uchar>(i,j));
					ctr2++;
				}
			}

			meanStdDev((rightPatch), rightMean, rightStd);

			if ( rightStd.val[0] != 0){
				leftPatch = leftPatch - leftMean.val[0]; //offset vector by mean
				rightPatch = rightPatch - rightMean.val[0];
				leftPatch = (1/leftStd.val[0])*(leftPatch); //divide vector by Stdev
				rightPatch = (1/rightStd.val[0])*rightPatch;

				tmp = float((leftPatch).dot(rightPatch));
				tmp = tmp/(float(winSize)*float(winSize));
				NCCScore = tmp;
			}
			else{
				NCCScore = 0.0;
				//NCCScore = FLT_MAX;
			}
		}
		else{
			NCCScore = 0.0;
			//NCCScore = FLT_MAX;
		}
	}
	return NCCScore;
}

void getSuperpixelPCScoreFP_fromDispMap(cv::Mat& dispMapRefit, std::vector<std::vector<cv::Vec3f> >& segmentsAllFinal,
		std::vector<float>& baselinePhotoCons, cv::Mat& leftImageG, cv::Mat& rightImageG,
		cv::Mat& leftPatch, cv::Mat& rightPatch, unsigned int winSize, double px, double py){

	//cout << "Starting baseline photoconsistency score computation." << endl;


	float NCCScoreSum = 0.0;
	float tmp = 0.0;

	for (unsigned int k = 0; k < segmentsAllFinal.size(); k++ ){ //for each superpixel
		NCCScoreSum = 0.0;
		if (segmentsAllFinal[k].size() > 0){ //if we found a valid plane and adjusted the points accordingly
			for (unsigned int p = 0; p < segmentsAllFinal[k].size(); p++){ //for each pixel of the superpixel
				int r = (int) round(segmentsAllFinal[k][p].val[0] + float(py)); //y
				int c = (int) round(segmentsAllFinal[k][p].val[1] + float(px)); //x
				float d = round(dispMapRefit.at<float>(r,c));

				if (d < 0){ //LATER ADD d > dmax??
					NCCScoreSum+= (-1*FLT_MAX); //point is behind the camera - penalize photoconsistency to -Inf
					//cout<< "behind camera error" << endl;
				}
				else{
					//tmp = getNCCScoreFP(leftImageG, rightImageG, leftPatch, rightPatch, r, c, d, winSize);
					//NCCScoreSum+= tmp;
					NCCScoreSum += getNCCScoreFPMMX(leftImageG, rightImageG,r, c, d, winSize);
				}
			}
		}
		//total photoconsistency score normalized by num pixels in superpixel
		if (segmentsAllFinal[k].size() == 0){ //avoid divide by 0 error
			baselinePhotoCons[k] = -1*FLT_MAX; //edited here
		}
		else{
			baselinePhotoCons[k] = NCCScoreSum/float((segmentsAllFinal[k].size()));
		}
	}
	//cout << "Ended baseline photoconsistency score computation." << endl;

}

void getSuperpixelAdjacencyDir(cv::Mat& superpixelAdjacencyMat,
		std::vector<std::vector<int> >& superpixelsLeft,
		std::vector<std::vector<int> >& superpixelsRight,
		std::vector<std::vector<int> >& superpixelsAbove,
		std::vector<std::vector<int> >& superpixelsBelow,
		std::vector<cv::Vec2f>& segmentCentroids){

	int idx_1 = 0;
	int idx_2 = 0;
	float y1 = 0.0;
	float y2 = 0.0;
	float x1 = 0.0;
	float x2 = 0.0;
	for (int k = 0; k < superpixelAdjacencyMat.rows; k++){
		for (int j = 0; j < superpixelAdjacencyMat.cols; j++){
			if (superpixelAdjacencyMat.at<unsigned char>(k,j) != 0){ //if adjacent
				idx_2 = k;
				idx_1 = j;
				y1 = segmentCentroids[idx_1].val[0];
				y2 = segmentCentroids[idx_2].val[0];
				x1 = segmentCentroids[idx_1].val[1];
				x2 = segmentCentroids[idx_2].val[1];
				float ang = 0.0;
				//find relationships
				if (y1 - y2 >= 0){
					if (x1 - x2 >= 0){
						ang = atan((x1 - x2)/(y2 - y2));
						if (ang < M_PI/4){
							//2 is above 1, row idx is above col idx
							superpixelAdjacencyMat.at<unsigned char>(idx_2,idx_1) = 2;
							superpixelsAbove[idx_1].push_back(idx_2); //idx2 is above idx1
						}
						else{
							//2 is left of 1
							superpixelAdjacencyMat.at<unsigned char>(idx_2,idx_1) = 3;
							superpixelsLeft[idx_1].push_back(idx_2); //idx2 is left of idx1
						}
					}
					else{ //x1 - x2 < 0
						ang = atan((x2 - x1)/(y1 - y2));
						if (ang < M_PI/4){
							//2 is above 1
							superpixelAdjacencyMat.at<unsigned char>(idx_2,idx_1) = 2;
							superpixelsAbove[idx_1].push_back(idx_2); //idx2 is above idx1
						}
						else{
							//2 is right of 1
							superpixelAdjacencyMat.at<unsigned char>(idx_2,idx_1) = 1;
							superpixelsRight[idx_1].push_back(idx_2); //idx2 is right of idx1
						}
					}
				}
				else{ // y1 - y2 < 0
					if (x1 - x2 >= 0){
						ang = atan((x1 - x2)/(y2 - y1));
						if (ang < M_PI/4){
							//2 is below 1
							superpixelAdjacencyMat.at<unsigned char>(idx_2,idx_1) = 4;
							superpixelsBelow[idx_1].push_back(idx_2); //idx2 is below idx1
						}
						else{
							//2 is left of 1
							superpixelAdjacencyMat.at<unsigned char>(idx_2,idx_1) = 3;
							superpixelsLeft[idx_1].push_back(idx_2); //idx2 is left of idx1
						}
					}
					else{ //x1 - x2 < 0
						ang = atan((x2-x1)/(y2-y1));
						if (ang < M_PI/4){
							//2 is below 1
							superpixelAdjacencyMat.at<unsigned char>(idx_2,idx_1) = 4;
							superpixelsBelow[idx_1].push_back(idx_2); //idx2 is below idx1
						}
						else{
							//2 is right of 1
							superpixelAdjacencyMat.at<unsigned char>(idx_2,idx_1) = 1;
							superpixelsRight[idx_1].push_back(idx_2); //idx2 is right of idx1
						}
					}
				}
			}
		}
	}
}

void getAdjacentSuperpixels(cv::Mat& superpixelAdjacencyMat, vl_uint32* segmentation, cv::Mat& leftImageG){
	int segIDX = 0;
	int segIDX_right = 0;
	int segIDX_below = 0;
	for (int i = 0; i < leftImageG.rows - 1; i++){
		for (int j = 0; j < leftImageG.cols - 1; j++){
			segIDX = (int) segmentation[j + leftImageG.cols*i];
			segIDX_right = (int) segmentation[j + 1 + leftImageG.cols*i];
			segIDX_below = (int) segmentation[j + leftImageG.cols*(i+1)];

			if (segIDX != segIDX_right){
				superpixelAdjacencyMat.at<unsigned char>(segIDX,segIDX_right) = 1;
				superpixelAdjacencyMat.at<unsigned char>(segIDX_right, segIDX) = 1;
			}
			if (segIDX != segIDX_below){
				superpixelAdjacencyMat.at<unsigned char>(segIDX,segIDX_below) = 1;
				superpixelAdjacencyMat.at<unsigned char>(segIDX_below, segIDX) = 1;
			}
		}
	}
}


void getSuperpixelCentroids( std::vector<cv::Vec2f>& segmentCentroids, std::vector<std::vector<cv::Vec3f> >& segmentsAll,
		double px, double py){
	float i_idx = 0.0;
	float j_idx = 0.0;
	for (unsigned int i = 0; i < segmentsAll.size(); i++){ //for each segment
		float i_idx_avg = 0.0;
		float j_idx_avg = 0.0;
		if (segmentsAll[i].size() > 0){
			for (unsigned int j = 0; j < segmentsAll[i].size(); j++){ //for each pixel in the segment
				i_idx = segmentsAll[i][j].val[0] + float(py);
				j_idx = segmentsAll[i][j].val[1] + float(px);
				i_idx_avg += i_idx;
				j_idx_avg += j_idx;
			}
			i_idx_avg = round(i_idx_avg/float(segmentsAll[i].size()));
			j_idx_avg = round(j_idx_avg/float(segmentsAll[i].size()));
			Vec2f tmp = Vec2f(i_idx_avg,j_idx_avg);
			segmentCentroids[i] = tmp;
		}
		else{
			Vec2f tmp = Vec2f(0.0, 0.0);
			segmentCentroids[i] = tmp;
		}
	}
}

float getNCCScoreInterp_RightCamReferenceMMX( cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& leftPatch, cv::Mat& rightPatch,
		int r, int c, double px, double py, cv::Vec4f& pl, unsigned int winSize, int winSize2, int maxDisparity){
	int vecsize = winSize*winSize;
	if(vecsize%16 > 0)
		vecsize += 16-vecsize%16;
	float NCCScore = 0.0;

	int w = rightImageG.cols;

	//plane parameters
	float A = pl.val[0]; float B = pl.val[1]; float C = pl.val[2]; float D = pl.val[3];

	float diffBest = -1*FLT_MAX; //largest difference in window x dimension
	float smallest = FLT_MAX; //smallest x values
	float largest = -1*FLT_MAX; //largest x value
	float diff = -1*FLT_MAX; //difference in x value

	float x = c - px;
	float y = r - py;

	 uint16* lvec = (uint16*)calloc(vecsize,sizeof(uint16));
	 uint16* rvec = (uint16*)calloc(vecsize,sizeof(uint16));
#pragma omp parallel
	 {
#pragma omp for
		for (int i = r - winSize2; i <= r + winSize2; i++){
			for (int j = c - winSize2; j <= c + winSize2; j++){
				rvec[(i-(r - winSize2))*(winSize) + (j - (c - winSize2))] = (uint16)rightImageG.at<uchar>(i,j);
			}
		}
	 }

	 double AR=0,CR=0;
#ifdef USE_AVX2
	NCCprecompute256(rvec,AR,CR,vecsize,winSize*winSize);
#else
	NCCprecompute(rvec,AR,CR,vecsize,winSize*winSize);
#endif

	unsigned int ctr2 = 0;
		//interpolation linear for left image patch,
		//which is necessarily smaller in x-dim than right image patch
		float j_exact, j_floor, j_ceil, final_val;
		float leftImageVal_floor, leftImageVal_ceil;
		float d_pl;

		for (int i = r - winSize2; i <= r + winSize2; i++){ //right image dim, row (y)
			y = float(i) - py;

			//bookeeping temporarily - to compute max x-dim for right image patch
			smallest = FLT_MAX; //reset these for each row
			largest = -1*FLT_MAX;
			diff = -1*FLT_MAX;

			for (int j = c - winSize2; j <= c + winSize2; j++){ //right image dim
				x = float(j) - px;
				//disparity using this plane, calculated at pixel from left image
				d_pl = ((-1*A)/(C+B))*y - (B/(C+B))*(x) - (D/(C+B)); //right view plane equation - x here is xR

				if (d_pl < 0){ //any point in window is behind camera, penalize NCC Score to -Inf
					NCCScore = -FLT_MAX;
					return NCCScore;
				}
				else{ //not behind camera
					j_exact = float(j) + d_pl; //calc. left image loc with disparity offset
					j_floor = float(floor(j_exact));
					j_ceil = float(ceil(j_exact));


					//temporarily add below two if statements for bookkeeping
					if (j_exact < smallest){
						smallest = j_exact;
					}
					if (j_exact > largest){
						largest = j_exact;
					}

					//if j is within bounds of image
					if (int(j_floor) >= 0 && int(j_ceil) <= w){
						leftImageVal_floor = float(leftImageG.at<uchar>(i,int(j_floor)));
						leftImageVal_ceil = float(leftImageG.at<uchar>(i,int(j_ceil)));
						if ((j_ceil - j_floor) == 0){ //if one pixel width only, pull exact value -- eliminate later?
							final_val = leftImageVal_floor;
						}
						else{ //if more than one pixel width, interpolate
							final_val = leftImageVal_floor + (leftImageVal_ceil - leftImageVal_floor)*((j_exact - j_floor)/(j_ceil - j_floor));
						}
						//set right patch vector to this value
						lvec[ctr2] = (uint16)final_val;
						ctr2++;
					}
					else{ //j is not within bounds of image... NCC score is 0
						NCCScore = 0.0;
						return NCCScore;
					}
				}
			}
			diff = largest - smallest;
			if (diff > diffBest){
				diffBest = diff;
			}
		}

		double AL=0,CL=0;
#ifdef USE_AVX2
		NCCprecompute256(lvec,AL,CL,vecsize,winSize*winSize);
#else
		NCCprecompute(lvec,AL,CL,vecsize,winSize*winSize);
#endif

#ifdef USE_AVX2
		float score = computencc256(lvec, rvec, vecsize,winSize*winSize, AL,  AR,  CL,  CR);
#else
		float score = computencc(lvec, rvec, vecsize,winSize*winSize, AL,  AR,  CL,  CR);
#endif

		delete [] lvec;
		delete [] rvec;
		return score;

}

//projection of plane on right camera is larger, so winsize x winsize window is on right cam
// (-B/C) < 0
// removed arg cv::Mat& segmentPixelMask,
float getNCCScoreInterp_RightCamReference( cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& leftPatch, cv::Mat& rightPatch,
		int r, int c, double px, double py, cv::Vec4f& pl, unsigned int winSize, int winSize2, int maxDisparity){

	//c input must be "col" in right window, cR = cL - d

	float NCCScore = 0.0;

	int w = rightImageG.cols;

	//plane parameters
	float A = pl.val[0]; float B = pl.val[1]; float C = pl.val[2]; float D = pl.val[3];

	float diffBest = -1*FLT_MAX; //largest difference in window x dimension
	float smallest = FLT_MAX; //smallest x values
	float largest = -1*FLT_MAX; //largest x value
	float diff = -1*FLT_MAX; //difference in x value

	float x = c - px;
	float y = r - py;

	unsigned int ctr = 0; //for left
	unsigned int ctr2 = 0; //for right

	//get image grayscale values - rightPatch is pixel-aligned window
	for (int i = r - winSize2; i <= r + winSize2; i++){
		for (int j = c - winSize2; j <= c + winSize2; j++){
			(rightPatch).at<float>(ctr, 0) = float(rightImageG.at<uchar>(i,j));
			ctr++;
		}
	}

	//get mean and stdev values
	float tmp;
	Scalar leftStd, rightStd, leftMean, rightMean;
	meanStdDev((rightPatch), rightMean, rightStd);

	if (rightStd.val[0] > FLT_EPSILON ){
		//interpolation linear for left image patch,
		//which is necessarily smaller in x-dim than right image patch
		float j_exact, j_floor, j_ceil, final_val;
		float leftImageVal_floor, leftImageVal_ceil;
		float d_pl;

		for (int i = r - winSize2; i <= r + winSize2; i++){ //right image dim, row (y)
			y = float(i) - py;

			//bookeeping temporarily - to compute max x-dim for right image patch
			smallest = FLT_MAX; //reset these for each row
			largest = -1*FLT_MAX;
			diff = -1*FLT_MAX;

			for (int j = c - winSize2; j <= c + winSize2; j++){ //right image dim
				x = float(j) - px;
				//disparity using this plane, calculated at pixel from left image
				d_pl = ((-1*A)/(C+B))*y - (B/(C+B))*(x) - (D/(C+B)); //right view plane equation - x here is xR

				if (d_pl < 0){ //any point in window is behind camera, penalize NCC Score to -Inf
					NCCScore = -FLT_MAX;
					return NCCScore;
				}
				else{ //not behind camera
					j_exact = float(j) + d_pl; //calc. left image loc with disparity offset
					j_floor = float(floor(j_exact));
					j_ceil = float(ceil(j_exact));

					//temporarily add below two if statements for bookkeeping
					if (j_exact < smallest){
						smallest = j_exact;
					}
					if (j_exact > largest){
						largest = j_exact;
					}

					//if j is within bounds of image
					if (int(j_floor) >= 0 && int(j_ceil) <= w){
						leftImageVal_floor = float(leftImageG.at<uchar>(i,int(j_floor)));
						leftImageVal_ceil = float(leftImageG.at<uchar>(i,int(j_ceil)));
						if ((j_ceil - j_floor) == 0){ //if one pixel width only, pull exact value -- eliminate later?
							final_val = leftImageVal_floor;
						}
						else{ //if more than one pixel width, interpolate
							final_val = leftImageVal_floor + (leftImageVal_ceil - leftImageVal_floor)*((j_exact - j_floor)/(j_ceil - j_floor));
						}
						//set right patch vector to this value
						leftPatch.at<float>(ctr2, 0) = final_val;
						ctr2++;
					}
					else{ //j is not within bounds of image... NCC score is 0
						NCCScore = 0.0;
						return NCCScore;
					}
				}
			}
			diff = largest - smallest;
			if (diff > diffBest){
				diffBest = diff;
			}
		}

		meanStdDev(leftPatch, leftMean, leftStd);

		if ( rightStd.val[0] > FLT_EPSILON){ //!= 0
			(leftPatch) = (leftPatch) - leftMean.val[0]; //offset vector by mean
			(rightPatch) = (rightPatch) - rightMean.val[0];
			(leftPatch) = (1/leftStd.val[0])*(leftPatch); //divide vector by Stdev
			(rightPatch) = (1/rightStd.val[0])*(rightPatch);

			tmp = float((leftPatch).dot((rightPatch)));
			tmp = tmp/(float(winSize)*float(winSize));
			NCCScore = tmp;
		}
		else{ // left std is 0
			//NCCScore = FLT_MAX;
			NCCScore = 0.0;
			//cout << "std dev left is 0" << endl;
			return NCCScore;
		}

	}
	else{ // right std is 0
		NCCScore = 0.0;
	}
	return NCCScore;
}

float getNCCScoreInterp_LeftCamReferenceMMX( cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& leftPatch, cv::Mat& rightPatch,
		int r, int c, double px, double py, cv::Vec4f& pl, unsigned int winSize, int winSize2, int maxDisparity){
	int vecsize = winSize*winSize;
	if(vecsize%16 > 0)
		vecsize += 16-vecsize%16;
	float NCCScore = 0.0;

	int w = rightImageG.cols;

	//plane parameters
	float A = pl.val[0]; float B = pl.val[1]; float C = pl.val[2]; float D = pl.val[3];

	float diffBest = -1*FLT_MAX; //largest difference in window x dimension
	float smallest = FLT_MAX; //smallest x values
	float largest = -1*FLT_MAX; //largest x value
	float diff = -1*FLT_MAX; //difference in x value

	float x = c - px;
	float y = r - py;

	 uint16* lvec = (uint16*)calloc(vecsize,sizeof(uint16));
	 uint16* rvec = (uint16*)calloc(vecsize,sizeof(uint16));
#pragma omp parallel
	 {
#pragma omp for
	for (int i = r - winSize2; i <= r + winSize2; i++){
		for (int j = c - winSize2; j <= c + winSize2; j++){
			lvec[(i-(r - winSize2))*(winSize) + (j - (c - winSize2))] = (uint16)leftImageG.at<uchar>(i,j);
		}
	}
	 }

	 double AL=0,CL=0;

#ifdef USE_AVX2
	NCCprecompute256(lvec,AL,CL,vecsize,winSize*winSize);
#else
	NCCprecompute(lvec,AL,CL,vecsize,winSize*winSize);
#endif

	unsigned int ctr2 = 0;

	//interpolation linear for right image patch,
	//which is necessarily smaller in x-dim than left image patch
	float j_exact, j_floor, j_ceil, final_val;
	float rightImageVal_floor, rightImageVal_ceil;
	float d_pl;

	for (int i = r - winSize2; i <= r + winSize2; i++){ //left image dim, row (y)
		y = float(i) - py;

		//bookeeping temporarily - to compute max x-dim for right image patch
			smallest = FLT_MAX; //reset these for each row
			largest = -1*FLT_MAX;
			diff = -1*FLT_MAX;

			for (int j = c - winSize2; j <= c + winSize2; j++){ //left image dim
				x = float(j) - px;
				//disparity using this plane, calculated at pixel from left image
				d_pl = ((-1*A)/C)*y - (B/C)*(x) - (D/C);

				if (d_pl < 0){ //any point in window is behind camera, penalize NCC Score to -Inf
					NCCScore = -FLT_MAX;
					return NCCScore;
				}
				else{ //not behind camera
					j_exact = float(j) - d_pl; //calc. right image loc with disparity offset
					j_floor = float(floor(j_exact));
					j_ceil = float(ceil(j_exact));

					//temporarily add below two if statements for bookkeeping
					if (j_exact < smallest){
						smallest = j_exact;
					}
					if (j_exact > largest){
						largest = j_exact;
					}

					//if j is within bounds of image
					if (int(j_floor) >= 0 && int(j_ceil) <= w){
						rightImageVal_floor = float(rightImageG.at<uchar>(i,int(j_floor)));
						rightImageVal_ceil = float(rightImageG.at<uchar>(i,int(j_ceil)));
						if ((j_ceil - j_floor) == 0){ //if one pixel width only, pull exact value -- eliminate later?
							final_val = rightImageVal_floor;
						}
						else{ //if more than one pixel width, interpolate
							final_val = rightImageVal_floor + (rightImageVal_ceil - rightImageVal_floor)*((j_exact - j_floor)/(j_ceil - j_floor));
						}
						//set right patch vector to this value
						rvec[ctr2] =(uint16) final_val;
						ctr2++;
					}
					else{ //j is not within bounds of image... NCC score is 0
						NCCScore = 0.0;
						return NCCScore;
					}
				}
			}
			diff = largest - smallest;
			if (diff > diffBest){
				diffBest = diff;
			}
		}

		double AR=0,CR=0;
#ifdef USE_AVX2
		NCCprecompute256(rvec,AR,CR,vecsize,winSize*winSize);
#else
		NCCprecompute(rvec,AR,CR,vecsize,winSize*winSize);
#endif

#ifdef USE_AVX2
		float score = computencc256(lvec, rvec, vecsize,winSize*winSize, AL,  AR,  CL,  CR);
#else
		float score = computencc(lvec, rvec, vecsize,winSize*winSize, AL,  AR,  CL,  CR);
#endif

		delete [] lvec;
		delete [] rvec;

		return score;

}

//projection of plane on left camera is larger, so winsize x winsize window is on left cam
// (-B/C) > 0
//removed arg cv::Mat& segmentPixelMask,
float getNCCScoreInterp_LeftCamReference( cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& leftPatch, cv::Mat& rightPatch,
		int r, int c, double px, double py, cv::Vec4f& pl, unsigned int winSize, int winSize2, int maxDisparity){

	float NCCScore = 0.0;

	int w = rightImageG.cols;

	//plane parameters
	float A = pl.val[0]; float B = pl.val[1]; float C = pl.val[2]; float D = pl.val[3];

	float diffBest = -1*FLT_MAX; //largest difference in window x dimension
	float smallest = FLT_MAX; //smallest x values
	float largest = -1*FLT_MAX; //largest x value
	float diff = -1*FLT_MAX; //difference in x value

	float x = c - px;
	float y = r - py;

	unsigned int ctr = 0; //for left
	unsigned int ctr2 = 0; //for right

	//get image grayscale values - leftPatch is pixel-aligned window
	for (int i = r - winSize2; i <= r + winSize2; i++){
		for (int j = c - winSize2; j <= c + winSize2; j++){

			(leftPatch).at<float>(ctr, 0) = float(leftImageG.at<uchar>(i,j));
			ctr++;
		}
	}

	//get mean and stdev values
	float tmp;
	Scalar leftStd, rightStd, leftMean, rightMean;
	meanStdDev((leftPatch), leftMean, leftStd);

	if (leftStd.val[0] > FLT_EPSILON ){
		//interpolation linear for right image patch,
		//which is necessarily smaller in x-dim than left image patch
		float j_exact, j_floor, j_ceil, final_val;
		float rightImageVal_floor, rightImageVal_ceil;
		float d_pl;

		for (int i = r - winSize2; i <= r + winSize2; i++){ //left image dim, row (y)
			y = float(i) - py;

			//bookeeping temporarily - to compute max x-dim for right image patch
			smallest = FLT_MAX; //reset these for each row
			largest = -1*FLT_MAX;
			diff = -1*FLT_MAX;

			for (int j = c - winSize2; j <= c + winSize2; j++){ //left image dim
				x = float(j) - px;
				//disparity using this plane, calculated at pixel from left image
				d_pl = ((-1*A)/C)*y - (B/C)*(x) - (D/C);

				if (d_pl < 0){ //any point in window is behind camera, penalize NCC Score to -Inf
					NCCScore = -FLT_MAX;
					return NCCScore;
				}
				else{ //not behind camera
					j_exact = float(j) - d_pl; //calc. right image loc with disparity offset
					j_floor = float(floor(j_exact));
					j_ceil = float(ceil(j_exact));

					//temporarily add below two if statements for bookkeeping
					if (j_exact < smallest){
						smallest = j_exact;
					}
					if (j_exact > largest){
						largest = j_exact;
					}

					//if j is within bounds of image
					if (int(j_floor) >= 0 && int(j_ceil) <= w){
						rightImageVal_floor = float(rightImageG.at<uchar>(i,int(j_floor)));
						rightImageVal_ceil = float(rightImageG.at<uchar>(i,int(j_ceil)));
						if ((j_ceil - j_floor) == 0){ //if one pixel width only, pull exact value -- eliminate later?
							final_val = rightImageVal_floor;
						}
						else{ //if more than one pixel width, interpolate
							final_val = rightImageVal_floor + (rightImageVal_ceil - rightImageVal_floor)*((j_exact - j_floor)/(j_ceil - j_floor));
						}
						//set right patch vector to this value
						rightPatch.at<float>(ctr2, 0) = final_val;
						ctr2++;
					}
					else{ //j is not within bounds of image... NCC score is 0
						NCCScore = 0.0;
						return NCCScore;
					}
				}
			}
			diff = largest - smallest;
			if (diff > diffBest){
				diffBest = diff;
			}
		}

		meanStdDev(rightPatch, rightMean, rightStd);

		if ( rightStd.val[0] > FLT_EPSILON){ //!= 0
			(leftPatch) = (leftPatch) - leftMean.val[0]; //offset vector by mean
			(rightPatch) = (rightPatch) - rightMean.val[0];
			(leftPatch) = (1/leftStd.val[0])*(leftPatch); //divide vector by Stdev
			(rightPatch) = (1/rightStd.val[0])*(rightPatch);

			tmp = float((leftPatch).dot((rightPatch)));
			tmp = tmp/(float(winSize)*float(winSize));
			NCCScore = tmp;
		}
		else{ // right std is 0
			//NCCScore = FLT_MAX;
			NCCScore = 0.0;
			//cout << "std dev right is 0" << endl;
			return NCCScore;
		}

	}
	else{ // left std is 0
		NCCScore = 0.0;
	}
	return NCCScore;
}

//removed arg cv::Mat& segmentPixelMask,
float getNCCScoreInterp( cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& leftPatch, cv::Mat& rightPatch,
		int r, int c, double px, double py, cv::Vec4f& pl, unsigned int winSize, int winSize2, int maxDisparity){

	float NCCScore = 0.0;
	int h = rightImageG.rows;
	int w = rightImageG.cols;

	//plane parameters
	float A = pl.val[0]; float B = pl.val[1]; float C = pl.val[2]; float D = pl.val[3];

	if (r >= winSize2 +1 && c >= winSize2 && r < h - winSize2 -1 && c < w - winSize2 ){ //within bounds of left image
		//check for plane orientation
		if ((-1*B)/C < 0){ //projection is larger on right image,
			//column idx moves to POV of right camera
			float x = c - px;
			float y = r - py;
			float d_pl = ((-1*A)/C)*y - (B/C)*(x) - (D/C);

			if (d_pl < 0){ //behind camera
				NCCScore = -1*FLT_MAX;
				//cout << "behind camera error" << endl;
			}
			else{
				int col = c - d_pl; //right image col = left image col - disparity
				if (col >= winSize2 && col < w - winSize2){ //within bounds of right image
					NCCScore = getNCCScoreInterp_RightCamReferenceMMX(leftImageG, rightImageG, leftPatch, rightPatch, r, col, px, py, pl, winSize, winSize2, maxDisparity);
					//NCCScore = getNCCScoreInterp_RightCamReference(leftImageG, rightImageG, leftPatch, rightPatch, r, col, px, py, pl, winSize, winSize2, maxDisparity);
				}
				else{ //not within bounds of right image, NCC score is 0
					NCCScore = 0.0;
				}
			}
		}
		else { //projection larger on left image
			//columnn idx stays in POV of left camera
			NCCScore = getNCCScoreInterp_LeftCamReferenceMMX(leftImageG, rightImageG, leftPatch, rightPatch, r, c, px, py, pl, winSize, winSize2, maxDisparity);
			//NCCScore = getNCCScoreInterp_LeftCamReference(leftImageG, rightImageG, leftPatch, rightPatch, r, c, px, py, pl, winSize, winSize2, maxDisparity);
		}
	}
	else{//not within bounds of left image, NCC score is 0
		NCCScore = 0.0;
		return NCCScore;
	}
	return NCCScore;
}

// removed arg cv::Mat& segmentPixelMask,
float getSuperpixelPCScore_interp_fromNewPlane( std::vector<cv::Vec3f>& segmentsAll, cv::Vec4f& pl,
		cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& leftPatch, cv::Mat& rightPatch,
		unsigned int winSize, double px, double py, int idx, int maxDisparity){

	int r = 0;
	int c = 0;
	float tmp_y, tmp_x, tmp_d = 0.0;

	float A, B, C, D;
	float pcScore = 0.0;
	int winSize2 = (int) floor(float(winSize)/2.0);

	if (segmentsAll.size() > 0){ //if we have pixels in the superpixels
		//TO-DO - missing model superpixels need to get guesses from neighbors

			for (unsigned int p = 0; p < segmentsAll.size(); p++){ //for each pixel in superpixel

				 A = pl.val[0];
				 B = pl.val[1];
				 C = pl.val[2];
				 D = pl.val[3];
				//remove below later

				 tmp_y = segmentsAll[p].val[0]; //y
				 tmp_x = segmentsAll[p].val[1]; //x
				 tmp_d = ((-1*A)/C)*tmp_y - (B/C)*tmp_x - (D/C); //refit to plane

				 r = (int) round(segmentsAll[p].val[0] + float(py)); //y
				 c = (int) round(segmentsAll[p].val[1] + float(px)); //x
				if (tmp_d < 0){ //point is behind the camera - penalize photoconsistency to -Inf
					pcScore+= (-1*FLT_MAX);
					//cout<< "behind camera error" << endl;
				}
				else{

					pcScore+= getNCCScoreInterp(leftImageG, rightImageG, leftPatch, rightPatch, r, c, px, py, pl, winSize, winSize2, maxDisparity);
				}
			}
			pcScore = pcScore/(segmentsAll.size()); //normalize by number of pixels

		}

	else{
		pcScore = -1*FLT_MAX; //no pixels to evaluate
	}
	return pcScore;
}



//fronto-parallel window assumption
float getSuperpixelPCScoreFP_fromNewPlane(std::vector<cv::Vec3f>& segmentsAll, cv::Vec4f& pl,
		cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& leftPatch, cv::Mat& rightPatch,
		unsigned int winSize, double px, double py, int idx){

	int r = 0;
	int c = 0;
	float tmp_y, tmp_x, tmp_d = 0.0;
	float A, B, C, D;
	float pcScore = 0.0;

	A = pl.val[0];
	B = pl.val[1];
	C = pl.val[2];
	D = pl.val[3];

	if (segmentsAll.size() > 0){ //if we have pixels in the superpixels
		//TO-DO - missing model superpixels need to get guesses from neighbors
		for (unsigned int p = 0; p < segmentsAll.size(); p++){ //for each pixel in superpixel
			tmp_y = segmentsAll[p].val[0]; //y
			tmp_x = segmentsAll[p].val[1]; //x
			tmp_d = ((-1*A)/C)*tmp_y - (B/C)*tmp_x - (D/C); //refit to plane
			tmp_d = round(tmp_d); // so can use in photoconsistency function
			r = (int) round(segmentsAll[p].val[0] + float(py)); //y
			c = (int) round(segmentsAll[p].val[1] + float(px)); //x
			if (tmp_d < 0){ //point is behind the camera - penalize photoconsistency to -Inf
				pcScore+= (-1*FLT_MAX);
				//cout<< "behind camera error" << endl;
			}
			else{
				//pcScore+= getNCCScoreFP(leftImageG, rightImageG, leftPatch, rightPatch, r, c, tmp_d, winSize);
				pcScore+=getNCCScoreFPMMX(leftImageG, rightImageG,r, c, tmp_d, winSize);
			}
		}
		pcScore = pcScore/(segmentsAll.size()); //normalize by number of pixels
	}
	else{
		pcScore = -1*FLT_MAX; //no pixels to evaluate
	}
	return pcScore;
}

//removed arg cv::Mat& segmentPixelMask,
void propagatePlaneOneDir( std::vector<vector<Vec4f> >& segmentsRefitPl,
		std::vector<cv::Vec3f>& segmentsAll, std::vector<int>& superpixelsAdj,
		float& baselinePhotoCons, cv::Mat& leftImageG, cv::Mat& rightImageG,
		cv::Mat& leftPatch, cv::Mat& rightPatch, unsigned int winSize, double px, double py, int i_idx,
		bool interpolation, int maxDisparity){

	unsigned int numAdj = superpixelsAdj.size();
	Vec4f pl;
	while (numAdj > 0){
		int j = superpixelsAdj[numAdj-1];
		if((segmentsRefitPl[j].size() > 0) && (segmentsAll.size() > 0)){
			//we found a model for the adj superpixel & current superpixel has pixels
			pl = segmentsRefitPl[j][0];
			float newPCScore;
			if (interpolation){
				newPCScore = getSuperpixelPCScore_interp_fromNewPlane(segmentsAll, pl, leftImageG, rightImageG, leftPatch, rightPatch, winSize, px, py, i_idx, maxDisparity );
			}
			else{
				newPCScore = getSuperpixelPCScoreFP_fromNewPlane(segmentsAll, pl, leftImageG, rightImageG, leftPatch, rightPatch, winSize, px, py, i_idx  );
			}
			if (newPCScore > baselinePhotoCons){
				baselinePhotoCons = newPCScore;
				if (segmentsRefitPl[i_idx].size()> 0){
					segmentsRefitPl[i_idx][0] = pl;
				}
				else{
					Vec4f tmp = Vec4f(pl);
					segmentsRefitPl[i_idx].push_back(tmp);
				}
				//cout << "propagation plane from: " << j << " to: " << i_idx << endl;
			}
		}
		numAdj--;
	}
}

void propagatePlanesAllDir(std::vector<vector<Vec4f> >& segmentsRefitPl,
		std::vector<std::vector<cv::Vec3f> >& segmentsAll,
		std::vector<vector<int> >& superpixelsLeft, std::vector<vector<int> >& superpixelsRight,
		std::vector<vector<int> >& superpixelsAbove, std::vector<vector<int> >& superpixelsBelow,
		std::vector<float>& baselinePhotoCons, cv::Mat& leftImageG, cv::Mat& rightImageG,
		unsigned int winSize, double px, double py, int propIter, bool interpolation, int maxDisparity){
	//cout << "Starting plane propagation." << endl;

	cv::Mat leftPatch =  cv::Mat::zeros(winSize*winSize, 1, CV_32F); //winSize
	cv::Mat rightPatch =  cv::Mat::zeros(winSize*winSize, 1, CV_32F); //winSize
	struct timeval  tv1, tv2;

	for (int p = 0; p < propIter; p++){
		//left

#ifdef M_TIME
		gettimeofday(&tv1, NULL);
#endif
		//cout << "propagating from left..." << endl;
		//myfile << "propagating from left..." << endl;
		for (unsigned int i = 0; i < segmentsAll.size(); i++){ //for each superpixel
			propagatePlaneOneDir(segmentsRefitPl, segmentsAll[i], superpixelsLeft[i], baselinePhotoCons[i], leftImageG, rightImageG, leftPatch, rightPatch, winSize, px, py, i, interpolation, maxDisparity); //propagate to left
		}
		//right
		//cout << "propagating from right..." << endl;
		//myfile << "propagating from right..." << endl;
		for (unsigned int i = 0; i < segmentsAll.size(); i++){ //for each superpixel
			propagatePlaneOneDir(segmentsRefitPl, segmentsAll[i], superpixelsRight[i], baselinePhotoCons[i], leftImageG, rightImageG, leftPatch, rightPatch, winSize, px, py, i, interpolation, maxDisparity); //propagate to left
		}
		//above
		//cout << "propagating from above..." << endl;
		//myfile << "propagating from above..." << endl;
		for (unsigned int i = 0; i < segmentsAll.size(); i++){ //for each superpixel
			propagatePlaneOneDir( segmentsRefitPl, segmentsAll[i], superpixelsAbove[i], baselinePhotoCons[i], leftImageG, rightImageG, leftPatch, rightPatch, winSize, px, py, i, interpolation, maxDisparity); //propagate to left
		}
		//below
		//cout << "propagating from below..." << endl;
		//myfile << "propagating from below..." << endl;
		for (unsigned int i = 0; i < segmentsAll.size(); i++){ //for each superpixel
			propagatePlaneOneDir(segmentsRefitPl, segmentsAll[i], superpixelsBelow[i], baselinePhotoCons[i], leftImageG, rightImageG, leftPatch, rightPatch, winSize, px, py, i, interpolation, maxDisparity); //propagate to left
		}

#ifdef M_TIME
		gettimeofday(&tv2, NULL);
		cout << "Iteration " << p+1 << " time: \n" <<
				(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
				(double) (tv2.tv_sec - tv1.tv_sec) << endl;
#endif
	}
	//cout << "Ended plane propagation." << endl;
}

void truncateToMaxDisparity(cv::Mat& dispMapProp, int maxDisparity){
#pragma omp parallel
	{
#pragma omp parallel
		for (int i = 0; i <  dispMapProp.rows; i++){
			for (int j = 0; j < dispMapProp.cols; j++){
				if (dispMapProp.at<float>(i,j) > maxDisparity){
					dispMapProp.at<float>(i,j) = maxDisparity;
				}
				else if (dispMapProp.at<float>(i,j) < 0 ){
					dispMapProp.at<float>(i,j) = 0;
				}
			}
		}
	}
}

void usage(){
	std::cout << "SPS usage" << std::endl;
	std::cout << "Arg1:\t Left png image" << std::endl;
	std::cout << "Arg2:\t Right png image" << std::endl;
	std::cout << "Arg3:\t Middlebury 2014 calibration file" << std::endl;
	std::cout << "Arg4:\t s " << std::endl;
	std::cout << "Arg5:\t v " << std::endl;
	std::cout << "Arg6:\t n"  << std::endl;
}
int main(int argc, char* argv[]) {
	omp_set_num_threads(THREADS);
	struct timeval  tvstart, tvend;
	gettimeofday(&tvstart, NULL);
	char leftImageName[100]; //view1 in Middlebury 06 or im0 in Middlebury 14
	char rightImageName[100]; //view5 in Middlebury or im1 in Middlebury 14
	float randPercent = 0.05;
	float randPercentPlProp = 0.25;
	int propIter = 3;

	if(argc<7){
		usage();
		return 0;
	}

	sprintf(leftImageName,"%s",argv[1]);
	sprintf(rightImageName,"%s",argv[2]);

	randPercent = atof(argv[4]);
	randPercentPlProp = atof(argv[5]);
	propIter =  atoi(argv[6]);

	unsigned int winSize = 5;
	int winSize_ncc = 15;

	cv::Mat leftImage = cv::imread(leftImageName, CV_LOAD_IMAGE_COLOR);
	cv::Mat rightImage = cv::imread(rightImageName, CV_LOAD_IMAGE_COLOR);
	cv::Mat leftImageG, rightImageG;
	cvtColor(leftImage, leftImageG, CV_BGR2GRAY);
	cvtColor(rightImage, rightImageG, CV_BGR2GRAY);

	double h = double(leftImage.rows);
	double w = double(leftImage.cols);
	double px = w/2.0;
	double py = h/2.0;
	int maxDisparity = getMaxDisparityFromFile(argv[3]);

	//------------------
	//DETECT FEATURES TO COMPUTE HOMOGRAPHY
	//------------------

	cv::Mat leftWarpedG, rightWarpedG, H_l, H_r, returnKeypoints;
	//function for y-only homography: y_new = a*y_old + b;
	compute_y_offset_ylinear( returnKeypoints, leftImageG, rightImageG, leftWarpedG, rightWarpedG, w, h, H_l, H_r);

	struct timeval  tv1,tv2;
	vl_uint32* segmentation = new vl_uint32[leftImage.rows*leftImage.cols];
#ifdef M_TIME
	gettimeofday(&tv1, NULL);
#endif

	segmentImage(leftImage, segmentation, REGION, REGULARIZATION, MIN_REGION);

#ifdef M_TIME
	gettimeofday(&tv2, NULL);
	cout << "SLIC, time in seconds\n" <<
							(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
							(double) (tv2.tv_sec - tv1.tv_sec) << endl;
#endif

	int numSegments = getMaxNumberofSegments(leftImageG, segmentation);
	cv::Mat segmentMap_CV_16U = cv::Mat::zeros(leftImage.rows, leftImage.cols, CV_16U);
	cv::Mat segmentMap_CV_16U_warped;


	segmentation_to_segmentMap(segmentation, segmentMap_CV_16U);

	warpSegmentMap(segmentMap_CV_16U, segmentMap_CV_16U_warped, H_l);

	std::vector<vector<Vec3f> > segmentsAll(numSegments);
	std::vector<vector<Vec3f> > segmentsNZRand(numSegments);
	std::vector<vector<Vec3f> > segmentsNZRand2(numSegments);
	std::vector<vector<Vec3f> > segmentsRand(numSegments);

	//start timer NCC


	//need segmentsAll for getting segment centroids below.
	//groupPixelsBySegments(segmentsAll, leftImageG, segmentation, px, py);//un-warped
	groupPixelsBySegments(segmentsAll, leftImageG, segmentMap_CV_16U_warped, px, py);//warped

	//need segmentsNZRand for RANSAC below
	unsigned int randomInitStart = 7;


	for (unsigned int randomInit = randomInitStart; randomInit < (randomInitStart+1); randomInit++){





		randomizePixelsInSegments(segmentsAll, randomInit);
		getVectorSubset(segmentsAll, segmentsRand, randPercent);

		//generate guess for disparities using NCC. modifies points in segmentsRand


#ifdef M_TIME
		gettimeofday(&tv1, NULL);
#endif

		//gettimeofday(&tv1, NULL);
		getNCCDisparities(segmentsRand, leftWarpedG, rightWarpedG, px, py, winSize_ncc, maxDisparity); //warped

#ifdef M_TIME
		gettimeofday(&tv2, NULL);
		cout << "NCC, time in seconds\n" <<
								(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
								(double) (tv2.tv_sec - tv1.tv_sec) << endl;
#endif


#ifdef M_TIME
	gettimeofday(&tv1, NULL);
#endif

	removeZeroPoints(segmentsRand, segmentsNZRand); //before plane fitting!

	randomizePixelsInSegments(segmentsAll, randomInit);
	getVectorSubset(segmentsAll, segmentsNZRand2, randPercentPlProp);

	unsigned int N = 5; //number of RANSAC iterations (overwritten if isAdaptive is true!)
	float t = 1; //distance threshold for inlier criteria
	bool isAdaptive = true; //true for adaptive termination RANSAC
	std::vector<vector<Vec3f> > segmentsPlPoints(numSegments);
	std::vector<vector<Vec3f> > inliers(numSegments);
	RANSACFitPlanes(segmentsPlPoints, inliers, segmentsNZRand, N, t, isAdaptive);


	//refit planes based on inliers
	std::vector<vector<Vec4f> > segmentsRefitPl(numSegments);
	refitPlanesInliers(segmentsRefitPl, segmentsPlPoints, inliers);

	//adjust disparities of all points based on plane fit, after adjustments for inliers
	std::vector<vector<Vec3f> > segmentsAllFinal(numSegments);
	adjustSegments2PlaneFit(segmentsRefitPl, segmentsAll, segmentsAllFinal);

#ifdef M_TIME
	gettimeofday(&tv2, NULL);
	cout << "RANSAC with plane re-fitting to inliers, time in seconds\n" <<
			(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
			(double) (tv2.tv_sec - tv1.tv_sec) << endl;
#endif

	cv::Mat dispMapRefit = cv::Mat::zeros(h, w, CV_32F);
	cv::Mat dispMapRefit8 = cv::Mat::zeros(h, w, CV_8U);
	planeAdjustedSegments2DispMap(segmentsAllFinal, dispMapRefit, px, py);


	//print result here before plane propagation! added 10-4-2015 (PFM)
	//inverse warp final disparity map dispMapProp before write
	cv::Mat dispMapRefit_unwarped;
	unwarpResultDisparity(dispMapRefit, dispMapRefit_unwarped, H_l, w, h);

	//get current photoconsistency scores per superpixel.
	std::vector<float> baselinePhotoCons(numSegments);
	//making these patch vectors here prevents allocation of too much memory in openCV
	cv::Mat leftPatch =  cv::Mat::zeros(winSize*winSize, 1, CV_32F);
	cv::Mat rightPatch =  cv::Mat::zeros(winSize*winSize, 1, CV_32F);

	getSuperpixelPCScoreFP_fromDispMap(dispMapRefit, segmentsNZRand2, baselinePhotoCons, leftWarpedG, rightWarpedG, leftPatch, rightPatch, winSize, px, py); //un-warped

#ifdef M_TIME
	gettimeofday(&tv2, NULL);
	cout << "Baseline photoconsistency computation, time in seconds\n" <<
			(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
			(double) (tv2.tv_sec - tv1.tv_sec) << endl;
#endif

	cv::Mat superpixelAdjacencyMat = cv::Mat::zeros(numSegments, numSegments, CV_8U);
	getAdjacentSuperpixels(superpixelAdjacencyMat, segmentation, leftImageG);
	delete[] segmentation;

	//adjacency matrix to list format
	std::vector<vector<int> > superpixelsLeft (numSegments);
	std::vector<vector<int> > superpixelsRight (numSegments);
	std::vector<vector<int> > superpixelsAbove (numSegments);
	std::vector<vector<int> > superpixelsBelow (numSegments);

	std::vector<Vec2f> superpixelCentroids(numSegments);
	getSuperpixelCentroids(superpixelCentroids, segmentsAll, px, py);

	getSuperpixelAdjacencyDir(superpixelAdjacencyMat, superpixelsLeft, superpixelsRight, superpixelsAbove, superpixelsBelow, superpixelCentroids);


	//edited here to use random prct of points
	propagatePlanesAllDir( segmentsRefitPl, segmentsNZRand2, superpixelsLeft, superpixelsRight, superpixelsAbove, superpixelsBelow,
			baselinePhotoCons, leftWarpedG, rightWarpedG, winSize, px, py, propIter, false, maxDisparity ); //warped

	//------------------
	//CREATE NEW DISPARITY MAP WITH PROPAGATED PLANES
	//------------------
	std::vector<vector<Vec3f> > segmentsAllFinal2(numSegments);//final segments after plane prop
	adjustSegments2PlaneFit(segmentsRefitPl, segmentsAll, segmentsAllFinal2);

	cv::Mat dispMapProp = cv::Mat::zeros(h, w, CV_32F);
	planeAdjustedSegments2DispMap(segmentsAllFinal2, dispMapProp, px, py);

#ifdef M_TIME
	gettimeofday(&tv2, NULL);
	cout << "Plane propagation, time in seconds\n" <<
							(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
							(double) (tv2.tv_sec - tv1.tv_sec) << endl;
#endif

	//ADDED 7-28-2015 cap pixels at maxdisparity to avoid RMS error blow-out
	//maybe remove later?
	truncateToMaxDisparity(dispMapProp, maxDisparity);

	//inverse warp final disparity map dispMapProp before write
	cv::Mat dispMapProp_unwarped=dispMapProp;
	//unwarpResultDisparity(dispMapProp, dispMapProp_unwarped, H_l, w, h);

	//write final disparity map to file in Midd/ImgName/result.png (8-bit)
	cv::Mat dispMapProp8 = cv::Mat::zeros(h, w, CV_8U);
	//dispMapProp.convertTo(dispMapProp8, CV_8U); //un-warped
	dispMapProp_unwarped.convertTo(dispMapProp8, CV_8U); //warped
	//cv::imwrite(saveDispMapImageName, dispMapProp8);

#ifdef M_TIME
	gettimeofday(&tv2, NULL);
	cout << "Plane propagation time in seconds\n" <<
			(double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
			(double) (tv2.tv_sec - tv1.tv_sec) << endl;
#endif

#ifdef M_TIME
	gettimeofday(&tv2, NULL);
	double time_pc = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
			(double) (tv2.tv_sec - tv1.tv_sec);
	cout << "Total baseline pc scoring and plane propagatin, time in seconds\n" << time_pc << endl;
#endif

	//convert final disparity map CV_32F to float* for WriteFilePFM
	//float* buffer = CV_32F_2float(dispMapProp); //un-warped
	//float* buffer = CV_32F_2float(dispMapProp_unwarped); //warped
	//float scale = 1.0/255.0;
	//float scale = 1.0/255.0;

	float* buffer = CV_32F_2float(dispMapProp_unwarped);
	float scale = 1.0/255.0;
	WriteFilePFM(buffer, int(w), int(h), "dispmap.pfm", scale);
	delete[] buffer;

	//------------------
	//EVALUATION
	//------------------
#ifdef M_TIME
	gettimeofday(&tvend, NULL);
	double time_sec = (double) (tvend.tv_usec - tvstart.tv_usec) / 1000000 +
			(double) (tvend.tv_sec - tvstart.tv_sec);
	cout <<  time_sec  << endl;
#endif

	}
	return 0;
}
