#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<highgui.h>
#include "header/camera.h"

#define cvQueryHistValue_1D( hist, idx0 ) \
    ((float)cvGetReal1D( (hist)->bins, (idx0)))

using namespace cv;
using namespace std;

const CvScalar CVX_WHITE = CV_RGB(0xff, 0xff, 0xff);
const CvScalar CVX_BLACK = CV_RGB(0x00, 0x00, 0x00);

const int IMAGE_SCALE = 2;

IplImage *frame, *Ismall, *birdsImg;
IplImage *frameRight, *IsmallRight;
IplImage *Iscratch, *Iscratch2;
IplImage *Imaskt, *Imask, *ImaskBird, *ImaskPlayers, *ImaskLines;
IplImage *ImaskSmall, *ImaskSmall2, *ImaskBirdt;
CvSize sz, szBird;
int cnt=0;

//contour
CvRect playerRect[30];
CvPoint playerCenter[30];
int playerCount=30;

//perspective
CvMat *H = cvCreateMat(3,3,CV_32F), *H_inv = cvCreateMat(3,3,CV_32F);
CvMat *H2 = cvCreateMat(3,3,CV_32F), *H_r2l = cvCreateMat(3,3,CV_32F);
CvPoint2D32f objPts[4], imgPts[4], objPtsRight[4];

//histogram
int max_idx_red[2], max_idx_blue[2], max_idx_green[2];
const int MARGIN = 10;
const int MARGIN_BLUE = 35;
const int MARGIN_GREEN = 40;
const int MARGIN_WHITE = 60;

void AllocImages(IplImage *frame){

	sz=cvGetSize(frame);
	sz.width=sz.width/2;sz.height=sz.height/2;
	szBird = cvGetSize(frame);
	//szBird.width/=2;szBird.height/=2;
	szBird.width=WIDTH;szBird.height=HEIGHT;

	Ismall = cvCreateImage(sz,frame->depth,frame->nChannels);
	birdsImg=cvCreateImage(szBird,frame->depth,frame->nChannels);
	IsmallRight=cvCreateImage(sz,frame->depth,frame->nChannels);

	Iscratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_32F,3);
	Iscratch2 = cvCreateImage(cvGetSize(frame),IPL_DEPTH_32F,3);

	Imask = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	Imaskt = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	ImaskBird = cvCreateImage(szBird, IPL_DEPTH_8U, 1);
	ImaskPlayers = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	ImaskLines = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);

	ImaskSmall = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	ImaskSmall2 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	ImaskBirdt = cvCreateImage(szBird, IPL_DEPTH_8U, 1);

	cvZero(Iscratch);cvZero(Iscratch2);
}

void DeallocateImages(){
	cvReleaseImage(&Iscratch);cvReleaseImage(&Iscratch2);
	cvReleaseImage(&Imaskt);
}

CvPoint transformPoint(const CvPoint point, const CvMat* matrix){

	/*float H[9];
	for(int i=0;i<3;++i){
		for(int j=0;j<3;++j){
			H[i*3+j]=cvmGet(matrix,i,j);
		}
	}
	CvMat *matrix2 = cvCreateMat(3,3,CV_32FC1);
	cvInitMatHeader(matrix2, 3,3,CV_32FC1,H);
	float coordinates[2]={point.x*1.0f, point.y*1.0f};
	CvMat *original = cvCreateMat(1,1,CV_32FC2);
	cvInitMatHeader(original, 1, 1, CV_32FC2, coordinates);
	CvMat *result = cvCreateMat(1,1,CV_32FC2);
	cvPerspectiveTransform(original, result, matrix2);
	float *pdata=(float*)(result->data.ptr);
	return cvPoint((int)(*pdata), (int)(*(pdata+1)));*/

	float coordinates[3]={point.x*1.0f, point.y*1.0f, 1.0f};
	CvMat originVector = cvMat(3,1,CV_32F,coordinates);
	CvMat transformedVector = cvMat(3,1,CV_32F,coordinates);
	CvMat *inv = cvCreateMat(3,3,CV_32F);
	cvInvert(matrix, inv);
	/*for(int i=0;i<3;++i)cout<<coordinates[i]<<' ';
	cout<<endl;
	for(int i=0;i<3;++i){
		for(int j=0;j<3;++j){
			cout<<cvmGet(matrix,i,j)<<' ';
		}
		cout<<endl;
	}*/
	cvMatMul(inv,&originVector,&transformedVector);
	/*for(int i=0;i<3;++i){
		cout<<cvmGet(&transformedVector,i,0)<<' ';
	}
	cout<<endl;*/
	//cvWaitKey();
	CvPoint result=cvPoint((int)(cvmGet(&transformedVector,0,0)/cvmGet(&transformedVector,2,0)),
			(int)(cvmGet(&transformedVector,1,0)/cvmGet(&transformedVector,2,0)));
	return result;
}

void find_connected_components(IplImage *mask, int find_ground=1, int poly1_hull0=1, float perimScale=60, int *num=NULL, CvRect *bbs=NULL, CvPoint *centers=NULL){
	static CvMemStorage *mem_storage=NULL;
	static CvSeq *contours=NULL;
	if(!find_ground){
		cvMorphologyEx(mask,mask,0,0,CV_MOP_OPEN,CVCLOSE_ITR_SMALL);//open
		cvMorphologyEx(mask,mask,0,0,CV_MOP_CLOSE,CVCLOSE_ITR);//close
		//cvErode(mask, mask, 0, 1);
		//cvDilate(mask ,mask, 0, 1);
	}else{
		cvMorphologyEx(mask,mask,0,0,CV_MOP_CLOSE,CVCLOSE_ITR);//close
	}
	if(mem_storage==NULL){
		mem_storage = cvCreateMemStorage(0);
	}else{
		cvClearMemStorage(mem_storage);
	}
	//find contours
	CvContourScanner scanner = cvStartFindContours(mask,mem_storage, 
		sizeof(CvContour),CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
	CvSeq *c;
	int numCont = 0;
	int maxarea = 0;
	while((c=cvFindNextContour(scanner))!=NULL){
		if(find_ground){
			double tmparea=fabs(cvContourArea(c));
			CvRect aRect = cvBoundingRect( c, 0 );
			if ((aRect.width/aRect.height)>5||(aRect.height/aRect.width)>5)
			{
				cvSubstituteContour(scanner, NULL); //删除宽高比例小于设定值的轮廓
				continue;
			}
			if(tmparea > maxarea)
			{
				maxarea = tmparea;
			}
		}
		if(!find_ground){
			double tmparea=fabs(cvContourArea(c));
			double len = cvContourPerimeter(c);
			CvRect aRect = cvBoundingRect( c, 0 );
			if(tmparea<50){//area too small
				cvSubstituteContour(scanner, NULL);
				continue;
			}
			/*if((aRect.y*1.0/mask->height*200+20)<(mask->height*1.0/perimScale)){
				cvSubstituteContour(scanner, NULL);
				continue;
			}*/
			if ((aRect.width/aRect.height)>2)//too fat
			{
				cvSubstituteContour(scanner, NULL); //删除宽高比例小于设定值的轮廓
				continue;
			}
			if(len/tmparea>3){//strange shape
				cvSubstituteContour(scanner, NULL);
				continue;
			}
		}
		double len = cvContourPerimeter(c);
		double q = (mask->height+mask->width)/perimScale;
		if(len<q){
			cvSubstituteContour(scanner,NULL);
		}else{
			/*CvSeq *c_new;
			if(poly1_hull0){
				c_new=cvApproxPoly(c, sizeof(CvContour), mem_storage, CV_POLY_APPROX_DP, CV_CONTOUR_APPROX_LEVEL, 0);
			}else{
				c_new=cvConvexHull2(c, mem_storage, CV_CLOCKWISE, 1);
			}
			cvSubstituteContour(scanner,c_new);//replace the contour with the smooth one*/
		}
		numCont++;
	}
	contours=cvEndFindContours(&scanner);
	
	cvZero(mask);
	IplImage *maskTemp;
	if(num!=NULL){
		int N=*num, numFilled=0, i=0;
		CvMoments moments;
		double M00,M01,M10;
		maskTemp = cvCloneImage(mask);
		for(i=0,c=contours;c!=NULL;c=c->h_next,i++){
			if(i<N){
				cvDrawContours(maskTemp, c, CVX_WHITE, CVX_WHITE, -1, CV_FILLED, 8);
				if(centers!=NULL){
					cvMoments(maskTemp, &moments, 1);
					M00 = cvGetSpatialMoment(&moments, 0, 0);
					M01 = cvGetSpatialMoment(&moments, 0, 1);
					M10 = cvGetSpatialMoment(&moments, 1, 0);
					centers[i].x=(int)(M10/M00);
					centers[i].y=(int)(M01/M00);
				}
				if(bbs!=NULL){
					bbs[i]=cvBoundingRect(c);
				}
				cvZero(maskTemp);
				numFilled++;
			}
			if(find_ground){
				double tmparea=fabs(cvContourArea(c));
				if(tmparea<maxarea-1)continue;
			}
			cvDrawContours(mask, c, CVX_WHITE, CVX_BLACK, -1, CV_FILLED, 8);
		}
		*num = numFilled;
		cvReleaseImage(&maskTemp);
	}else{
		for(c=contours; c!=NULL; c=c->h_next){
			if(find_ground){
				double tmparea=fabs(cvContourArea(c));
				if(tmparea<maxarea-1)continue;
			}
			cvDrawContours(mask, c, CVX_WHITE, CVX_BLACK, -1, CV_FILLED, 8);
		}
	}
}

void cvMouseCallback(int mouseEvent, int x, int y, int flags, void* param){
	switch(mouseEvent){
		case CV_EVENT_LBUTTONDOWN:
			if(cnt<4){
				objPts[cnt].x=x;
				objPts[cnt].y=y;
				cnt++;
			}else if(cnt<8){
				objPtsRight[cnt-4].x=x;
				objPtsRight[cnt-4].y=y;
				cnt++;
			}
			break;
		default:
			break;
	}
}

void showHist(IplImage *frame, int isRight){
	IplImage *scratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);
	IplImage *blue = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *green = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *red = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	cvCvtScale(frame, scratch, 1, 0);
	cvSplit(scratch,blue,green,red,0);
	//cvCvtColor(frame, gray, CV_BGR2GRAY);
	int hist_size=255;
	int hist_height=256;
	float range[] = {1, 255};
	float *ranges[] = {range};
	CvHistogram *blue_hist=cvCreateHist(1,&hist_size,CV_HIST_ARRAY, ranges);
	CvHistogram *green_hist=cvCreateHist(1,&hist_size,CV_HIST_ARRAY, ranges);
	CvHistogram *red_hist=cvCreateHist(1,&hist_size,CV_HIST_ARRAY, ranges);
	cvCalcHist(&blue, blue_hist, 0, 0);
	cvCalcHist(&green, green_hist, 0, 0);
	cvCalcHist(&red, red_hist, 0, 0);
	cvNormalizeHist(blue_hist, 1.0);cvNormalizeHist(green_hist, 1.0);cvNormalizeHist(red_hist, 1.0);

	float max_value = 0;
	cvGetMinMaxHistValue(red_hist, 0,&max_value,0,max_idx_red+isRight);
	cvGetMinMaxHistValue(blue_hist, 0,&max_value,0,max_idx_blue+isRight);
	cvGetMinMaxHistValue(green_hist, 0,&max_value,0,max_idx_green+isRight);
	/*IplImage *hist_img = cvCreateImage(cvSize(hist_size*3,hist_height),IPL_DEPTH_8U,3);
	cvZero(hist_img);
	float max_value = 0;
	cvGetMinMaxHistValue(red_hist, 0, &max_value, 0, &max_idx_red);
	float real_max=0;
	for(int i=1;i<hist_size;++i){
		float bin_val=cvQueryHistValue_1D(red_hist,i);
		if(bin_val>=real_max){max_idx_red = i;real_max=bin_val;}
		int intensity=cvRound(bin_val*hist_height/max_value);
		cvRectangle(hist_img,cvPoint(i,hist_height-1),
			cvPoint((i+1),hist_height-intensity),
			CV_RGB(255,0,0));
	}
	cvGetMinMaxHistValue(blue_hist, 0, &max_value, 0, &max_idx_blue);
	real_max=0;
	for(int i=1;i<hist_size;++i){
		float bin_val=cvQueryHistValue_1D(blue_hist,i);
		if(bin_val>=real_max){max_idx_blue = i;real_max=bin_val;}
		int intensity=cvRound(bin_val*hist_height/max_value);
		cvRectangle(hist_img,cvPoint(i+hist_size,hist_height-1),
			cvPoint((i+1)+hist_size,hist_height-intensity),
			CV_RGB(0,255,0));
	}
	cvGetMinMaxHistValue(green_hist, 0, &max_value, 0, &max_idx_green);
	real_max=0;
	for(int i=1;i<hist_size;++i){
		float bin_val=cvQueryHistValue_1D(green_hist,i);
		if(bin_val>=real_max){max_idx_green = i;real_max=bin_val;}
		int intensity=cvRound(bin_val*hist_height/max_value);
		cvRectangle(hist_img,cvPoint(i+hist_size*2,hist_height-1),
			cvPoint((i+1)+hist_size*2,hist_height-intensity),
			CV_RGB(0,0,255));
	}
	//cout<<max_idx_red<<' '<<max_idx_green<<' '<<max_idx_blue<<endl;
	cvShowImage("gray",hist_img);
	cvReleaseImage(&blue);cvReleaseImage(&green);cvReleaseImage(&red);
	cvReleaseImage(&hist_img);*/
}

void findGround(IplImage *frame, IplImage *Imask, int isRight){

	IplImage *scratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);
	IplImage *blue = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *green = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *red = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	cvCvtScale(frame, scratch, 1, 0);
	cvSplit(scratch,blue,green,red,0);

	cvInRangeS(blue, cvScalar(max_idx_blue[isRight]-MARGIN_BLUE), cvScalar(max_idx_blue[isRight]+MARGIN_BLUE), Imask);
	cvInRangeS(green, cvScalar(max_idx_green[isRight]-MARGIN_GREEN), cvScalar(max_idx_green[isRight]+MARGIN_GREEN), Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvInRangeS(red, cvScalar(max_idx_red[isRight]-MARGIN), cvScalar(max_idx_red[isRight]+MARGIN), Imaskt);
	cvOr(Imask,Imaskt,Imask);
	//cvCmp(green, blue, Imaskt, CV_CMP_GT);
	cvSub(green, blue, Imaskt);
	cvCmpS(Imaskt, 5, Imaskt, CV_CMP_GT);
	cvAnd(Imask,Imaskt,Imask);
	//cvCmp(green, red, Imaskt, CV_CMP_GT);
	cvSub(green, red, Imaskt);
	cvCmpS(Imaskt, 5, Imaskt, CV_CMP_GT);
	cvAnd(Imask,Imaskt,Imask);

	cvReleaseImage(&blue);cvReleaseImage(&green);cvReleaseImage(&red);
}

void findLines(IplImage *frame, IplImage *Imask, int isRight){
	IplImage *scratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);
	IplImage *blue = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *green = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *red = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	cvCvtScale(frame, scratch, 1, 0);
	cvSplit(scratch,blue,green,red,0);

	cvInRangeS(blue, cvScalar(max_idx_blue[isRight]-80), cvScalar(max_idx_blue[isRight]+80), Imask);
	/*cvShowImage("display", Imask);
	cout<<"blue"<<endl;
	cvWaitKey();*/
	cvInRangeS(green, cvScalar(max_idx_green[isRight]-90), cvScalar(max_idx_green[isRight]+90), Imaskt);
	/*cvShowImage("display", Imaskt);
	cout<<"green"<<endl;
	cvWaitKey();*/
	cvOr(Imask,Imaskt,Imask);
	cvInRangeS(red, cvScalar(max_idx_red[isRight]-5), cvScalar(max_idx_red[isRight]+5), Imaskt);
	/*cvShowImage("display", Imaskt);
	cout<<"red"<<endl;
	cvWaitKey();*/
	cvOr(Imask,Imaskt,Imask);
	//cvNot(Imask, Imask);
	/*cvShowImage("display", Imask);
	cvWaitKey();*/

	cvReleaseImage(&blue);cvReleaseImage(&green);cvReleaseImage(&red);
}

int main(int argc,char **argv){

	cvNamedWindow("gray",WINDOW_AUTOSIZE);
	namedWindow("display",WINDOW_AUTOSIZE);
	namedWindow("displayRight", WINDOW_AUTOSIZE);
	namedWindow("bird",WINDOW_AUTOSIZE);
	namedWindow("bird2",WINDOW_AUTOSIZE);
	CvCapture *capture = cvCreateFileCapture(argv[1]);
	CvCapture *captureRight = cvCreateFileCapture(argv[2]);
	frame=cvQueryFrame(capture);
	frameRight=cvQueryFrame(captureRight);

	AllocImages(frame);

	cvResize(frame, Ismall);
	cvShowImage("display",Ismall);
	cvResize(frameRight, IsmallRight);
	cvShowImage("displayRight", IsmallRight);
	showHist(frame,0);
	showHist(frameRight,1);

	setMouseCallback("display",cvMouseCallback);
	setMouseCallback("displayRight",cvMouseCallback);
	while(cnt<8)cvWaitKey(0);//click to set obj points

	imgPts[0].x=szBird.width/2; imgPts[0].y=szBird.height/2;
	imgPts[1].x=szBird.width; imgPts[1].y=szBird.height/2;
	imgPts[2].x=szBird.width/2; imgPts[2].y=szBird.height;
	imgPts[3].x=szBird.width; imgPts[3].y=szBird.height;
	
	cvGetPerspectiveTransform(imgPts, objPts, H);
	cvGetPerspectiveTransform(imgPts, objPtsRight, H2);
	cvInvert(H, H_inv);
	cvGEMM(H2, H_inv, 1.0, NULL, 0.0, H_r2l);

	cvWarpPerspective(Ismall, birdsImg, H, CV_INTER_LINEAR|
			CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);
	cvShowImage("bird",birdsImg);
	
	while(1){
		frame=cvQueryFrame(capture);
		frameRight=cvQueryFrame(captureRight);
		if(!frame||!frameRight)break;
		cvResize(frame, Ismall);
		cvWarpPerspective(Ismall, birdsImg, H, CV_INTER_LINEAR|
			CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);

		findGround(frame, Imask,0);
		cvCvtScale(Imask,ImaskPlayers,1,0);
		find_connected_components(Imask);
		cvNot(ImaskPlayers, ImaskPlayers);
		cvAnd(Imask, ImaskPlayers, ImaskPlayers);
		//find_connected_components(ImaskPlayers, 0);
		find_connected_components(ImaskPlayers, 0, 0, 60, &playerCount, playerRect, playerCenter);
		
		cvZero(birdsImg);
		for(int i=0;i<playerCount;++i){
			CvPoint pt = cvPoint(playerRect[i].x+playerRect[i].width/2, playerRect[i].y+playerRect[i].height);
			pt.x=pt.x/IMAGE_SCALE;
			pt.y=pt.y/IMAGE_SCALE;
			//pt = transformPoint(pt, H);
			//cvCircle(birdsImg, pt, 5, CVX_WHITE , CV_FILLED);
			cvRectangle(ImaskPlayers, cvPoint(playerRect[i].x,playerRect[i].y),
				cvPoint(playerRect[i].x+playerRect[i].width,playerRect[i].y+playerRect[i].height), CVX_WHITE);
		}
		playerCount=30;
		//cvShowImage("display", frame);
		cvShowImage("display", ImaskPlayers);
		/*cvResize(ImaskPlayers, ImaskSmall);
		cvWarpPerspective(ImaskSmall, ImaskBird, H, CV_INTER_LINEAR|
			CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);*/
		cvShowImage("bird", birdsImg);

		findGround(frameRight, Imask,0);
		cvCvtScale(Imask, ImaskPlayers, 1, 0);
		find_connected_components(Imask);
		cvNot(ImaskPlayers, ImaskPlayers);
		cvAnd(Imask, ImaskPlayers, ImaskPlayers);
		find_connected_components(ImaskPlayers,0);
		cvResize(ImaskPlayers, ImaskSmall);
		cvWarpPerspective(ImaskSmall, ImaskBirdt, H2, CV_INTER_LINEAR|
			CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);
		cvShowImage("bird2", ImaskBirdt);
		cvAnd(ImaskBird, ImaskBirdt, ImaskBird);

		char c = cvWaitKey(33);
		if(c==27)break;
	}
	DeallocateImages();
	cvReleaseCapture(&capture);
	cvDestroyWindow("display");
	cvDestroyWindow("displayRight");
	cvDestroyWindow("bird");
	cvDestroyWindow("bird2");
	return 0;
}
