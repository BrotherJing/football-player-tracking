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

IplImage *frame, *Ismall, *birdsImg;
IplImage *frameRight, *IsmallRight;
CvSize sz, szBird;
CvPoint2D32f objPts[4], imgPts[4], objPtsRight[4];
int cnt=0;

IplImage *IavgF, *IdiffF, *IprevF, *IhiF, *IlowF;
IplImage *Iscratch, *Iscratch2;
IplImage *Iblue, *Igreen, *Ired;
IplImage *Ilow1, *Ilow2, *Ilow3;
IplImage *Ihi1, *Ihi2, *Ihi3;
IplImage *Imaskt, *Imask, *ImaskBird, *ImaskPlayers, *ImaskLines;
float Icount;

//histogram
int max_idx_red=0, max_idx_blue=0, max_idx_green=0;
const int MARGIN = 10;
const int MARGIN_BLUE = 35;
const int MARGIN_GREEN = 40;
const int MARGIN_WHITE = 10;

void AllocImages(IplImage *frame){

	sz=cvGetSize(frame);
	sz.width=sz.width/2;sz.height=sz.height/2;
	szBird = cvGetSize(frame);
	szBird.width=WIDTH;szBird.height=HEIGHT;

	Ismall = cvCreateImage(sz,frame->depth,frame->nChannels);
	birdsImg=cvCreateImage(szBird,frame->depth,frame->nChannels);
	IsmallRight=cvCreateImage(sz,frame->depth,frame->nChannels);

	//CvSize sz=cvGetSize(i);
	IavgF = cvCreateImage(sz,IPL_DEPTH_32F,3);
	IdiffF = cvCreateImage(sz,IPL_DEPTH_32F,3);
	IprevF = cvCreateImage(sz,IPL_DEPTH_32F,3);
	IhiF = cvCreateImage(sz,IPL_DEPTH_32F,3);
	IlowF = cvCreateImage(sz,IPL_DEPTH_32F,3);

	Ilow1 = cvCreateImage(sz,IPL_DEPTH_32F,1);
	Ilow2 = cvCreateImage(sz,IPL_DEPTH_32F,1);
	Ilow3 = cvCreateImage(sz,IPL_DEPTH_32F,1);
	Ihi1 = cvCreateImage(sz,IPL_DEPTH_32F,1);
	Ihi2 = cvCreateImage(sz,IPL_DEPTH_32F,1);
	Ihi3 = cvCreateImage(sz,IPL_DEPTH_32F,1);

	cvZero(IavgF);cvZero(IdiffF);cvZero(IprevF);cvZero(IhiF);cvZero(IlowF);
	Icount=0.00001;
	Iscratch = cvCreateImage(sz,IPL_DEPTH_32F,3);
	Iscratch2 = cvCreateImage(sz,IPL_DEPTH_32F,3);
	Iblue = cvCreateImage(sz,IPL_DEPTH_32F,1);
	Igreen = cvCreateImage(sz,IPL_DEPTH_32F,1);
	Ired = cvCreateImage(sz,IPL_DEPTH_32F,1);
	Imaskt = cvCreateImage(sz,IPL_DEPTH_8U,1);

	Imask = cvCreateImage(sz,IPL_DEPTH_8U,1);
	ImaskBird = cvCreateImage(szBird, IPL_DEPTH_8U, 1);
	ImaskPlayers = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	ImaskLines = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	cvZero(Iscratch);cvZero(Iscratch2);
}

void DeallocateImages(){
	cvReleaseImage(&IavgF);
	cvReleaseImage(&IdiffF);
	cvReleaseImage(&IprevF);
	cvReleaseImage(&IhiF);
	cvReleaseImage(&IlowF);
	cvReleaseImage(&Ilow1);cvReleaseImage(&Ilow2);cvReleaseImage(&Ilow3);
	cvReleaseImage(&Ihi1);cvReleaseImage(&Ihi2);cvReleaseImage(&Ihi3);
	cvReleaseImage(&Iscratch);cvReleaseImage(&Iscratch2);
	cvReleaseImage(&Iblue);cvReleaseImage(&Igreen);cvReleaseImage(&Ired);
	cvReleaseImage(&Imaskt);
}

//accumulate avg and diff
void accumulateBackground(IplImage *i){
	static int first=1;
	cvCvtScale(i,Iscratch,1,0);
	if(!first){
		cvAcc(Iscratch,IavgF);
		cvAbsDiff(Iscratch,IprevF,Iscratch2);
		cvAcc(Iscratch2,IdiffF);
		Icount+=1.0;
	}
	first=0;
	cvCopy(Iscratch,IprevF);
}

void setHighThreshold(float scale){
	cvConvertScale(IdiffF,Iscratch,scale);
	cvAdd(Iscratch,IavgF,IhiF);
	cvSplit(IhiF,Ihi1,Ihi2,Ihi3,0);
}

void setLowThreshold(float scale){
	cvConvertScale(IdiffF,Iscratch,scale);
	cvSub(IavgF,Iscratch,IlowF);
	cvSplit(IlowF,Ilow1,Ilow2,Ilow3,0);
}

//divide accumulated value to get avg and diff
void createModelFromStats(){
	cvConvertScale(IavgF,IavgF,(double)(1.0/Icount));
	cvConvertScale(IdiffF,IdiffF,(double)(1.0/Icount));
	cvAddS(IdiffF,cvScalar(1.0,1.0,1.0),IdiffF);
	setHighThreshold(2.5);
	setLowThreshold(2.5);
}

//generate mask for foreground
void backgroudDiff(IplImage *i, IplImage *Imask){
	cvCvtScale(i,Iscratch,1,0);
	cvSplit(Iscratch,Iblue,Igreen,Ired,0);
	
	cvInRange(Iblue,Ilow1,Ihi1,Imask);//in this range? background
	cvInRange(Igreen,Ilow2,Ihi2,Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvInRange(Ired,Ilow3,Ihi3,Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvSubRS(Imask,cvScalar(255,255,255),Imask);
}

void find_connected_components(IplImage *mask, int find_ground=1, int poly1_hull0=1, float perimScale=60, int *num=NULL, CvRect *bbs=NULL, CvPoint *centers=NULL){
	static CvMemStorage *mem_storage=NULL;
	static CvSeq *contours=NULL;
	if(!find_ground){
		cvMorphologyEx(mask,mask,0,0,CV_MOP_OPEN,CVCLOSE_ITR_SMALL);//open
		cvMorphologyEx(mask,mask,0,0,CV_MOP_CLOSE,CVCLOSE_ITR_SMALL);//close
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
			if(tmparea<80){
				cvSubstituteContour(scanner, NULL);
				continue;
			}
			if ((aRect.width/aRect.height)>2)
			{
				cvSubstituteContour(scanner, NULL); //删除宽高比例小于设定值的轮廓
				continue;
			}
			if(len/tmparea>4){
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
			cvDrawContours(mask, c, CVX_WHITE, CVX_WHITE, -1, CV_FILLED, 8);//draw contours on mask
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

void showHist(IplImage *frame){
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
	CvHistogram *blue_hist=cvCreateHist(1,&hist_size,CV_HIST_ARRAY);
	CvHistogram *green_hist=cvCreateHist(1,&hist_size,CV_HIST_ARRAY);
	CvHistogram *red_hist=cvCreateHist(1,&hist_size,CV_HIST_ARRAY);
	cvCalcHist(&blue, blue_hist, 0, 0);
	cvCalcHist(&green, green_hist, 0, 0);
	cvCalcHist(&red, red_hist, 0, 0);
	cvNormalizeHist(blue_hist, 1.0);cvNormalizeHist(green_hist, 1.0);cvNormalizeHist(red_hist, 1.0);

	IplImage *hist_img = cvCreateImage(cvSize(hist_size*3,hist_height),IPL_DEPTH_8U,3);
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
	cout<<max_idx_red<<' '<<max_idx_green<<' '<<max_idx_blue<<endl;
	cvShowImage("gray",hist_img);
	cvReleaseImage(&blue);cvReleaseImage(&green);cvReleaseImage(&red);
	cvReleaseImage(&hist_img);
}

void findGround(IplImage *frame, IplImage *Imask){

	IplImage *scratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);
	IplImage *blue = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *green = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *red = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	cvCvtScale(frame, scratch, 1, 0);
	cvSplit(scratch,blue,green,red,0);

	cvInRangeS(blue, cvScalar(max_idx_blue-MARGIN_BLUE), cvScalar(max_idx_blue+MARGIN_BLUE), Imask);
	cvInRangeS(green, cvScalar(max_idx_green-MARGIN_GREEN), cvScalar(max_idx_green+MARGIN_GREEN), Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvInRangeS(red, cvScalar(max_idx_red-MARGIN), cvScalar(max_idx_red+MARGIN), Imaskt);
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

void findLines(IplImage *frame, IplImage *Imask){
	IplImage *scratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);
	IplImage *blue = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *green = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *red = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	cvCvtScale(frame, scratch, 1, 0);
	cvSplit(scratch,blue,green,red,0);

	cvInRangeS(blue, cvScalar(255-MARGIN_WHITE), cvScalar(255), Imask);
	cvInRangeS(green, cvScalar(255-MARGIN_GREEN), cvScalar(255), Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvInRangeS(red, cvScalar(255-MARGIN_WHITE), cvScalar(255), Imaskt);
	cvOr(Imask,Imaskt,Imask);

	cvReleaseImage(&blue);cvReleaseImage(&green);cvReleaseImage(&red);
}

int main(int argc,char **argv){

	float train_size = 1;
	int has_model = 0;

	cvNamedWindow("gray",WINDOW_AUTOSIZE);
	namedWindow("display",WINDOW_AUTOSIZE);
	namedWindow("displayRight", WINDOW_AUTOSIZE);
	namedWindow("bird",WINDOW_AUTOSIZE);
	CvCapture *capture = cvCreateFileCapture(argv[1]);
	CvCapture *captureRight = cvCreateFileCapture(argv[2]);
	frame=cvQueryFrame(capture);
	frameRight=cvQueryFrame(captureRight);

	AllocImages(frame);

	cvResize(frame, Ismall);
	cvShowImage("display",Ismall);
	cvResize(frameRight, IsmallRight);
	cvShowImage("displayRight", IsmallRight);
	showHist(frame);
	setMouseCallback("display",cvMouseCallback);
	setMouseCallback("displayRight",cvMouseCallback);
	while(cnt<8)cvWaitKey(0);//click to set obj points

	CvMat *H = cvCreateMat(3,3,CV_32F), *H_inv = cvCreateMat(3,3,CV_32F);
	CvMat *H2 = cvCreateMat(3,3,CV_32F), *H_r2l = cvCreateMat(3,3,CV_32F);
	imgPts[0].x=szBird.width/2; imgPts[0].y=szBird.height/2;
	imgPts[1].x=szBird.width; imgPts[1].y=szBird.height/2;
	imgPts[2].x=szBird.width/2; imgPts[2].y=szBird.height;
	imgPts[3].x=szBird.width; imgPts[3].y=szBird.height;
	
	/*imgPts[0].x=LEFT; imgPts[0].y=TOP;
	imgPts[1].x=RIGHT; imgPts[1].y=TOP;
	imgPts[2].x=LEFT; imgPts[2].y=BOTTOM;
	imgPts[3].x=RIGHT; imgPts[3].y=BOTTOM;*/

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
		cvResize(frameRight, IsmallRight);
		cvWarpPerspective(IsmallRight, IsmallRight, H_r2l, CV_INTER_LINEAR|
			CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);

		if(!has_model&&Icount<train_size){
			accumulateBackground(Ismall);
		}else{
			if(!has_model){
				has_model=1;
				createModelFromStats();
			}
		}
		if(has_model){
			//backgroudDiff(Ismall, Imask);
			//find_connected_components(Imask);
			/*cvWarpPerspective(Imask, ImaskBird, H, CV_INTER_LINEAR|
				CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);
			cvShowImage("display",ImaskBird);*/
			findGround(Ismall, Imask);
			//cvShowImage("display", Imask);
			//findLines(Ismall, ImaskLines);
			cvCvtScale(Imask,ImaskPlayers,1,0);
			find_connected_components(Imask);
			cvNot(ImaskPlayers, ImaskPlayers);
			cvAnd(Imask, ImaskPlayers, ImaskPlayers);
			find_connected_components(ImaskPlayers,0);
			cvShowImage("display", ImaskPlayers);
			cvShowImage("displayRight", IsmallRight);
		}

		cvShowImage("bird",birdsImg);
		char c = cvWaitKey(33);
		if(c==27)break;
	}
	DeallocateImages();
	cvReleaseCapture(&capture);
	cvDestroyWindow("display");
	cvDestroyWindow("displayRight");
	cvDestroyWindow("bird");
	return 0;
}
