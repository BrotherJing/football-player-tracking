#include<cstdio>
#include<cmath>
#include<opencv2/opencv.hpp>
#include<highgui.h>
#include "package_bgs/FrameDifferenceBGS.h"
#include "package_bgs/AdaptiveBackgroundLearning.h"
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
CvPoint ptsOnLine[4];
bool foundPerspective=false;

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

void find_connected_components(IplImage *mask, int find_ground=1, int poly1_hull0=1, float perimScale=60, int *num=NULL, CvRect *bbs=NULL, CvPoint *centers=NULL, int find_lines=0){
	static CvMemStorage *mem_storage=NULL;
	static CvSeq *contours=NULL;
	if(!find_lines){
		if(!find_ground){
			cvMorphologyEx(mask,mask,0,0,CV_MOP_OPEN,CVCLOSE_ITR_SMALL);//open
			cvMorphologyEx(mask,mask,0,0,CV_MOP_CLOSE,CVCLOSE_ITR);//close
			//cvErode(mask, mask, 0, 1);
			//cvDilate(mask ,mask, 0, 1);
		}else{
			cvMorphologyEx(mask,mask,0,0,CV_MOP_CLOSE,CVCLOSE_ITR);//close
		}
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
		if(!find_ground&&!find_lines){
			double tmparea=fabs(cvContourArea(c));
			double len = cvContourPerimeter(c);
			CvRect aRect = cvBoundingRect(c, 0);
			if(tmparea<100){//area too small
				cvSubstituteContour(scanner, NULL);
				continue;
			}
			if((aRect.y*1.0/mask->height*60+10)>aRect.height){
				cvSubstituteContour(scanner, NULL);
				continue;
			}
			if ((aRect.width*1.0/aRect.height)>1.5)//too fat
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
			continue;
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
	//int hist_height=256;
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
	cout<<max_idx_red[isRight]<<' '<<max_idx_green[isRight]<<' '<<max_idx_blue[isRight]<<endl;
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

#define ZERO 1e-8
#define DIS(a,b) sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y))
#define SGN(x) (fabs(x)<ZERO?0:(x>0?1:-1))
#define CROSS(a,b,c) ((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x))
#define CMP(a,b) (a.x<b.x||(SGN(a.x-b.x)==0&&a.y<b.y))
int hull_size=0;
inline void ipush(CvPoint *S, CvPoint pt){S[hull_size++]=pt;}
inline CvPoint ipop(CvPoint *S){return S[--hull_size];}
inline void iswap(CvPoint *p, int x, int y){
	CvPoint pt=p[x];
	p[x]=p[y];
	p[y]=pt;
}
inline bool icompare(CvPoint a,CvPoint b,CvPoint c){
	int tmp=SGN(CROSS(a,b,c));
	if(tmp!=0)return tmp>0;
	else return DIS(a,b)<DIS(a,c);
}
void isort(CvPoint *p, int l,int r){
	CvPoint tmp = p[(l + r) / 2];  
	int i = l;  
	int j = r;  
	do{  
		while(icompare(p[0],p[i],tmp))i++;  
		while(icompare(p[0],tmp,p[j]))j--;  
		if(i <= j)  {  
			iswap(p,i,j);  
			i++;  
			j--;  
		}  
	}while(i <=j);  
	if(i < r)isort(p,i,r);  
	if(j > l)isort(p,l,j); 
}
int findHull2(IplImage *Imask, CvPoint* pts, int cnt, CvPoint *hull_p){
	int min=-1;
	hull_size=0;
	for(int j=0;j<cnt;++j){
		if(min==-1||CMP(pts[j],pts[min]))
			min=j;
	}
	if(min!=0)iswap(pts,0,min);
	isort(pts,1,cnt-1);
	ipush(hull_p,pts[0]);
	ipush(hull_p,pts[1]);
	ipush(hull_p,pts[2]);

	/*cout<<"after sort"<<endl;
	for(int i=0;i<cnt;++i){
		cvCircle(Imask, pts[i], 5, CVX_WHITE , CV_FILLED);
		cout<<"("<<pts[i].x<<","<<pts[i].y<<")"<<endl;
		cvShowImage("display",Imask);
		cvWaitKey();
	}*/

	/*cvLine(Imask,pts[0],pts[1],cvScalar(255,0,255));//为了看清运行过程而加的  
	cvLine(Imask,pts[1],pts[2],cvScalar(255,0,255));//为了看清运行过程而加的  
	cvShowImage("display",Imask);  */

	for(int i=3;i<cnt;++i){
		while(true){
			float k1 = ((hull_p[hull_size-1].y-hull_p[hull_size-2].y)*1.0f/(hull_p[hull_size-1].x-hull_p[hull_size-2].x));
			float k2 = ((pts[i].y-hull_p[hull_size-2].y)*1.0f/(pts[i].x-hull_p[hull_size-2].x));
			if(CROSS(hull_p[hull_size-2], hull_p[hull_size-1],pts[i])<0||fabs(atan(k1)-atan(k2))<CV_PI/60){
				/*cvLine(Imask,hull_p[hull_size - 2],hull_p[hull_size-1],cvScalar(0,0,0));//为了看清运行过程而加的  
				cvShowImage("display",Imask);*/
				ipop(hull_p);
			}else{
				break;
			}
		}
		/*cvLine(Imask,hull_p[hull_size - 1],pts[i],cvScalar(255,0,255));//为了看清运行过程而加的  
		cvShowImage("display",Imask);  */

		ipush(hull_p,pts[i]);
	}

	//connect the last node to the first
	while(true){
		float k1 = ((hull_p[hull_size-1].y-hull_p[hull_size-2].y)*1.0f/(hull_p[hull_size-1].x-hull_p[hull_size-2].x));
		float k2 = ((pts[0].y-hull_p[hull_size-2].y)*1.0f/(pts[0].x-hull_p[hull_size-2].x));
		if(CROSS(hull_p[hull_size-2], hull_p[hull_size-1],pts[0])<0||fabs(atan(k1)-atan(k2))<CV_PI/60){
			/*cvLine(Imask,hull_p[hull_size - 2],hull_p[hull_size-1],cvScalar(0,0,0));//为了看清运行过程而加的  
			cvShowImage("display",Imask);*/
			ipop(hull_p);
		}else{
			break;
		}
	}

	return hull_size;
}

bool findLines(IplImage *frame, IplImage *ImaskGround, IplImage *Imask, int isRight, CvPoint* result=NULL){

	static CvMemStorage *storage;
	if(storage==NULL){
		storage = cvCreateMemStorage(0);
	}else{
		cvClearMemStorage(storage);
	}
	CvSeq *lineseq=0;

	IplImage *scratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);
	IplImage *blue = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *green = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	IplImage *red = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	cvCvtScale(frame, scratch, 1, 0);
	cvSplit(scratch,blue,green,red,0);
	
	cvInRangeS(blue, cvScalar(0), cvScalar(max_idx_blue[isRight]+40), Imask);
	cvInRangeS(green, cvScalar(0), cvScalar(max_idx_green[isRight]+85), Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvInRangeS(red, cvScalar(0), cvScalar(max_idx_red[isRight]+5), Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvNot(Imask, Imask);
	cvAnd(Imask, ImaskGround, Imask);

	//cvSmooth(Imask,Imask,CV_MEDIAN,3,3);
	//find_connected_components(Imask,0, 0, 200, NULL, NULL, NULL,1);

	cvCanny(Imask,Imask,150,100);

	//find lines
	if(result==NULL)
		lineseq=cvHoughLines2(Imask,storage,CV_HOUGH_PROBABILISTIC,1,CV_PI/180,50,50,10);
	else
		lineseq=cvHoughLines2(Imask,storage,CV_HOUGH_PROBABILISTIC,1,CV_PI/180,100,200,50);

	//remove overlapping lines.
	cvZero(Imask);
	CvPoint **lines = new CvPoint*[lineseq->total];
	float **k_b = new float*[lineseq->total];
	for(int i=0;i<lineseq->total;++i){
		lines[i]=new CvPoint[2];
		k_b[i]=new float[2];
	}
	int total=0;
	bool found=false;
	for(int i=0;i<lineseq->total;++i){
		found=false;
		CvPoint* line = (CvPoint*)cvGetSeqElem(lineseq,i);
		float k2 = (line[0].y-line[1].y)*1.0/(line[0].x-line[1].x);
		float b2 = line[0].y-line[0].x*k2;
		for(int j=0;j<total;++j){
			CvPoint *prevLine = lines[j];
			float k1 = (prevLine[0].y-prevLine[1].y)*1.0/(prevLine[0].x-prevLine[1].x);
			float b1 = prevLine[0].y-prevLine[0].x*k1;
			if(fabs(atan(k1)-atan(k2))<5e-1&&fabs(b1-b2)<30){//too close to each other
				float len1 = (prevLine[0].y-prevLine[1].y)*(prevLine[0].y-prevLine[1].y)+(prevLine[0].x-prevLine[1].x)*(prevLine[0].x-prevLine[1].x);
				float len2 = (line[0].y-line[1].y)*(line[0].y-line[1].y)+(line[0].x-line[1].x)*(line[0].x-line[1].x);
				if(len1<len2){
					prevLine[0]=line[0];
					prevLine[1]=line[1];
					k_b[j][0]=k2;
					k_b[j][1]=b2;
				}
				found=true;
				break;
			}
		}
		if(!found){
			lines[total][0]=line[0];
			lines[total][1]=line[1];
			k_b[total][0]=k2;
			k_b[total][1]=b2;
			total++;
		}
		cvLine(Imask,line[0],line[1],CVX_WHITE,3,CV_AA,0);
	}
	if(result==NULL)return false;//only need lines, return.

	//find cross point
	CvSize img_size=cvGetSize(frame);
	CvPoint *cross=new CvPoint[total*(total-1)/2];
	int cross_cnt=0;
	for(int i=0;i<total-1;++i){
		for(int j=i+1;j<total;++j){
			if(fabs(atan(k_b[j][0])-atan(k_b[i][0]))<CV_PI/12)continue;
			int x=(int)(-(k_b[j][1]-k_b[i][1])/(k_b[j][0]-k_b[i][0]));
			int y=(int)(k_b[i][0]*x+k_b[i][1]);
			if(x>img_size.width*1.3||x<-img_size.width*0.3||
				y>img_size.height*1.3||y<-img_size.height*0.3){
				continue;
			}
			int len_to1= max((x-lines[i][0].x)*(x-lines[i][0].x)+(y-lines[i][0].y)*(y-lines[i][0].y),
					(x-lines[i][1].x)*(x-lines[i][1].x)+(y-lines[i][1].y)*(y-lines[i][1].y));
			int len_to2= max((x-lines[j][0].x)*(x-lines[j][0].x)+(y-lines[j][0].y)*(y-lines[j][0].y),
					(x-lines[j][1].x)*(x-lines[j][1].x)+(y-lines[j][1].y)*(y-lines[j][1].y));
			int len1 = (lines[i][1].x-lines[i][0].x)*(lines[i][1].x-lines[i][0].x)+(lines[i][1].y-lines[i][0].y)*(lines[i][1].y-lines[i][0].y);
			int len2 = (lines[j][1].x-lines[j][0].x)*(lines[j][1].x-lines[j][0].x)+(lines[j][1].y-lines[j][0].y)*(lines[j][1].y-lines[j][0].y);

			if(len_to1>len1*1.7&&len_to2>len2*1.7)continue;
			cross[cross_cnt].x=x;
			cross[cross_cnt].y=y;
			cross_cnt++;
		}
	}

	//find hull, and playground boundary
	cvZero(Imask);
	CvPoint *final=new CvPoint[cross_cnt];
	int hull_cnt = findHull2(Imask, cross,cross_cnt,final);
	if(hull_cnt!=4)return false;
	for(int i=0;i<hull_cnt;++i){
		cout<<"("<<final[i].x<<","<<final[i].y<<")"<<endl;
	}
	if(result!=NULL){
		result[0]=final[1];
		result[1]=final[2];
		result[2]=final[0];
		result[3]=final[3];
	}

	/*cvClearMemStorage(storage);
	lines=cvHoughLines2(Imask,storage,CV_HOUGH_PROBABILISTIC,1,CV_PI/180,200,200,50);
	cvZero(Imask);
	for(int i=0;i<lines->total;++i){
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
		cvLine(Imask,line[0],line[1],CVX_WHITE,1,CV_AA,0);
	}
	cvShowImage("display", Imask);*/

	cvReleaseImage(&blue);cvReleaseImage(&green);cvReleaseImage(&red);
	return true;
}

void getPerspectiveTransform(CvPoint *pts){

	imgPts[0].x=0; imgPts[0].y=szBird.height/2;
	imgPts[1].x=szBird.width; imgPts[1].y=szBird.height/2;
	imgPts[2].x=0; imgPts[2].y=szBird.height;
	imgPts[3].x=szBird.width; imgPts[3].y=szBird.height;
	
	for(int i=0;i<4;++i){
		objPts[i].x=(float)pts[i].x/2;
		objPts[i].y=(float)pts[i].y/2;
	}
	cvGetPerspectiveTransform(imgPts, objPts, H);
}

int main(int argc,char **argv){

	namedWindow("display",WINDOW_AUTOSIZE);
	namedWindow("displayRight", WINDOW_AUTOSIZE);
	namedWindow("lines",WINDOW_AUTOSIZE);
	namedWindow("bird",WINDOW_AUTOSIZE);
	namedWindow("bird2",WINDOW_AUTOSIZE);
	CvCapture *capture = cvCreateFileCapture(argv[1]);
	//CvCapture *captureRight = cvCreateFileCapture(argv[2]);
	frame=cvQueryFrame(capture);
	//frameRight=cvQueryFrame(captureRight);

	AllocImages(frame);

	cvResize(frame, Ismall);
	cvShowImage("display",Ismall);
	showHist(frame,0);
	/*cvResize(frameRight, IsmallRight);
	cvShowImage("displayRight", IsmallRight);
	showHist(frameRight,1);*/

	setMouseCallback("display",cvMouseCallback);
	//setMouseCallback("displayRight",cvMouseCallback);
	//while(cnt<4)cvWaitKey(0);//click to set obj points

	/*cvGetPerspectiveTransform(imgPts, objPtsRight, H2);
	cvInvert(H, H_inv);
	cvGEMM(H2, H_inv, 1.0, NULL, 0.0, H_r2l);

	cvWarpPerspective(Ismall, birdsImg, H, CV_INTER_LINEAR|
			CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);
	cvShowImage("bird",birdsImg);*/
	
	while(1){
		frame=cvQueryFrame(capture);
		//frameRight=cvQueryFrame(captureRight);
		if(!frame
			//||!frameRight
			)
			break;

		findGround(frame, Imask,0);
		cvCvtScale(Imask,ImaskPlayers,1,0);
		find_connected_components(Imask);

		cvResize(frame, Ismall);
		if(!foundPerspective){
			foundPerspective = findLines(frame, Imask, ImaskLines, 0, ptsOnLine);
			getPerspectiveTransform(ptsOnLine);
		}else{
			cvWarpPerspective(Ismall, birdsImg, H, CV_INTER_LINEAR|
				CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);
		}

		findLines(frame, Imask, ImaskLines, 0);
		cvNot(ImaskPlayers, ImaskPlayers);
		cvSub(ImaskPlayers, ImaskLines, ImaskPlayers);
		cvAnd(Imask,ImaskPlayers,ImaskPlayers);
		find_connected_components(ImaskPlayers, 0, 0, 60, &playerCount, playerRect, playerCenter);
		for(int i=0;i<playerCount;++i){
			CvPoint pt = cvPoint(playerRect[i].x+playerRect[i].width/2, playerRect[i].y+playerRect[i].height);
			pt.x=pt.x/IMAGE_SCALE;
			pt.y=pt.y/IMAGE_SCALE;
			pt = transformPoint(pt, H);
			cvCircle(birdsImg, pt, 5, CVX_WHITE , CV_FILLED);
			/*cvRectangle(ImaskPlayers, cvPoint(playerRect[i].x,playerRect[i].y),
				cvPoint(playerRect[i].x+playerRect[i].width,playerRect[i].y+playerRect[i].height), CVX_WHITE);*/
		}
		playerCount=30;
		cvShowImage("bird", birdsImg);

		char c = cvWaitKey(33);
		if(c==27)break;
	}
	DeallocateImages();
	cvReleaseCapture(&capture);
	cvDestroyWindow("display");
	cvDestroyWindow("displayRight");
	cvDestroyWindow("lines");
	cvDestroyWindow("bird");
	cvDestroyWindow("bird2");
	return 0;
}
