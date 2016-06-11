#include "header/camera.h"

#define cvQueryHistValue_1D( hist, idx0 ) \
    ((float)cvGetReal1D( (hist)->bins, (idx0)))

using namespace cv;
using namespace std;

IplImage *frame, *Ismall, *birdsImg, *Iground, *Itrace;
IplImage *frameRight, *IsmallRight;
IplImage *Iscratch, *Iscratch2;
IplImage *Imaskt, *Imask, *ImaskBird, *ImaskPlayers, *ImaskLines;
IplImage *ImaskSmall, *ImaskSmall2, *ImaskBirdt;
IplImage *scratch;
IplImage *blue, *green, *red;
CvSize sz, szBird;
int cnt=0;

// tracking
// Variables needed by callback func ...
vector<QcvCAMshiftTracker> camShiftTrackers;
bool selectObject = false;
cv::Point origin;
cv::Rect selection;
Mat hsv, hue;
int ch[] = {0, 0};

vector<Tracker> trackers;

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
bool foundGround = false;

//histogram
int max_idx_red[2], max_idx_blue[2], max_idx_green[2];

void AllocImages(IplImage *frame){

	sz=cvGetSize(frame);
	sz.width=sz.width/2;sz.height=sz.height/2;
	szBird = cvGetSize(frame);
	//szBird.width/=2;szBird.height/=2;
	szBird.width=WIDTH;szBird.height=HEIGHT;

	Ismall = cvCreateImage(sz,frame->depth,frame->nChannels);
	birdsImg=cvCreateImage(szBird,frame->depth,frame->nChannels);
	IsmallRight=cvCreateImage(sz,frame->depth,frame->nChannels);
	Itrace = cvCreateImage(sz, frame->depth,frame->nChannels);

	Iscratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_32F,3);
	Iscratch2 = cvCreateImage(cvGetSize(frame),IPL_DEPTH_32F,3);
	scratch = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);

	Iground = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	Imask = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	Imaskt = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	ImaskBird = cvCreateImage(szBird, IPL_DEPTH_8U, 1);
	ImaskPlayers = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	ImaskLines = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);

	ImaskSmall = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	ImaskSmall2 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	ImaskBirdt = cvCreateImage(szBird, IPL_DEPTH_8U, 1);

 	blue = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
 	green = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
 	red = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);

	cvZero(Iscratch);cvZero(Iscratch2);
	cvZero(Itrace);

}

void DeallocateImages(){
	cvReleaseImage(&Ismall);cvReleaseImage(&birdsImg);cvReleaseImage(&IsmallRight);
	cvReleaseImage(&Iscratch);cvReleaseImage(&Iscratch2);
	cvReleaseImage(&Imaskt);cvReleaseImage(&ImaskBird);cvReleaseImage(&ImaskPlayers);cvReleaseImage(&ImaskLines);
	cvReleaseImage(&ImaskSmall);cvReleaseImage(&ImaskSmall2);cvReleaseImage(&ImaskBirdt);

	cvReleaseImage(&Iground);
}

void splitFrame(IplImage *frame){
	cvCvtScale(frame, scratch, 1, 0);
	cvSplit(scratch,blue,green,red,0);
}

// Callback function used in labeler image show window ...
static void camShiftLabelerOnMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        // 这个是干什么的
        selection &= cv::Rect(0, 0, QcvCAMshiftTracker::getMainImage().cols, QcvCAMshiftTracker::getMainImage().rows);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN://鼠标向下
        origin = cv::Point(x,y);
        selection = cv::Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP://鼠标向上
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
        {
            // 加入一个新的 newTracker
            QcvCAMshiftTracker newTracker;//these two lines are important!!
            newTracker.setCurrentRect(selection);
            camShiftTrackers.push_back(newTracker);
        }
        break;
    }
}

void showHist(IplImage *frame, int isRight){
	splitFrame(frame);
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
}

void findGround(IplImage *frame, IplImage *Imask, int isRight){

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
}

CvPoint addLines[4][2];
int addLineCnt=0;
void callbackAddLines(int event, int x, int y, int flag, void *ustc){
	static bool isButtonFirstDown = true;
	static CvPoint startPoint;
	if(addLineCnt==4)return;
	if(event==CV_EVENT_LBUTTONDOWN){
		if(isButtonFirstDown){
			isButtonFirstDown=false;
			startPoint.x=x*IMAGE_SCALE;
			startPoint.y=y*IMAGE_SCALE;
		}else{
			isButtonFirstDown=true;
			CvPoint endPoint = cvPoint(x*IMAGE_SCALE, y*IMAGE_SCALE);
			addLines[addLineCnt][0]=startPoint;
			addLines[addLineCnt][1]=endPoint;
			cvLine(frame, startPoint,endPoint ,CVX_RED, 2);
			cvResize(frame,Ismall);
			cvShowImage("display",Ismall);
			addLineCnt++;
		}
	}
}

bool findLines(IplImage *frame, IplImage *ImaskGround, IplImage *Imask, int isRight, CvPoint* result=NULL){

	static CvMemStorage *storage;
	if(storage==NULL){
		storage = cvCreateMemStorage(0);
	}else{
		cvClearMemStorage(storage);
	}
	CvSeq *lineseq=0;

	cvInRangeS(blue, cvScalar(0), cvScalar(max_idx_blue[isRight]+40), Imask);
	cvInRangeS(green, cvScalar(0), cvScalar(max_idx_green[isRight]+85), Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvInRangeS(red, cvScalar(0), cvScalar(max_idx_red[isRight]+5), Imaskt);
	cvOr(Imask,Imaskt,Imask);
	cvNot(Imask, Imask);
	cvAnd(Imask, ImaskGround, Imask);
	/*cvShowImage("displayRight", Imask);
	cvWaitKey();*/

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
	CvPoint **lines = new CvPoint*[lineseq->total+4];
	float **k_b = new float*[lineseq->total+4];
	for(int i=0;i<lineseq->total+4;++i){
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
			if(result!=NULL)//only show lines when finding the points
				cvLine(frame,line[0],line[1],CVX_RED,1,CV_AA,0);
		}
		cvLine(Imask,line[0],line[1],CVX_WHITE,3,CV_AA,0);
	}
	if(result==NULL){
		for(int i=0;i<lineseq->total+4;i++){delete lines[i]; delete k_b[i];}
		return false;//only need lines, return.
	}

	//if there are no enough line, add some line manually
	cvResize(frame, Ismall);
	cvShowImage("display",Ismall);
	setMouseCallback("display",callbackAddLines);
	cvWaitKey();
	for(int i=0;i<addLineCnt;++i){
		found=false;
		CvPoint* line = addLines[i];
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
			cout<<"new line: ("<<line[0].x<<","<<line[0].y<<")->("<<line[1].x<<","<<line[1].y<<")"<<endl;
		}
	}
	addLineCnt=0;

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
	for(int i=0;i<cross_cnt;++i){
		cvCircle(frame, cross[i], 5, CVX_RED , CV_FILLED);
	}
	cvResize(frame, Ismall);
	cvShowImage("display",Ismall);
	cvWaitKey();

	//find hull, and playground boundary
	cvZero(Imask);
	CvPoint *final=new CvPoint[cross_cnt];
	int hull_cnt = findHull2(Imask, cross,cross_cnt,final);
	for(int i=0;i<hull_cnt;++i){
		cout<<"("<<final[i].x<<","<<final[i].y<<")"<<endl;
	}
	if(hull_cnt!=4)return false;

	//find points in order
	int min_x=final[0].x,max_x=final[0].x,min_y=final[0].y,max_y=final[0].y;
	int min_x_id, min_y_id, max_x_id, max_y_id;
	for(int i=0;i<hull_cnt;++i){
		if(final[i].x<=min_x){min_x=final[i].x;min_x_id=i;}
		if(final[i].x>=max_x){max_x=final[i].x;max_x_id=i;}
		if(final[i].y<=min_y){min_y=final[i].y;min_y_id=i;}
		if(final[i].y>=max_y){max_y=final[i].y;max_y_id=i;}
	}
	int upper_left=0;
	if(min_x_id==min_y_id){upper_left=min_x_id;}// found upper left
	else if(max_x_id==max_y_id){upper_left=(max_x_id+2)%4;}//found lower right
	else if(min_x_id==max_y_id){upper_left=(min_x_id+1)%4;}//found lower left
	else if(max_x_id==min_y_id){upper_left=(max_x_id+3)%4;}//found upper right
	else{
		cout<<final[max_y_id].x<<' '<<img_size.width<<endl;
		if(final[max_y_id].x>img_size.width/2)upper_left=min_y_id;//right camera
		else upper_left=min_x_id;
	}
	if(result!=NULL){
		result[0]=final[upper_left];
		result[1]=final[(upper_left+1)%4];
		result[2]=final[(upper_left+3)%4];
		result[3]=final[(upper_left+2)%4];
	}

	for(int i=0;i<lineseq->total+4;i++){delete lines[i]; delete k_b[i];}
	delete [] lines; delete [] k_b; delete [] cross; delete [] final;
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

	bool paused=false;
	cv::Mat m_frame;

	namedWindow("display",WINDOW_AUTOSIZE);
	namedWindow("displayRight", WINDOW_AUTOSIZE);
	namedWindow("lines",WINDOW_AUTOSIZE);
	namedWindow("bird",WINDOW_AUTOSIZE);
	namedWindow("bird2",WINDOW_AUTOSIZE);
	CvCapture *capture = cvCreateFileCapture(argv[1]);
	frame=cvQueryFrame(capture);

	AllocImages(frame);

	cvResize(frame, Ismall);
	cvShowImage("display",Ismall);
	showHist(frame,0);
	
	while(1){
		if(!paused){
			frame=cvQueryFrame(capture);
			splitFrame(frame);
			//frameRight=cvQueryFrame(captureRight);
			if(!frame
				//||!frameRight
				)
				break;

			findGround(frame, Imask,0);// exclude player and lines
			if(!foundGround){
				foundGround = true;
				cvCvtScale(Imask, Iground, 1,0);
				find_connected_components(Iground);
			}

			if(!foundPerspective){
				foundPerspective = findLines(frame, Iground, ImaskLines, 0, ptsOnLine);
				if(foundPerspective){
					getPerspectiveTransform(ptsOnLine);
					//setMouseCallback( "display", camShiftLabelerOnMouse, 0 );
				}
				continue;
			}

			cvResize(frame, Ismall);
			cvWarpPerspective(Ismall, birdsImg, H, CV_INTER_LINEAR|
				CV_WARP_INVERSE_MAP|CV_WARP_FILL_OUTLIERS);

			findLines(frame, Iground, ImaskLines, 0);
			cvNot(Imask, Imask);
			cvSub(Imask, ImaskLines, Imask);
			cvAnd(Iground,Imask,Imask);
			find_connected_components(Imask, 0, 0, 60, &playerCount, playerRect, playerCenter);
			//trackPlayers(trackers, playerRect, playerCenter, playerCount);
			//cout<<trackers.size()<<endl;
			find_player_teams(green, Imask, playerRect, NULL, playerCount);
			for(int i=0;i<playerCount;++i){
				/*CvPoint pt = cvPoint(playerRect[i].x+playerRect[i].width/2, playerRect[i].y+playerRect[i].height);
				pt.x=pt.x/IMAGE_SCALE;
				pt.y=pt.y/IMAGE_SCALE;*/
				//pt = transformPoint(pt, H);
				//cvCircle(birdsImg, pt, 5, CVX_WHITE , CV_FILLED);
				/*cvRectangle(Ismall, cvPoint(playerRect[i].x/IMAGE_SCALE,playerRect[i].y/IMAGE_SCALE),
					cvPoint(playerRect[i].x/IMAGE_SCALE+playerRect[i].width/IMAGE_SCALE,playerRect[i].y/IMAGE_SCALE+playerRect[i].height/IMAGE_SCALE),
					CVX_WHITE);*/
			}
			/*for(vector<Tracker>::iterator it = trackers.begin();it != trackers.end();++it){
				cvRectangle(Ismall, cvPoint(it->bbox.x/IMAGE_SCALE, it->bbox.y/IMAGE_SCALE),
					cvPoint((it->bbox.x+it->bbox.width)/IMAGE_SCALE, (it->bbox.y+it->bbox.height)/IMAGE_SCALE),
					CVX_WHITE);
				if(it->no_found_cnt==0)
					cvLine(Itrace,cvPoint(it->last.x/IMAGE_SCALE, it->last.y/IMAGE_SCALE),cvPoint(it->center.x/IMAGE_SCALE, it->center.y/IMAGE_SCALE), CVX_WHITE,1);
			}
			cvAdd(Ismall, Itrace, Ismall);*/
			playerCount=30;
			cvShowImage("bird", green);
			//cvWaitKey(0);

			/*m_frame=Mat(Ismall);
			QcvCAMshiftTracker::setMainImage(m_frame);
			for(int i=0; i<camShiftTrackers.size(); i++){
				if(camShiftTrackers[i].trackCurrentRect().boundingRect().area() <= 1)
					continue;
				cv::ellipse(m_frame, camShiftTrackers[i].trackCurrentRect(), cv::Scalar(0, 255, 0), 2, CV_AA);
				cv::rectangle(m_frame, camShiftTrackers[i].trackCurrentRect().boundingRect(), cv::Scalar(0, 255, 0), 2, CV_AA);
			}*/
		}

		/*if( selectObject && selection.width > 0 && selection.height > 0 ){
			cv::Mat roi(m_frame, selection);
			cv::bitwise_not(roi, roi);
 		}*/

		cvShowImage("display", Ismall);

		char c = cvWaitKey(33);
		if(c==27)break;
		switch(c){
		case 'p':paused = !paused;break;
		//case 'k':camShiftTrackers.clear();break;
		default:break;
		}
	}
	//camShiftTrackers.clear();
	DeallocateImages();
	cvReleaseCapture(&capture);
	cvDestroyWindow("display");
	cvDestroyWindow("displayRight");
	cvDestroyWindow("lines");
	cvDestroyWindow("bird");
	cvDestroyWindow("bird2");
	return 0;
}
