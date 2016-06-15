#include "header/camera.h"

using namespace std;
using namespace cv;

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

//graham algorithm
int findHull2(IplImage *Imask, CvPoint* pts, int cnt, CvPoint *hull_p){
	int min=-1;
	hull_size=0;
	for(int j=0;j<cnt;++j){
		if(min==-1||CMP(pts[j],pts[min]))
			min=j;//find point with mininum x
	}
	if(min!=0)iswap(pts,0,min);
	isort(pts,1,cnt-1);
	ipush(hull_p,pts[0]);
	ipush(hull_p,pts[1]);
	//ipush(hull_p,pts[2]);

	for(int i=2;i<cnt;++i){
		while(hull_size>=2){
			float k1 = ((hull_p[hull_size-1].y-hull_p[hull_size-2].y)*1.0f/(hull_p[hull_size-1].x-hull_p[hull_size-2].x));
			float k2 = ((pts[i].y-hull_p[hull_size-1].y)*1.0f/(pts[i].x-hull_p[hull_size-1].x));
			float x1= hull_p[hull_size-1].x-hull_p[hull_size-2].x;
			float y1 = hull_p[hull_size-1].y-hull_p[hull_size-2].y;
			float x2= pts[i].x-hull_p[hull_size-1].x;
			float y2 = pts[i].y-hull_p[hull_size-1].y;
			cout<<atan(k1)<<' '<<atan(k2)<<endl;
			if(CROSS(hull_p[hull_size-2], hull_p[hull_size-1],pts[i])<0||
				(fabs(atan(k1)-atan(k2))<CV_PI/45&&(x1*x2+y1*y2>0))){// >180, or near 180 and same orientation
				//cout<<"pop "<<hull_p[hull_size-1].x<<' '<<hull_p[hull_size-1].y<<endl;
				ipop(hull_p);
			}else{
				break;
			}
		}
		ipush(hull_p,pts[i]);
		//cout<<"push "<<pts[i].x<<' '<<pts[i].y<<endl;
	}

	//connect the last node to the first
	while(true){
		float k1 = ((hull_p[hull_size-1].y-hull_p[hull_size-2].y)*1.0f/(hull_p[hull_size-1].x-hull_p[hull_size-2].x));
		float k2 = ((pts[0].y-hull_p[hull_size-1].y)*1.0f/(pts[0].x-hull_p[hull_size-1].x));
		if(CROSS(hull_p[hull_size-2], hull_p[hull_size-1],pts[0])<0||fabs(atan(k1)-atan(k2))<CV_PI/45){
			ipop(hull_p);
		}else{
			break;
		}
	}

	return hull_size;
}

//transform point using perspective transformation matrix
CvPoint transformPoint(const CvPoint point, const CvMat* matrix){

	float coordinates[3]={point.x*1.0f, point.y*1.0f, 1.0f};
	CvMat originVector = cvMat(3,1,CV_32F,coordinates);
	CvMat transformedVector = cvMat(3,1,CV_32F,coordinates);
	CvMat *inv = cvCreateMat(3,3,CV_32F);
	cvInvert(matrix, inv);
	cvMatMul(inv,&originVector,&transformedVector);
	CvPoint result=cvPoint((int)(cvmGet(&transformedVector,0,0)/cvmGet(&transformedVector,2,0)),
			(int)(cvmGet(&transformedVector,1,0)/cvmGet(&transformedVector,2,0)));
	return result;
}

void find_connected_components(IplImage *mask, int find_ground, float perimScale, int *num, CvRect *bbs, CvPoint *centers, bool draw){
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
			double tmparea=cvContourArea(c);
			double len = cvContourPerimeter(c);
			CvRect aRect = cvBoundingRect(c, 0);
			if(tmparea<100){//area too small
				cvSubstituteContour(scanner, NULL);
				continue;
			}
			if((aRect.y*1.0/mask->height*60+5)>aRect.height){
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
	
	if(draw)
		cvZero(mask);
	if(num!=NULL){
		int N=*num, numFilled=0, i=0;
		for(i=0,c=contours;c!=NULL;c=c->h_next,i++){
			if(i<N){
				if(bbs!=NULL){
					bbs[i]=cvBoundingRect(c);
				}
				if(centers!=NULL){
					centers[i].x=bbs[i].x+bbs[i].width/2;
					centers[i].y=bbs[i].y+bbs[i].height/2;
				}
				numFilled++;
			}
			if(find_ground){
				double tmparea=fabs(cvContourArea(c));
				if(tmparea<maxarea-1)continue;
			}
			if(draw)
				cvDrawContours(mask, c, CVX_WHITE, CVX_BLACK, -1, CV_FILLED, 8);
		}
		*num = numFilled;
	}else{
		for(c=contours; c!=NULL; c=c->h_next){
			if(find_ground){
				double tmparea=fabs(cvContourArea(c));
				if(tmparea<maxarea-1)continue;
			}
			if(draw)
				cvDrawContours(mask, c, CVX_WHITE, CVX_BLACK, -1, CV_FILLED, 8);
		}
	}
}

void find_player_teams(IplImage *frame, IplImage *mask, CvRect *bbs, int *labels, int cnt){
	/*cv::Rect roi;
	cv::Mat hist;
	float range[] = {0, 255};
	const int histSize = 256;
	const float *ranges[] = {range};
	for(int i=0;i<cnt;++i){
		roi.x=bbs[i].x; roi.y=bbs[i].y;
		roi.width=bbs[i].width; roi.height=bbs[i].height;
		Mat roi_mat(Mat(frame), roi);
		Mat roi_mask(Mat(mask), roi);
		IplImage *Imask = cvCreateImage(cvSize(roi.width, roi.height), IPL_DEPTH_8U, 1);
		cvZero(Imask);

		IplImage roi_img = roi_mat;
		cvInRangeS(&roi_img, cvScalar(0), cvScalar(10), Imask);
		int area = cvCountNonZero(Imask);
		cout<<"team 1 area: "<<area<<endl;
		cvZero(Imask);
		cvInRangeS(&roi_img, cvScalar(40), cvScalar(60), Imask);
		area = cvCountNonZero(Imask);
		cout<<"team 2 area: "<<area<<endl;
		cvRectangle(frame, cvPoint(roi.x,roi.y),
	                          cvPoint(roi.x+roi.width, roi.y+roi.height),
	                          cvScalar(0,0,0), -1, 8 );
		cvShowImage("bird", frame);
		cvWaitKey(0);
	}*/
}

extern CvRNG rng;

Tracker::Tracker(CvRect c, CvPoint p){
	no_found_cnt = 0;
	bbox_id = -1;

	bbox = c;
	context = cvRect(c.x-c.width/2, c.y-c.height/2, (int)(c.width*2), (int)(c.height*2));
	center = p;
	last = center;
	foot = cvPoint(center.x, center.y+bbox.height/2+10);
	color = cvScalar(cvRandInt(&rng)%255, cvRandInt(&rng)%255, cvRandInt(&rng)%255);
}

const int MAX_NO_FOUND = 5;
bool *isNewPoint = new bool[30];
int *numTrackerPerPoint = new int[30];

//vector<Tracker> trackersRight;

void trackPlayersSimple(vector<Tracker> &trackers, CvRect *bbs, CvPoint *centers, int cnt){

	for(int i=0;i<cnt;++i){
		isNewPoint[i]=true;
		numTrackerPerPoint[i]=0;
	}

	for (vector<Tracker>::iterator it = trackers.begin();it != trackers.end();){
		float dist_min = numeric_limits<float>::max();
		int best_choice=-1;
		int contain_tracker = -1;
		for(int i=0;i<cnt;++i){
			if(centers[i].x>it->context.x&&centers[i].x<it->context.x+it->context.width&&
				centers[i].y>it->context.y&&centers[i].y<it->context.y+it->context.height){
				isNewPoint[i]=false;
				float dist=DIS(it->center, centers[i]);
				if(dist<dist_min){
					dist_min = dist;
					best_choice=i;
				}
			}
			if(it->center.x>bbs[i].x&&it->center.x<bbs[i].x+bbs[i].width&&
				it->center.y>bbs[i].y&&it->center.y<bbs[i].y+bbs[i].height)contain_tracker = i;
		}
		if(best_choice!=-1){// candidate found
			it->no_found_cnt=0;
			int x=bbs[best_choice].x, y=bbs[best_choice].y, w=bbs[best_choice].width, h=(int)(it->bbox.height*LOWPASS_FILTER_RATE+bbs[best_choice].height*(1.0 - LOWPASS_FILTER_RATE));
			it->context = cvRect(x-w/2, y-h/2, (int)(w*2), (int)(h*2));
			it->bbox = cvRect(x, y, w, h);
			it->last = it->center;
			it->center = cvPoint(centers[best_choice].x, (int)(it->center.y*LOWPASS_FILTER_RATE+centers[best_choice].y*(1- LOWPASS_FILTER_RATE)));
			it->foot = cvPoint(it->center.x, it->center.y+h/2+5);
			it->bbox_id = best_choice;
			numTrackerPerPoint[best_choice]++;
		}else if(contain_tracker!=-1){
			it->no_found_cnt=0;
			int x=(int)(it->bbox.x*LOWPASS_FILTER_RATE+bbs[contain_tracker].x*(1.0 - LOWPASS_FILTER_RATE));
			int y=(int)(it->bbox.y*LOWPASS_FILTER_RATE+bbs[contain_tracker].y*(1.0 - LOWPASS_FILTER_RATE));
			int w=(int)(it->bbox.width*LOWPASS_FILTER_RATE+bbs[contain_tracker].width*(1.0 - LOWPASS_FILTER_RATE));
			int h=(int)(it->bbox.height*LOWPASS_FILTER_RATE+bbs[contain_tracker].height*(1.0 - LOWPASS_FILTER_RATE));
			it->context = cvRect(x-w/2, y-h/2, (int)(w*2), (int)(h*2));
			it->bbox = cvRect(x, y, w, h);
			it->last = it->center;
			it->center = cvPoint((int)(it->center.x*LOWPASS_FILTER_RATE+centers[contain_tracker].x*(1- LOWPASS_FILTER_RATE)), 
				(int)(it->center.y*LOWPASS_FILTER_RATE+centers[contain_tracker].y*(1- LOWPASS_FILTER_RATE)));
			it->foot = cvPoint(it->center.x, it->center.y+h/2+5);
			it->bbox_id = contain_tracker;
			numTrackerPerPoint[contain_tracker]++;
		}else{
			it->no_found_cnt++;
			it->bbox_id = -1;
			if(it->no_found_cnt>MAX_NO_FOUND){
				it = trackers.erase(it);
				continue;
			}
		}
		++it;
	}

	for(int i=0;i<cnt;++i){
		if(isNewPoint[i]){
			trackers.push_back(Tracker(bbs[i], centers[i]));
		}
	}
}

void trackPlayers(vector<Tracker> &trackers, CvRect *bbs, CvPoint *centers, int cnt, vector<Tracker> &trackersRight, CvMat *H_r2l){

	for(int i=0;i<cnt;++i){
		isNewPoint[i]=true;
		numTrackerPerPoint[i]=0;
	}

	for (vector<Tracker>::iterator it = trackers.begin();it != trackers.end();){
		float dist_min = numeric_limits<float>::max();
		int best_choice=-1;
		int contain_tracker = -1;
		for(int i=0;i<cnt;++i){
			if(centers[i].x>it->context.x&&centers[i].x<it->context.x+it->context.width&&
				centers[i].y>it->context.y&&centers[i].y<it->context.y+it->context.height){
				isNewPoint[i]=false;
				float dist=DIS(it->center, centers[i]);
				if(dist<dist_min){
					dist_min = dist;
					best_choice=i;
				}
			}
			if(it->center.x>bbs[i].x&&it->center.x<bbs[i].x+bbs[i].width&&
				it->center.y>bbs[i].y&&it->center.y<bbs[i].y+bbs[i].height)contain_tracker = i;
		}
		if(best_choice!=-1){// candidate found
			/*it->no_found_cnt=0;
			int x=bbs[best_choice].x, y=bbs[best_choice].y, w=bbs[best_choice].width, h=bbs[best_choice].height;
			it->context = cvRect(x-w/2, y-h/2, (int)(w*2), (int)(h*2));
			it->bbox = bbs[best_choice];
			it->last = it->center;
			it->center = cvPoint(centers[best_choice].x, centers[best_choice].y);*/
			it->bbox_id = best_choice;
			numTrackerPerPoint[best_choice]++;
		}else if(contain_tracker!=-1){
			it->bbox_id = contain_tracker;
			numTrackerPerPoint[contain_tracker]++;
		}else{
			it->no_found_cnt++;
			it->bbox_id = -1;
			if(it->no_found_cnt>MAX_NO_FOUND){
				it = trackers.erase(it);
				continue;
			}
		}
		++it;
	}

	for (vector<Tracker>::iterator it = trackers.begin();it != trackers.end();){
		if(it->bbox_id!=-1&&numTrackerPerPoint[it->bbox_id]==1){// candidate found, and is unique for this tracker
			it->no_found_cnt=0;
			int x=bbs[it->bbox_id].x, y=bbs[it->bbox_id].y, w=bbs[it->bbox_id].width, h=bbs[it->bbox_id].height;
			it->context = cvRect(x-w/2, y-h/2, (int)(w*2), (int)(it->context.height*LOWPASS_FILTER_RATE+h*2*(1.0 - LOWPASS_FILTER_RATE)));
			it->bbox = cvRect(x, y, w, (int)(it->bbox.height*LOWPASS_FILTER_RATE+h*(1.0 - LOWPASS_FILTER_RATE)));
			it->last = it->center;
			it->center = cvPoint(centers[it->bbox_id].x, (int)(it->center.y*LOWPASS_FILTER_RATE+centers[it->bbox_id].y*(1- LOWPASS_FILTER_RATE)));
			it->foot = cvPoint(centers[it->bbox_id].x, centers[it->bbox_id].y+bbs[it->bbox_id].height/2);
		}else if(it->bbox_id!=-1){//not unique candidate. occlude!! resort to the other camera
			float dist_min = numeric_limits<float>::max();
			int best_choice=-1;
			for(int i=0;i<trackersRight.size();++i){
				CvPoint footRight = transformPoint(trackersRight[i].foot, H_r2l);
				if(footRight.x>it->context.x&&footRight.x<it->context.x+it->context.width&&
					footRight.y-it->bbox.height/2>it->context.y&&
					footRight.y-it->bbox.height/2<it->context.y+it->context.height){
					float dist=DIS(it->foot, footRight);
					if(dist<dist_min){
						dist_min = dist;
						best_choice=i;
					}
				}
			}
			if(best_choice!=-1){// candidate found
				it->no_found_cnt=0;
				CvPoint footBest = transformPoint(trackersRight[best_choice].foot, H_r2l);
				int x=footBest.x, y=footBest.y-it->bbox.height/2;//center point
				//it->context = cvRect(x-w/2, y-h/2, (int)(w*2), (int)(h*2));
				it->context.x = x-it->context.width/2;
				it->context.y = y-it->context.height/2;
				it->bbox.x = x-it->bbox.width/2;
				it->bbox.y = y-it->bbox.height/2;
				it->last = it->center;
				it->center.x = x;
				it->center.y = y;
				it->foot.x = x;
				it->foot.y = y+it->bbox.height/2;
			}else{
				it->no_found_cnt++;
				it->bbox_id = -1;
				if(it->no_found_cnt>MAX_NO_FOUND){
					it = trackers.erase(it);
					continue;
				}
			}
		}
		++it;
	}

	for(int i=0;i<cnt;++i){
		if(isNewPoint[i]){
			trackers.push_back(Tracker(bbs[i], centers[i]));
		}
	}
}