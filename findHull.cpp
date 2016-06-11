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

void find_connected_components(IplImage *mask, int find_ground, int poly1_hull0, float perimScale, int *num, CvRect *bbs, CvPoint *centers, int find_lines){
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

void find_player_teams(IplImage *frame, CvRect *bbs, int *labels){

}

Tracker::Tracker(CvRect c, CvPoint p){
	no_found_cnt = 0;
	move_dist = numeric_limits<float>::max();

	bbox = c;
	context = cvRect(c.x-c.width/2, c.y-c.height/2, (int)(c.width*1.5), (int)(c.height*1.5));
	center = p;
	last = center;
}

const int MAX_NO_FOUND = 5;

void trackPlayers(vector<Tracker> &trackers, CvRect *bbs, CvPoint *centers, int cnt){

	bool *isNewPoint = new bool[cnt];
	for(int i=0;i<cnt;++i)isNewPoint[i]=true;

	for (vector<Tracker>::iterator it = trackers.begin();it != trackers.end();){
		float dist_min = numeric_limits<float>::max();
		int best_choice=-1;
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
		}	
		if(best_choice!=-1){// candidate found
			it->no_found_cnt=0;
			int x=bbs[best_choice].x, y=bbs[best_choice].y, w=bbs[best_choice].width, h=bbs[best_choice].height;
			it->context = cvRect(x-w/2, y-h/2, (int)(w*1.5), (int)(h*1.5));
			it->bbox = bbs[best_choice];
			it->last = it->center;
			it->center = cvPoint(centers[best_choice].x, centers[best_choice].y);
		}else{
			it->no_found_cnt++;
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