#ifndef HEADER_CONSTANTS
#define HEADER_CONSTANTS

#include <cstdio>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <limits.h>
// k-means
#include <cxcore.h>
#include "qcvcamshifttracker.h"

#define CV_CONTOUR_APPROX_LEVEL 1
#define CVCLOSE_ITR 2
#define CVCLOSE_ITR_SMALL 1

using namespace std;

const int WIDTH = 500;
const int HEIGHT = 680;
const int TOP = 552;
const int BOTTOM = 680;
const int LEFT = 106;
const int RIGHT = 394;
const int CIRCLE_LEFT = 177;
const int CIRCLE_RIGHT = 323;

const CvScalar CVX_WHITE = CV_RGB(0xff, 0xff, 0xff);
const CvScalar CVX_RED = CV_RGB(0xff, 0x00, 0x00);
const CvScalar CVX_BLACK = CV_RGB(0x00, 0x00, 0x00);

const int IMAGE_SCALE = 2;

//histogram
const int MARGIN = 10;
const int MARGIN_BLUE = 35;
const int MARGIN_GREEN = 40;
const int MARGIN_WHITE = 60;

//some function
#define ZERO 1e-8
#define DIS(a,b) sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y))
#define SGN(x) (fabs(x)<ZERO?0:(x>0?1:-1))
#define CROSS(a,b,c) ((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x))
#define CMP(a,b) (a.x<b.x||(SGN(a.x-b.x)==0&&a.y<b.y))

int hull_size=0;
int findHull2(IplImage *Imask, CvPoint* pts, int cnt, CvPoint *hull_p);
CvPoint transformPoint(const CvPoint point, const CvMat* matrix);
void find_connected_components(IplImage *mask, int find_ground=1, int poly1_hull0=1, float perimScale=60, 
	int *num=NULL, CvRect *bbs=NULL, CvPoint *centers=NULL, int find_lines=0);


//tracking
class Tracker{
public:
	CvRect context;
	CvRect bbox;
	CvPoint center, last;
	float move_dist;
	int no_found_cnt;

	Tracker(CvRect, CvPoint);
};
void trackPlayers(vector<Tracker> &trackers, CvRect *bbs, CvPoint *centers, int cnt);
void find_player_teams(IplImage *frame, IplImage *mask,  CvRect *bbs, int *labels, int cnt);

#endif