// 3DIVS.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>


#define FILE_TYPE_VIDEO_CVMAT
//#define FILE_TYPE_SINGLE_IPLIMAGE

using namespace cv;
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	cv::Mat mergeRows(cv::Mat A, cv::Mat B);
	cv::Mat mergeCols(cv::Mat A, cv::Mat B);

#ifdef FILE_TYPE_SINGLE_CVMAT
	vector<Rect> people;
	Mat src = imread("C:\\peopletest\\6.jpg");
	//define HOG, use default parameter
	HOGDescriptor defaultHog;
	//setting SVM
	defaultHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	//multiscale detection, return rect
	defaultHog.detectMultiScale(src,people,0,Size(8,8),Size(32,32),1.03,2);
	//defaultHog.detectMultiScale(src, people, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 1);
	//draw rect, mark people
	for (int i=0;i<people.size();i++)
	{
		Rect r = people[i];
		rectangle(src, r.tl(), r.br(), Scalar(0, 0, 255), 3);
	}

	int headcout = people.size();

	namedWindow("3DIVS", CV_WINDOW_AUTOSIZE);

	imshow("3DIVS", src);
	waitKey(0);
#endif

#ifdef FILE_TYPE_VIDEO_CVMAT
	
	//open video file
	const string source = "C:\\peopletest\\videos\\MOV_0616_1280x720.mp4";
	string::size_type pAt = source.find_last_of('.');
	//const string result = source.substr(0, pAt) + "_SVMHOGResult" + ".avi";
	const string result = source.substr(0, pAt) + "_SVMHOG" + ".avi";
	VideoCapture capture(source);

	//check open status
	if(!capture.isOpened())
		cout<<"fail to open!"<<endl;
	//total frame number
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout<<"The video is total "<<totalFrameNumber<<"frames"<<endl;
	

	//set start frame
	long frameToStart = 0;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	cout << "From " << frameToStart << " Frame" << endl;

	//set stop frame
	int frameToStop = 4000;

	if (frameToStop < frameToStart)
	{
		cout << "Stop frame is beyond end frame, program quit!" << endl;
		return -1;
	}
	else
	{
		cout << "End frame is " << frameToStop << " Frame" << endl;
	}

	//get FPS
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "FPS is: " << rate << endl;

	//define stop flag
	bool stop = false;

	Mat frame;

	//frame interval 
	int delay = 1000 / rate;
	long currentFrame = frameToStart;

	//filter
	//int kernel_size = 3;
	//Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);

	///detection people number
	//set HOG
	HOGDescriptor defaultHog;
	//set SVM
	defaultHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	vector<Rect> people;

	///set videoWriter
	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video  
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video  
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	VideoWriter oVideoWriter(result, CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true); //initialize the VideoWriter object   

	while (!stop)
	{
		//check null
		if (!capture.read(frame))
		{
			cout << "Read video failed" << endl;
			return -1;
		}
		//show original frame
		//Mat resizedFrame;
		//resize(frame, resizedFrame, Size(frame.cols / 2, frame.rows / 2), 0, 0, CV_INTER_LINEAR);
		//namedWindow("Origin Video", CV_WINDOW_AUTOSIZE);
		//imshow("Origin Video", frame);
		Mat afterFrame;
		frame.copyTo(afterFrame);
		//multiscale detection, return rect
		defaultHog.detectMultiScale(afterFrame, people, 0, Size(8, 8), Size(0, 0), 1.05, 2);
		//defaultHog.detectMultiScale(frame, people, 0, Size(8, 8), Size(32, 32), 1.03, 2); //big body
		//defaultHog.detectMultiScale(src, people, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 1);
		//draw rect, mark people
		for (int i = 0; i<people.size(); i++)
		{
			Rect r = people[i];
			rectangle(afterFrame, r.tl(), r.br(), Scalar(0, 0, 255), 3);
		}
		//show marked frame
		imshow("before",frame);
		imshow("bafter", afterFrame);
		//imshow("before vs after", mergeRows(frame, afterFrame));
		
		cout << "Reading frame " << currentFrame << endl;
		cout << "People detected " <<people.size() << endl;
		long leftFrameNumber = totalFrameNumber - currentFrame;
		cout << "Left frames is " << leftFrameNumber << endl;

		//video writer
		oVideoWriter.write(afterFrame); //writer the frame into the file 

		int c = waitKey(delay);
		//ECS or to specified frame No.
		if ((char)c == 270 || currentFrame > frameToStop)
		{
			stop = true;
		}
		//pause
		if (c >= 0)
		{
			waitKey(0);
		}
		currentFrame++;
	}
	cout << "new video file save as " << result << endl;
	//close video file
	capture.release();
	waitKey(0);


#endif

#ifdef FILE_TYPE_SINGLE_IPLIMAGE
	//qing:load image
	IplImage *test;
	test = cvLoadImage("C:\\ALEX\\I2R\\color\\Picture1.jpg");
	cvNamedWindow("3DIVS_demo",1);
	cvShowImage("3DIVS_demo",test);
	cvWaitKey(0);
	cvDestroyWindow("3DIVS_demo");
	cvReleaseImage(&test);
#endif

#ifdef FILE_TYPE_VIDEO_IPLIMAGE
	//qing: load video
	int key = 0;

	// Initialize camera or OpenCV image
	//CvCapture* capture = cvCaptureFromCAM( 0 );
	CvCapture* capture = cvCaptureFromAVI("C:\\ALEX\\I2R\\color\\Video1.avi");
	IplImage* frame = cvQueryFrame(capture);

	// Check 
	if (!capture)
	{
		fprintf(stderr, "Cannot open AVI!\n");
		return 1;
	}

	// Get the fps, needed to set the delay
	int fps = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);

	// Create a window to display the video
	cvNamedWindow("video", CV_WINDOW_AUTOSIZE);

	while (key != 'x')
	{
		// get the image frame 
		frame = cvQueryFrame(capture);

		// exit if unsuccessful
		if (!frame) break;

		// display current frame 
		cvShowImage("video", frame);

		// exit if user presses 'x'        
		key = cvWaitKey(1000 / fps);
	}

	// Tidy up
	cvDestroyWindow("video");
	cvReleaseCapture(&capture);
#endif

	return 0;
}
//by rows
cv::Mat mergeRows(cv::Mat A, cv::Mat B)
{
	// cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	int totalRows = A.rows + B.rows;
	cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
	cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);
	return mergedDescriptors;
}
//by cols
cv::Mat mergeCols(cv::Mat A, cv::Mat B)
{
	// cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	int totalCols = A.cols + B.cols;
	cv::Mat mergedDescriptors(A.rows, totalCols, A.type());
	cv::Mat submat = mergedDescriptors.colRange(0, A.cols);
	A.copyTo(submat);
	submat = mergedDescriptors.colRange(A.cols, totalCols);
	B.copyTo(submat);
	return mergedDescriptors;
}

