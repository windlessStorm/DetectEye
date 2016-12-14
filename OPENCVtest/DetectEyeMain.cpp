/* ********************************************************************************
	This program uses openCV Library to process video feed from webcam and detect 
	faces and open/close eyes in each face. It does face and eye detection using 
	Haar Feature-based Cascade Classifiers (haarcascade xml files) which is freely 
	available online. This is a very basic program for what it does, but sometimes 
	gives many false detections. So it have many great scopes for improvements that
	I will put in TODO list.
	TODO:
	1. Instead of using vague and general haarcascade for detecting face and eyes, 
		train our own classifiers for detecting open and close eye.
	2. Processing the frame and extracting contour detail and using circle detection 
		algorithm like Hough Circle to accurately detect and classify pupil will 
		greatly increase the accuracy and precission of this program. 
    3. More preprocessing the frame before using classifiers to make eye stand out and also to avoid false detections.

*********************************************************************************** */


#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdlib.h>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Detects the face and eyes per face in each frame, marks it with circle and display it
void detectAndDisplay(Mat frame);

//Global Variables
String face_cascade_name = "../haarcascades/haarcascade_frontalface_alt2.xml";
String eyes_cascade_name = "../haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Open Eye Detection";
//RNG rng(12345);

// main function 
int main(int argc, const char** argv)
{
	
	Mat frame;

	// Loading the haarcascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading haarcascade for face detection\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading haarcascade for eye detection\n"); return -1; };

	// Reading the video stream from webcam
	VideoCapture capture(0);
	if (capture.isOpened())
	{
		while (true)
		{
			capture.read(frame); // read single frame 

			// Pass the frame to our processing function detectAndDisplay()
			if (!frame.empty())
			{
				detectAndDisplay(frame);
				//imshow("Output", frame);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

			// Keep processing and displaying frames until 'c' is pressed
			int c = waitKey(10);
			if ((char)c == 'c') { break; }
		}
	}
	return 0;
} // main ends here



// Accepts a frame, searches for faces in it, for each face searches for eyes in it, put circles around them and display it
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	#ifdef DEBUG
		cout << "Converting to grayscale!" << endl;
	#endif

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
//		frame_gray = frame;
	#ifdef DEBUG
		cout << "Equalizing hist!" << endl;
	#endif // DEBUG

	equalizeHist(frame_gray, frame_gray);

	#ifdef DEBUG
		cout << "Detecting faces.."<<endl;
	#endif // DEBUG


	// Detecting faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	system("cls");
	if (faces.size() == 0)
		cout << "No Faces found in the feed!" << endl;
	else
		cout << "Number of faces found : " << faces.size() << "." << endl;
	

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);

		#ifdef DEBUG
			cout << "putting frame around detected faces!" << endl;
		#endif // DEBUG
	
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 4, 8, 0);
		
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		
		#ifdef DEBUG
			cout << "Detecting eyes for face("<< i+1 <<").."<< endl;
		#endif 

		// For each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		#ifdef DEBUG
			cout << "Number of eyes found for face(" << i + 1 << ") : " << eyes.size() << "." << endl;
			cout << "Putting frame around eye for face(" << i+1 << ")!" << endl;
		#endif

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.15);
		 	circle(frame, center, radius, Scalar(0, 0, 255), 4, 8, 0);
		}
		cout << "Eyes  open for face(" << i + 1 << ") : " << eyes.size() << "." << endl;
		
	}
	// Display
	imshow(window_name, frame);
	
}