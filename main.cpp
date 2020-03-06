// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>

#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/types_c.h>
#include <opencv2/core/cuda.hpp>

#include <opencv2/bgsegm.hpp>
#include "./DeepAppearanceDescriptor/FeatureTensor.h"

#include "KalmanFilter/tracker.h"
#include "CountingBees.h"

const char *keys =
    "{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
    "{image i        |<none>| input image   }"
    "{video v       |<none>| input video   }";

// yolo parameter
// Initialize the parameters
const float confThreshold = 0.5; // Confidence threshold
const float nmsThreshold = 0.4;   // Non-maximum suppression threshold
const int inpWidth = 416;         // Width of network's input image
const int inpHeight = 416;        // Height of network's input image
std::vector<std::string> classes;

//Deep SORT parameter

const int nn_budget = 100;
const float max_cosine_distance = 0.2;
// Remove the bounding boxes with low confidence using non-maxima suppression
//void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &out, DETECTIONS &d);
void postprocess(cv::Mat &frame, const cv::Mat &output, DETECTIONS &d);
// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame);

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net &net);

void get_detections(DETECTBOX box, float confidence, DETECTIONS &d);

static long gettimeU()
{
  struct timeval tv;
  struct timezone tz;

  gettimeofday(&tv, &tz);

  std::cout << "==" << tv.tv_usec << std::endl;

  return tv.tv_usec;
}

static void timeusePrint(const std::string &mod, const long time_start, const long time_end)
{
  std::cout << mod << " cost " << (time_end - time_start) / 1000 << " ms .." << std::endl;
}

//static std::string itos(int i) // convert int to string
//{
//  std::stringstream s;
//  s << i;
//  return s.str();
//}

using namespace cv;

int main(int argc, char **argv)
{
  int num_devices = cv::cuda::getCudaEnabledDeviceCount();
  if (num_devices <= 0)
  {
    std::cout << "there is no device ." << std::endl;
    return -1;
  }

  std::cout << "num_devices = " << num_devices << std::endl;

  int enable_device_id = -1;
  for (int i = 0; i < num_devices; i++) //cv::cuda::DeviceInfo::deviceID() ;
  {
    cv::cuda::DeviceInfo dev_info(i);
    if (dev_info.isCompatible())
    {
      enable_device_id = i;
      std::cout << "enable_device_id = " << enable_device_id << std::endl;
    }
  }

  cv::cuda::setDevice(enable_device_id);

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
  if (parser.has("help"))
  {
    parser.printMessage();
    return 0;
  }

  //deep SORT
  tracker mytracker(max_cosine_distance, nn_budget);
  //yolo
  // Load names of classes
  std::string classesFile = "coco.names";
  std::ifstream ifs(classesFile.c_str());
  std::string line;
  while (getline(ifs, line))
    classes.push_back(line);

  // counter with the gate area of bee box
  CountingBees counter(190, 190, 1665, 1665);
  int cnInBees = 0, cnOutBees = 0;

  // Give the configuration and weight files for the model

  //cv::String modelConfiguration = "yolov3.cfg";
  //cv::String modelWeights = "yolov3.weights";

  //cv::String modelConfiguration = "deployssd.prototxt";
  //cv::String modelWeights = "mobilenet_iter_73000.caffemodel";


  cv::String modelConfiguration = "/home/frankyu/github/DeepSORT/model1/frozen_inference_graph.pb";
  cv::String modelWeights = "/home/frankyu/github/DeepSORT/model1/bee.pbtxt";


  // Load the network
  //cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
  //cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelConfiguration, modelWeights);
  cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelConfiguration, modelWeights);
  
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

  //net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  // Open a video file or an image file or a camera stream.
  std::string str, outputFile;
  cv::VideoCapture cap;
  cv::VideoWriter video;
  cv::Mat frame, blob;
  try
  {

    outputFile = "yolo_out_cpp.avi";
    if (parser.has("image"))
    {
      // Open the image file
      str = parser.get<cv::String>("image");
      std::ifstream ifile(str);
      if (!ifile)
        throw("error");
      //cap.open(str);
      cap.open("http://192.168.31.85:8080/?action=stream?dummy=param.mjpg");
      str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg");
      outputFile = str;
    }
    else if (parser.has("video"))
    {
      // Open the video file
      str = parser.get<cv::String>("video");
      std::ifstream ifile(str);
      if (!ifile)
        throw("error");
      cap.open(str);
      str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
      outputFile = str;
    }
    else
    {
      cap.open(0);
    }
    // Open the webcaom
    // else cap.open(parser.get<int>("device"));
  }
  catch (...)
  {
    std::cout << "Could not open the input image/video stream" << std::endl;
    return 0;
  }

  // Get the video writer initialized to save the output video
  if (!parser.has("image"))
  {
    video.open(outputFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 28.0,
               cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));
  }

  // Create a window
  static const std::string kWinName = "Multiple Object Tracking";
  namedWindow(kWinName, cv::WINDOW_NORMAL);

  // The index of video frame
  int nFrame = 0;

  //cap.open("http://169.254.92.99:8080/?action=stream?dummy=param.mjpg");

  long time_start = 0;
  long time_end = 0;

  // Process frames.
  while (cv::waitKey(1) < 0)
  {
    time_start = gettimeU();

    // get frame from the video
    cap >> frame;

    // Stop the program if reached end of video
    if (frame.empty())
    {
      std::cout << "Done processing !!!" << std::endl;
      std::cout << "Output file is stored as " << outputFile << std::endl;
      cv::waitKey(3000);
      break;
    }
    // Create a 4D blob from a frame.
    //cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);
    blob=cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(), true, false);

    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    //std::vector<cv::Mat> outs;
    cv::Mat outs;
    //net.forward(outs, getOutputsNames(net));
    outs = net.forward();

    // Remove the bounding boxes with low confidence
    DETECTIONS detections;
    postprocess(frame, outs, detections);

    std::cout << "Detections size:" << detections.size() << std::endl;
    if (!FeatureTensor::getInstance()->getRectsFeature(frame, detections))
    {
      std::cout << "Tensorflow get feature failed!" << std::endl;
      usleep(1000);
      continue;
    }

    {
      std::cout << "Tensorflow get feature succeed!" << std::endl;
      mytracker.predict();
      mytracker.update(detections);
      std::vector<RESULT_DATA> result;
      for (Track &track : mytracker.tracks)
      {
        if (!track.is_confirmed() || track.time_since_update > 1)
          continue;
        result.push_back(std::make_pair(track.track_id, track.to_tlwh()));

        DETECTBOX bbox = track.to_tlwh();
        counter.Update(nFrame, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]);
      }
      for (unsigned int k = 0; k < detections.size(); k++)
      {
        DETECTBOX tmpbox = detections[k].tlwh;
        cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 4);

        // cvScalar的储存顺序是B-G-R，CV_RGB的储存顺序是R-G-B

        for (unsigned int k = 0; k < result.size(); k++)
        {
          DETECTBOX tmp = result[k].second;
          cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
          rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);

          std::string label = cv::format("%d", result[k].first);
          cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        }
      }
    }

    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time for a frame : %.2f ms", t);
    putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

    // Write the frame with the detection boxes
    cv::Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);
    if (parser.has("image"))
      imwrite(outputFile, detectedFrame);
    else
      video.write(detectedFrame);

    time_end = gettimeU();
    timeusePrint("total", time_start, time_end);

    imshow(kWinName, frame);

    std::cout << "nFrame = " << nFrame << std::endl;
    nFrame++;
    usleep(1000);
    //if (0 == (nFrame % 30)){
	counter.Count(cnInBees,cnOutBees);
	std::cout << "-------------------"
	          << "cnInBees  =  " << cnInBees << " , "
              << "cnOutBees = " << cnOutBees << std::endl;
    //}
  }
  
  // Count the number of bees which are in&out of the bee box
  counter.Count(cnInBees,cnOutBees);
  std::cout << "cnInBees  =  " << cnInBees << " , "
              << "cnOutBees = " << cnOutBees << std::endl;

  cap.release();
  if (!parser.has("image"))
    video.release();

  return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat &frame, const cv::Mat &output, DETECTIONS &d)
{
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, (cv::Scalar *)output.ptr<float>());

  for (int i = 0; i < detectionMat.rows; i++)
  {

    float confidence = detectionMat.at<float>(i, 2);

    if (confidence > confThreshold)
    {

      size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

      int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
      int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
      int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
      int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

      int width = xRightTop - xLeftBottom;
      int height = yLeftBottom - yRightTop;
      int left = xLeftBottom;
      int top = yRightTop;

      cv::Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));

      cv::rectangle(frame, object, Scalar(0, 255, 0), 2);

      // String label = String(classNames[objectClass]) + ": " + conf;
      String label = cv::format("%.2f", confidence);

      int baseLine = 0;
      Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      cv::Rect Matafterzh = cv::Rect(Point(xLeftBottom, yLeftBottom - labelSize.height), Size(labelSize.width, labelSize.height + baseLine));
      if (Matafterzh.x < 0 || Matafterzh.y < 0 || Matafterzh.x > frame.cols || Matafterzh.y > frame.rows || (Matafterzh.x + Matafterzh.width) > frame.cols || (Matafterzh.y + Matafterzh.height) > frame.rows)
        continue;

      cv::rectangle(frame, Matafterzh, Scalar(0, 255, 0), cv::FILLED);
      putText(frame, label, Point(xLeftBottom, yLeftBottom), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
      classIds.push_back(objectClass);
      confidences.push_back(confidence);

      boxes.push_back(Matafterzh);
    }
  }

  std::vector<int> indices;
  std::cout << "boxesize before nms:" << boxes.size() << std::endl;
  cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  std::cout << "boxesize after nms:" << boxes.size() << std::endl;
  for (size_t i = 0; i < indices.size(); ++i)
  {
    size_t idx = static_cast<size_t>(indices[i]);
    cv::Rect box = boxes[idx];
    //目标检测 代码的可视化
    //drawPred(classIds[idx], confidences[idx], box.x, box.y,box.x + box.width, box.y + box.height, frame);
    get_detections(DETECTBOX(box.x, box.y, box.width, box.height), confidences[idx], d);
  }

  /* imwrite("image.jpg", frame);
     std::cout<<"havewriteten"<<std::endl;
     waitKey(5000);
     return;*/
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame)
{
  //Draw a rectangle displaying the bounding box
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

  //Get the label for the class name and its confidence
  std::string label = cv::format("%.2f", conf);
  if (!classes.empty())
  {
    CV_Assert(classId < (int)classes.size());
    label = classes[classId] + ":" + label;
  }

  //Display the label at the top of the bounding box
  int baseLine;
  cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = cv::max(top, labelSize.height);
  cv::rectangle(frame, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
  cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net &net)
{
  static std::vector<cv::String> names;
  if (names.empty())
  {
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    std::vector<int> outLayers = net.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    std::vector<cv::String> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
      names[i] = layersNames[outLayers[i] - 1];
  }
  return names;
}

void get_detections(DETECTBOX box, float confidence, DETECTIONS& d)
{
  DETECTION_ROW tmpRow;
  tmpRow.tlwh = box; //DETECTBOX(x, y, w, h);

  tmpRow.confidence = confidence;
  d.push_back(tmpRow);
}
