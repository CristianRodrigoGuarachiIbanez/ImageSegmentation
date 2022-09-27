#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <string>
#include <vector>

namespace watershed{
    class WaterShed{
        public:
        WaterShed(const char * file);
        ~WaterShed();
        void calculateMakers(bool morph);
        void waterShed();
        void highlightingMarkers();
        cv::Mat segmentedImage(){
            if(dst.empty()){
                std::cerr << "Error\n";
                std::cerr << "Cannot Read Segmented Image\n";
            }else{
                return dst;
            }
        }
        void showBinaryImg(){
            showImg("binary image", bin_img);
            cv::waitKey(0);
        }

        cv::Mat getMarkers(){
            if(original_img.empty())
            {
                std::cerr << "Error\n";
                std::cerr << "Cannot Read Image\n";

            }else{
                return markers;
            }

        }
        cv::Mat shiftedFilter(){
            cv:: Mat shifted;
            cv::pyrMeanShiftFiltering(this->original_img, shifted, 21, 51);
            // showImg("shifted", shifted);
            return shifted;
        }

        void grayImg(cv::Mat&src, cv::Mat&gray_img ){
            cv::cvtColor(src, gray_img, cv::COLOR_BGR2GRAY);
            //cv::threshold(gray_img, gray_img, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        }

        cv::Mat binaryImg(){
            cv::Mat gray_img;
            cv::Mat imgLaplacian;
            cv::Mat imgResult;
            kernels(original_img, imgLaplacian, imgResult);
            grayImg(imgResult, gray_img);
            cv::threshold(gray_img, bin_img, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            return bin_img;
        }

        void background(cv::Mat&src, cv::Mat&dst){
            getBackground(src, dst);
            // showImg("sure background", dst);
        }

        void foreground(cv::Mat&src, cv::Mat&dst){
            getForeground(src, dst);
            // showImg("sure foreground", dst);
        }

        private:
        cv::Mat original_img;
        cv::Mat bin_img;
        cv::Mat markers;
        cv::Mat dst;
        std::vector<std::vector<cv::Point>> contours;

        void bgWhiteToBlack(cv::Mat&src){
            cv::Mat mask;
            cv::inRange(src, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), mask);
            src.setTo(cv::Scalar(0, 0, 0), mask);
            // showImg("WB", mask);
        }
        void kernels(cv::Mat&src, cv::Mat&imgLaplacian, cv::Mat&imgResult);
        void findMarkers(const cv::Mat& sureBg, cv::Mat& markers, std::vector<std::vector<cv::Point>>& contours);
        void getBackground(const cv::Mat&source, cv::Mat&dst);
        void getForeground(const cv::Mat&source, cv::Mat&dst);
        void bitWise(cv::Mat mark);
        void getRandomColor(std::vector<cv::Vec3b>& colors,size_t size);
        void morphologyEx(cv::Mat&thresh, cv::Mat&output);
        void showImg(const std::string&windowName, const cv::Mat&img){
            cv::imshow(windowName,img);
        }
    };
}