#include "waterShed.h"


namespace watershed{

WaterShed::WaterShed(const char * file){
    this->original_img = cv::imread(file);

	if(original_img.empty())
	{
		std::cerr << "Error\n";
		std::cerr << "Cannot Read Image\n";
	}
}


WaterShed::~WaterShed(){

}

void WaterShed::kernels(cv::Mat&src, cv::Mat&imgLaplacian, cv::Mat&imgResult){
    cv::Mat kernel = (cv::Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    cv::filter2D(src, imgLaplacian, CV_32F, kernel);
    cv::Mat sharp;
    src.convertTo(sharp, CV_32F);
    imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
}
void WaterShed::getBackground(const cv::Mat& source,cv::Mat& dst) {
	cv::dilate(source, dst, cv::Mat::ones(3,3,CV_8U)); //Kernel 3x3
}

void WaterShed::getForeground(const cv::Mat& source,cv::Mat& dst) {
    //
	cv::distanceTransform(source, dst, cv::DIST_L2, 3, CV_32F);
	cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
}

void WaterShed::findMarkers(const cv::Mat& sureBg, cv::Mat& markers, std::vector<std::vector<cv::Point>>& contours){
	cv::findContours(sureBg, contours,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	// Draw the foreground markers
	size_t size = contours.size();
	for (size_t i = 0; i < size; i++){
			cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
	}
}

void WaterShed::waterShed(){
    cv::watershed(original_img, markers);
}

void WaterShed::morphologyEx(cv::Mat&thresh, cv::Mat&output){
    cv::morphologyEx(thresh, output,3, cv::Mat::ones(2,2,CV_8U));
}

void WaterShed::calculateMakers(bool morph){

    cv::Mat binary_img = binaryImg();
    cv::Mat sure_bg, sure_fg, dist_8u;
    if(morph){
        morphologyEx(binary_img, binary_img);
    }
    showImg("binary inbetween", binary_img);
    foreground(binary_img, sure_fg);
    cv::threshold(sure_fg, sure_fg, 0.4, 1.0, cv::THRESH_BINARY);
    background(sure_fg, sure_bg);

    this->markers = cv::Mat::zeros(sure_bg.size(), CV_32S);
    sure_bg.convertTo(dist_8u, CV_8U);
	findMarkers(dist_8u, markers, this->contours);
	cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1); //Drawing Circle around the marker
	cv::Mat mark;
	markers.convertTo(mark, CV_8U, 10);
    bitWise(mark);
}

void WaterShed::bitWise(cv::Mat mark){
	cv::bitwise_not(mark, mark); //Convert white to black and black to white
	showImg("MARKER", mark);
}

void WaterShed::getRandomColor(std::vector<cv::Vec3b>&colors, size_t size)
{
	for (int i = 0; i < size ; ++i){
			int b = cv::theRNG().uniform(0, 256);
			int g = cv::theRNG().uniform(0, 256);
			int r = cv::theRNG().uniform(0, 256);
			colors.emplace_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
}

void WaterShed::highlightingMarkers(){

    std::vector<cv::Vec3b> colors;
    getRandomColor(colors,contours.size());

    // Create the result image
    this->dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++){
            for (int j = 0; j < markers.cols; j++)
            {
                    int index = markers.at<int>(i,j);
                    if (index > 0 && index <= static_cast<int>(contours.size()))
                            dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
    }
}
}
int main(){
    watershed::WaterShed ws("scene_749.png");
    ws.calculateMakers(1);
    ws.waterShed();
    ws.highlightingMarkers();
    ws.showColoredImg();
    ws.showBinaryImg();
    return 0;
}