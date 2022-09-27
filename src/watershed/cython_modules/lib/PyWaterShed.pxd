#distutils: language = c++
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

ctypedef unsigned char uchar

# For cv::Mat usage
cdef extern from "opencv2/core/core.hpp":
  cdef int  CV_WINDOW_AUTOSIZE
  cdef int CV_8UC3
  cdef int CV_8UC1
  cdef int CV_32FC1
  cdef int CV_8U
  cdef int CV_32F

cdef extern from "opencv2/core/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    void create(int, int, int)
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()


cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Vec3b:
        Vec3d()
        Vec3d(uchar v0, uchar v1, uchar v2)

    cdef cppclass Point:
        Point()
        Point (int _x, int _y)


# For Buffer usage
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO


cdef extern from "../waterShed.h" namespace "watershed":
    cdef cppclass WaterShed:
        WaterShed(const char * file) except +
        void calculateMakers(bool morph);
        void waterShed();
        void highlightingMarkers();
        Mat segmentedImage()
        void showBinaryImg()
        Mat getMarkers()
        Mat shiftedFilter()
        void grayImg(Mat&src, Mat&gray_img)

        Mat binaryImg()

        void background(Mat&src, Mat&dst)

        void foreground(Mat&src, Mat&dst)

        Mat original_img;
        Mat bin_img;
        Mat markers;
        Mat dst;
        vector[vector[Point]] contours;

        void bgWhiteToBlack(Mat&src)
        void kernels(Mat&src, Mat&imgLaplacian, Mat&imgResult);
        void findMarkers(const Mat& sureBg, Mat& markers, vector[vector[Point]]&contours);
        void getBackground(const Mat&source, Mat&dst);
        void getForeground(const Mat&source, Mat&dst);
        void bitWise(Mat mark);
        void getRandomColor(vector[Vec3b]&colors, size_t size);
        void morphologyEx(Mat&thresh, Mat&output);
        void showImg(const string&windowName, const Mat&img)