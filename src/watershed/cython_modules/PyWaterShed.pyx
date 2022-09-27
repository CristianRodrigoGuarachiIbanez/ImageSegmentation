from lib.PyWaterShed cimport *
from cython cimport boundscheck, wraparound, cdivision
from numpy import asarray, dstack, uint8, float32, zeros, ascontiguousarray, any
from numpy cimport ndarray, uint8_t


cdef class PyWaterShed:
    cdef:
        vector[Mat] segImages
    def __cinit__(self, list dir, size_t length):
        self.mainLoop(dir, length)

    @boundscheck(False)
    @wraparound(False)
    cdef void mainLoop(self, list dir, size_t length):

        cdef:
            size_t i
            string ext
            WaterShed * WS
            Mat image
        for i in range(length):
            ext = dir[i].encode("utf_8")[-4:]
            if(ext==b".png"):
                WS = new WaterShed(dir[i].encode("utf_8"))
                WS.calculateMakers(True)
                WS.waterShed()
                WS.highlightingMarkers()
                image = WS.segmentedImage()
                self.segImages.push_back(image)

            else:
                print("pass")

    @boundscheck(False)
    @wraparound(False)
    cdef ndarray[uint8_t, ndim=4] waterShed(self, vector[Mat]&segImages):
        cdef:
            int i
            Mat temp
        output = []

        for i in range(segImages.size()):
            temp = segImages[i]
            img = self.Mat2np(temp)
            output.append(img)
        return asarray(output, dtype=uint8)


    @boundscheck(False)
    @wraparound(False)
    cdef inline object Mat2np(self, Mat&m):
        # Create buffer to transfer data from m.data
        cdef Py_buffer buf_info

        # Define the size / len of data
        cdef size_t len = m.rows*m.cols*m.elemSize()  #m.channels()*sizeof(CV_8UC3)

        # Fill buffer
        PyBuffer_FillInfo(&buf_info, NULL, m.data, len, 1, PyBUF_FULL_RO)

        # Get Pyobject from buffer data
        Pydata  = PyMemoryView_FromBuffer(&buf_info)

        # Create ndarray with data
        # the dimension of the output array is 2 if the image is grayscale
        if (m.channels()==2 ):
            shape_array = (m.rows, m.cols, m.channels())
        elif(m.channels()==3):
            shape_array = (m.rows, m.cols, m.channels())
        else:
            shape_array = (m.rows, m.cols)

        if m.depth() == CV_32F :
            array = ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=float32)
        else :
            #8-bit image
            array = ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=uint8)

        if m.channels() == 3:
            # BGR -> RGB
            array = dstack((array[...,2], array[...,1], array[...,0]))

        return asarray(array, dtype=uint8)

    def pyWaterShed(self):
        return self.waterShed(self.segImages)