from watershed.cython_modules.PyWaterShed import PyWaterShed
from os import listdir
from cv2 import imshow, imwrite, waitKey
import pytest

dir = listdir("./")
length = len(dir)

@pytest.mark.parametrize("dir_path, length", [
    (dir, len(dir))
])
def test_watershed(dir_path, length):
    ws = PyWaterShed(dir_path, length)
    images = ws.pyWaterShed()
    assert images.ndim>0, "No Working"
    imwrite("./image.png", images[0])

if __name__ == "__main__":

    ws = PyWaterShed(dir, length)
    images = ws.pyWaterShed()
    print(images.shape)
    imshow("image", images[0])
    waitKey(0)