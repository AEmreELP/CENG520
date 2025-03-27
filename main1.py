import sys
import cv2 as cv


def main(path):
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    src = cv.imread(path, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image: ' + "image.jpg")
        return -1

    # src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    print(grad_y.shape)
    print(grad_x.shape)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    print(abs_grad_x.shape)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    print(abs_grad_y.shape)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.imshow(window_name, grad)
    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main("image.jpg")