import cv2 as cv
import numpy as np

image = cv.imread('test_image.jpeg')
resized_image = cv.resize(image, (1280, 720), interpolation=cv.INTER_AREA)
lane_image = np.copy(resized_image)

def canny(resized_image):
    gray = cv.cvtColor(resized_image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny


def region_of_interest(resized_image):
    height = resized_image.shape[0]
    triangle = np.array([
        [(400, height), (1100, height), (620, 555)]])
    mask = np.zeros_like(resized_image)
    cv.fillPoly(mask, triangle, 255)
    masked_image = cv.bitwise_and(resized_image, mask)
    return masked_image


def display_lines(resized_image, lines):
    line_image = np.zeros_like(resized_image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=10)
    return line_image


def make_coordinates(resized_image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = resized_image.shape[0]
    y2 = int(y1*(4.2/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(resized_image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(resized_image, left_fit_average)
    right_line = make_coordinates(resized_image, right_fit_average)
    return np.array([left_line, right_line])

def img():
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, averaged_lines)
    combined_image = cv.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv.imshow('frame', combined_image)
    cv.waitKey(0)

def video():
    cap = cv.VideoCapture("Recordings/Recording4_014.mp4")
    while(cap.isOpened()):
        _, frame = cap.read()
        resized_frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_AREA)
        canny_image = canny(resized_frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv.HoughLinesP(cropped_image, 2, np.pi / 180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(resized_frame, lines)
        line_image = display_lines(resized_frame, averaged_lines)
        combined_image = cv.addWeighted(resized_frame, 0.8, line_image, 1, 1)
        cv.imshow("result", combined_image)
        cv.waitKey(50)

img()
# video()


