import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

class LaneTracker:
    def __init__(self):
        # Initialize Kalman filter parameters
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-5
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

    def update(self, lane):
        if lane is not None:
            self.kf.correct(np.array([[lane[0]], [lane[1]]], np.float32))

    def predict(self):
        return self.kf.predict()

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # Draw blue lines
    return line_image

def region_of_interest(image):
    height, width = image.shape[:2]
    polygon_points = np.array([[0, 400], [0, 480], [640, 480], [640, 320], [365, 270]], np.int32)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], (255))
    return mask

def detect_lanes(frame):
    canny_image = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
    roi_mask = region_of_interest(canny_image)
    roi_canny_image = cv2.bitwise_and(canny_image, roi_mask)
    
    lines = cv2.HoughLinesP(roi_canny_image, rho=7,
                            theta=np.pi / 180,
                            threshold=100,
                            minLineLength=10,
                            maxLineGap=100)

    return lines
