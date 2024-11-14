import cv2
import numpy as np
from ultralytics import YOLO
from lane import detect_lanes, display_lines

# Load YOLOv8 segmentation model
try:
    model = YOLO("/home/sm/Desktop/yolov8/yolov8m-seg-custom.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open video file
cap = cv2.VideoCapture('/home/sm/Desktop/yolov8/test2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (640, 480))

    # Perform segmentation
    results = model(frame)

    # Check if results are available
    if results:
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  

                for i in range(masks.shape[0]):  
                    mask = masks[i]  
                    mask = (mask * 255).astype(np.uint8)  
                    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    mask_colored = cv2.resize(mask_colored, (640, 480))
                    frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.5, 0)

                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) > 0:
                            top_left = tuple(contour[contour[:, :, 1].argmin()][0])
                            same_y_points = contour[contour[:, 0, 1] == top_left[1]]
                            if same_y_points.size > 0:
                                max_x = same_y_points[:, 0, 0].max()
                                top_right = (max_x, top_left[1])

                                # Draw points on the frame for visualization
                                cv2.circle(frame, top_left, 5, (0, 255, 0), -1)  
                                cv2.circle(frame, top_right, 5, (255, 0, 0), -1)

            # Detect lanes using the lane detection function imported from lane.py
            lines = detect_lanes(frame)

            # Draw detected lanes on the frame
            line_image = display_lines(frame.copy(), lines)
            combined_image = cv2.addWeighted(frame.copy(), 0.8, line_image, 1.0, 0)

            # Display the resulting frame with segmentation and lane detection
            cv2.imshow('YOLOv8 Segmentation and Lane Detection', combined_image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
