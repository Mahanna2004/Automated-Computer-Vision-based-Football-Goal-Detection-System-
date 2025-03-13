from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time
import threading
from playsound import playsound

# Initialize Serial Communication with Arduino 
arduino = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2)  # Allow Arduino to initialize

# Load YOLO model
model = YOLO('D:\\Goal detection using ML model_all files\\c13best.pt')

# Define color range for white detection (goal line detection)
lower_white = np.array([170, 170, 170], dtype=np.uint8)  # Lower BGR bound for white
upper_white = np.array([255, 255, 255], dtype=np.uint8)  # Upper BGR bound for white

# Variables for goal detection
goal_detected = False
ball_left_of_goal = False
ball_right_of_goal = False
goal_message_counter = 0
blue_line_x = None  # X-coordinate of the blue line

# Path to Goal Audio
goal_audio_path = "D:\\Goal detection using ML model_all files\\Goal_audio.mp3"

# Function to play goal sound
def play_goal_audio():
    playsound(goal_audio_path)

# Minimum area threshold for a contour to be considered as the goal line
min_contour_area = 500  # Adjust this value based on your requirements

# Maximum ball length relative to video dimensions (0.4 times video width or height)
max_ball_length_ratio = 0.5

# Maximum ball area
max_ball_area_ratio = 0.25

# Minimum ball area
min_ball_area_ratio = 0.05

# Function to detect the goal line
def detect_goal_line(frame):
    global blue_line_x

    goal_line_frame = frame.copy()
    white_mask = cv2.inRange(goal_line_frame, lower_white, upper_white)  # Mask for white pixels
    white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)  # Apply Gaussian blur to reduce noise

    # Finding contours in the mask
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_with_lines = frame.copy()

    if contours:
        # Filter contours based on area threshold
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        if filtered_contours:
            # Find the contour with the maximum length (perimeter)
            goal_contour = max(filtered_contours, key=lambda cnt: cv2.arcLength(cnt, closed=True))

            # Draw the largest contour in red
            cv2.drawContours(frame_with_lines, [goal_contour], -1, (0, 0, 255), 3)

            # Extract contour points
            contour_points = goal_contour[:, 0, :]  # Shape: (N, 2)

            # Sort contour points by x-coordinate
            sorted_points = contour_points[np.argsort(contour_points[:, 0])]

            # Calculate the median x-coordinate of the 10% rightmost points
            num_points = len(sorted_points)
            right_points = sorted_points[-num_points // 10:]  # 10% rightmost points

            # Use the median to determine the x-coordinate for the right vertical line
            blue_line_x = int(np.median(right_points[:, 0]))  # Median x-coordinate of right points

            # Ensure the blue line is within the bounds of the contour
            if blue_line_x > sorted_points[-1][0]:  # If blue_line_x is righter than the rightmost point
                blue_line_x = sorted_points[-1][0]

            # Draw the vertical blue line
            cv2.line(frame_with_lines, (blue_line_x, 0), (blue_line_x, frame.shape[0]), (255, 0, 0), 2)  # Blue line

    return frame_with_lines, blue_line_x

# Function to process video for goal detection
def process_video_for_goal_detection(frame):
    global goal_detected, ball_left_of_goal, ball_right_of_goal, goal_message_counter, blue_line_x

    frame = cv2.resize(frame, (1280, 780))  # Resize for consistent processing

    # Detect the goal line and update blue_line_x
    frame, blue_line_x = detect_goal_line(frame)

    if blue_line_x is None:
        return frame  # Skip goal detection if no goal line is found

    # Ball detection using YOLO model
    results = model(frame)
    smallest_box = None
    smallest_area = float('inf')

    for result in results[0].boxes.data:  # Access detected boxes
        x1, y1, x2, y2, confidence, cls = result.cpu().numpy()
        
        if int(cls) != 0:  # Assuming class 0 is the ball, discard others
            continue

        # Calculate ball bounding box dimensions
        ball_width = x2 - x1
        ball_height = y2 - y1

        # Check if ball length exceeds the maximum allowed ratio
        if ball_width > max_ball_length_ratio * frame.shape[1] or ball_height > max_ball_length_ratio * frame.shape[0]:
            continue  # Skip this detection if the ball is too large

        area = (x2 - x1) * (y2 - y1)  # Calculate area before checking
        
        if confidence > 0.5 and area < (max_ball_area_ratio * (1280 * 780)) and area > (min_ball_area_ratio * (1280 * 780)):  # Confidence threshold
            if area < smallest_area:
                smallest_area = area
                smallest_box = (int(x1), int(y1), int(x2), int(y2), confidence)

    # Draw ball bounding box and perform goal detection logic
    if smallest_box:
        x1, y1, x2, y2, confidence = smallest_box
        ball_center_x = (x1 + x2) // 2
        ball_radius = (x2 - x1) // 2

        # Drawing the ball bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Ball {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Goal detection logic based on blue line
        if blue_line_x is not None:
            if ball_center_x + ball_radius < blue_line_x and not ball_left_of_goal:
                ball_left_of_goal = True
            elif ball_center_x - ball_radius > blue_line_x and ball_left_of_goal:
                goal_detected = True
                ball_left_of_goal = False

            if ball_center_x - ball_radius > blue_line_x and not ball_right_of_goal:
                ball_right_of_goal = True
            elif ball_center_x + ball_radius < blue_line_x and ball_right_of_goal:
                goal_detected = True
                ball_right_of_goal = False

    # Displaying goal message
    if goal_detected:
        text = "Goal Detected!"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 3)
        cv2.rectangle(frame, (10, 10), (10 + text_width + 20, 10 + text_height + 20), (0, 255, 0), -1)
        cv2.putText(frame, text, (20, 10 + text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 3)

        goal_message_counter += 1
        if goal_message_counter > 30:
            goal_detected = False
            goal_message_counter = 0

        # Send '1' to Arduino to trigger LED and LCD
        arduino.write(b'1\n')

    # Play goal sound using threading
    if goal_message_counter > 0:
        threading.Thread(target=play_goal_audio).start()


    return frame

# Capture video from webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define video writer to save the captured frames
output_file = "captured_video.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Record video until 's' is pressed
print("Recording video... Press 's' to stop.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow("Webcam Recording", frame)

    # Write the frame to the output video file
    out.write(frame)

    # Stop recording if 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Recording stopped.")
        break

# Release resources for recording
cap.release()
out.release()
cv2.destroyAllWindows()

# Reopen the saved video for goal detection
cap = cv2.VideoCapture(output_file)
if not cap.isOpened():
    print("Error: Could not open saved video file.")
    exit()

# Process the saved video for goal detection
print("Running goal detection on the saved video...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame = process_video_for_goal_detection(frame)  # Process the frame

    cv2.imshow("Goal Detection (Saved Video)", frame)

    key = cv2.waitKey(20)
    if key & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources for goal detection
cap.release()
cv2.destroyAllWindows()