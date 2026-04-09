import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

# Imports exactly as shown in your documentation
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def analyze_live():
    # Setup options for Video mode
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    # Initialize the detector
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture("input_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        y_values = []
        
        print("Starting analysis using Tasks API... Press 'q' to quit.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert frame to MediaPipe Image object as required by docs
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Use detect_for_video (requires timestamp in ms)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            bpm = 0
            if result.pose_landmarks:
                # Track Landmark #0 (Nose) as per the result example in docs
                # The result is a list of poses; we take the first one [0]
                nose = result.pose_landmarks[0][0] 
                
                # 1 - y because normalized coordinates 0 is top
                y_pos = 1 - nose.y
                y_values.append(y_pos)

                # Calculate BPM after we have 2 seconds of data
                if len(y_values) > (fps * 2):
                    peaks, _ = find_peaks(y_values, distance=fps/4, prominence=0.01)
                    duration = len(y_values) / fps
                    bpm = (len(peaks) / duration) * 60

            # UI Overlay
            cv2.rectangle(frame, (10, 10), (420, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"LIVE BPM: {int(bpm)}", (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            cv2.imshow('BPM Tracker - MediaPipe Tasks', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1)

if __name__ == "__main__":
    analyze_live()