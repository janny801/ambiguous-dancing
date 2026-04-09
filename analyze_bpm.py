import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.signal import find_peaks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def analyze_live():
    # Setup for Lite model for better performance on MacBook M2
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture("input_video.mp4")
        
        # --- GET ACTUAL FPS FROM VIDEO ---
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Fallback if the video file has weird metadata
        if fps <= 0:
            fps = 30.0
            
        # Calculate ideal time per frame in milliseconds
        frame_delay = int(1000 / fps)
        
        y_values = []
        frame_count = 0
        process_every_n_frames = 2 
        bpm = 0

        print(f"Detected Video FPS: {fps}")
        print(f"Target playback delay: {frame_delay}ms")

        while cap.isOpened():
            start_time = time.time() # Track start of frame processing
            
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Only process specific frames to keep playback smooth
            if frame_count % process_every_n_frames == 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                # Calculate timestamp based on frame count rather than metadata for consistency
                timestamp_ms = int((frame_count / fps) * 1000)
                
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    # landmark[0][0] is the Nose
                    nose = result.pose_landmarks[0][0] 
                    y_values.append(1 - nose.y)

                    # Calculate BPM using the actual FPS from the file
                    # We need at least 2 seconds of tracked data
                    min_samples = int((fps / process_every_n_frames) * 2)
                    if len(y_values) > min_samples:
                        # Find peaks with a distance constraint based on FPS
                        peaks, _ = find_peaks(y_values, distance=fps/8, prominence=0.01)
                        
                        # duration = samples / (samples per second)
                        duration = len(y_values) / (fps / process_every_n_frames)
                        bpm = (len(peaks) / duration) * 60

            # UI Overlay
            cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"LIVE BPM: {int(bpm)}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            cv2.imshow('BPM Tracker - Natural Speed', frame)

            # --- DYNAMIC DELAY CALCULATION ---
            # Calculate how long the processing (AI + Drawing) actually took
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # The wait time is (target delay) - (time already spent)
            actual_delay = max(1, frame_delay - elapsed_ms)

            if cv2.waitKey(actual_delay) & 0xFF == ord('q'): 
                break

        cap.release()
        cv2.destroyAllWindows()
        # Ensure macOS window cleanup
        for i in range(5):
            cv2.waitKey(1)

if __name__ == "__main__":
    analyze_live()