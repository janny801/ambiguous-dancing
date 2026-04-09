"""
NOTE: THIS FILE IS FOR TESTING PURPOSES ONLY.
IT IS NOT USED FOR THE MAIN GAME SCRIPT.

PURPOSE: This script uses MediaPipe Pose Landmarking to calculate BPM 
based on the physical vertical movement of the dancer's nose.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from scipy.signal import find_peaks

# EXACT IMPORTS FROM YOUR DOCUMENTATION
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def analyze_live():
    # 1. PLAYLIST SETUP
    video_folder = "videos"
    if not os.path.exists(video_folder):
        print(f"Error: '{video_folder}' folder not found. Run download_video.py first.")
        return
        
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4')])
    
    if not video_files:
        print("No videos found in the 'videos' folder!")
        return

    current_idx = 0
    
    # 2. GLOBAL TIMESTAMP (Moved outside loops to prevent crash)
    global_timestamp_ms = 0
    
    # MediaPipe Task Setup
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        
        while True: # OUTER LOOP: Controls which video is playing
            video_name = video_files[current_idx]
            video_path = os.path.join(video_folder, video_name)
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            frame_delay = 1.0 / fps
            
            y_values = []
            internal_frame_count = 0 
            bpm = 0
            
            print(f"Testing Movement BPM for: {video_name}")

            while cap.isOpened(): # INNER LOOP
                start_time = time.time()
                success, frame = cap.read()
                
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    internal_frame_count = 0
                    success, frame = cap.read()
                    if not success: break

                internal_frame_count += 1
                
                # Increment global timestamp even when switching videos
                global_timestamp_ms += int(1000 / fps)
                h, w, _ = frame.shape

                if internal_frame_count % 2 == 0:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    
                    # Use global_timestamp_ms to ensure it is always increasing
                    result = landmarker.detect_for_video(mp_image, global_timestamp_ms)

                    if result.pose_landmarks:
                        nose = result.pose_landmarks[0][0]
                        cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 8, (0, 0, 255), cv2.FILLED)
                        
                        y_values.append(1 - nose.y)
                        if len(y_values) > 150: y_values.pop(0)

                        if len(y_values) > int(fps):
                            peaks, _ = find_peaks(y_values, distance=fps/8, prominence=0.01)
                            duration = len(y_values) / (fps / 2)
                            bpm = (len(peaks) / duration) * 60

                # --- UI OVERLAY ---
                cv2.rectangle(frame, (10, 10), (450, 110), (0, 0, 0), -1)
                cv2.putText(frame, f"TESTING: {video_name}", (20, 40), 2, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MOVEMENT BPM: {int(bpm)}", (20, 85), 2, 1.1, (0, 255, 255), 3)
                
                cv2.putText(frame, "Arrows: Scroll | Q: Quit", (20, h - 20), 2, 0.6, (255, 255, 255), 1)

                cv2.imshow('BPM Movement Test - Not Main Game', frame)

                # --- CONTROLS ---
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return 
                
                elif key == 3 or key == ord('n'): # Right Arrow
                    current_idx = (current_idx + 1) % len(video_files)
                    break 
                    
                elif key == 2 or key == ord('p'): # Left Arrow
                    current_idx = (current_idx - 1) % len(video_files)
                    break 

                # Maintain playback speed
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

            cap.release()

if __name__ == "__main__":
    analyze_live()