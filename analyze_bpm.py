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
    # This looks inside your new 'videos' folder for the reels
    video_folder = "videos"
    if not os.path.exists(video_folder):
        print(f"Error: '{video_folder}' folder not found. Run download_video.py first.")
        return
        
    # Get all mp4 files and sort them (input_video1, 2, 3...)
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4')])
    
    if not video_files:
        print("No videos found in the 'videos' folder!")
        return

    current_idx = 0
    
    # MediaPipe Task Setup
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        
        while True: # OUTER LOOP: Controls which video is playing
            video_path = os.path.join(video_folder, video_files[current_idx])
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            frame_delay = int(1000 / fps)
            
            # State variables for the CURRENT video
            y_values = []
            internal_frame_count = 0 
            cumulative_frame_count = 0 
            bpm = 0
            
            print(f"Now playing: {video_files[current_idx]}")

            while cap.isOpened(): # INNER LOOP: Plays the specific video
                start_time = time.time()
                success, frame = cap.read()
                
                # Auto-Loop this specific video
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    internal_frame_count = 0
                    success, frame = cap.read()
                    if not success: break

                internal_frame_count += 1
                cumulative_frame_count += 1
                h, w, _ = frame.shape

                # Processing Logic (Every 2 frames)
                if internal_frame_count % 2 == 0:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    timestamp_ms = int((cumulative_frame_count / fps) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if result.pose_landmarks:
                        for pose in result.pose_landmarks:
                            # Tracking Nose (Landmark 0)
                            nose = pose[0]
                            cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 8, (0, 0, 255), cv2.FILLED)
                            
                            y_values.append(1 - nose.y)
                            if len(y_values) > 150: y_values.pop(0)

                            if len(y_values) > int(fps):
                                peaks, _ = find_peaks(y_values, distance=fps/8, prominence=0.01)
                                duration = len(y_values) / (fps / 2)
                                bpm = (len(peaks) / duration) * 60

                # --- UI OVERLAY ---
                # Show BPM and Playlist Position
                cv2.rectangle(frame, (10, 10), (400, 110), (0, 0, 0), -1)
                cv2.putText(frame, f"VIDEO: {video_files[current_idx]}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"BPM: {int(bpm)}", (20, 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                cv2.putText(frame, "[N] Next | [P] Prev | [Q] Quit", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow('Ambiguous Dancing - Reel Scroller', frame)

                # --- CONTROLS ---
                key = cv2.waitKey(max(1, frame_delay - int((time.time() - start_time) * 1000))) & 0xFF
                
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return # Exit the entire function
                
                elif key == ord('n'): # NEXT VIDEO
                    current_idx = (current_idx + 1) % len(video_files)
                    break # Break inner loop to load next path
                    
                elif key == ord('p'): # PREVIOUS VIDEO
                    current_idx = (current_idx - 1) % len(video_files)
                    break # Break inner loop to load next path

            cap.release()

if __name__ == "__main__":
    analyze_live()