import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.signal import find_peaks

# EXACT IMPORTS FROM YOUR DOCUMENTATION
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def analyze_live():
    # SETUP OPTIONS
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    # CREATE THE TASK
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture("input_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
            
        frame_delay = int(1000 / fps)
        y_values = []
        
        # We need two counters:
        # 1. Internal frame count (resets on loop)
        # 2. Cumulative frame count (never resets, used for MediaPipe timestamps)
        internal_frame_count = 0 
        cumulative_frame_count = 0 
        
        process_every_n_frames = 2 
        bpm = 0

        while cap.isOpened():
            start_time = time.time()
            success, frame = cap.read()
            
            # --- LOOP LOGIC ---
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                internal_frame_count = 0
                # Note: We do NOT reset cumulative_frame_count or y_values here
                # so the BPM stays consistent and the timestamp keeps increasing.
                success, frame = cap.read()
                if not success: break
            # ------------------

            internal_frame_count += 1
            cumulative_frame_count += 1
            h, w, _ = frame.shape

            if internal_frame_count % process_every_n_frames == 0:
                # PREPARE DATA
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                # FIX: Use cumulative_frame_count so timestamp always increases
                timestamp_ms = int((cumulative_frame_count / fps) * 1000)
                
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # HANDLE AND DISPLAY RESULTS
                if result.pose_landmarks:
                    for pose in result.pose_landmarks:
                        for landmark in pose:
                            px, py = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

                        nose = pose[0]
                        cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 8, (0, 0, 255), cv2.FILLED)
                        
                        y_pos = 1 - nose.y
                        y_values.append(y_pos)

                        # Maintain a sliding window of the last 150 samples (~10 seconds)
                        # to keep the BPM calculation relevant to the current movement
                        if len(y_values) > 150:
                            y_values.pop(0)

                        min_samples = int((fps / process_every_n_frames) * 2)
                        if len(y_values) > min_samples:
                            peaks, _ = find_peaks(y_values, distance=fps/8, prominence=0.01)
                            duration = len(y_values) / (fps / process_every_n_frames)
                            bpm = (len(peaks) / duration) * 60

            # UI OVERLAY
            cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"LIVE BPM: {int(bpm)}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            cv2.imshow('BPM Tracker - Looping Enabled', frame)

            # DYNAMIC DELAY
            elapsed_ms = int((time.time() - start_time) * 1000)
            actual_delay = max(1, frame_delay - elapsed_ms)

            if cv2.waitKey(actual_delay) & 0xFF == ord('q'): 
                break

    cap.release()
    cv2.destroyAllWindows()
    for i in range(5): cv2.waitKey(1)

if __name__ == "__main__":
    analyze_live()