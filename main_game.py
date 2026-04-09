import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyaudio
import librosa
import os
import subprocess 
from librosa.onset import onset_strength
from librosa.beat import beat_track

# MediaPipe Imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- GLOBAL VARIABLES ---
mic_bpm = 0
current_volume = 0
video_audio_bpm = 0 
global_timestamp_ms = 0 
audio_process = None 

def audio_thread_function():
    global mic_bpm, current_volume
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 22050
    CHUNK = 1024 * 2
    p = pyaudio.PyAudio()
    last_onset = 0
    intervals = []
    
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                        input=True, frames_per_buffer=CHUNK)
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            rms = np.sqrt(np.mean(audio_data**2))
            current_volume = min(1.0, rms * 12) 

            o_env = onset_strength(y=audio_data, sr=RATE)
            peak_intensity = np.max(o_env)
             
            if peak_intensity > 3.0:  # live bpm sensitivity modifier
                now = time.time()
                if (now - last_onset) > 0.25:
                    if last_onset > 0:
                        diff = now - last_onset
                        if 0.3 < diff < 1.5:
                            intervals.append(diff)
                            if len(intervals) > 5: intervals.pop(0)
                            mic_bpm = int(60 / np.mean(intervals))
                    last_onset = now
            
            if time.time() - last_onset > 0.5:
                mic_bpm = 0
                intervals = []
                
    except: pass
    finally: p.terminate()

def stop_audio():
    global audio_process
    if audio_process:
        audio_process.terminate()
        audio_process = None

def play_audio(path):
    global audio_process
    stop_audio()
    audio_process = subprocess.Popen(["afplay", path])

def analyze_game():
    global mic_bpm, current_volume, video_audio_bpm, global_timestamp_ms
    
    threading.Thread(target=audio_thread_function, daemon=True).start()

    video_folder = "videos"
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4')])
    current_idx = 0

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.VIDEO
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True: 
            video_path = os.path.join(video_folder, video_files[current_idx])
            y_video, sr = librosa.load(video_path, sr=22050)
            full_onset_env = onset_strength(y=y_video, sr=sr)
            
            play_audio(video_path)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # --- STREAK & TIMING TRACKER ---
            match_streak = 0 
            STREAK_THRESHOLD = 90 # 3 seconds @ 30fps
            last_check_time = time.time()
            is_matching = False

            while cap.isOpened():
                start_loop = time.time()
                success, frame = cap.read()
                
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    play_audio(video_path)
                    match_streak = 0 
                    continue

                curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                global_timestamp_ms += int(1000 / fps)
                h, w, _ = frame.shape

                # 1. Lively Audio BPM Analysis
                audio_idx = int((curr_frame / fps) * (sr / 512))
                window = full_onset_env[max(0, audio_idx-50):audio_idx+1]
                if len(window) > 10:
                    t, _ = librosa.beat.beat_track(onset_envelope=window, sr=sr)
                    video_audio_bpm = int(t[0]) if isinstance(t, np.ndarray) else int(t)

                # 2. Every 2 Seconds: Evaluate the Match
                current_time = time.time()
                if current_time - last_check_time >= 0.5:
                    is_matching = (abs(video_audio_bpm - mic_bpm) <= 10 and video_audio_bpm > 0)
                    last_check_time = current_time

                # 3. Update Streak 
                if is_matching:
                    match_streak = min(match_streak + 1, 150)
                else:
                    match_streak = max(0, match_streak - 3)

                # 4. Render Matching & Locked In Effects
                if is_matching:
                    # Apply green overlay
                    overlay = frame.copy()
                    # Alpha builds from 0.1 up to 0.4 intensity
                    alpha = max(0.1, min(0.4, (match_streak / 150) * 0.4))
                    overlay[:] = (0, 255, 0) # Green
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # Same format for both states
                    text = "LOCKED IN!" if match_streak >= STREAK_THRESHOLD else "MATCHING"
                    
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    scale = 1.5
                    thick = 4 # Bold for visibility
                    color = (255, 255, 255) # Pure white
                    
                    # Center text
                    tsize = cv2.getTextSize(text, font, scale, thick)[0]
                    tx = (w - tsize[0]) // 2
                    ty = (h + tsize[1]) // 2
                    cv2.putText(frame, text, (tx, ty), font, scale, color, thick)

                # MediaPipe Visualization (Visual Dot only)
                if int(curr_frame) % 2 == 0:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    res = landmarker.detect_for_video(mp_image, global_timestamp_ms)
                    if res.pose_landmarks:
                        n = res.pose_landmarks[0][0]
                        cv2.circle(frame, (int(n.x*w), int(n.y*h)), 8, (0,0,255), -1)

                # UI OVERLAY
                cv2.rectangle(frame, (10, 10), (420, 160), (0, 0, 0), -1)
                cv2.putText(frame, f"REEL: {video_files[current_idx]}", (20, 45), 2, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"LIVE SONG BPM: {video_audio_bpm}", (20, 85), 2, 0.8, (255,255,255), 2)
                cv2.putText(frame, f"YOUR BEAT: {mic_bpm}", (20, 135), 2, 0.9, (0, 255, 0), 2)

                vol_w = int(current_volume * (w - 100))
                cv2.rectangle(frame, (50, h-50), (w-50, h-30), (50,50,50), -1)
                cv2.rectangle(frame, (50, h-50), (50+vol_w, h-30), (0,255,0), -1)

                cv2.imshow('Rhythm Game', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_audio()
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == 3 or key == ord('n'):
                    current_idx = (current_idx + 1) % len(video_files)
                    stop_audio(); break
                elif key == 2 or key == ord('p'):
                    current_idx = (current_idx - 1) % len(video_files)
                    stop_audio(); break

                elapsed = time.time() - start_loop
                if elapsed < (1/fps):
                    time.sleep((1/fps) - elapsed)

            cap.release()

if __name__ == "__main__":
    analyze_game()