import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyaudio
import librosa
import os
import pygame
from librosa.onset import onset_strength
from librosa.beat import beat_track

# Use specific imports to avoid the 'AttributeError'
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- GLOBAL VARIABLES ---
mic_bpm = 0
current_volume = 0
video_audio_bpm = 0 
global_timestamp_ms = 0 

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
            current_volume = min(1.0, rms * 15) 
            
            o_env = onset_strength(y=audio_data, sr=RATE)
            if np.max(o_env) > 2.5:
                now = time.time()
                if last_onset > 0:
                    diff = now - last_onset
                    if 0.3 < diff < 1.5:
                        intervals.append(diff)
                        if len(intervals) > 5: intervals.pop(0)
                        mic_bpm = int(60 / np.mean(intervals))
                last_onset = now
            if time.time() - last_onset > 2.0: mic_bpm = 0
    except: pass
    finally: p.terminate()

def analyze_game():
    global mic_bpm, current_volume, video_audio_bpm, global_timestamp_ms
    
    # Pre-initialize mixer to avoid SDL conflicts
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    
    threading.Thread(target=audio_thread_function, daemon=True).start()

    video_folder = "videos"
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4')])
    current_idx = 0

    # FIX: Use the imported classes directly to avoid the AttributeError
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.VIDEO
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True: 
            video_path = os.path.join(video_folder, video_files[current_idx])
            
            # Use 'ffmpeg' as the primary decoder to solve the PySoundFile warning
            y_video, sr = librosa.load(video_path, sr=22050)
            full_onset_env = onset_strength(y=y_video, sr=sr)
            
            pygame.mixer.music.load(video_path)
            pygame.mixer.music.play(-1)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            while cap.isOpened():
                start_loop = time.time()
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pygame.mixer.music.rewind()
                    success, frame = cap.read()
                    if not success: break

                curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                global_timestamp_ms += int(1000 / fps)
                h, w, _ = frame.shape

                # Lively Audio BPM
                audio_idx = int((curr_frame / fps) * (sr / 512))
                window = full_onset_env[max(0, audio_idx-50):audio_idx+1]
                
                if len(window) > 10:
                    t, _ = librosa.beat.beat_track(onset_envelope=window, sr=sr)
                    video_audio_bpm = int(t[0]) if isinstance(t, np.ndarray) else int(t)

                # Nose tracking for visualization
                if int(curr_frame) % 2 == 0:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    res = landmarker.detect_for_video(mp_image, global_timestamp_ms)
                    if res.pose_landmarks:
                        n = res.pose_landmarks[0][0]
                        cv2.circle(frame, (int(n.x*w), int(n.y*h)), 8, (0,0,255), -1)

                # UI
                cv2.rectangle(frame, (10, 10), (420, 160), (0, 0, 0), -1)
                cv2.putText(frame, f"REEL: {video_files[current_idx]}", (20, 45), 2, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"LIVE SONG BPM: {video_audio_bpm}", (20, 85), 2, 0.8, (255,255,255), 2)
                cv2.putText(frame, f"YOUR BEAT: {mic_bpm}", (20, 135), 2, 0.9, (0, 255, 0), 2)

                vol_w = int(current_volume * (w - 100))
                cv2.rectangle(frame, (50, h-50), (w-50, h-30), (50,50,50), -1)
                cv2.rectangle(frame, (50, h-50), (50+vol_w, h-30), (0,255,0), -1)

                if abs(video_audio_bpm - mic_bpm) < 10 and video_audio_bpm > 0:
                    cv2.putText(frame, "LOCKED IN!", (w//2-120, h//2), 2, 1.5, (0,255,0), 4)

                cv2.imshow('Rhythm Game', frame)
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): return
                elif k in [2, 3, ord('n'), ord('p')]:
                    if k in [3, ord('n')]: current_idx = (current_idx + 1) % len(video_files)
                    else: current_idx = (current_idx - 1) % len(video_files)
                    pygame.mixer.music.stop()
                    break

                time.sleep(max(0, (1/fps) - (time.time() - start_loop)))
            cap.release()

if __name__ == "__main__":
    analyze_game()