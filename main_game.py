import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyaudio
import librosa
import os
import subprocess # NEW: To run native macOS afplay
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
audio_process = None # To track the afplay process

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
            
            # 1. Volume visualization scaling
            rms = np.sqrt(np.mean(audio_data**2))
            current_volume = min(1.0, rms * 12) 

            # 2. Onset Detection
            o_env = onset_strength(y=audio_data, sr=RATE)
            peak_intensity = np.max(o_env)
            
            # ADJUST THIS: 3.5 is a middle ground. 
            # If still 0, try 3.0. If still too high, try 4.0.
            if peak_intensity > 3.0: 
                now = time.time()
                
                # Debounce: Ignore sounds that happen within 0.25s of each other
                # (prevents one clap from being counted twice)
                if (now - last_onset) > 0.25:
                    if last_onset > 0:
                        diff = now - last_onset
                        # Standard human rhythm window (40 to 200 BPM)
                        if 0.3 < diff < 1.5:
                            intervals.append(diff)
                            if len(intervals) > 5: intervals.pop(0)
                            mic_bpm = int(60 / np.mean(intervals))
                    
                    last_onset = now
            
            # Reset if quiet for 2 seconds
            if time.time() - last_onset > 2.0:
                mic_bpm = 0
                intervals = []
                
    except: pass
    finally: p.terminate()


def stop_audio():
    """Stops the current macOS audio process."""
    global audio_process
    if audio_process:
        audio_process.terminate()
        audio_process = None

def play_audio(path):
    """Starts playing audio using macOS native afplay."""
    global audio_process
    stop_audio()
    # afplay is a native macOS command that plays audio from files
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
            
            # 1. LOAD AUDIO FEATURES (LIVELY BPM)
            y_video, sr = librosa.load(video_path, sr=22050)
            full_onset_env = onset_strength(y=y_video, sr=sr)
            
            # 2. START NATIVE AUDIO
            play_audio(video_path)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            while cap.isOpened():
                start_loop = time.time()
                success, frame = cap.read()
                
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    play_audio(video_path) # Restart audio for loop
                    continue

                curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                global_timestamp_ms += int(1000 / fps)
                h, w, _ = frame.shape

                # Lively Audio BPM
                audio_idx = int((curr_frame / fps) * (sr / 512))
                window = full_onset_env[max(0, audio_idx-50):audio_idx+1]
                if len(window) > 10:
                    t, _ = librosa.beat.beat_track(onset_envelope=window, sr=sr)
                    video_audio_bpm = int(t[0]) if isinstance(t, np.ndarray) else int(t)

                # Visualization
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
                
                # CONTROLS
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_audio()
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == 3 or key == ord('n'): # Right
                    current_idx = (current_idx + 1) % len(video_files)
                    stop_audio()
                    break
                elif key == 2 or key == ord('p'): # Left
                    current_idx = (current_idx - 1) % len(video_files)
                    stop_audio()
                    break

                elapsed = time.time() - start_loop
                if elapsed < (1/fps):
                    time.sleep((1/fps) - elapsed)

            cap.release()

if __name__ == "__main__":
    analyze_game()