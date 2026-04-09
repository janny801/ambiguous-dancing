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
movement_bpm = 0
global_timestamp_ms = 0 
audio_process = None 
game_mode = "AUDIO" # Default mode

def audio_thread_function():
    global mic_bpm, current_volume, game_mode
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
            
            # Base mic sensitivity
            threshold = 2.5 
             
            if peak_intensity > threshold: 
                now = time.time()
                if (now - last_onset) > 0.25:
                    if last_onset > 0:
                        diff = now - last_onset
                        if 0.3 < diff < 1.5:
                            intervals.append(diff)
                            if len(intervals) > 5: intervals.pop(0)
                            mic_bpm = int(60 / np.mean(intervals))
                    last_onset = now
            
            if time.time() - last_onset > 2.0:
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
    global mic_bpm, current_volume, video_audio_bpm, movement_bpm, global_timestamp_ms, game_mode
    
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
            
            match_streak = 0 
            STREAK_THRESHOLD = 90
            last_check_time = time.time()
            is_matching = False
            
            prev_y = None
            move_intervals = []
            last_move_peak = 0

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

                # 1. AUDIO BPM ANALYSIS
                audio_idx = int((curr_frame / fps) * (sr / 512))
                window = full_onset_env[max(0, audio_idx-50):audio_idx+1]
                if len(window) > 10:
                    t, _ = librosa.beat.beat_track(onset_envelope=window, sr=sr)
                    video_audio_bpm = int(t[0]) if isinstance(t, np.ndarray) else int(t)

                # 2. MOVEMENT BPM ANALYSIS
                if int(curr_frame) % 2 == 0:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    res = landmarker.detect_for_video(mp_image, global_timestamp_ms)
                    if res.pose_landmarks:
                        n = res.pose_landmarks[0][0]
                        curr_y = n.y
                        cv2.circle(frame, (int(n.x*w), int(n.y*h)), 8, (0,0,255), -1)
                        
                        if prev_y is not None:
                            velocity = abs(curr_y - prev_y)
                            if velocity > 0.01:
                                now = time.time()
                                if (now - last_move_peak) > 0.3:
                                    diff = now - last_move_peak
                                    if 0.3 < diff < 1.5:
                                        move_intervals.append(diff)
                                        if len(move_intervals) > 5: move_intervals.pop(0)
                                        movement_bpm = int(60 / np.mean(move_intervals))
                                    last_move_peak = now
                        prev_y = curr_y

                # 3. SELECT & MODIFY TARGET BPM
                if game_mode == "AUDIO":
                    target_bpm = video_audio_bpm
                else:
                    # HIGHER BPM MODIFIER: Increase the movement BPM by 10% or a flat 10
                    # This makes the "Target" faster when tracking the dancer
                    target_bpm = movement_bpm + 10 

                # 4. EVALUATE MATCH (Every 2 seconds)
                current_time = time.time()
                if current_time - last_check_time >= 2.0:
                    is_matching = (abs(target_bpm - mic_bpm) <= 20 and target_bpm > 0)
                    last_check_time = current_time

                # 5. RENDER EFFECTS
                if is_matching:
                    match_streak = min(match_streak + 1, 150)
                else:
                    match_streak = max(0, match_streak - 3)

                if is_matching:
                    overlay = frame.copy()
                    alpha = max(0.1, min(0.4, (match_streak / 150) * 0.4))
                    overlay[:] = (0, 255, 0)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    text = "LOCKED IN!" if match_streak >= STREAK_THRESHOLD else "MATCHING"
                    tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 4)[0]
                    cv2.putText(frame, text, ((w - tsize[0]) // 2, (h + tsize[1]) // 2), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 4)

                # UI OVERLAY
                cv2.rectangle(frame, (10, 10), (450, 200), (0, 0, 0), -1)
                cv2.putText(frame, f"MODE: {game_mode} (Press V)", (20, 40), 2, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"TARGET BPM: {target_bpm}", (20, 80), 2, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"YOUR BEAT: {mic_bpm}", (20, 130), 2, 0.9, (0, 255, 0), 2)
                
                vol_w = int(current_volume * (w - 100))
                cv2.rectangle(frame, (50, h-50), (w-50, h-30), (50,50,50), -1)
                cv2.rectangle(frame, (50, h-50), (50+vol_w, h-30), (0,255,0), -1)

                cv2.imshow('Rhythm Game', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_audio(); cap.release(); cv2.destroyAllWindows(); return
                elif key == ord('v'):
                    game_mode = "MOVEMENT" if game_mode == "AUDIO" else "AUDIO"
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