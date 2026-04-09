import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyaudio
import librosa
from librosa.onset import onset_strength
from scipy.signal import find_peaks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- GLOBAL VARIABLES ---
mic_bpm = 0
current_volume = 0
last_onset_time = 0
beat_intervals = []

def audio_thread_function():
    """Tracks microphone volume and calculates live BPM based on hits."""
    global mic_bpm, current_volume, last_onset_time, beat_intervals
    
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 22050
    CHUNK = 1024 * 2 # Smaller chunk for better latency

    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                        input=True, frames_per_buffer=CHUNK)

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            # 1. VOLUME BAR LOGIC
            # Calculate RMS (root mean square) for volume visualization
            rms = np.sqrt(np.mean(audio_data**2))
            current_volume = min(1.0, rms * 10) # Scaling for visibility

            # 2. LIVE BPM LOGIC (Onset Detection)
            # We look for a sudden spike in energy (a beat)
            o_env = onset_strength(y=audio_data, sr=RATE)
            if np.max(o_env) > 2.5: # Threshold for a "hit"
                now = time.time()
                if last_onset_time > 0:
                    interval = now - last_onset_time
                    # Only accept intervals between 40 and 200 BPM
                    if 0.3 < interval < 1.5:
                        beat_intervals.append(interval)
                        if len(beat_intervals) > 5: beat_intervals.pop(0)
                        
                        # Average the last few hits for a stable live BPM
                        avg_interval = np.mean(beat_intervals)
                        mic_bpm = int(60 / avg_interval)
                
                last_onset_time = now
            
            # Decay the BPM slowly if no hits are detected
            if time.time() - last_onset_time > 2.0:
                mic_bpm = 0
                beat_intervals = []

    except Exception as e:
        print(f"Audio Error: {e}")
    finally:
        p.terminate()

def analyze_game():
    global mic_bpm, current_volume
    
    audio_thread = threading.Thread(target=audio_thread_function, daemon=True)
    audio_thread.start()

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture("input_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = int(1000 / fps)
        
        y_values, cumulative_count, video_bpm = [], 0, 0

        while cap.isOpened():
            start_time = time.time()
            success, frame = cap.read()
            
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
                if not success: break

            cumulative_count += 1
            h, w, _ = frame.shape

            if cumulative_count % 2 == 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                timestamp_ms = int((cumulative_count / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    nose = result.pose_landmarks[0][0]
                    cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 8, (0, 0, 255), -1)
                    
                    y_values.append(1 - nose.y)
                    if len(y_values) > 100: y_values.pop(0)
                    if len(y_values) > 30:
                        peaks, _ = find_peaks(y_values, distance=fps/8, prominence=0.01)
                        duration = len(y_values) / (fps / 2)
                        video_bpm = (len(peaks) / duration) * 60

            # --- VISUALIZATION ---
            # 1. BPM Stats
            cv2.putText(frame, f"VIDEO BPM: {int(video_bpm)}", (20, 50), 2, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"YOUR BEAT: {mic_bpm}", (20, 100), 2, 0.8, (0, 255, 0), 2)

            # 2. VOLUME BAR (Bottom of screen)
            bar_width = int(current_volume * (w - 100))
            cv2.rectangle(frame, (50, h - 50), (w - 50, h - 30), (50, 50, 50), -1) # Background
            cv2.rectangle(frame, (50, h - 50), (50 + bar_width, h - 30), (0, 255, 0), -1) # Active Bar
            cv2.putText(frame, "MIC INPUT", (50, h - 60), 2, 0.5, (0, 255, 0), 1)

            # 3. Match Feedback
            if abs(video_bpm - mic_bpm) < 8 and video_bpm > 0:
                cv2.putText(frame, "PERFECT!", (w//2 - 80, h//2), 2, 1.5, (0, 255, 0), 3)

            cv2.imshow('Rhythm Game', frame)
            
            wait = max(1, frame_delay - int((time.time() - start_time) * 1000))
            if cv2.waitKey(wait) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_game()