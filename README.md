# Ambiguous Dancing - Rhythm Game 🕺

A real-time rhythm game built with Python that matches your physical tapping speed against the live tempo of Instagram Reels. The game uses computer vision to track the dancer and native macOS audio processing for a seamless experience.

## 🚀 Features
- Lively BPM Detection: Analyzes the audio track of the video in real-time to determine the target tempo.
- Microphone Integration: Listens to your claps, taps, or hits to calculate your personal BPM.
- Pose Visualization: Uses MediaPipe to track the dancer's movement for visual feedback.
- Reel Scroller: Easily navigate through your downloaded videos using keyboard shortcuts.
- macOS Optimized: Uses native 'afplay' to ensure audio plays correctly within VS Code environments.

## 🛠️ Prerequisites
- macOS (Optimized for Apple Silicon/M2)
- Python 3.10+
- FFmpeg (Required for audio/video merging)
  Command: brew install ffmpeg

## 📦 Installation
1. Clone this repository or enter your project folder.
2. Install the required Python libraries:
   Command: pip install opencv-python mediapipe numpy pyaudio librosa pygame yt-dlp

3. Ensure you have the MediaPipe model file in your directory:
   - pose_landmarker_lite.task

## 🎮 How to Play
1. Download Content:
   Run the downloader script to populate your videos/ folder:
   Command: python3 download_video.py

2. Start the Game:
   Command: python3 main_game.py

3. Match the Beat:
   Watch the LIVE SONG BPM and tap your desk or clap your hands to see your beat appear under YOUR BEAT.

4. Goal: Try to get the "LOCKED IN!" message by matching the song's tempo within a 10 BPM margin.

## ⌨️ Controls
- Right Arrow / N: Next Video (Reel)
- Left Arrow / P: Previous Video (Reel)
- Q: Quit Game

## 📁 Project Structure
- main_game.py: The primary game engine (Audio, Vision, and UI).
- download_video.py: Script to fetch Reels from Instagram.
- analyze_bpm.py: Testing script for movement-based BPM analysis.
- videos/: Directory where your levels (mp4 files) are stored.

## 🧰 Dependencies Breakdown

| Library | Purpose | Command |
| :--- | :--- | :--- |
| **FFmpeg** | Merges audio/video and decodes MP4s. | `brew install ffmpeg` |
| **PortAudio** | Required for PyAudio to access the mic. | `brew install portaudio` |
| **OpenCV** | Handles video playback and UI windows. | `pip install opencv-python` |
| **MediaPipe** | AI engine for nose tracking visuals. | `pip install mediapipe` |
| **Librosa** | Analyzes video audio for live BPM. | `pip install librosa` |
| **PyAudio** | Records your claps/taps for your BPM. | `pip install pyaudio` |
| **Pygame** | Used for audio logic and mixer setup. | `pip install pygame` |
| **YT-DLP** | Downloads the Instagram Reels. | `pip install yt-dlp` |

## 🛠 Troubleshooting (Mac-Specific)

- **No Audio in VS Code:** The game uses macOS native `afplay`. If you hear nothing, ensure your Mac's system volume is up and that the terminal has permission to "Access Desktop" where your videos are stored.
- **BPM Too High:** If "Your Beat" is consistently at 200+, try moving further away from the laptop speakers. The microphone might be picking up the music vibrations as claps.
- **PyAudio Install Error:** If `pip install pyaudio` fails on your M2, run `brew install portaudio` first, then try the pip install again.
- **MediaPipe Crash:** Ensure `pose_landmarker_lite.task` is in the same folder as `main_game.py`. If it is named differently, update the `model_asset_path` in the code.
- **Permissions:** When running for the first time, macOS will ask for Microphone and Camera permissions. You must click **OK** for the game to function.