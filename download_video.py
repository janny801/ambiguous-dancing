import yt_dlp
import os

def download_reels(url_list):
    # 1. Create the videos folder if it doesn't exist
    folder = "videos"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, url in enumerate(url_list):
        # Logic to keep your first video as is, and name others 2, 3, 4, 5
        video_number = i + 1
        video_name = f"input_video{video_number}.mp4"
        output_path = os.path.join(folder, video_name)

        # Skip the first one if you already have it, or skip if file exists
        if video_number == 1:
            print(f"Skipping {video_name} per your request.")
            continue

        if os.path.exists(output_path):
            print(f"Skipping {video_name}, already exists in /videos.")
            continue

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': False,
        }

        print(f"Downloading {url} as {video_name}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    # Your provided links
    links = [
        "https://www.instagram.com/reel/DP3qHlXE_Ip/", # input_video1 (skipped)
        "https://www.instagram.com/reel/DW3c0u1RtTd/", # input_video2
        "https://www.instagram.com/reel/DWyi4mEEUi5/", # input_video3
        "https://www.instagram.com/reel/DWq092YETaX/", # input_video4
        "https://www.instagram.com/reel/DWlnsZwkbbT/"  # input_video5
    ]
    
    download_reels(links)