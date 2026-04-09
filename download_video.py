import yt_dlp

def download_reel(url):
    ydl_opts = {
        # 'best' ensures you get a compatible mp4 for OpenCV
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'input_video.mp4', 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    # The URL of the video you showed me
    reel_url = "https://www.instagram.com/reel/DP3qHlXE_Ip/"
    download_reel(reel_url)