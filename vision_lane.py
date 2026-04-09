#PHASE 3 : Vision lane
# This lane is all about processing video from YouTube and describing it.

import cv2
import yt_dlp
import base64
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def download_youtube_video(url, output_filename="temp_video"):
    """Downloads the lowest-res MP4 of a video to save time and space."""
    print(f"🎬 1. Downloading low-res video from: {url}")
    
    ydl_opts = {
        # Grab the worst quality MP4 (Vision models don't need 4K!)
        'format': 'worstvideo[ext=mp4]', 
        'outtmpl': f'{output_filename}.%(ext)s',
        'quiet': True,
        'no_warnings': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    print("   ✅ Video download complete!")
    return f"{output_filename}.mp4"

def process_video_frames(video_path, interval_seconds=5):
    """Takes a screenshot every X seconds and asks Llama to describe it."""
    print(f"👀 2. Loading Llama-3.2-Vision to analyze frames...")
    llm = ChatOllama(model="llama3.2-vision", temperature=0)
    
    print(f"📸 3. Extracting a frame every {interval_seconds} seconds...")
    video = cv2.VideoCapture(video_path)
    
    # Get the Frames Per Second (FPS) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    current_frame = 0
    chunks = []
    
    print("\n--- 🖼️ EXTRACTED VISUAL CHUNKS ---\n")
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break # Video is over!
            
        # Only process the frame if it hits our 5-second interval
        if current_frame % frame_interval == 0:
            # Calculate the timestamp in seconds
            timestamp = int(current_frame / fps)
            
            # Convert the frame into a format the AI can read (Base64)
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Ask the Vision model to describe the screenshot
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Briefly describe the key action, people, or text visible in this image in one sentence."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            )
            
            response = llm.invoke([message])
            description = response.content.strip()
            
            # Format the chunk
            chunk = f"[{timestamp}s] Visual: {description}"
            chunks.append(chunk)
            print(chunk)
            
        current_frame += 1

    video.release()
    return chunks

if __name__ == "__main__":
    print("🤖 Welcome to the Vision Lane!")
    url = input("🔗 Paste a YouTube URL: ")
    
    try:
        video_file = download_youtube_video(url)
        # We process 1 frame every 5 seconds. You can change this number!
        visual_chunks = process_video_frames(video_file, interval_seconds=5)
        
        print(f"\n🎉 Success! Extracted {len(visual_chunks)} visual chunks.")
        
    finally:
        # Clean up the video file
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
            print("🧹 Cleaned up temporary video files.")