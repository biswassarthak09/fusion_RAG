# PHASE 3 : Audio lane
# This lane is all about processing audio from YouTube videos.
import yt_dlp
import whisper
import os

def download_youtube_audio(url, output_filename="temp_audio"):
    """Downloads a YouTube video and extracts just the audio as an MP3."""
    print(f"🎧 1. Downloading audio from: {url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # Tell it exactly what to name the file
        'outtmpl': f'{output_filename}.%(ext)s', 
        'quiet': True,
        'no_warnings': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    print("   ✅ Audio download complete!")
    return f"{output_filename}.mp3"

def transcribe_with_timestamps(audio_path):
    """Runs Whisper locally to transcribe audio and grab timestamps."""
    print("🧠 2. Loading local Whisper AI (using 'base' model for M2 speed)...")
    
    # We use the 'base' model here. It's fast and highly accurate for English.
    # You can upgrade to 'small' or 'medium' later if you want.
    model = whisper.load_model("base") 
    
    print("✍️ 3. Transcribing audio (this takes a moment)...")
    result = model.transcribe(audio_path)
    
    chunks = []
    print("\n--- 🎬 EXTRACTED TIMESTAMP CHUNKS ---\n")
    
    # Whisper automatically breaks the text into logical "segments"
    for segment in result["segments"]:
        # Round to 1 decimal place instead of chopping it off!
        start = round(segment["start"], 1)
        end = round(segment["end"], 1)
        text = segment["text"].strip()
        
        # Format it exactly how our RAG system will want to read it
        chunk = f"[{start}s - {end}s] {text}"
        chunks.append(chunk)
        print(chunk)
        
    return chunks

if __name__ == "__main__":
    # A great test video: A 1-minute clip of Sam Altman talking
    print("🤖 Welcome to the Audio Lane!")
    url = input("🔗 Paste a YouTube URL: ")
    
    try:
        # Run the pipeline
        audio_file = download_youtube_audio(url)
        timestamped_chunks = transcribe_with_timestamps(audio_file)
        
        print(f"\n🎉 Success! Extracted {len(timestamped_chunks)} chunks.")
        print("These chunks are now ready to be injected into your Vector Database!")
        
    finally:
        # Clean up the heavy MP3 file so we don't clutter your Mac
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
            print("🧹 Cleaned up temporary audio files.")