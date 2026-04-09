# PHASE 3 : This is the grand conductor that runs the video ingestion, 
# builds the database, and boots up your Agents.

import os
from vector_db import create_retriever
from graph_builder import build_graph

# Import our custom Video RAG tools
from audio_lane import download_youtube_audio, transcribe_with_timestamps
from vision_lane import download_youtube_video, process_video_frames
from fusion_lane import fuse_multimodal_chunks

def process_youtube_url(url):
    """Runs the 3-step Multimodal ingestion pipeline."""
    print("\n" + "="*50)
    print("🎥 INITIALIZING MULTIMODAL INGESTION PIPELINE")
    print("="*50)
    
    audio_file = download_youtube_audio(url)
    audio_chunks = transcribe_with_timestamps(audio_file)
    
    video_file = download_youtube_video(url)
    vision_chunks = process_video_frames(video_file, interval_seconds=5)
    
    super_chunks = fuse_multimodal_chunks(audio_chunks, vision_chunks)
    
    # Clean up the heavy media files
    if os.path.exists("temp_audio.mp3"): os.remove("temp_audio.mp3")
    if os.path.exists("temp_video.mp4"): os.remove("temp_video.mp4")
        
    return super_chunks

def main():
    print("🤖 Welcome to the Multimodal Agentic RAG System!")
    url = input("🔗 Paste a YouTube URL to analyze: ")
    
    try:
        # 1. Rıp the video into Super-Chunks
        documents = process_youtube_url(url)
        
        # 2. Build the local memory (ChromaDB)
        print("\n🧠 Pushing Super-Chunks into Vector Database...")
        retriever = create_retriever(documents)
        
        # 3. Boot up the LangGraph Multi-Agent Team
        print("🔗 Waking up the Router, Writer, and Critic...")
        app = build_graph(retriever)
        
        print("\n✅ System Ready! Your AI has watched the video.")
        print("-" * 50)
        
        # 4. Chat with your video!
        while True:
            query = input("\n🔎 Ask a question about the video (or 'quit'): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            result = app.invoke({"question": query}) # type: ignore
            
            print(f"\n🤖 Final Verified Answer:\n{result['generation']}\n")
            print("-" * 50)
            
    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")

if __name__ == "__main__":
    main()