# PHASE 3 : The Fusion lane

# use a technique called Time Bucketing. 
# We will divide the video into 5-second "buckets" 
# and pour both the audio and the visuals into the same bucket.

import re
from langchain_core.documents import Document

# Import the tools we built in Phase 4!
from audio_lane import download_youtube_audio, transcribe_with_timestamps
from vision_lane import download_youtube_video, process_video_frames

def fuse_multimodal_chunks(audio_chunks, vision_chunks, bucket_size=5):
    """Glues Audio and Vision chunks together into 5-second Super-Chunks."""
    print(f"\n🧬 4. Fusing Audio and Vision into {bucket_size}-second Super-Chunks...")
    
    super_chunks = {}
    
    # --- 1. Process Vision (The Anchor) ---
    # We use the vision timestamps (0s, 5s, 10s) as the main buckets
    vision_regex = r"\[(\d+)s\] Visual: (.*)"
    
    for chunk in vision_chunks:
        match = re.search(vision_regex, chunk)
        if match:
            time_sec = int(match.group(1))
            description = match.group(2)
            
            super_chunks[time_sec] = {
                "vision": description,
                "audio": [] # We will fill this next
            }
            
    # --- 2. Process Audio (The Filler) ---
    audio_regex = r"\[([\d\.]+)s - ([\d\.]+)s\] (.*)"
    
    for chunk in audio_chunks:
        match = re.search(audio_regex, chunk)
        if match:
            start_time = float(match.group(1))
            text = match.group(3)
            
            # Figure out which 5-second bucket this audio belongs to
            # e.g., 6.2 seconds belongs in the 5s bucket. 11.4 belongs in the 10s bucket.
            bucket_key = int(start_time // bucket_size) * bucket_size
            
            # If the bucket exists, add the audio text to it!
            if bucket_key in super_chunks:
                super_chunks[bucket_key]["audio"].append(text)

    # --- 3. Format for LangChain / ChromaDB ---
    final_documents = []
    print("\n--- 🦸‍♂️ THE SUPER-CHUNKS ---\n")
    
    for time_sec in sorted(super_chunks.keys()):
        data = super_chunks[time_sec]
        
        # Glue all the audio in this bucket into one sentence
        audio_combined = " ".join(data["audio"]) if data["audio"] else "[Silence / No speech]"
        
        # Create the final text representation
        fused_text = (
            f"Timestamp: [{time_sec}s - {time_sec + bucket_size}s]\n"
            f"Visual Context: {data['vision']}\n"
            f"Audio Transcript: {audio_combined}"
        )
        
        print(fused_text)
        print("-" * 30)
        
        # Wrap it in a LangChain Document object
        # Now it looks exactly like a PDF chunk to your existing Phase 2 loop!
        doc = Document(page_content=fused_text, metadata={"source": "youtube", "start_time": time_sec})
        final_documents.append(doc)
        
    return final_documents

if __name__ == "__main__":
    print("🤖 Welcome to the Multimodal Fusion Pipeline!")
    url = input("🔗 Paste a YouTube URL: ")
    
    # 1. Run the Audio Lane
    audio_file = download_youtube_audio(url)
    audio_chunks = transcribe_with_timestamps(audio_file)
    
    # 2. Run the Vision Lane
    video_file = download_youtube_video(url)
    vision_chunks = process_video_frames(video_file, interval_seconds=5)
    
    # 3. Fuse them together!
    final_docs = fuse_multimodal_chunks(audio_chunks, vision_chunks)
    
    print(f"\n🎉 Success! Created {len(final_docs)} Multimodal Super-Chunks.")
    print("These are perfectly formatted to be inserted straight into your ChromaDB!")