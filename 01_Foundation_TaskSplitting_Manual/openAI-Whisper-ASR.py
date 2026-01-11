#!/usr/bin/env python3
"""
Transcript generator using OpenAI Whisper - base model
Excludes follow-up folders for ML baseline comparison
"""

import os
import sys
import whisper
from tqdm import tqdm
from pathlib import Path
import logging
import time
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio_fast(audio_file, model):
    """Fast transcription using OpenAI Whisper"""
    try:
        start_time = time.time()
        logging.info(f"Starting transcription of {audio_file}")

        # Transcribe with OpenAI Whisper
        result = model.transcribe(str(audio_file))
        
        total_time = time.time() - start_time
        logging.info(f"Transcription completed in {total_time:.2f} seconds")
        logging.info(f"Detected language: {result['language']}")

        return result["segments"]
        
    except Exception as e:
        logging.error(f"Error processing {audio_file}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def main():
    start_time = time.time()
    logging.info("Starting transcript generation with OpenAI Whisper")
    
    # Set paths
    audio_root = "/path/to/volumes/Databackup2025/conversational speech 18-0456/Participant Data and Forms"
    output_root = "/path/to/user/Desktop/CSA_Research_OpenAIWhisperTranscript"
    
    logging.info(f"Input root: {audio_root}")
    logging.info(f"Output root: {output_root}")
    
    # Load OpenAI Whisper model
    model = whisper.load_model("base")
    logging.info("OpenAI Whisper model loaded successfully")
    
    # Get all wav files recursively
    audio_root_path = Path(audio_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    
    all_audio_files = list(audio_root_path.rglob("*.wav"))
    
    # Filter out files in follow-up folders (case-insensitive)
    audio_files = []
    excluded_files = []
    for audio_file in all_audio_files:
        path_parts = [part.lower() for part in audio_file.parts]
        # Check if any path component contains both "follow" and "up"
        is_followup = any("follow" in part and "up" in part for part in path_parts)
        if is_followup:
            excluded_files.append(audio_file)
        else:
            audio_files.append(audio_file)
    
    logging.info(f"Found {len(all_audio_files)} total wav files")
    logging.info(f"Excluded {len(excluded_files)} follow-up files")
    logging.info(f"Processing {len(audio_files)} files")
    
    if excluded_files:
        logging.info("Excluded folders:")
        excluded_folders = set(f.parent for f in excluded_files)
        for folder in sorted(excluded_folders):
            logging.info(f"  - {folder.relative_to(audio_root_path)}")
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        file_start_time = time.time()
        
        logging.info(f"Processing: {audio_file.name}")
        
        try:
            segments = process_audio_fast(audio_file, model)
            
            if not segments:
                logging.warning(f"No segments generated for {audio_file.name}")
                continue
            
            # Calculate relative path to maintain folder structure
            relative_path = audio_file.relative_to(audio_root_path)
            output_dir = output_root_path / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save transcript with timestamps
            transcript_file = output_dir / f"{audio_file.stem}_transcript.txt"
            with open(transcript_file, "w") as f:
                f.write(f"Transcript: {audio_file.name}\n")
                f.write(f"Source: {relative_path}\n")
                f.write(f"Model: OpenAI Whisper (base)\n")
                f.write("=" * 80 + "\n\n")
                
                for segment in segments:
                    start = format_timestamp(segment['start'])
                    end = format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    f.write(f"[{start} - {end}] {text}\n")
            
            logging.info(f"Saved transcript to {transcript_file}")
            
            file_processing_time = time.time() - file_start_time
            logging.info(f"Completed {audio_file.name} in {file_processing_time:.2f} seconds\n")
        
        except Exception as e:
            logging.error(f"Failed to process {audio_file.name}: {str(e)}")
            logging.error(traceback.format_exc())

    total_processing_time = time.time() - start_time
    logging.info(f"All transcripts completed in {total_processing_time/60:.1f} minutes")

if __name__ == "__main__":
    main()