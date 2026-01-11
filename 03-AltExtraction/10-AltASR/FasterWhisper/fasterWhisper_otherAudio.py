#!/usr/bin/env python3
"""
Fast transcript generator for task-separated audio
Maintains folder structure from input to output
"""

import os
import sys
from faster_whisper import WhisperModel
from tqdm import tqdm
from pathlib import Path
import logging
import time
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio_fast(audio_file, model):
    """Fast transcription using faster-whisper"""
    try:
        start_time = time.time()
        logging.info(f"Starting transcription of {audio_file}")

        # Transcribe with faster-whisper
        segments, info = model.transcribe(
            audio_file,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Convert generator to list
        segments = list(segments)
        
        total_time = time.time() - start_time
        logging.info(f"Transcription completed in {total_time:.2f} seconds")
        logging.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        return segments, info
        
    except Exception as e:
        logging.error(f"Error processing {audio_file}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def check_existing_transcript(output_file):
    """Check if transcript already exists"""
    return output_file.exists()

def main():
    start_time = time.time()
    logging.info("Starting fast transcript generation for task-separated audio")
    
    # Set paths - UPDATED FOR YOUR CURRENT NEED
    audio_root = "/path/to/user/Desktop/AudioThreeTest/otherAudio"
    output_root = "/path/to/user/Desktop/AudioThreeTest/FasterWhisper_Transcripts"
    
    logging.info(f"Input root: {audio_root}")
    logging.info(f"Output root: {output_root}")
    
    # Initialize model
    logging.info("Loading Whisper model (base)...")
    model = WhisperModel(
        "base",
        device="auto",
        compute_type="float32"
    )
    logging.info("Model loaded successfully")
    
    # Get all wav files recursively
    audio_root_path = Path(audio_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(audio_root_path.rglob("*.wav"))
    logging.info(f"Found {len(audio_files)} audio files to process")
    
    # Track statistics
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        file_start_time = time.time()
        
        logging.info(f"\nProcessing: {audio_file.name}")
        
        try:
            # Calculate relative path to maintain folder structure
            relative_path = audio_file.relative_to(audio_root_path)
            output_dir = output_root_path / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output file
            transcript_file = output_dir / f"{audio_file.stem}_transcript.txt"
            
            # Check if already processed
            if check_existing_transcript(transcript_file):
                logging.info(f"✓ Transcript already exists, skipping: {transcript_file.name}")
                skipped_count += 1
                continue
            
            # Process audio
            segments, info = process_audio_fast(str(audio_file), model)
            
            if not segments:
                logging.warning(f"No segments generated for {audio_file.name}")
                failed_count += 1
                continue
            
            # Save transcript with timestamps
            with open(transcript_file, "w") as f:
                f.write(f"Transcript: {audio_file.name}\n")
                f.write(f"Source: {relative_path}\n")
                f.write(f"Language: {info.language} (probability: {info.language_probability:.2f})\n")
                f.write("=" * 80 + "\n\n")
                
                for segment in segments:
                    start = format_timestamp(segment.start)
                    end = format_timestamp(segment.end)
                    text = segment.text.strip()
                    f.write(f"[{start} - {end}] {text}\n")
            
            logging.info(f"✓ Saved transcript to {transcript_file}")
            processed_count += 1
            
            file_processing_time = time.time() - file_start_time
            logging.info(f"Completed {audio_file.name} in {file_processing_time:.2f} seconds")
        
        except Exception as e:
            logging.error(f"✗ Failed to process {audio_file.name}: {str(e)}")
            logging.error(traceback.format_exc())
            failed_count += 1

    # Final summary
    total_processing_time = time.time() - start_time
    logging.info("\n" + "=" * 80)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Total files found: {len(audio_files)}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped (already exist): {skipped_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"Total time: {total_processing_time/60:.1f} minutes")
    logging.info(f"Output directory: {output_root}")

if __name__ == "__main__":
    main()
