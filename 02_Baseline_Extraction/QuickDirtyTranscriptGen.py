#!/usr/bin/env python3
"""
Fast transcript generator - base model, no diarization, no features
Just timestamped transcripts for quick review
"""

import os
import sys
import whisperx
import gc 
import torch
from tqdm import tqdm
from pathlib import Path
import logging
import time
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio_fast(audio_file, device, batch_size, compute_type):
    """Fast transcription only - no diarization or features"""
    try:
        start_time = time.time()
        logging.info(f"Starting transcription of {audio_file}")

        # Use base model instead of large-v2 for speed
        try:
            model = whisperx.load_model("base", device, compute_type=compute_type)
        except ValueError as e:
            if "unsupported device mps" in str(e):
                logging.warning(f"MPS device not supported. Falling back to CPU for {audio_file}")
                device = "cpu"
                model = whisperx.load_model("base", device, compute_type=compute_type)
            else:
                raise

        # Transcribe
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        # Align for better timestamps
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        del model_a
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        logging.info(f"Transcription completed in {total_time:.2f} seconds")

        return result['segments']
        
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
    logging.info("Starting fast transcript generation")
    
    # Command line arguments
    if len(sys.argv) >= 3:
        audio_root = sys.argv[1]
        output_root = sys.argv[2]
    else:
        audio_root = "/path/to/user/Desktop/ClinicWavsToProcess"
        output_root = "/path/to/user/Desktop/ClinicWavsToProcess/TranscriptsToReview"
    
    logging.info(f"Input root: {audio_root}")
    logging.info(f"Output root: {output_root}")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")
    
    batch_size = 16
    compute_type = "float32"
    
    # Get all wav files
    audio_root_path = Path(audio_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(audio_root_path.glob("*.wav"))
    logging.info(f"Found {len(audio_files)} audio files to process")
    
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        # Check transcript at processing time
        transcript_file = output_root_path / f"{audio_file.stem}_transcript.txt"
        if transcript_file.exists():
            logging.info(f"⊘ Skipping (transcript exists): {audio_file.name}")
            continue
        file_start_time = time.time()
        
        logging.info(f"Processing: {audio_file.name}")
        
        try:
            segments = process_audio_fast(str(audio_file), device, batch_size, compute_type)
            
            if not segments:
                logging.warning(f"No segments generated for {audio_file.name}")
                continue
            
            # Save transcript with timestamps
            transcript_file = output_root_path / f"{audio_file.stem}_transcript.txt"
            with open(transcript_file, "w") as f:
                f.write(f"Transcript: {audio_file.name}\n")
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
            if "unsupported device mps" in str(e) and device != "cpu":
                logging.info(f"Retrying {audio_file.name} with CPU")
                try:
                    device = "cpu"
                    segments = process_audio_fast(str(audio_file), device, batch_size, compute_type)
                    # Save transcript (same code as above)
                    transcript_file = output_root_path / f"{audio_file.stem}_transcript.txt"
                    with open(transcript_file, "w") as f:
                        f.write(f"Transcript: {audio_file.name}\n")
                        f.write("=" * 80 + "\n\n")
                        for segment in segments:
                            start = format_timestamp(segment['start'])
                            end = format_timestamp(segment['end'])
                            text = segment['text'].strip()
                            f.write(f"[{start} - {end}] {text}\n")
                    logging.info(f"Saved transcript to {transcript_file}")
                except Exception as e:
                    logging.error(f"CPU retry failed for {audio_file.name}: {str(e)}")

    total_processing_time = time.time() - start_time
    logging.info(f"All transcripts completed in {total_processing_time/60:.1f} minutes")

if __name__ == "__main__":
    main()
