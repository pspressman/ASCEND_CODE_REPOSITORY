#!/usr/bin/env python3
"""
WhisperX Word-Level Timestamp Extractor
Fast ASR with word-level timestamps only
No acoustic feature extraction
"""

print("SCRIPT STARTED")

import os
import sys
import gc
import json
import argparse
import pandas as pd
from pathlib import Path
import whisperx
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class WhisperXProcessor:
    """Extract WhisperX word-level timestamps only"""
    
    def __init__(self, device="cpu", model_size="large-v2", compute_type="float32", batch_size=16):
        """
        Initialize WhisperX model
        
        Args:
            device: "cpu", "cuda", or "mps"
            model_size: WhisperX model size (default: "large-v2")
            compute_type: "float32" or "float16"
            batch_size: Batch size for WhisperX
        """
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.batch_size = batch_size
        
        # Initialize WhisperX
        logging.info(f"Loading WhisperX model: {model_size} on {device}")
        try:
            self.whisper_model = whisperx.load_model(
                model_size, 
                device, 
                compute_type=compute_type
            )
        except ValueError as e:
            if "unsupported device mps" in str(e):
                logging.warning("MPS not supported, falling back to CPU")
                self.device = "cpu"
                device = "cpu"
                self.whisper_model = whisperx.load_model(
                    model_size,
                    "cpu",
                    compute_type=compute_type
                )
            else:
                raise
        logging.info("✓ WhisperX model loaded")
    
    def extract_word_timestamps(self, audio_file):
        """
        Extract word-level timestamps using WhisperX
        
        Returns:
            - words: List of word dicts with start/end times
            - full_transcript: Complete transcript string
        """
        logging.info("  → Running WhisperX ASR...")
        
        # Load and transcribe audio
        audio = whisperx.load_audio(str(audio_file))
        result = self.whisper_model.transcribe(
            audio, 
            batch_size=self.batch_size,
            language="en"
        )
        
        # Run alignment to get word-level timestamps
        logging.info("  → Running WhisperX alignment...")
        model_a, metadata = whisperx.load_align_model(
            language_code="en", 
            device=self.device
        )
        
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            self.device,
            return_char_alignments=False
        )
        
        # Clean up alignment model
        del model_a
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Extract words
        words = []
        full_transcript_segments = []
        
        for segment in result["segments"]:
            segment_text = segment.get("text", "").strip()
            full_transcript_segments.append(segment_text)
            
            if "words" in segment:
                for word_info in segment["words"]:
                    word_start = word_info.get("start")
                    word_end = word_info.get("end")
                    word_text = word_info.get("word", "").strip()
                    
                    if word_start is not None and word_end is not None and word_text:
                        words.append({
                            "word": word_text,
                            "start": round(word_start, 3),
                            "end": round(word_end, 3),
                            "duration": round(word_end - word_start, 3)
                        })
        
        full_transcript = " ".join(full_transcript_segments)
        
        logging.info(f"  ✓ Extracted {len(words)} words")
        
        return words, full_transcript
    
    def process_audio_file(self, audio_path, output_dir):
        """
        Process audio file: extract word timestamps and save
        
        Args:
            audio_path: Path to audio file
            output_dir: Output directory
        """
        base_name = Path(audio_path).stem
        
        # Output files
        word_timing_file = output_dir / f"{base_name}_word_timing.json"
        word_timing_csv = output_dir / f"{base_name}_word_timing.csv"
        transcript_file = output_dir / f"{base_name}_transcript.txt"
        
        logging.info(f"Processing: {base_name}")
        
        # Check if already done
        if word_timing_file.exists() and word_timing_csv.exists() and transcript_file.exists():
            logging.info(f"  ✓ Already exists, skipping")
            return
        
        # Extract word timestamps
        words, full_transcript = self.extract_word_timestamps(audio_path)
        
        # Save as JSON
        with open(word_timing_file, 'w') as f:
            json.dump({
                "audio_file": audio_path.name,
                "total_words": len(words),
                "words": words
            }, f, indent=2)
        logging.info(f"  ✓ Saved JSON: {word_timing_file.name}")
        
        # Save as CSV
        df = pd.DataFrame(words)
        df.to_csv(word_timing_csv, index=False)
        logging.info(f"  ✓ Saved CSV: {word_timing_csv.name}")
        
        # Save transcript
        with open(transcript_file, 'w') as f:
            f.write(full_transcript)
        logging.info(f"  ✓ Saved transcript: {transcript_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract WhisperX word-level timestamps (no acoustic features)'
    )
    parser.add_argument('--input', required=True, help='Input directory with WAV files')
    parser.add_argument('--output', required=True, help='Output directory for timestamp files')
    parser.add_argument('--model', default='large-v2', help='WhisperX model size (default: large-v2)')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("WHISPERX WORD-LEVEL TIMESTAMP EXTRACTOR")
    logging.info("="*80)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    logging.info(f"Input:  {input_path.absolute()}")
    logging.info(f"Output: {output_path.absolute()}")
    logging.info(f"Model:  {args.model}")
    logging.info(f"Input exists: {input_path.exists()}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find WAV files (recursively)
    wav_files = list(input_path.rglob("*.wav"))
    logging.info(f"Found {len(wav_files)} WAV files")
    
    if not wav_files:
        logging.error(f"No WAV files found in {input_path}")
        sys.exit(1)
    
    for wf in wav_files:
        logging.info(f"  - {wf.name}")
    
    logging.info("="*80)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logging.info(f"Device: {device}")
    logging.info("="*80)
    
    # Initialize processor
    processor = WhisperXProcessor(
        device=device,
        model_size=args.model,
        compute_type="float32",
        batch_size=16
    )
    
    # Process files
    logging.info("PROCESSING FILES")
    logging.info("="*80)
    
    for i, wav_file in enumerate(wav_files, 1):
        logging.info(f"\n[{i}/{len(wav_files)}]")
        try:
            processor.process_audio_file(wav_file, output_path)
        except Exception as e:
            logging.error(f"ERROR: {e}")
            import traceback
            logging.error(traceback.format_exc())
            continue
    
    logging.info("="*80)
    logging.info(f"✅ Complete! Check {output_path}")
    logging.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    except SystemExit as e:
        print(f"SystemExit: {e}")
