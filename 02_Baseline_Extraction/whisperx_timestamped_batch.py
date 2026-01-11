#!/usr/bin/env python3
"""
WhisperX Word-Level Timestamp Extractor - Batch Processing
Uses existing RTTM files for speaker labels, extracts word-level timestamps
"""

import os
import sys
import json
from pathlib import Path
import whisperx
import torch
import logging
from typing import Dict, List, Optional
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Processing configuration
PROCESS_CONFIGS = [
    {
        "input": Path("/Volumes/Databackup2025/ForLOCUS/CleanedSplicedAudio/CSA-Research"),
        "output": Path("/path/to/user/Desktop/TIMESTAMPED_CLEAN_OUTPUT/CSA-Research"),
        "skip_dirs": [
            "Mac_Audacity_10MinConvo_withCoordinatorSpeech",
            "Mac_Audacity_10MinuteConvo_withoutCoordinatorSpeech",
            "Mac_Audacity_Grandfather_Passage",
            "Mac_Audacity_Motor_Speech_Evaluation",
            "Mac_Audacity_Picnic_Description",
            "Mac_Audacity_Spontaneous_Speech"
        ]
    },
    {
        "input": Path("/Volumes/Databackup2025/ForLOCUS/CleanedSplicedAudio/Clinic_Plus_segmented"),
        "output": Path("/path/to/user/Desktop/TIMESTAMPED_CLEAN_OUTPUT/Clinic_Plus_segmented"),
        "skip_dirs": []
    }
]

MODEL_SIZE = "large-v2"
BATCH_SIZE = 8

class RTTMParser:
    """Parse RTTM files to get speaker segments"""
    
    @staticmethod
    def parse_rttm(rttm_path: Path) -> List[Dict]:
        """
        Parse RTTM file into list of speaker segments
        
        RTTM format: SPEAKER <filename> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
        
        Returns: [{"start": float, "end": float, "speaker": str}, ...]
        """
        segments = []
        
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == 'SPEAKER':
                    start = float(parts[3])
                    duration = float(parts[4])
                    end = start + duration
                    speaker = parts[7]
                    
                    segments.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker
                    })
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        return segments
    
    @staticmethod
    def find_speaker_for_word(word_start: float, word_end: float, 
                              speaker_segments: List[Dict]) -> Optional[str]:
        """
        Find which speaker spoke a word based on temporal overlap
        
        Uses midpoint of word for assignment
        """
        word_midpoint = (word_start + word_end) / 2
        
        # Find segment containing word midpoint
        for segment in speaker_segments:
            if segment['start'] <= word_midpoint <= segment['end']:
                return segment['speaker']
        
        # If no exact match, find closest segment
        min_distance = float('inf')
        closest_speaker = None
        
        for segment in speaker_segments:
            distance = min(
                abs(word_midpoint - segment['start']),
                abs(word_midpoint - segment['end'])
            )
            if distance < min_distance:
                min_distance = distance
                closest_speaker = segment['speaker']
        
        return closest_speaker

class WhisperXWordExtractor:
    def __init__(self, device: str = "cpu", model_size: str = "large-v2", 
                 compute_type: str = "float32", batch_size: int = 16):
        """
        Initialize WhisperX models for ASR and alignment
        
        Args:
            device: "cpu", "cuda", or "mps"
            model_size: WhisperX model size
            compute_type: "float32" or "float16"
            batch_size: Batch size for processing
        """
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.batch_size = batch_size
        
        logging.info(f"Loading WhisperX model: {model_size} on {device}")
        self.model = whisperx.load_model(
            model_size, 
            device, 
            compute_type=compute_type
        )
        
        logging.info("WhisperX model loaded")
    
    def extract_words(self, audio_path: Path, rttm_path: Path) -> Dict:
        """
        Extract word-level timestamps with speaker labels
        
        Args:
            audio_path: Path to audio file (cleaned or raw)
            rttm_path: Path to existing RTTM file
        
        Returns:
            {
                "audio_file": str,
                "words": [
                    {"word": str, "start": float, "end": float, "speaker": str},
                    ...
                ]
            }
        """
        # Load RTTM speaker segments
        logging.info(f"  Loading RTTM: {rttm_path.name}")
        speaker_segments = RTTMParser.parse_rttm(rttm_path)
        logging.info(f"  Found {len(speaker_segments)} speaker segments")
        
        # Run WhisperX ASR
        logging.info(f"  Running ASR on: {audio_path.name}")
        audio = whisperx.load_audio(str(audio_path))
        result = self.model.transcribe(
            audio, 
            batch_size=self.batch_size,
            language="en"
        )
        
        # Run alignment to get word-level timestamps
        logging.info(f"  Running alignment...")
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
        
        # Extract words with speaker assignments
        words_with_speakers = []
        
        for segment in result["segments"]:
            if "words" in segment:
                for word_info in segment["words"]:
                    word_start = word_info.get("start")
                    word_end = word_info.get("end")
                    word_text = word_info.get("word", "").strip()
                    
                    if word_start is not None and word_end is not None and word_text:
                        # Find speaker for this word
                        speaker = RTTMParser.find_speaker_for_word(
                            word_start, word_end, speaker_segments
                        )
                        
                        words_with_speakers.append({
                            "word": word_text,
                            "start": round(word_start, 3),
                            "end": round(word_end, 3),
                            "speaker": speaker if speaker else "UNKNOWN"
                        })
        
        logging.info(f"  Extracted {len(words_with_speakers)} words")
        
        return {
            "audio_file": audio_path.name,
            "rttm_file": rttm_path.name,
            "total_words": len(words_with_speakers),
            "words": words_with_speakers
        }

def process_directory(input_dir: Path, output_dir: Path, extractor: WhisperXWordExtractor, skip_dirs: List[str] = None):
    """Process all audio files with RTTM in a directory"""
    
    if skip_dirs is None:
        skip_dirs = []
    
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Find all RTTM files recursively
    rttm_files = list(input_dir.rglob("*.rttm"))
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing: {input_dir}")
    logging.info(f"Found {len(rttm_files)} RTTM files")
    if skip_dirs:
        logging.info(f"Skipping subdirectories: {', '.join(skip_dirs)}")
    logging.info(f"{'='*80}\n")
    
    processed = 0
    skipped = 0
    skipped_dir = 0
    errors = 0
    
    for idx, rttm_path in enumerate(rttm_files, 1):
        # Check if file is in a skip directory
        skip_this = False
        for skip_dir in skip_dirs:
            if skip_dir in rttm_path.parts:
                logging.info(f"[{idx}/{len(rttm_files)}] ⏭ SKIP (excluded dir): {rttm_path.relative_to(input_dir)}")
                skipped_dir += 1
                skip_this = True
                break
        
        if skip_this:
            continue
        
        # Find corresponding audio file
        # Try both cleaned and original versions
        audio_candidates = [
            rttm_path.parent / f"{rttm_path.stem}_cleaned.wav",
            rttm_path.parent / f"{rttm_path.stem}.wav"
        ]
        
        audio_path = None
        for candidate in audio_candidates:
            if candidate.exists():
                audio_path = candidate
                break
        
        if not audio_path:
            logging.warning(f"[{idx}/{len(rttm_files)}] ✗ No audio file found for {rttm_path.name}")
            errors += 1
            continue
        
        # Create output path maintaining directory structure
        rel_path = rttm_path.relative_to(input_dir)
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_json = output_subdir / f"{rttm_path.stem}_word_timing.json"
        
        # Check if already processed
        if output_json.exists():
            logging.info(f"[{idx}/{len(rttm_files)}] ✓ SKIP (exists): {rel_path}")
            skipped += 1
            continue
        
        logging.info(f"[{idx}/{len(rttm_files)}] Processing: {rel_path}")
        
        try:
            # Extract word-level timestamps
            result = extractor.extract_words(audio_path, rttm_path)
            
            # Save to JSON
            with open(output_json, 'w') as f:
                json.dump(result, f, indent=2)
            
            logging.info(f"  ✓ Saved: {output_json.name}")
            processed += 1
            
        except Exception as e:
            logging.error(f"  ✗ Error processing {rttm_path.stem}: {e}")
            errors += 1
            continue
    
    # Summary for this directory
    logging.info(f"\n{'='*80}")
    logging.info(f"Directory Summary: {input_dir.name}")
    logging.info(f"{'='*80}")
    logging.info(f"Processed: {processed}")
    logging.info(f"Skipped (already done): {skipped}")
    logging.info(f"Skipped (excluded dirs): {skipped_dir}")
    logging.info(f"Errors: {errors}")
    logging.info(f"Total RTTM files: {len(rttm_files)}")

def main():
    """Batch process multiple directories"""
    
    # Detect device
    # Note: faster-whisper (used by WhisperX) does not support MPS
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"  # Use CPU for Mac and other systems
    
    logging.info("="*80)
    logging.info("WHISPERX WORD-LEVEL TIMESTAMP EXTRACTOR - BATCH MODE")
    logging.info("="*80)
    logging.info(f"Model:  {MODEL_SIZE}")
    logging.info(f"Device: {device}")
    logging.info(f"Batch:  {BATCH_SIZE}")
    logging.info("="*80)
    
    # Initialize extractor once for all directories
    extractor = WhisperXWordExtractor(
        device=device,
        model_size=MODEL_SIZE,
        compute_type="float32",
        batch_size=BATCH_SIZE
    )
    
    # Process each configured directory
    for config in PROCESS_CONFIGS:
        try:
            process_directory(
                config['input'],
                config['output'],
                extractor,
                config.get('skip_dirs', [])
            )
        except KeyboardInterrupt:
            logging.info("\n\n⚠ Processing interrupted by user")
            sys.exit(1)
        except Exception as e:
            logging.error(f"\n✗ Error processing directory {config['input']}: {e}\n")
    
    logging.info("\n" + "="*80)
    logging.info("ALL PROCESSING COMPLETE")
    logging.info("="*80)

if __name__ == "__main__":
    main()
