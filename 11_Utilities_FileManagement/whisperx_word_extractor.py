#!/usr/bin/env python3
"""
WhisperX Word-Level Timestamp Extractor
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

def main():
    """
    Usage: python whisperx_word_extractor.py <audio_root> <output_root> [model_size]
    
    Expects:
    - Audio files and RTTM files in same directories under audio_root
    - RTTM files named: {basename}.rttm
    - Audio files: {basename}.wav or {basename}_cleaned.wav
    
    Example:
    python whisperx_word_extractor.py /path/to/user/Desktop/AudioThreeTest/otherAudio /path/to/user/Desktop/WordTimings large-v2
    """
    
    if len(sys.argv) < 3:
        print("Usage: python whisperx_word_extractor.py <audio_root> <output_root> [model_size]")
        sys.exit(1)
    
    audio_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    model_size = sys.argv[3] if len(sys.argv) > 3 else "large-v2"
    
    # Detect device - Force CPU since faster-whisper doesn't support MPS
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logging.info("="*80)
    logging.info("WHISPERX WORD-LEVEL EXTRACTOR")
    logging.info("="*80)
    logging.info(f"Audio root: {audio_root}")
    logging.info(f"Output:     {output_root}")
    logging.info(f"Model:      {model_size}")
    logging.info(f"Device:     {device}")
    logging.info("="*80)
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = WhisperXWordExtractor(
        device=device,
        model_size=model_size,
        compute_type="float32",
        batch_size=16
    )
    
    # Find all RTTM files
    rttm_files = list(audio_root.rglob("*.rttm"))
    logging.info(f"Found {len(rttm_files)} RTTM files")
    
    if not rttm_files:
        logging.error("No RTTM files found!")
        sys.exit(1)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, rttm_path in enumerate(rttm_files, 1):
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
            logging.warning(f"[{idx}/{len(rttm_files)}] No audio file found for {rttm_path.name}")
            errors += 1
            continue
        
        # Check if already processed
        output_json = output_root / f"{rttm_path.stem}_word_timing.json"
        if output_json.exists():
            logging.info(f"[{idx}/{len(rttm_files)}] ✓ SKIP (exists): {rttm_path.stem}")
            skipped += 1
            continue
        
        logging.info(f"[{idx}/{len(rttm_files)}] Processing: {rttm_path.stem}")
        
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
    
    # Summary
    logging.info("="*80)
    logging.info("EXTRACTION SUMMARY")
    logging.info("="*80)
    logging.info(f"Processed: {processed}")
    logging.info(f"Skipped:   {skipped}")
    logging.info(f"Errors:    {errors}")
    logging.info("="*80)

if __name__ == "__main__":
    main()
