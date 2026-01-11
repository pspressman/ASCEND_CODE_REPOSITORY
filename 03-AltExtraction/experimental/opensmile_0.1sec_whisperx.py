#!/usr/bin/env python3
"""
OpenSMILE 0.1-Second Feature Extractor with WhisperX Transcripts
Extracts eGEMAPS and ComParE features in 0.1-second windows
Non-diarized version (whole file processing)
Also generates WhisperX transcript for reference
"""

print("SCRIPT STARTED")

import os
import sys
import gc
import argparse
import pandas as pd
from pathlib import Path
import soundfile as sf
import whisperx
import torch
import opensmile
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Suppress opensmile warnings
warnings.filterwarnings('ignore', category=UserWarning, module='opensmile')


def drop_egemaps_from_compare(df):
    """Remove eGeMEPS columns from ComParE output to avoid duplication"""
    egemaps_prefixes = [
        'F0semitoneFrom27.5Hz', 'jitterLocal', 'shimmerLocaldB',
        'HNRdBACF', 'logRelF0-H1-H2', 'logRelF0-H1-A3',
        'F1frequency', 'F1bandwidth', 'F1amplitudeLogRelF0',
        'F2frequency', 'F2bandwidth', 'F2amplitudeLogRelF0',
        'F3frequency', 'F3bandwidth', 'F3amplitudeLogRelF0',
        'alphaRatio', 'hammarbergIndex', 'spectralSlope0-500',
        'spectralSlope500-1500', 'F0final', 'voicedSegmentsPerSec',
        'equivalentSoundLevel', 'loudness'
    ]
    
    cols_to_drop = []
    for col in df.columns:
        if any(prefix in col for prefix in egemaps_prefixes):
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logging.debug(f"  Dropped {len(cols_to_drop)} eGeMEPS columns from ComParE")
    
    return df


class OpenSmileProcessor:
    """Extract openSMILE features + WhisperX transcripts"""
    
    def __init__(self, device="cpu", model_size="large-v2", compute_type="float32", 
                 batch_size=16, extract_compare=True, extract_egemaps=True):
        """
        Initialize WhisperX and openSMILE extractors
        
        Args:
            device: "cpu", "cuda", or "mps"
            model_size: WhisperX model size (default: "large-v2")
            compute_type: "float32" or "float16"
            batch_size: Batch size for WhisperX
            extract_compare: Extract ComParE features (default: True)
            extract_egemaps: Extract eGEMAPS features (default: True)
        """
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.extract_compare = extract_compare
        self.extract_egemaps = extract_egemaps
        
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
        
        # Initialize openSMILE extractors
        if self.extract_compare:
            logging.info("Initializing ComParE feature extractor...")
            self.smile_compare = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            logging.info("✓ ComParE extractor ready")
        
        if self.extract_egemaps:
            logging.info("Initializing eGEMAPS feature extractor...")
            self.smile_egemaps = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            logging.info("✓ eGEMAPS extractor ready")
    
    def extract_transcript(self, audio_file):
        """
        Extract transcript using WhisperX
        
        Returns:
            - full_transcript: Complete transcript string
        """
        logging.info("  → Running WhisperX transcription...")
        
        # Load and transcribe audio
        audio = whisperx.load_audio(str(audio_file))
        result = self.whisper_model.transcribe(
            audio, 
            batch_size=self.batch_size,
            language="en"
        )
        
        # Extract full transcript
        full_transcript_segments = []
        for segment in result["segments"]:
            segment_text = segment.get("text", "").strip()
            full_transcript_segments.append(segment_text)
        
        full_transcript = " ".join(full_transcript_segments)
        
        logging.info(f"  ✓ Transcript extracted ({len(full_transcript)} characters)")
        
        # Clean up
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return full_transcript
    
    def extract_opensmile_windows(self, audio, sr):
        """
        Extract openSMILE features in 0.1-second windows
        
        Returns:
            - compare_df: ComParE features (if enabled)
            - egemaps_df: eGEMAPS features (if enabled)
        """
        window_size = 0.1
        window_samples = int(window_size * sr)
        
        compare_features = []
        egemaps_features = []
        
        total_windows = (len(audio) - window_samples) // window_samples
        logging.info(f"  → Extracting openSMILE features for ~{total_windows} windows...")
        
        for window_start in range(0, len(audio) - window_samples, window_samples):
            window_end = window_start + window_samples
            window_audio = audio[window_start:window_end]
            
            window_time_start = window_start / sr
            window_time_center = (window_start + window_end) / (2 * sr)
            window_time_end = window_end / sr
            
            try:
                # Extract ComParE
                if self.extract_compare:
                    compare_feats = self.smile_compare.process_signal(window_audio, sr)
                    compare_feats = compare_feats.copy()
                    compare_feats['window_start'] = window_time_start
                    compare_feats['window_center'] = window_time_center
                    compare_feats['window_end'] = window_time_end
                    compare_feats['window_duration'] = window_size
                    compare_features.append(compare_feats)
                
                # Extract eGEMAPS
                if self.extract_egemaps:
                    egemaps_feats = self.smile_egemaps.process_signal(window_audio, sr)
                    egemaps_feats = egemaps_feats.copy()
                    egemaps_feats['window_start'] = window_time_start
                    egemaps_feats['window_center'] = window_time_center
                    egemaps_feats['window_end'] = window_time_end
                    egemaps_feats['window_duration'] = window_size
                    egemaps_features.append(egemaps_feats)
                    
            except Exception as e:
                logging.debug(f"  Window at {window_time_start:.1f}s failed: {e}")
                continue
            
            # Progress update every 500 windows
            current_count = len(compare_features) if compare_features else len(egemaps_features)
            if current_count > 0 and current_count % 500 == 0:
                logging.info(f"    Processed {current_count} windows...")
        
        # Convert to DataFrames
        compare_df = None
        egemaps_df = None
        
        if compare_features:
            compare_df = pd.concat(compare_features, ignore_index=True)
            # Drop eGEMAPS columns from ComParE to avoid duplication
            compare_df = drop_egemaps_from_compare(compare_df)
            # Reorder columns: metadata first
            metadata_cols = ['window_start', 'window_center', 'window_end', 'window_duration']
            feature_cols = [col for col in compare_df.columns if col not in metadata_cols]
            compare_df = compare_df[metadata_cols + feature_cols]
            logging.info(f"  ✓ ComParE: {len(compare_df)} windows, {len(feature_cols)} features")
        
        if egemaps_features:
            egemaps_df = pd.concat(egemaps_features, ignore_index=True)
            # Reorder columns: metadata first
            metadata_cols = ['window_start', 'window_center', 'window_end', 'window_duration']
            feature_cols = [col for col in egemaps_df.columns if col not in metadata_cols]
            egemaps_df = egemaps_df[metadata_cols + feature_cols]
            logging.info(f"  ✓ eGEMAPS: {len(egemaps_df)} windows, {len(feature_cols)} features")
        
        return compare_df, egemaps_df
    
    def process_audio_file(self, audio_path, output_dir):
        """
        Process audio file:
        - Extract WhisperX transcript
        - Extract 0.1s openSMILE ComParE features
        - Extract 0.1s openSMILE eGEMAPS features
        
        Args:
            audio_path: Path to audio file
            output_dir: Output directory
        """
        base_name = Path(audio_path).stem
        
        # Output files
        compare_file = output_dir / f"{base_name}_compare_0.1sec.csv"
        egemaps_file = output_dir / f"{base_name}_egemaps_0.1sec.csv"
        transcript_file = output_dir / f"{base_name}_transcript.txt"
        
        logging.info(f"Processing: {base_name}")
        
        # Check what's already done
        compare_exists = compare_file.exists() and compare_file.stat().st_size > 0
        egemaps_exists = egemaps_file.exists() and egemaps_file.stat().st_size > 0
        transcript_exists = transcript_file.exists() and transcript_file.stat().st_size > 0
        
        all_done = (
            (not self.extract_compare or compare_exists) and
            (not self.extract_egemaps or egemaps_exists) and
            transcript_exists
        )
        
        if all_done:
            logging.info(f"  ✓ All outputs exist, skipping")
            return
        
        # Extract transcript if needed
        if not transcript_exists:
            transcript = self.extract_transcript(audio_path)
            with open(transcript_file, 'w') as f:
                f.write(transcript)
            logging.info(f"  ✓ Saved transcript: {transcript_file.name}")
        else:
            logging.info(f"  ✓ Transcript already exists")
        
        # Extract openSMILE features if needed
        if (self.extract_compare and not compare_exists) or (self.extract_egemaps and not egemaps_exists):
            # Load audio
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=0)
            
            duration = len(audio) / sr
            logging.info(f"  Duration: {duration:.1f}s")
            
            # Extract features
            compare_df, egemaps_df = self.extract_opensmile_windows(audio, sr)
            
            # Save ComParE
            if self.extract_compare and compare_df is not None and not compare_exists:
                compare_df.to_csv(compare_file, index=False)
                logging.info(f"  ✓ Saved ComParE: {compare_file.name}")
            
            # Save eGEMAPS
            if self.extract_egemaps and egemaps_df is not None and not egemaps_exists:
                egemaps_df.to_csv(egemaps_file, index=False)
                logging.info(f"  ✓ Saved eGEMAPS: {egemaps_file.name}")
        else:
            logging.info(f"  ✓ openSMILE features already exist")


def main():
    parser = argparse.ArgumentParser(
        description='Extract openSMILE features in 0.1-second windows with WhisperX transcript'
    )
    parser.add_argument('--input', required=True, help='Input directory with WAV files')
    parser.add_argument('--output', required=True, help='Output directory for CSV files')
    parser.add_argument('--model', default='large-v2', help='WhisperX model size (default: large-v2)')
    parser.add_argument('--compare-only', action='store_true', help='Only extract ComParE features')
    parser.add_argument('--egemaps-only', action='store_true', help='Only extract eGEMAPS features')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("OPENSMILE 0.1-SECOND EXTRACTOR WITH WHISPERX TRANSCRIPTS")
    logging.info("="*80)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Determine what to extract
    extract_compare = not args.egemaps_only
    extract_egemaps = not args.compare_only
    
    logging.info(f"Input:  {input_path.absolute()}")
    logging.info(f"Output: {output_path.absolute()}")
    logging.info(f"Model:  {args.model}")
    logging.info(f"Extract ComParE: {extract_compare}")
    logging.info(f"Extract eGEMAPS: {extract_egemaps}")
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
    processor = OpenSmileProcessor(
        device=device,
        model_size=args.model,
        compute_type="float32",
        batch_size=16,
        extract_compare=extract_compare,
        extract_egemaps=extract_egemaps
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
