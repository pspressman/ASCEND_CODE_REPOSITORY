#!/usr/bin/env python3
"""
openSMILE ComParE Feature Extractor for ASCEND Project
Extracts ComParE_2016 acoustic features from pre-transcribed/diarized audio

Extraction:
- Per-speaker aggregated (ALL tasks)
- Non-diarized combined (ALL tasks)
- ~6,000 ComParE features (drops eGeMAPS columns already extracted)

NO per-segment or granular extraction - aggregated only.

Usage:
    python opensmile_compare_extractor.py --input /path/to/audio --output /path/to/output
"""

import sys
import logging
import time
import traceback
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import opensmile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_rttm(rttm_file):
    """Load speaker diarization from RTTM file."""
    segments = []
    
    with open(rttm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                
                # Extract speaker number
                match = re.search(r'(\d+)', speaker)
                speaker_num = int(match.group(1)) if match else 0
                
                segments.append({
                    'start': start,
                    'end': start + duration,
                    'speaker': speaker_num
                })
    
    return pd.DataFrame(segments)


def get_egemaps_columns():
    """
    Return list of eGeMAPS column names to drop from ComParE output.
    These were already extracted in the previous run.
    """
    # eGeMAPS feature names (88 total from eGeMAPSv02)
    egemaps_features = [
        # Frequency related parameters (frequency, formants)
        'F0semitoneFrom27.5Hz_sma3nz_amean',
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
        'F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
        'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
        'F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
        'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
        'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
        'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
        'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
        'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
        
        'F1frequency_sma3nz_amean',
        'F1frequency_sma3nz_stddevNorm',
        'F1bandwidth_sma3nz_amean',
        'F1bandwidth_sma3nz_stddevNorm',
        
        'F2frequency_sma3nz_amean',
        'F2frequency_sma3nz_stddevNorm',
        'F2bandwidth_sma3nz_amean',
        'F2bandwidth_sma3nz_stddevNorm',
        
        'F3frequency_sma3nz_amean',
        'F3frequency_sma3nz_stddevNorm',
        
        # Energy/amplitude related
        'Loudness_sma3_amean',
        'Loudness_sma3_stddevNorm',
        'Loudness_sma3_percentile20.0',
        'Loudness_sma3_percentile50.0',
        'Loudness_sma3_percentile80.0',
        'Loudness_sma3_pctlrange0-2',
        'Loudness_sma3_meanRisingSlope',
        'Loudness_sma3_stddevRisingSlope',
        'Loudness_sma3_meanFallingSlope',
        'Loudness_sma3_stddevFallingSlope',
        
        # Spectral (balance)
        'alphaRatio_sma3_amean',
        'alphaRatio_sma3_stddevNorm',
        'hammarbergIndex_sma3_amean',
        'hammarbergIndex_sma3_stddevNorm',
        'spectralSlope0-500_sma3nz_amean',
        'spectralSlope0-500_sma3nz_stddevNorm',
        'spectralSlope500-1500_sma3nz_amean',
        'spectralSlope500-1500_sma3nz_stddevNorm',
        
        # Spectral (formant position)
        'F1amplitudeLogRelF0_sma3nz_amean',
        'F1amplitudeLogRelF0_sma3nz_stddevNorm',
        'F2amplitudeLogRelF0_sma3nz_amean',
        'F2amplitudeLogRelF0_sma3nz_stddevNorm',
        'F3amplitudeLogRelF0_sma3nz_amean',
        'F3amplitudeLogRelF0_sma3nz_stddevNorm',
        
        # Spectral (harmonic difference)
        'HNRdBACF_sma3nz_amean',
        'HNRdBACF_sma3nz_stddevNorm',
        
        # Temporal features
        'jitterLocal_sma3nz_amean',
        'jitterLocal_sma3nz_stddevNorm',
        'shimmerLocaldB_sma3nz_amean',
        'shimmerLocaldB_sma3nz_stddevNorm',
        
        # MFCCs
        'mfcc1_sma3_amean',
        'mfcc1_sma3_stddevNorm',
        'mfcc2_sma3_amean',
        'mfcc2_sma3_stddevNorm',
        'mfcc3_sma3_amean',
        'mfcc3_sma3_stddevNorm',
        'mfcc4_sma3_amean',
        'mfcc4_sma3_stddevNorm',
        
        # Additional functionals on LLDs
        'F0semitoneFrom27.5Hz_sma3nz_meanAbsDelta',
        'F1frequency_sma3nz_meanAbsDelta',
        'F2frequency_sma3nz_meanAbsDelta',
        'F3frequency_sma3nz_meanAbsDelta',
        'Loudness_sma3_meanAbsDelta',
        'spectralFlux_sma3_amean',
        
        # Equivalent sound level
        'equivalentSoundLevel_dBp',
        
        # Various functionals with different suffixes
        'loudnessPeaksPerSec',
        'VoicedSegmentsPerSec',
        'MeanVoicedSegmentLengthSec',
        'StddevVoicedSegmentLengthSec',
        'MeanUnvoicedSegmentLength',
        'StddevUnvoicedSegmentLength',
    ]
    
    return egemaps_features


def extract_compare_features(audio_array, sr, smile):
    """
    Extract ComParE features using openSMILE.
    Drops eGeMAPS columns that were already extracted.
    """
    try:
        features = smile.process_signal(audio_array, sr)
        
        # Drop eGeMAPS columns
        egemaps_cols = get_egemaps_columns()
        cols_to_drop = [col for col in egemaps_cols if col in features.columns]
        
        if cols_to_drop:
            features = features.drop(columns=cols_to_drop)
        
        return features
    
    except Exception as e:
        logging.error(f"ComParE feature extraction failed: {e}")
        return pd.DataFrame()


def extract_full_task_features(audio_file, rttm_file, output_dir, base_name, smile):
    """
    Extract per-speaker aggregated + non-diarized combined ComParE features.
    ALL tasks get this extraction.
    """
    logging.info(f"  Extracting ComParE features (aggregated only)")
    
    # Load audio
    audio, sr = sf.read(audio_file)
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)
    
    # 1. NON-DIARIZED COMBINED (all speakers together)
    combined_output = output_dir / f"{base_name}_combined_compare_features.csv"
    
    if combined_output.exists() and combined_output.stat().st_size > 0:
        logging.info(f"    ✓ Combined: Already exists, skipping")
    else:
        logging.info(f"    Combined: Extracting features")
        features = extract_compare_features(audio, sr, smile)
        
        if not features.empty:
            features.to_csv(combined_output, index=False)
            logging.info(f"    ✓ Combined: Saved {len(features.columns)} features")
        else:
            logging.warning(f"    Combined: No features extracted")
    
    # 2. DIARIZED PER-SPEAKER
    # Load diarization
    diarization = load_rttm(rttm_file)
    
    if diarization.empty:
        logging.warning(f"  No diarization segments found")
        return
    
    # Group by speaker
    speakers = diarization['speaker'].unique()
    
    for speaker in speakers:
        output_file = output_dir / f"{base_name}_speaker{speaker}_compare_features.csv"
        
        # Check if already exists
        if output_file.exists() and output_file.stat().st_size > 0:
            logging.info(f"    ✓ Speaker {speaker}: Already exists, skipping")
            continue
        
        speaker_segments = diarization[diarization['speaker'] == speaker]
        logging.info(f"    Speaker {speaker}: {len(speaker_segments)} segments")
        
        # Concatenate all speaker segments
        speaker_audio = []
        
        for _, seg in speaker_segments.iterrows():
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            
            if start_sample >= end_sample or end_sample > len(audio):
                continue
            
            audio_segment = audio[start_sample:end_sample]
            
            if len(audio_segment) < sr * 0.1:  # Skip segments < 0.1s
                continue
            
            speaker_audio.append(audio_segment)
        
        if not speaker_audio:
            logging.warning(f"    Speaker {speaker}: No valid audio segments")
            continue
        
        # Concatenate all segments
        speaker_audio_concat = np.concatenate(speaker_audio)
        
        # Extract features from concatenated audio
        features = extract_compare_features(speaker_audio_concat, sr, smile)
        
        if not features.empty:
            features.to_csv(output_file, index=False)
            logging.info(f"    ✓ Speaker {speaker}: Saved {len(features.columns)} features")
        else:
            logging.warning(f"    Speaker {speaker}: No features extracted")


def process_file(audio_file, output_dir, smile):
    """Process single audio file."""
    base_name = audio_file.stem
    
    # Check for required inputs
    rttm_file = audio_file.parent / f"{base_name}.rttm"
    
    if not rttm_file.exists():
        logging.warning(f"  Missing diarization: {rttm_file.name}")
        return
    
    # Extract ComParE features (aggregated only)
    extract_full_task_features(audio_file, rttm_file, output_dir, base_name, smile)


def main():
    parser = argparse.ArgumentParser(
        description='Extract openSMILE ComParE_2016 features from pre-transcribed audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extracts ~6,000 ComParE_2016 features (drops eGeMAPS columns already extracted).
Per-speaker aggregated + non-diarized combined for ALL tasks.

Example:
  python opensmile_compare_extractor.py --input /path/to/audio --output /path/to/output
        """
    )
    
    parser.add_argument('--input', required=True, help='Input directory containing audio files')
    parser.add_argument('--output', required=True, help='Output directory for ComParE features')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    logging.info("="*80)
    logging.info("OPENSMILE COMPARE FEATURE EXTRACTOR")
    logging.info("="*80)
    logging.info(f"Input:  {args.input}")
    logging.info(f"Output: {args.output}")
    logging.info("="*80)
    
    # Initialize openSMILE with ComParE_2016
    logging.info("Loading openSMILE ComParE_2016 model...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    logging.info("✓ Model loaded")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logging.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)
    
    # Find all audio files
    audio_files = list(input_path.rglob("*.wav"))
    logging.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        logging.warning("No audio files found. Exiting.")
        return
    
    # Process each file
    for idx, audio_file in enumerate(audio_files, 1):
        logging.info(f"\n[{idx}/{len(audio_files)}] {audio_file.name}")
        
        # Determine output subdirectory
        rel_path = audio_file.relative_to(input_path)
        output_subdir = output_path / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            process_file(audio_file, output_subdir, smile)
        except Exception as e:
            logging.error(f"ERROR processing {audio_file.name}")
            logging.error(f"  {str(e)}")
            logging.error(traceback.format_exc())
            continue
    
    elapsed = time.time() - start_time
    logging.info("="*80)
    logging.info(f"COMPLETE: {len(audio_files)} files in {elapsed/60:.1f} minutes")
    logging.info(f"Average: {elapsed/len(audio_files):.1f}s per file")
    logging.info("="*80)


if __name__ == "__main__":
    main()
