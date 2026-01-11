#!/usr/bin/env python3
"""
Standalone MSE/Paralang Parselmouth 0.1-Second Extractor for ASCEND Project
Dedicated high-resolution motor speech analysis script for VOX computer

Extracts Parselmouth features in 0.1-second windows for:
- MSE (Motor Speech Evaluation) tasks - all cohorts
- Paralang (Paralinguistic) tasks - CSA Research only

This script runs independently on VOX for specialized motor speech analysis
with different QC requirements and potential multiple runs for verification.

Usage:
    python mse_parselmouth_0.1sec.py --input /path/to/mse/audio --output /path/to/output
"""

import sys
import logging
import time
import traceback
import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth

# VoiceSpeechHealth imports
try:
    from voicespeechhealth.parselmouth_wrapper import (
        extract_speech_rate,
        extract_pitch_values,
        extract_pitch,
        extract_intensity,
        extract_harmonicity,
        extract_slope_tilt,
        extract_cpp,
        measure_formants,
        extract_Spectral_Moments,
    )
    VOICESPEECHHEALTH_AVAILABLE = True
except ImportError:
    logging.warning("VoiceSpeechHealth not available. Install with: pip install git+https://github.com/nickcummins41/VoiceSpeechHealth.git")
    VOICESPEECHHEALTH_AVAILABLE = False
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def is_mse_or_paralang_task(audio_file_path):
    """
    Determine if audio file is from MSE or Paralang task.
    """
    path_str = str(audio_file_path).lower()
    
    target_keywords = [
        'mse', 'motorspeech', 'motor_speech', 'motor-speech',
        'paralang', 'paralinguistic', 'para_ling'
    ]
    
    for keyword in target_keywords:
        if keyword in path_str:
            return True
    
    return False


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


def extract_parselmouth_features(audio_array, sr):
    """
    Extract all Parselmouth features for an audio segment.
    Returns dictionary of features.
    """
    if not VOICESPEECHHEALTH_AVAILABLE:
        return {}
    
    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=0)
    
    # Create Parselmouth Sound object
    try:
        snd = parselmouth.Sound(audio_array, sampling_frequency=sr)
    except Exception as e:
        logging.error(f"Failed to create Parselmouth Sound: {e}")
        return {}
    
    features = {}
    
    try:
        # Get dynamic pitch range FIRST
        pitch_vals = extract_pitch_values(snd)
        floor = pitch_vals.get('pitch_floor', 75)
        ceiling = pitch_vals.get('pitch_ceiling', 500)
        
        # Speech rate features (5 features)
        speech_rate = extract_speech_rate(snd)
        features.update(speech_rate)
        
        # Pitch features (4 features: 2 base + 2 Prosodyad)
        pitch = extract_pitch(snd, floor, ceiling, 0.005)
        features['mean_F0'] = pitch.get('mean_F0', np.nan)
        features['stdev_F0_Semitone'] = pitch.get('stdev_F0_Semitone', np.nan)
        features['F0_floor'] = pitch.get('min_F0', np.nan)
        features['F0_max'] = pitch.get('max_F0', np.nan)
        
        # Intensity features (5 features: 2 base + 3 Prosodyad)
        intensity = extract_intensity(snd, floor, 0.005)
        features['mean_dB'] = intensity.get('mean_dB', np.nan)
        features['range_dB_ratio'] = intensity.get('range_dB_ratio', np.nan)
        features['Intensity_min'] = intensity.get('min_dB', np.nan)
        features['Intensity_max'] = intensity.get('max_dB', np.nan)
        features['Intensity_SD'] = intensity.get('stdev_dB', np.nan)
        
        # Quality features (4 features)
        harmonicity = extract_harmonicity(snd, floor, 0.005)
        features['HNR_dB'] = harmonicity.get('HNR_dB', np.nan)
        
        slope_tilt = extract_slope_tilt(snd, floor, ceiling)
        features['Spectral_Slope'] = slope_tilt.get('spectral_slope', np.nan)
        features['Spectral_Tilt'] = slope_tilt.get('spectral_tilt', np.nan)
        
        cpp = extract_cpp(snd, floor, ceiling, 0.005)
        features['Cepstral_Peak_Prominence'] = cpp.get('CPP', np.nan)
        
        # Formant features (8 features)
        formants = measure_formants(snd, floor, ceiling, 0.005)
        features.update({
            'F1_mean': formants.get('F1_mean', np.nan),
            'F1_Std': formants.get('F1_std', np.nan),
            'B1_mean': formants.get('B1_mean', np.nan),
            'B1_Std': formants.get('B1_std', np.nan),
            'F2_mean': formants.get('F2_mean', np.nan),
            'F2_Std': formants.get('F2_std', np.nan),
            'B2_mean': formants.get('B2_mean', np.nan),
            'B2_Std': formants.get('B2_std', np.nan),
        })
        
        # Spectral moments (4 features)
        spectral = extract_Spectral_Moments(snd, floor, ceiling, 0.025, 0.005)
        features.update({
            'Spectral_Gravity': spectral.get('spectral_cog', np.nan),
            'Spectral_Std_Dev': spectral.get('spectral_std', np.nan),
            'Spectral_Skewness': spectral.get('spectral_skewness', np.nan),
            'Spectral_Kurtosis': spectral.get('spectral_kurtosis', np.nan),
        })
        
        # HF500 (1 feature) - ratio of energy above/below 500 Hz
        try:
            spectrum = snd.to_spectrum()
            freqs = spectrum.xs()
            power = spectrum.values[0]
            
            idx_500 = np.argmin(np.abs(freqs - 500))
            energy_below = np.sum(power[:idx_500])
            energy_above = np.sum(power[idx_500:])
            
            features['HF500'] = energy_above / (energy_below + 1e-10) if energy_below > 0 else np.nan
        except Exception as e:
            logging.warning(f"Failed to calculate HF500: {e}")
            features['HF500'] = np.nan
        
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        return {}
    
    return features


def extract_mse_granular(audio_file, rttm_file, output_dir, base_name):
    """
    Extract 0.1-second granular features for MSE/Paralang tasks.
    """
    logging.info(f"  Extracting 0.1-second granular features")
    
    # Load audio
    audio, sr = sf.read(audio_file)
    
    # Load diarization
    diarization = load_rttm(rttm_file)
    
    if diarization.empty:
        logging.warning(f"  No diarization segments found")
        return
    
    # Group by speaker
    speakers = diarization['speaker'].unique()
    
    window_size = 0.1  # 0.1 second windows
    window_samples = int(window_size * sr)
    
    for speaker in speakers:
        output_file = output_dir / f"{base_name}_speaker{speaker}_parselmouth_0.1sec.csv"
        
        # Check if already exists (resume capability)
        if output_file.exists() and output_file.stat().st_size > 0:
            logging.info(f"    ✓ Speaker {speaker}: Already exists, skipping")
            continue
        
        speaker_segments = diarization[diarization['speaker'] == speaker]
        
        total_duration = speaker_segments['end'].sum() - speaker_segments['start'].sum()
        estimated_windows = int(total_duration / window_size)
        
        logging.info(f"    Speaker {speaker}: {len(speaker_segments)} segments, ~{estimated_windows} windows")
        
        # Extract features in 0.1s windows
        all_windows = []
        
        for seg_idx, (_, seg) in enumerate(speaker_segments.iterrows()):
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            
            if start_sample >= end_sample or end_sample > len(audio):
                continue
            
            # Slide 0.1s windows across segment with no overlap
            for window_start in range(start_sample, end_sample - window_samples, window_samples):
                window_end = window_start + window_samples
                
                audio_window = audio[window_start:window_end]
                
                features = extract_parselmouth_features(audio_window, sr)
                
                if features:
                    # Add window metadata
                    features['segment_id'] = seg_idx
                    features['window_start'] = window_start / sr
                    features['window_end'] = window_end / sr
                    features['window_center'] = (window_start + window_end) / (2 * sr)
                    features['window_duration'] = window_size
                    
                    all_windows.append(features)
        
        if not all_windows:
            logging.warning(f"    Speaker {speaker}: No valid windows extracted")
            continue
        
        # Save as CSV
        df = pd.DataFrame(all_windows)
        
        # Reorder columns: metadata first, then features
        metadata_cols = ['segment_id', 'window_start', 'window_center', 'window_end', 'window_duration']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        df = df[metadata_cols + feature_cols]
        
        df.to_csv(output_file, index=False)
        logging.info(f"    ✓ Speaker {speaker}: Saved {len(all_windows)} windows ({len(feature_cols)} features)")


def process_file(audio_file, output_dir):
    """Process single MSE/Paralang audio file."""
    base_name = audio_file.stem
    
    # Check for required inputs
    rttm_file = output_dir / f"{base_name}.rttm"
    
    if not rttm_file.exists():
        logging.warning(f"  Missing diarization: {rttm_file.name}")
        logging.warning(f"  Expected location: {rttm_file}")
        return
    
    # Verify this is MSE or Paralang task
    if not is_mse_or_paralang_task(audio_file):
        logging.info(f"  SKIP: Not an MSE or Paralang task")
        return
    
    # Extract granular features
    extract_mse_granular(audio_file, rttm_file, output_dir, base_name)


def main():
    parser = argparse.ArgumentParser(
        description='Extract 0.1-second Parselmouth features for MSE and Paralang tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This standalone script is designed for VOX computer processing of:
  - MSE (Motor Speech Evaluation) tasks - all cohorts
  - Paralang (Paralinguistic) tasks - CSA Research only

Extracts high-resolution (0.1-second) acoustic features for detailed
motor speech analysis and dysarthria detection.

Example:
  python mse_parselmouth_0.1sec.py --input /path/to/mse/audio --output /path/to/output
        """
    )
    
    parser.add_argument('--input', required=True, help='Input directory containing MSE/Paralang audio files')
    parser.add_argument('--output', required=True, help='Output directory for 0.1-second features')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    logging.info("="*80)
    logging.info("MSE/PARALANG PARSELMOUTH 0.1-SECOND EXTRACTOR")
    logging.info("="*80)
    logging.info(f"Input:  {args.input}")
    logging.info(f"Output: {args.output}")
    logging.info("="*80)
    
    if not VOICESPEECHHEALTH_AVAILABLE:
        logging.error("VoiceSpeechHealth not available. Exiting.")
        sys.exit(1)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logging.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = list(input_path.rglob("*.wav"))
    logging.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        logging.warning("No audio files found. Exiting.")
        return
    
    # Filter to MSE/Paralang tasks
    mse_paralang_files = [f for f in audio_files if is_mse_or_paralang_task(f)]
    logging.info(f"Found {len(mse_paralang_files)} MSE/Paralang task files")
    
    if not mse_paralang_files:
        logging.warning("No MSE or Paralang tasks found. Exiting.")
        return
    
    # Process each file
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, audio_file in enumerate(mse_paralang_files, 1):
        logging.info(f"\n[{idx}/{len(mse_paralang_files)}] {audio_file.name}")
        
        # Determine output subdirectory
        rel_path = audio_file.relative_to(input_path)
        output_subdir = output_path / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            process_file(audio_file, output_subdir)
            processed += 1
        except Exception as e:
            logging.error(f"ERROR processing {audio_file.name}")
            logging.error(f"  {str(e)}")
            logging.error(traceback.format_exc())
            errors += 1
            continue
    
    elapsed = time.time() - start_time
    
    logging.info("="*80)
    logging.info("PROCESSING SUMMARY")
    logging.info("="*80)
    logging.info(f"Total files found:     {len(audio_files)}")
    logging.info(f"MSE/Paralang files:    {len(mse_paralang_files)}")
    logging.info(f"Successfully processed: {processed}")
    logging.info(f"Errors:                {errors}")
    logging.info(f"Total time:            {elapsed/60:.1f} minutes")
    if processed > 0:
        logging.info(f"Average per file:      {elapsed/processed:.1f} seconds")
    logging.info("="*80)


if __name__ == "__main__":
    main()
