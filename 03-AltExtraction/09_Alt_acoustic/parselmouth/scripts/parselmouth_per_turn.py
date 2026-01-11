#!/usr/bin/env python3
"""
Parselmouth Per-Turn Feature Extractor
Extracts features for each speaker turn from pyannote diarization
"""

print("SCRIPT STARTED", flush=True)

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import soundfile as sf
import logging

from voicespeechhealth.audio import load_audio_file
from voicespeechhealth.parselmouth_wrapper import (
    extract_cpp, extract_harmonicity, extract_intensity,
    extract_pitch, extract_pitch_values, extract_slope_tilt,
    extract_Spectral_Moments, extract_speech_rate, measure_formants,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def parse_rttm(rttm_path):
    """Parse pyannote RTTM file to get speaker turns"""
    turns = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                turns.append({
                    'speaker': speaker,
                    'start': start_time,
                    'end': start_time + duration,
                    'duration': duration
                })
    return turns


def extract_turn_features(snd, pitch_floor, pitch_ceiling):
    """Extract all features from a speaker turn"""
    features = {}
    
    try:
        speech_rate_out = extract_speech_rate(snd)
        features['Speaking_Rate'] = speech_rate_out.get("speaking_rate")
        features['Articulation_Rate'] = speech_rate_out.get("articulation_rate")
        features['Phonation_Ratio'] = speech_rate_out.get("phonation_ratio")
        features['Pause_Rate'] = speech_rate_out.get("pause_rate")
        features['Mean_Pause_Duration'] = speech_rate_out.get("mean_pause_dur")
        
        pitch_out = extract_pitch(snd, pitch_floor, pitch_ceiling, frame_shift=0.005)
        features['mean_F0'] = pitch_out.get("mean_F0")
        features['stdev_F0_Semitone'] = pitch_out.get("stdev_F0_semitone")
        
        intensity_out = extract_intensity(snd, pitch_floor, frame_shift=0.005)
        features['mean_dB'] = intensity_out.get("mean_dB")
        features['range_ratio_dB'] = intensity_out.get("range_dB_ratio")
        
        harmonicity_out = extract_harmonicity(snd, pitch_floor, frame_shift=0.005)
        features['HNR_dB'] = harmonicity_out.get("HNR_db")
        
        slope_tilt_out = extract_slope_tilt(snd, pitch_floor, pitch_ceiling)
        features['Spectral_Slope'] = slope_tilt_out.get("spc_slope")
        features['Spectral_Tilt'] = slope_tilt_out.get("spc_tilt")
        
        cpp_out = extract_cpp(snd, pitch_floor, pitch_ceiling, frame_shift=0.005)
        features['Cepstral_Peak_Prominence'] = cpp_out.get("mean_cpp")
        
        formants_out = measure_formants(snd, pitch_floor, pitch_ceiling, frame_shift=0.005)
        features['mean_F1_Loc'] = formants_out.get("F1_mean")
        features['std_F1_Loc'] = formants_out.get("F1_Std")
        features['mean_B1_Loc'] = formants_out.get("B1_mean")
        features['std_B1_Loc'] = formants_out.get("B1_Std")
        features['mean_F2_Loc'] = formants_out.get("F2_mean")
        features['std_F2_Loc'] = formants_out.get("F2_Std")
        features['mean_B2_Loc'] = formants_out.get("B2_mean")
        features['std_B2_Loc'] = formants_out.get("B2_Std")
        
        spectral_moments_out = extract_Spectral_Moments(
            snd, pitch_floor, pitch_ceiling,
            window_size=0.025, frame_shift=0.005
        )
        features['Spectral_Gravity'] = spectral_moments_out.get("spc_gravity")
        features['Spectral_Std_Dev'] = spectral_moments_out.get("spc_std_dev")
        features['Spectral_Skewness'] = spectral_moments_out.get("spc_skewness")
        features['Spectral_Kurtosis'] = spectral_moments_out.get("spc_kurtosis")
        
    except Exception as e:
        logging.debug(f"Feature extraction warning: {e}")
        return None
    
    return features


def process_audio_file(audio_path, rttm_path, output_dir, input_base):
    """Process single audio file using speaker turns from RTTM"""
    
    base_name = Path(audio_path).stem
    
    # Create mirrored directory structure
    relative_path = audio_path.parent.relative_to(input_base)
    output_subdir = output_dir / relative_path
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_subdir / f"{base_name}_parselmouth_per_turn.csv"
    
    logging.info(f"Processing: {base_name}")
    logging.info(f"  Output: {relative_path}")
    
    if output_file.exists() and output_file.stat().st_size > 0:
        logging.info(f"  ✓ Already exists, skipping")
        return
    
    # Parse RTTM file
    turns = parse_rttm(rttm_path)
    logging.info(f"  Found {len(turns)} speaker turns")
    
    # Load audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)
    
    all_turns = []
    temp_wav = '/tmp/temp_turn.wav'
    
    for i, turn in enumerate(turns):
        start_sample = int(turn['start'] * sr)
        end_sample = int(turn['end'] * sr)
        turn_audio = audio[start_sample:end_sample]
        
        # Skip very short turns
        if len(turn_audio) < sr * 0.1:  # Less than 0.1 seconds
            logging.debug(f"  Turn {i+1} too short ({turn['duration']:.2f}s), skipping")
            continue
        
        sf.write(temp_wav, turn_audio, sr)
        
        try:
            snd = load_audio_file(temp_wav)
            pitch_values = extract_pitch_values(snd)
            pitch_floor = pitch_values.get("pitch_floor", 75)
            pitch_ceiling = pitch_values.get("pitch_ceiling", 500)
            
            features = extract_turn_features(snd, pitch_floor, pitch_ceiling)
            
            if features:
                features['speaker'] = turn['speaker']
                features['turn_start'] = turn['start']
                features['turn_end'] = turn['end']
                features['turn_duration'] = turn['duration']
                all_turns.append(features)
        
        except Exception as e:
            logging.debug(f"  Turn {i+1} at {turn['start']:.1f}s failed: {e}")
            continue
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    
    if not all_turns:
        logging.warning(f"  No valid turns extracted")
        return
    
    # Create DataFrame with metadata columns first
    df = pd.DataFrame(all_turns)
    metadata_cols = ['speaker', 'turn_start', 'turn_end', 'turn_duration']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + feature_cols]
    
    df.to_csv(output_file, index=False)
    logging.info(f"  ✓ Saved {len(all_turns)} turns ({len(feature_cols)} features)")


def main():
    parser = argparse.ArgumentParser(description='Extract Parselmouth features per speaker turn')
    parser.add_argument('--input', required=True, help='Input directory with WAV files')
    parser.add_argument('--output', required=True, help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("PARSELMOUTH PER-TURN EXTRACTOR")
    logging.info("="*80)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    logging.info(f"Input:  {input_path.absolute()}")
    logging.info(f"Output: {output_path.absolute()}")
    
    if not input_path.exists():
        logging.error(f"ERROR: Input directory does not exist: {input_path}")
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find WAV files recursively
    wav_files = list(input_path.rglob("*.wav"))
    logging.info(f"Found {len(wav_files)} WAV files")
    
    if not wav_files:
        logging.error(f"ERROR: No WAV files found in {input_path}")
        sys.exit(1)
    
    logging.info("="*80)
    
    # Process each file
    processed = 0
    skipped = 0
    
    for i, wav_file in enumerate(wav_files, 1):
        # Look for matching RTTM file
        rttm_file = wav_file.with_suffix('.rttm')
        
        if not rttm_file.exists():
            logging.warning(f"[{i}/{len(wav_files)}] {wav_file.name}: No RTTM file, skipping")
            skipped += 1
            continue
        
        logging.info(f"\n[{i}/{len(wav_files)}]")
        try:
            process_audio_file(wav_file, rttm_file, output_path, input_path)
            processed += 1
        except Exception as e:
            logging.error(f"ERROR processing {wav_file.name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            continue
    
    logging.info("="*80)
    logging.info(f"✅ Complete! Processed {processed}/{len(wav_files)} files ({skipped} skipped)")
    logging.info(f"📁 Output: {output_path}")
    logging.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
