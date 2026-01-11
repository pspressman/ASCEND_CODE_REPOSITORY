#!/usr/bin/env python3
"""
Parselmouth 0.1-Second Feature Extractor
Based on VoiceSpeechHealth_deploy2.py
"""

print("SCRIPT STARTED")

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


def extract_window_features(snd, pitch_floor, pitch_ceiling):
    """Extract all features from a sound window"""
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


def process_audio_file(audio_path, output_dir):
    """Process single audio file in 0.1-second windows"""
    
    base_name = Path(audio_path).stem
    output_file = output_dir / f"{base_name}_parselmouth_0.1sec.csv"
    
    logging.info(f"Processing: {base_name}")
    
    if output_file.exists() and output_file.stat().st_size > 0:
        logging.info(f"  ✓ Already exists, skipping")
        return
    
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)
    
    duration = len(audio) / sr
    window_size = 0.1
    window_samples = int(window_size * sr)
    estimated_windows = int(duration / window_size)
    
    logging.info(f"  Duration: {duration:.1f}s, ~{estimated_windows} windows")
    
    all_windows = []
    temp_wav = '/tmp/temp_window.wav'
    
    for window_start in range(0, len(audio) - window_samples, window_samples):
        window_end = window_start + window_samples
        window_audio = audio[window_start:window_end]
        
        sf.write(temp_wav, window_audio, sr)
        
        try:
            snd = load_audio_file(temp_wav)
            pitch_values = extract_pitch_values(snd)
            pitch_floor = pitch_values.get("pitch_floor", 75)
            pitch_ceiling = pitch_values.get("pitch_ceiling", 500)
            
            features = extract_window_features(snd, pitch_floor, pitch_ceiling)
            
            if features:
                features['window_start'] = window_start / sr
                features['window_end'] = window_end / sr
                features['window_center'] = (window_start + window_end) / (2 * sr)
                features['window_duration'] = window_size
                all_windows.append(features)
        
        except Exception as e:
            logging.debug(f"  Window at {window_start/sr:.1f}s failed: {e}")
            continue
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    
    if not all_windows:
        logging.warning(f"  No valid windows extracted")
        return
    
    df = pd.DataFrame(all_windows)
    metadata_cols = ['window_start', 'window_center', 'window_end', 'window_duration']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + feature_cols]
    
    df.to_csv(output_file, index=False)
    logging.info(f"  ✓ Saved {len(all_windows)} windows ({len(feature_cols)} features)")


def main():
    parser = argparse.ArgumentParser(description='Extract Parselmouth features in 0.1-second windows')
    parser.add_argument('--input', required=True, help='Input directory with WAV files')
    parser.add_argument('--output', required=True, help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("PARSELMOUTH 0.1-SECOND EXTRACTOR")
    logging.info("="*80)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    logging.info(f"Input:  {input_path.absolute()}")
    logging.info(f"Output: {output_path.absolute()}")
    logging.info(f"Input exists: {input_path.exists()}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(input_path.glob("*.wav"))
    logging.info(f"Found {len(wav_files)} WAV files")
    
    if not wav_files:
        logging.error(f"No WAV files found in {input_path}")
        sys.exit(1)
    
    for wf in wav_files:
        logging.info(f"  - {wf.name}")
    
    logging.info("="*80)
    
    for i, wav_file in enumerate(wav_files, 1):
        logging.info(f"\n[{i}/{len(wav_files)}]")
        try:
            process_audio_file(wav_file, output_path)
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
