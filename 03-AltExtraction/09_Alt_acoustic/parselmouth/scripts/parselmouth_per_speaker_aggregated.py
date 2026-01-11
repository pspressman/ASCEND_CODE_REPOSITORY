#!/usr/bin/env python3
"""
Parselmouth Per-Speaker Aggregated Feature Extractor
Aggregates features across all turns for each speaker in each file
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


def extract_speaker_features(snd, pitch_floor, pitch_ceiling):
    """Extract all features from aggregated speaker audio"""
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


def process_audio_file(audio_path, rttm_path):
    """Process single audio file, return list of speaker feature dicts"""
    
    base_name = Path(audio_path).stem
    logging.info(f"Processing: {base_name}")
    
    # Parse RTTM file
    turns = parse_rttm(rttm_path)
    
    # Group turns by speaker
    speaker_turns = {}
    for turn in turns:
        speaker = turn['speaker']
        if speaker not in speaker_turns:
            speaker_turns[speaker] = []
        speaker_turns[speaker].append(turn)
    
    logging.info(f"  Found {len(speaker_turns)} speakers")
    
    # Load audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)
    
    all_speakers = []
    temp_wav = '/tmp/temp_speaker.wav'
    
    # Process each speaker
    for speaker, speaker_turn_list in speaker_turns.items():
        logging.info(f"  Speaker {speaker}: {len(speaker_turn_list)} turns")
        
        # Concatenate all turns for this speaker
        speaker_audio_segments = []
        for turn in speaker_turn_list:
            start_sample = int(turn['start'] * sr)
            end_sample = int(turn['end'] * sr)
            speaker_audio_segments.append(audio[start_sample:end_sample])
        
        # Concatenate all segments
        import numpy as np
        speaker_audio = np.concatenate(speaker_audio_segments)
        
        # Skip if total audio too short
        if len(speaker_audio) < sr * 0.5:  # Less than 0.5 seconds total
            logging.warning(f"  Speaker {speaker} total audio too short, skipping")
            continue
        
        sf.write(temp_wav, speaker_audio, sr)
        
        try:
            snd = load_audio_file(temp_wav)
            pitch_values = extract_pitch_values(snd)
            pitch_floor = pitch_values.get("pitch_floor", 75)
            pitch_ceiling = pitch_values.get("pitch_ceiling", 500)
            
            features = extract_speaker_features(snd, pitch_floor, pitch_ceiling)
            
            if features:
                features['filename'] = base_name
                features['speaker'] = speaker
                features['total_duration'] = len(speaker_audio) / sr
                features['num_turns'] = len(speaker_turn_list)
                all_speakers.append(features)
        
        except Exception as e:
            logging.error(f"  Speaker {speaker} failed: {e}")
            continue
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    
    return all_speakers


def main():
    parser = argparse.ArgumentParser(description='Extract aggregated Parselmouth features per speaker')
    parser.add_argument('--input', required=True, help='Input directory with WAV files')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("PARSELMOUTH PER-SPEAKER AGGREGATED EXTRACTOR")
    logging.info("="*80)
    
    input_path = Path(args.input)
    output_file = Path(args.output)
    
    logging.info(f"Input:  {input_path.absolute()}")
    logging.info(f"Output: {output_file.absolute()}")
    
    if not input_path.exists():
        logging.error(f"ERROR: Input directory does not exist: {input_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Find WAV files
    wav_files = list(input_path.rglob("*.wav"))
    logging.info(f"Found {len(wav_files)} WAV files")
    
    if not wav_files:
        logging.error(f"ERROR: No WAV files found in {input_path}")
        sys.exit(1)
    
    logging.info("="*80)
    
    # Process all files
    all_results = []
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
            results = process_audio_file(wav_file, rttm_file)
            all_results.extend(results)
            processed += 1
        except Exception as e:
            logging.error(f"ERROR processing {wav_file.name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            continue
    
    # Save all results to single CSV
    if all_results:
        df = pd.DataFrame(all_results)
        metadata_cols = ['filename', 'speaker', 'total_duration', 'num_turns']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        df = df[metadata_cols + feature_cols]
        
        df.to_csv(output_file, index=False)
        logging.info("="*80)
        logging.info(f"✅ Complete! Processed {processed}/{len(wav_files)} files ({skipped} skipped)")
        logging.info(f"📊 Total speakers extracted: {len(all_results)}")
        logging.info(f"📁 Output: {output_file}")
        logging.info("="*80)
    else:
        logging.error("="*80)
        logging.error("❌ No results to save")
        logging.error("="*80)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
