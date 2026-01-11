#!/usr/bin/env python3
"""
Librosa Per-Speaker Aggregated Feature Extractor
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
import librosa
import numpy as np

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


def extract_librosa_features(audio, sr):
    """Extract all 103 librosa features from audio segment"""
    features = {}
    
    try:
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)
        
        # 1. SPECTRAL FEATURES (48 features total)
        
        # Spectral centroid (4 features)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(centroid)
        features['spectral_centroid_std'] = np.std(centroid)
        features['spectral_centroid_min'] = np.min(centroid)
        features['spectral_centroid_max'] = np.max(centroid)
        
        # Spectral bandwidth (4 features)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)
        features['spectral_bandwidth_min'] = np.min(bandwidth)
        features['spectral_bandwidth_max'] = np.max(bandwidth)
        
        # Spectral rolloff (4 features)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        features['spectral_rolloff_std'] = np.std(rolloff)
        features['spectral_rolloff_min'] = np.min(rolloff)
        features['spectral_rolloff_max'] = np.max(rolloff)
        
        # Spectral contrast - 7 bands (28 features: 7 bands × 4 stats)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        for i in range(7):
            features[f'spectral_contrast_band{i+1}_mean'] = np.mean(contrast[i])
            features[f'spectral_contrast_band{i+1}_std'] = np.std(contrast[i])
            features[f'spectral_contrast_band{i+1}_min'] = np.min(contrast[i])
            features[f'spectral_contrast_band{i+1}_max'] = np.max(contrast[i])
        
        # Spectral flatness (4 features)
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)
        features['spectral_flatness_min'] = np.min(flatness)
        features['spectral_flatness_max'] = np.max(flatness)
        
        # Zero crossing rate (4 features)
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)
        features['zero_crossing_rate_min'] = np.min(zcr)
        features['zero_crossing_rate_max'] = np.max(zcr)
        
        # 2. MFCCS (26 features: 13 coefficients × 2 stats)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
        
        # 3. CHROMA FEATURES (24 features: 12 bins × 2 stats)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        for i in range(12):
            features[f'chroma{i+1}_mean'] = np.mean(chroma[i])
            features[f'chroma{i+1}_std'] = np.std(chroma[i])
        
        # 4. ENERGY (4 features)
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_energy_mean'] = np.mean(rms)
        features['rms_energy_std'] = np.std(rms)
        features['rms_energy_min'] = np.min(rms)
        features['rms_energy_max'] = np.max(rms)
        
        # 5. HF500 (1 feature) - ratio of energy above/below 500 Hz
        stft = np.abs(librosa.stft(y=audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        idx_500 = np.argmin(np.abs(freqs - 500))
        energy_below = np.sum(stft[:idx_500, :]**2)
        energy_above = np.sum(stft[idx_500:, :]**2)
        
        features['HF500'] = energy_above / (energy_below + 1e-10) if energy_below > 0 else np.nan
        
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
    
    # Load audio using librosa (soundfile has issues with some files)
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    audio_duration = len(audio) / sr
    logging.info(f"  Audio file: {len(audio)} samples, {audio_duration:.2f}s @ {sr} Hz")
    
    all_speakers = []
    
    # Process each speaker
    for speaker, speaker_turn_list in speaker_turns.items():
        logging.info(f"  Speaker {speaker}: {len(speaker_turn_list)} turns")
        
        # Concatenate all turns for this speaker
        speaker_audio_segments = []
        total_expected_duration = 0
        for i, turn in enumerate(speaker_turn_list):
            start_sample = int(turn['start'] * sr)
            end_sample = int(turn['end'] * sr)
            segment = audio[start_sample:end_sample]
            
            if i < 3:  # Log first 3 turns for debugging
                logging.info(f"    Turn {i+1}: start={turn['start']:.2f}s ({start_sample} samples), end={turn['end']:.2f}s ({end_sample} samples), extracted={len(segment)} samples")
            
            speaker_audio_segments.append(segment)
            total_expected_duration += turn['duration']
        
        # Concatenate all segments
        speaker_audio = np.concatenate(speaker_audio_segments)
        actual_duration = len(speaker_audio) / sr
        
        logging.info(f"    Total duration: {actual_duration:.2f}s (expected: {total_expected_duration:.2f}s)")
        logging.info(f"    Audio length: {len(speaker_audio)} samples, sr: {sr}")
        
        # Skip if total audio too short
        if len(speaker_audio) < sr * 0.5:  # Less than 0.5 seconds total
            logging.warning(f"  Speaker {speaker} total audio too short, skipping")
            continue
        
        try:
            features = extract_librosa_features(speaker_audio, sr)
            
            if features:
                features['filename'] = base_name
                features['speaker'] = speaker
                features['total_duration'] = len(speaker_audio) / sr
                features['num_turns'] = len(speaker_turn_list)
                all_speakers.append(features)
        
        except Exception as e:
            logging.error(f"  Speaker {speaker} failed: {e}")
            continue
    
    return all_speakers


def main():
    parser = argparse.ArgumentParser(description='Extract aggregated Librosa features per speaker')
    parser.add_argument('--input', required=True, help='Input directory with WAV files')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("LIBROSA PER-SPEAKER AGGREGATED EXTRACTOR")
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
        
        # Round to 3 decimal places
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(3)
        
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
