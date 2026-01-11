#!/usr/bin/env python3
"""
In-Place PHI De-identification Script (MASTER CSV VERSION)
Applies HYBRID de-identification to PHI regions within full-length audio recordings.

Modified to work with a SINGLE master CSV containing all participants' PHI timestamps.

Input:  - Original full-length audio files
        - MASTER PHI timestamp CSV with columns: participant_id, start_time, end_time, phi_word, wav_filename, json_source
Output: - Full-length recordings with selective PHI processing
        - Metadata CSV documenting processed segments

Usage:
    python phi_inplace_deidentifier_MASTER_CSV.py --audio_dir /path/to/audio --master_csv /path/to/phi_timestamps_master.csv --output_dir /path/to/output --method hybrid
"""

import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime
import sys

# Import the PHI deidentification components
try:
    from phi_deidentification_pipeline import PHIDeidentifier
except ImportError:
    print("ERROR: Could not import PHIDeidentifier from phi_deidentification_pipeline.py")
    print("Make sure phi_deidentification_pipeline.py is in the same directory or in PYTHONPATH")
    sys.exit(1)


class InPlacePHIDeidentifier:
    """Apply HYBRID de-identification to PHI regions within full audio files."""
    
    def __init__(self, method: str = 'hybrid'):
        """
        Initialize in-place de-identifier.
        
        Args:
            method: De-identification method ('blanket', 'surgical', or 'hybrid')
        """
        self.method = method.lower()
        self.deidentifier = None  # Will be initialized per-file with correct SR
        
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file preserving original sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            audio: Audio signal
            sr: Original sample rate
        """
        # Load with original sample rate (sr=None means preserve original)
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return audio, sr
    
    def extract_segment(self, audio: np.ndarray, sr: int, 
                       start_time: float, end_time: float) -> Tuple[np.ndarray, int, int]:
        """
        Extract a single segment from audio.
        
        Args:
            audio: Full audio signal
            sr: Sample rate
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            segment: Extracted audio segment
            start_sample: Starting sample index
            end_sample: Ending sample index
        """
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Boundary check - clip to audio duration
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        segment = audio[start_sample:end_sample]
        return segment, start_sample, end_sample
    
    def process_full_file(self, audio_path: Path, timestamps_df: pd.DataFrame, 
                         output_path: Path) -> Dict:
        """
        Process a full audio file with in-place PHI de-identification.
        
        Args:
            audio_path: Path to original audio file
            timestamps_df: DataFrame with PHI timestamps for this file
            output_path: Where to save processed audio
            
        Returns:
            metadata: Processing metadata
        """
        participant_id = audio_path.stem
        print(f"\nProcessing: {participant_id}")
        
        # Load full audio
        audio, sr = self.load_audio(audio_path)
        duration = len(audio) / sr
        print(f"  Original duration: {duration:.2f}s at {sr} Hz")
        
        # Deduplicate timestamps
        timestamps_df = timestamps_df.drop_duplicates(subset=['start_time', 'end_time'], keep='first')
        
        # Create output audio as copy of original
        output_audio = audio.copy()
        
        # Initialize deidentifier with this file's sample rate
        self.deidentifier = PHIDeidentifier(method=self.method, sr=sr)
        
        # Track processing metadata
        processed_segments = []
        total_phi_duration = 0.0
        
        print(f"  Found {len(timestamps_df)} PHI timestamp(s)")
        
        # Process each PHI timestamp
        for idx, row in timestamps_df.iterrows():
            # Skip if no valid timestamps
            if pd.isna(row['start_time']) or pd.isna(row['end_time']):
                continue
            
            start_time = float(row['start_time'])
            end_time = float(row['end_time'])
            phi_word = row.get('phi_word', 'UNKNOWN')
            
            print(f"    Processing PHI segment {idx+1}: [{start_time:.2f}s - {end_time:.2f}s] ({phi_word})")
            
            # Extract segment from original
            segment, start_sample, end_sample = self.extract_segment(
                audio, sr, start_time, end_time
            )
            
            # Apply de-identification to this segment
            try:
                processed_segment = self.deidentifier.process_audio(segment, sr)
                
                # Replace samples in output audio
                output_audio[start_sample:end_sample] = processed_segment
                
                # Record metadata
                processed_segments.append({
                    'segment_index': idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'phi_word': phi_word,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'status': 'success'
                })
                
                total_phi_duration += (end_time - start_time)
                
            except Exception as e:
                print(f"      WARNING: Failed to process segment: {e}")
                processed_segments.append({
                    'segment_index': idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'phi_word': phi_word,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save output audio preserving original format
        print(f"  Saving processed audio...")
        sf.write(output_path, output_audio, sr, subtype='PCM_16')
        
        # Calculate statistics
        num_successful = sum(1 for seg in processed_segments if seg['status'] == 'success')
        phi_percentage = (total_phi_duration / duration) * 100 if duration > 0 else 0
        
        print(f"  âœ“ Saved: {output_path.name}")
        print(f"    PHI segments processed: {num_successful}/{len(timestamps_df)}")
        print(f"    Total PHI duration: {total_phi_duration:.2f}s ({phi_percentage:.1f}% of audio)")
        print(f"    Non-PHI audio: Untouched original")
        
        # Create metadata
        metadata = {
            'participant_id': participant_id,
            'original_audio_path': str(audio_path),
            'output_audio_path': str(output_path),
            'original_duration': duration,
            'sample_rate': sr,
            'num_phi_segments': len(timestamps_df),
            'num_successful': num_successful,
            'total_phi_duration': total_phi_duration,
            'phi_percentage': phi_percentage,
            'segments': processed_segments,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return metadata


def normalize_filename(filename: str) -> str:
    """
    Normalize filename by stripping common prefixes/suffixes.
    
    Args:
        filename: Original filename
        
    Returns:
        Normalized filename
    """
    # Strip anon_ prefix
    if filename.startswith('anon_'):
        filename = filename[5:]
    
    # Strip _clean suffix (before extension)
    if filename.endswith('_clean.wav'):
        filename = filename[:-10] + '.wav'
    elif filename.endswith('_clean.WAV'):
        filename = filename[:-10] + '.WAV'
    
    return filename


def build_audio_file_index(audio_dir: Path) -> Dict[str, Path]:
    """
    Build an index of all audio files in directory and subdirectories.
    Maps ACTUAL basename (filename only) to full path - NO normalization.
    
    Args:
        audio_dir: Root directory to search
        
    Returns:
        Dictionary mapping actual_basename -> full Path
    """
    print(f"Building audio file index from: {audio_dir}")
    audio_index = {}
    
    # Search recursively for all WAV files
    wav_files = list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.WAV"))
    
    print(f"Found {len(wav_files)} WAV files in directory tree")
    
    for wav_path in wav_files:
        # Use ACTUAL filename as key (no normalization)
        actual_name = wav_path.name
        
        if actual_name in audio_index:
            print(f"  WARNING: Duplicate filename: {actual_name}")
            print(f"    Existing: {audio_index[actual_name]}")
            print(f"    New: {wav_path}")
            print(f"    Using: {wav_path} (newer entry)")
        
        audio_index[actual_name] = wav_path
    
    print(f"Indexed {len(audio_index)} unique audio files\n")
    return audio_index


def process_batch_from_master_csv(audio_dir: Path, master_csv_path: Path, 
                                  output_dir: Path, method: str = 'hybrid') -> Dict:
    """
    Process all audio files using a master CSV with all timestamps.
    
    Args:
        audio_dir: Directory containing original audio files
        master_csv_path: Path to master PHI timestamps CSV
        output_dir: Where to save processed audio files
        method: De-identification method
        
    Returns:
        summary: Processing summary statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build audio file index FIRST
    audio_index = build_audio_file_index(audio_dir)
    
    # Load master CSV
    print(f"Loading master CSV: {master_csv_path}")
    try:
        master_df = pd.read_csv(master_csv_path)
    except Exception as e:
        print(f"ERROR: Could not load master CSV: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['start_time', 'end_time', 'wav_filename']
    missing_cols = [col for col in required_cols if col not in master_df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in master CSV: {missing_cols}")
        print(f"Available columns: {list(master_df.columns)}")
        sys.exit(1)
    
    print(f"Master CSV loaded: {len(master_df)} total PHI timestamps")
    
    # Group by wav_filename to get unique files
    unique_files = master_df['wav_filename'].unique()
    print(f"Found {len(unique_files)} unique audio files with PHI timestamps")
    print(f"Using method: {method.upper()}\n")
    print("="*60)
    
    processor = InPlacePHIDeidentifier(method=method)
    
    all_metadata = []
    success_count = 0
    error_count = 0
    not_found_count = 0
    total_phi_duration = 0.0
    total_audio_duration = 0.0
    
    for wav_filename in unique_files:
        # Get timestamps for this file
        file_timestamps = master_df[master_df['wav_filename'] == wav_filename].copy()
        
        # Look up audio file in index (matches by basename)
        if wav_filename not in audio_index:
            print(f"Skipping {wav_filename}: Not found in audio file index")
            not_found_count += 1
            continue
        
        audio_path = audio_index[wav_filename]
        print(f"\nFound: {wav_filename}")
        print(f"  Location: {audio_path}")
        
        # Generate output path PRESERVING folder structure
        # Calculate relative path from audio_dir to this file
        try:
            relative_path = audio_path.relative_to(audio_dir)
        except ValueError:
            # File is not under audio_dir, just use filename
            relative_path = Path(audio_path.name)
        
        # Create output path with same subdirectory structure
        participant_id = audio_path.stem  # Remove .wav extension
        output_filename = f"{participant_id}_{method.upper()}_deidentified.wav"
        
        # Preserve subdirectory: output_dir / subdir / filename
        output_path = output_dir / relative_path.parent / output_filename
        
        # Create subdirectories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process this file
        try:
            metadata = processor.process_full_file(audio_path, file_timestamps, output_path)
            metadata['status'] = 'success'
            all_metadata.append(metadata)
            
            success_count += 1
            total_phi_duration += metadata.get('total_phi_duration', 0)
            total_audio_duration += metadata.get('original_duration', 0)
            
        except Exception as e:
            print(f"  âœ— ERROR processing {wav_filename}: {e}")
            metadata = {
                'participant_id': participant_id,
                'wav_filename': wav_filename,
                'status': 'error',
                'error': str(e),
                'original_audio_path': str(audio_path)
            }
            all_metadata.append(metadata)
            error_count += 1
    
    # Calculate overall statistics
    overall_phi_percentage = (total_phi_duration / total_audio_duration * 100) if total_audio_duration > 0 else 0
    
    # Generate summary
    summary = {
        'total_files_in_csv': len(unique_files),
        'files_not_found': not_found_count,
        'successful': success_count,
        'errors': error_count,
        'method': method.upper(),
        'total_audio_duration_minutes': total_audio_duration / 60,
        'total_phi_duration_minutes': total_phi_duration / 60,
        'overall_phi_percentage': overall_phi_percentage,
        'master_csv_path': str(master_csv_path),
        'audio_directory': str(audio_dir),
        'output_directory': str(output_dir),
        'processing_timestamp': datetime.now().isoformat(),
        'per_file_metadata': all_metadata
    }
    
    # Save summary JSON
    summary_path = output_dir / 'inplace_deidentification_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save metadata CSV for easy analysis
    csv_records = []
    for m in all_metadata:
        if m['status'] == 'success':
            csv_records.append({
                'participant_id': m['participant_id'],
                'original_duration_sec': m['original_duration'],
                'num_phi_segments': m['num_phi_segments'],
                'num_successful': m['num_successful'],
                'total_phi_duration_sec': m['total_phi_duration'],
                'phi_percentage': m['phi_percentage'],
                'sample_rate': m['sample_rate'],
                'output_path': m['output_audio_path']
            })
    
    if csv_records:
        csv_path = output_dir / 'processing_metadata.csv'
        pd.DataFrame(csv_records).to_csv(csv_path, index=False)
        print(f"\nMetadata CSV saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("IN-PLACE PHI DE-IDENTIFICATION SUMMARY")
    print("="*60)
    print(f"Files in master CSV:        {len(unique_files)}")
    print(f"Audio files not found:      {not_found_count}")
    print(f"Successfully processed:     {success_count}")
    print(f"Errors:                     {error_count}")
    print(f"Method used:                {method.upper()}")
    print(f"\nTotal audio duration:       {total_audio_duration/60:.1f} minutes")
    print(f"Total PHI duration:         {total_phi_duration/60:.1f} minutes")
    print(f"PHI percentage:             {overall_phi_percentage:.1f}%")
    print(f"Non-PHI audio:              Untouched ({100-overall_phi_percentage:.1f}%)")
    print(f"\nOutput directory:           {output_dir}")
    print(f"Summary saved to:           {summary_path}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Apply in-place PHI de-identification using MASTER CSV with all timestamps'
    )
    
    parser.add_argument('--audio_dir', type=Path, required=True,
                       help='Directory containing original audio files')
    parser.add_argument('--master_csv', type=Path, required=True,
                       help='Path to master PHI timestamps CSV (phi_timestamps_master.csv)')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Directory to save processed audio files')
    parser.add_argument('--method', type=str, default='hybrid',
                       choices=['blanket', 'surgical', 'hybrid'],
                       help='De-identification method (default: hybrid)')
    
    args = parser.parse_args()
    
    # Process all files from master CSV
    process_batch_from_master_csv(args.audio_dir, args.master_csv, 
                                  args.output_dir, args.method)


if __name__ == '__main__':
    main()