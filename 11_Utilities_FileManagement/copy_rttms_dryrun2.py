#!/usr/bin/env python3
"""
RTTM Copy Script - DRY RUN MODE
Matches WAV files in otherAudio/ with RTTM files in Transcripts-Gemeps-rttm/
and copies RTTMs to sit next to their corresponding WAVs.

This is a DRY RUN - no files will be copied, only logged.
"""

import os
from pathlib import Path
from collections import defaultdict
import datetime

# Configuration
AUDIO_ROOT = Path("/path/to/user/Desktop/AudioThreeTest/otherAudio")
RTTM_ROOT = Path("/path/to/user/Desktop/AudioThreeTest/Transcripts-Gemeps-rttm")
LOG_FILE = Path("/path/to/user/Desktop/AudioThreeTest/rttm_copy_log.txt")

def get_base_filename(filepath):
    """
    Extract base filename without extension or suffix.
    Strips: .wav, .rttm, _transcript
    """
    name = filepath.stem
    # Remove _transcript suffix if present
    if name.endswith('_transcript'):
        name = name[:-11]  # Remove '_transcript'
    return name

def find_all_rttms(rttm_root):
    """
    Recursively find all RTTM files and index them by base filename.
    Returns dict: {base_filename: [list of full paths]}
    """
    rttm_index = defaultdict(list)
    
    print(f"Scanning for RTTM files in: {rttm_root}")
    rttm_files = list(rttm_root.rglob("*.rttm"))
    print(f"Found {len(rttm_files)} RTTM files total")
    
    for rttm_path in rttm_files:
        base = get_base_filename(rttm_path)
        rttm_index[base].append(rttm_path)
    
    return rttm_index

def find_all_wavs(audio_root):
    """
    Recursively find all WAV files.
    Returns list of Path objects.
    """
    print(f"Scanning for WAV files in: {audio_root}")
    wav_files = list(audio_root.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files total")
    return wav_files

def main():
    print("="*80)
    print("RTTM COPY SCRIPT - DRY RUN MODE")
    print("="*80)
    print(f"Started: {datetime.datetime.now()}\n")
    
    # Build RTTM index
    print("Step 1: Indexing all RTTM files...")
    rttm_index = find_all_rttms(RTTM_ROOT)
    print(f"Indexed {len(rttm_index)} unique base filenames\n")
    
    # Find all WAV files
    print("Step 2: Finding all WAV files...")
    wav_files = find_all_wavs(AUDIO_ROOT)
    print()
    
    # Match and log
    print("Step 3: Matching WAVs to RTTMs...\n")
    
    matches = []
    no_matches = []
    multiple_matches = []
    
    for wav_path in wav_files:
        base = get_base_filename(wav_path)
        
        if base not in rttm_index:
            no_matches.append((wav_path, base))
        elif len(rttm_index[base]) > 1:
            multiple_matches.append((wav_path, base, rttm_index[base]))
        else:
            rttm_path = rttm_index[base][0]
            target_path = wav_path.parent / rttm_path.name
            matches.append((wav_path, rttm_path, target_path))
    
    # Write log file
    print(f"Writing log to: {LOG_FILE}\n")
    with open(LOG_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RTTM COPY SCRIPT - DRY RUN LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Audio Root: {AUDIO_ROOT}\n")
        f.write(f"RTTM Root: {RTTM_ROOT}\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total WAV files found: {len(wav_files)}\n")
        f.write(f"Successful matches: {len(matches)}\n")
        f.write(f"No RTTM found: {len(no_matches)}\n")
        f.write(f"Multiple RTTMs found: {len(multiple_matches)}\n\n")
        
        # Successful matches
        f.write("="*80 + "\n")
        f.write("SUCCESSFUL MATCHES (WOULD BE COPIED)\n")
        f.write("="*80 + "\n\n")
        for wav, rttm, target in matches:
            f.write(f"WAV:    {wav}\n")
            f.write(f"RTTM:   {rttm}\n")
            f.write(f"TARGET: {target}\n")
            f.write(f"EXISTS: {target.exists()}\n")
            f.write("-"*80 + "\n")
        
        # No matches
        if no_matches:
            f.write("\n" + "="*80 + "\n")
            f.write("WAV FILES WITH NO MATCHING RTTM\n")
            f.write("="*80 + "\n\n")
            for wav, base in no_matches:
                f.write(f"Base: {base}\n")
                f.write(f"WAV:  {wav}\n")
                f.write("-"*80 + "\n")
        
        # Multiple matches
        if multiple_matches:
            f.write("\n" + "="*80 + "\n")
            f.write("WAV FILES WITH MULTIPLE MATCHING RTTMS (NEEDS REVIEW)\n")
            f.write("="*80 + "\n\n")
            for wav, base, rttm_list in multiple_matches:
                f.write(f"Base: {base}\n")
                f.write(f"WAV:  {wav}\n")
                f.write(f"Found {len(rttm_list)} RTTMs:\n")
                for rttm in rttm_list:
                    f.write(f"  - {rttm}\n")
                f.write("-"*80 + "\n")
    
    # Print summary to console
    print("="*80)
    print("DRY RUN COMPLETE - SUMMARY")
    print("="*80)
    print(f"Total WAV files: {len(wav_files)}")
    print(f"✅ Successful matches: {len(matches)}")
    print(f"❌ No RTTM found: {len(no_matches)}")
    print(f"⚠️  Multiple RTTMs found: {len(multiple_matches)}")
    print(f"\nDetailed log written to: {LOG_FILE}")
    print("\nThis was a DRY RUN - no files were copied.")
    print("Review the log file before running the actual copy.")

if __name__ == "__main__":
    main()
