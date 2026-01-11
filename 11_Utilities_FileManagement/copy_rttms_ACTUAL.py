#!/usr/bin/env python3
"""
RTTM Copy Script - ACTUAL COPY MODE
Matches WAV files in otherAudio/ with RTTM files in Transcripts-Gemeps-rttm/
and copies RTTMs to sit next to their corresponding WAVs.

THIS WILL ACTUALLY COPY FILES.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import datetime

# Configuration
AUDIO_ROOT = Path("/path/to/user/Desktop/AudioThreeTest/otherAudio")
RTTM_ROOT = Path("/path/to/user/Desktop/AudioThreeTest/Transcripts-Gemeps-rttm")
LOG_FILE = Path("/path/to/user/Desktop/AudioThreeTest/rttm_copy_ACTUAL_log.txt")

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

def get_cohort_from_wav_path(wav_path, audio_root):
    """Extract cohort identifier from WAV path."""
    try:
        rel_path = wav_path.relative_to(audio_root)
        return rel_path.parts[0] if rel_path.parts else None
    except:
        return None

def get_cohort_mapping():
    """Map audio cohort names to RTTM cohort folder names."""
    return {
        '004': 'CSA-004',
        'CSA-Clinical': 'CSA-clinic',
        'CSA-Research': 'CSA-research-X-section',
        'Clinic_Plus_segmented': 'clinic_PLUS1',
        'FUsegmented': 'CSA-research-long',
        'LIIA': 'LIIA'
    }

def choose_best_rttm(rttm_list, wav_path, audio_root):
    """
    Choose the best RTTM from multiple matches.
    Strategy:
    1. Prefer RTTM from matching cohort folder
    2. Exclude paths with "Duplicate" folder
    3. If still multiple, take the first one
    Returns: (chosen_rttm, reason)
    """
    # Get cohort from WAV path
    wav_cohort = get_cohort_from_wav_path(wav_path, audio_root)
    cohort_mapping = get_cohort_mapping()
    expected_rttm_cohort = cohort_mapping.get(wav_cohort)
    
    # Try to match by cohort first
    if expected_rttm_cohort:
        cohort_matches = [r for r in rttm_list if expected_rttm_cohort in str(r)]
        if len(cohort_matches) == 1:
            return cohort_matches[0], f"matched_cohort_{expected_rttm_cohort}"
        elif len(cohort_matches) > 1:
            # Multiple in same cohort, filter duplicates
            non_dup = [r for r in cohort_matches if "Duplicate" not in str(r)]
            if non_dup:
                return non_dup[0], f"matched_cohort_{expected_rttm_cohort}_excluded_duplicates"
            return cohort_matches[0], f"matched_cohort_{expected_rttm_cohort}_first"
    
    # Fallback: filter out Duplicates folder
    non_duplicates = [r for r in rttm_list if "Duplicate" not in str(r)]
    
    if len(non_duplicates) == 1:
        return non_duplicates[0], "excluded_duplicates_folder"
    elif len(non_duplicates) > 1:
        return non_duplicates[0], "first_after_excluding_duplicates"
    else:
        # All were in Duplicates folder, just take first
        return rttm_list[0], "first_from_duplicates_folder"

def main():
    print("="*80)
    print("RTTM COPY SCRIPT - ACTUAL COPY MODE")
    print("="*80)
    print(f"Started: {datetime.datetime.now()}\n")
    print("⚠️  THIS WILL ACTUALLY COPY FILES ⚠️\n")
    
    # Build RTTM index
    print("Step 1: Indexing all RTTM files...")
    rttm_index = find_all_rttms(RTTM_ROOT)
    print(f"Indexed {len(rttm_index)} unique base filenames\n")
    
    # Find all WAV files
    print("Step 2: Finding all WAV files...")
    wav_files = find_all_wavs(AUDIO_ROOT)
    print()
    
    # Match and copy
    print("Step 3: Matching and copying RTTMs...\n")
    
    copied = []
    skipped_already_exists = []
    skipped_no_match = []
    multiples_handled = []
    errors = []
    
    for i, wav_path in enumerate(wav_files, 1):
        if i % 100 == 0:
            print(f"Processing {i}/{len(wav_files)}...")
        
        base = get_base_filename(wav_path)
        
        # No RTTM found
        if base not in rttm_index:
            skipped_no_match.append((wav_path, base))
            continue
        
        # Choose RTTM (handle singles and multiples)
        rttm_list = rttm_index[base]
        if len(rttm_list) == 1:
            rttm_path = rttm_list[0]
            reason = "single_match"
        else:
            rttm_path, reason = choose_best_rttm(rttm_list, wav_path, AUDIO_ROOT)
            multiples_handled.append((wav_path, base, rttm_list, rttm_path, reason))
        
        # Target path
        target_path = wav_path.parent / rttm_path.name
        
        # Check if already exists
        if target_path.exists():
            skipped_already_exists.append((wav_path, rttm_path, target_path))
            continue
        
        # Copy the file
        try:
            shutil.copy2(rttm_path, target_path)
            copied.append((wav_path, rttm_path, target_path))
        except Exception as e:
            errors.append((wav_path, rttm_path, target_path, str(e)))
    
    print(f"\nProcessing complete!\n")
    
    # Write log file
    print(f"Writing detailed log to: {LOG_FILE}\n")
    with open(LOG_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RTTM COPY SCRIPT - ACTUAL COPY LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Audio Root: {AUDIO_ROOT}\n")
        f.write(f"RTTM Root: {RTTM_ROOT}\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total WAV files processed: {len(wav_files)}\n")
        f.write(f"✅ Successfully copied: {len(copied)}\n")
        f.write(f"⏭️  Already existed (skipped): {len(skipped_already_exists)}\n")
        f.write(f"❌ No RTTM found (skipped): {len(skipped_no_match)}\n")
        f.write(f"🔀 Multiple RTTMs (handled): {len(multiples_handled)}\n")
        f.write(f"❗ Errors: {len(errors)}\n\n")
        
        # Successfully copied
        f.write("="*80 + "\n")
        f.write("SUCCESSFULLY COPIED\n")
        f.write("="*80 + "\n\n")
        for wav, rttm, target in copied:
            f.write(f"WAV:    {wav}\n")
            f.write(f"RTTM:   {rttm}\n")
            f.write(f"COPIED: {target}\n")
            f.write("-"*80 + "\n")
        
        # Already existed
        if skipped_already_exists:
            f.write("\n" + "="*80 + "\n")
            f.write("ALREADY EXISTS (SKIPPED)\n")
            f.write("="*80 + "\n\n")
            for wav, rttm, target in skipped_already_exists:
                f.write(f"Target already exists: {target}\n")
                f.write(f"WAV:  {wav}\n")
                f.write(f"RTTM: {rttm}\n")
                f.write("-"*80 + "\n")
        
        # No matches
        if skipped_no_match:
            f.write("\n" + "="*80 + "\n")
            f.write("NO RTTM FOUND (SKIPPED)\n")
            f.write("="*80 + "\n\n")
            for wav, base in skipped_no_match:
                f.write(f"Base: {base}\n")
                f.write(f"WAV:  {wav}\n")
                f.write("-"*80 + "\n")
        
        # Multiples handled
        if multiples_handled:
            f.write("\n" + "="*80 + "\n")
            f.write("MULTIPLE RTTMs FOUND (STRATEGY APPLIED)\n")
            f.write("="*80 + "\n\n")
            for wav, base, rttm_list, chosen, reason in multiples_handled:
                f.write(f"Base: {base}\n")
                f.write(f"WAV:  {wav}\n")
                f.write(f"Found {len(rttm_list)} RTTMs, chose based on: {reason}\n")
                f.write(f"CHOSEN: {chosen}\n")
                f.write(f"Others:\n")
                for rttm in rttm_list:
                    if rttm != chosen:
                        f.write(f"  - {rttm}\n")
                f.write("-"*80 + "\n")
        
        # Errors
        if errors:
            f.write("\n" + "="*80 + "\n")
            f.write("ERRORS\n")
            f.write("="*80 + "\n\n")
            for wav, rttm, target, error in errors:
                f.write(f"ERROR: {error}\n")
                f.write(f"WAV:    {wav}\n")
                f.write(f"RTTM:   {rttm}\n")
                f.write(f"TARGET: {target}\n")
                f.write("-"*80 + "\n")
    
    # Print summary to console
    print("="*80)
    print("COPY COMPLETE - SUMMARY")
    print("="*80)
    print(f"Total WAV files processed: {len(wav_files)}")
    print(f"✅ Successfully copied: {len(copied)}")
    print(f"⏭️  Already existed (skipped): {len(skipped_already_exists)}")
    print(f"❌ No RTTM found (skipped): {len(skipped_no_match)}")
    print(f"🔀 Multiple RTTMs (handled): {len(multiples_handled)}")
    print(f"❗ Errors: {len(errors)}")
    print(f"\nDetailed log written to: {LOG_FILE}")
    
    if errors:
        print("\n⚠️  WARNING: Some errors occurred. Check the log file for details.")
    else:
        print("\n✅ All operations completed successfully!")

if __name__ == "__main__":
    main()
