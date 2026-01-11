#!/usr/bin/env python3
"""
Quick diagnostic: Show samples of WAV files with no matching RTTM
"""

import os
from pathlib import Path
from collections import defaultdict, Counter

# Configuration
AUDIO_ROOT = Path("/path/to/user/Desktop/AudioThreeTest/otherAudio")
RTTM_ROOT = Path("/path/to/user/Desktop/AudioThreeTest/Transcripts-Gemeps-rttm")

def get_base_filename(filepath):
    """Extract base filename without extension or suffix."""
    name = filepath.stem
    if name.endswith('_transcript'):
        name = name[:-11]
    return name

def get_cohort_and_task(wav_path, audio_root):
    """Extract cohort and task from path structure."""
    try:
        rel_path = wav_path.relative_to(audio_root)
        parts = rel_path.parts
        cohort = parts[0] if len(parts) > 0 else "unknown"
        task = parts[1] if len(parts) > 1 else "unknown"
        return cohort, task
    except:
        return "unknown", "unknown"

def main():
    print("Scanning for RTTMs...")
    rttm_index = set()
    for rttm_path in RTTM_ROOT.rglob("*.rttm"):
        base = get_base_filename(rttm_path)
        rttm_index.add(base)
    
    print(f"Found {len(rttm_index)} unique RTTM base filenames\n")
    
    print("Scanning for WAVs...")
    wav_files = list(AUDIO_ROOT.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files\n")
    
    # Find WAVs with no RTTM
    no_matches = []
    for wav_path in wav_files:
        base = get_base_filename(wav_path)
        if base not in rttm_index:
            cohort, task = get_cohort_and_task(wav_path, AUDIO_ROOT)
            no_matches.append((wav_path, base, cohort, task))
    
    print(f"Found {len(no_matches)} WAVs with no matching RTTM\n")
    
    # Group by cohort and task
    by_cohort_task = defaultdict(list)
    for wav, base, cohort, task in no_matches:
        by_cohort_task[(cohort, task)].append((wav, base))
    
    # Print summary by cohort/task
    print("="*80)
    print("MISSING RTTMs BY COHORT/TASK")
    print("="*80)
    for (cohort, task), items in sorted(by_cohort_task.items()):
        print(f"\n{cohort} / {task}: {len(items)} missing")
        print("-"*80)
        # Show first 5 examples
        for wav, base in items[:5]:
            print(f"  {base}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")

if __name__ == "__main__":
    main()
