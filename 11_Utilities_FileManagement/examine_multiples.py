#!/usr/bin/env python3
"""
Examine WAV files with multiple matching RTTMs
Shows details to help decide which RTTM to use
"""

import os
from pathlib import Path
from collections import defaultdict
import datetime

# Configuration
AUDIO_ROOT = Path("/path/to/user/Desktop/AudioThreeTest/otherAudio")
RTTM_ROOT = Path("/path/to/user/Desktop/AudioThreeTest/Transcripts-Gemeps-rttm")

def get_base_filename(filepath):
    """Extract base filename without extension or suffix."""
    name = filepath.stem
    if name.endswith('_transcript'):
        name = name[:-11]
    return name

def get_file_info(filepath):
    """Get file info including size and modification time."""
    stat = filepath.stat()
    return {
        'size': stat.st_size,
        'modified': datetime.datetime.fromtimestamp(stat.st_mtime),
        'path': filepath
    }

def main():
    print("Scanning for RTTMs...")
    rttm_index = defaultdict(list)
    for rttm_path in RTTM_ROOT.rglob("*.rttm"):
        base = get_base_filename(rttm_path)
        rttm_index[base].append(rttm_path)
    
    print(f"Found {len(rttm_index)} unique RTTM base filenames\n")
    
    print("Scanning for WAVs...")
    wav_files = list(AUDIO_ROOT.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files\n")
    
    # Find multiples
    multiple_matches = []
    for wav_path in wav_files:
        base = get_base_filename(wav_path)
        if base in rttm_index and len(rttm_index[base]) > 1:
            multiple_matches.append((wav_path, base, rttm_index[base]))
    
    print(f"Found {len(multiple_matches)} WAVs with multiple RTTMs\n")
    
    # Analyze patterns
    print("="*80)
    print("ANALYZING MULTIPLE MATCHES")
    print("="*80)
    
    # Show first 20 examples with details
    print("\nFirst 20 examples:\n")
    for i, (wav, base, rttm_list) in enumerate(multiple_matches[:20]):
        print(f"\n{i+1}. Base filename: {base}")
        print(f"   WAV: {wav.relative_to(AUDIO_ROOT)}")
        print(f"   Found {len(rttm_list)} RTTMs:")
        
        for j, rttm in enumerate(rttm_list, 1):
            info = get_file_info(rttm)
            rel_path = rttm.relative_to(RTTM_ROOT)
            print(f"     {j}) {rel_path}")
            print(f"        Size: {info['size']:,} bytes")
            print(f"        Modified: {info['modified']}")
            
            # Check for common patterns in path
            path_str = str(rel_path)
            if 'DeID' in path_str or 'deID' in path_str or 'deid' in path_str:
                print(f"        ⚠️  Contains 'DeID' in path")
            if 'Duplicate' in path_str:
                print(f"        ⚠️  Contains 'Duplicate' in path")
            if 'original' in path_str.lower():
                print(f"        ✓ Contains 'original' in path")
        
        print("-"*80)
    
    if len(multiple_matches) > 20:
        print(f"\n... and {len(multiple_matches) - 20} more cases")
    
    # Pattern analysis
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    patterns = {
        'has_deID': 0,
        'has_duplicate': 0,
        'has_original': 0,
        'different_sizes': 0,
        'same_size': 0
    }
    
    for wav, base, rttm_list in multiple_matches:
        paths = [str(r.relative_to(RTTM_ROOT)) for r in rttm_list]
        sizes = [r.stat().st_size for r in rttm_list]
        
        if any('deID' in p or 'DeID' in p for p in paths):
            patterns['has_deID'] += 1
        if any('Duplicate' in p for p in paths):
            patterns['has_duplicate'] += 1
        if any('original' in p.lower() for p in paths):
            patterns['has_original'] += 1
        
        if len(set(sizes)) > 1:
            patterns['different_sizes'] += 1
        else:
            patterns['same_size'] += 1
    
    print(f"\nOut of {len(multiple_matches)} cases with multiples:")
    print(f"  - {patterns['has_deID']} have 'DeID' in path")
    print(f"  - {patterns['has_duplicate']} have 'Duplicate' in path")
    print(f"  - {patterns['has_original']} have 'original' in path")
    print(f"  - {patterns['different_sizes']} have different file sizes")
    print(f"  - {patterns['same_size']} have identical file sizes")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\nBased on the patterns, consider:")
    print("1. Exclude paths containing 'DeID' or 'Duplicate'")
    print("2. Prefer paths containing 'original'")
    print("3. If still multiple, take most recent modification time")
    print("4. Or manually review and clean up the source directories")

if __name__ == "__main__":
    main()
