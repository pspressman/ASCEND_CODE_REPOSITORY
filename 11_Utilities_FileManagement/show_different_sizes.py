#!/usr/bin/env python3
"""
Show only the cases where multiple RTTMs have DIFFERENT file sizes
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
    
    # Find multiples with different sizes
    different_sizes = []
    for wav_path in wav_files:
        base = get_base_filename(wav_path)
        if base in rttm_index and len(rttm_index[base]) > 1:
            rttm_list = rttm_index[base]
            sizes = [r.stat().st_size for r in rttm_list]
            if len(set(sizes)) > 1:  # Different sizes
                different_sizes.append((wav_path, base, rttm_list))
    
    print(f"Found {len(different_sizes)} cases with different file sizes\n")
    
    print("="*80)
    print("CASES WITH DIFFERENT FILE SIZES")
    print("="*80)
    
    for i, (wav, base, rttm_list) in enumerate(different_sizes, 1):
        print(f"\n{i}. Base filename: {base}")
        print(f"   WAV: {wav.relative_to(AUDIO_ROOT)}")
        print(f"   Found {len(rttm_list)} RTTMs with DIFFERENT sizes:")
        
        # Sort by size for easier comparison
        rttm_info = [(r, get_file_info(r)) for r in rttm_list]
        rttm_info.sort(key=lambda x: x[1]['size'], reverse=True)
        
        for j, (rttm, info) in enumerate(rttm_info, 1):
            rel_path = rttm.relative_to(RTTM_ROOT)
            print(f"\n     {j}) {rel_path}")
            print(f"        Size: {info['size']:,} bytes")
            print(f"        Modified: {info['modified']}")
            
            # Check for patterns
            path_str = str(rel_path)
            flags = []
            if 'Duplicate' in path_str:
                flags.append("⚠️ Duplicate folder")
            if 'original' in path_str.lower():
                flags.append("✓ original")
            if flags:
                print(f"        {' | '.join(flags)}")
        
        print("-"*80)
    
    print("\n" + "="*80)
    print("MANUAL REVIEW NEEDED")
    print("="*80)
    print(f"\nThese {len(different_sizes)} cases need manual decision:")
    print("- Different file sizes suggest different content or versions")
    print("- Review which version is correct before running copy script")
    print("- Consider removing incorrect versions from source")

if __name__ == "__main__":
    main()
