#!/usr/bin/env python3
"""
Universal Audio Noise Cleaner
Uses stationary noise estimation for all recordings
"""

import os
import sys
from pathlib import Path
import soundfile as sf
import noisereduce as nr
import logging
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class UniversalNoiseCleaner:
    def __init__(self, prop_decrease: float = 1.0):
        """
        Initialize universal noise cleaner
        
        Args:
            prop_decrease: How much to reduce noise (0.0-1.0)
                          1.0 = maximum reduction
                          0.8 = moderate reduction
        """
        self.prop_decrease = prop_decrease
    
    def clean_audio(self, audio_path: Path, output_path: Path) -> bool:
        """
        Clean audio file using stationary noise estimation
        
        This mimics Audacity's noise reduction by:
        1. Automatically estimating the noise floor across the entire recording
        2. Applying spectral gating to reduce stationary noise (hum, AC, etc.)
        
        Returns: success (bool)
        """
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Convert stereo to mono if needed
            # noisereduce requires mono input
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
                logging.info(f"  Converted stereo to mono")
            
            # Apply noise reduction with explicit STFT parameters
            # These parameters ensure scipy's constraint: noverlap < nperseg
            cleaned_audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,                # Assumes stationary noise (hum, AC)
                prop_decrease=self.prop_decrease,  # How much to reduce
                n_fft=2048,                     # FFT window size
                hop_length=512,                 # Step size (n_fft / 4)
                win_length=2048                 # Window length (matches n_fft)
            )
            
            # Save cleaned audio (as mono)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, cleaned_audio, sr)
            
            return True
            
        except Exception as e:
            logging.error(f"Error cleaning {audio_path.name}: {e}")
            return False

def main():
    """
    Usage: python noise_cleaner.py <input_audio_root> <output_root> [prop_decrease]
    
    Example:
    python noise_cleaner.py /path/to/user/Desktop/AudioThreeTest/otherAudio /path/to/user/Desktop/CleanedAudio 1.0
    
    prop_decrease: 0.0-1.0 (default 1.0)
        1.0 = maximum noise reduction
        0.8 = moderate reduction
    """
    
    if len(sys.argv) < 3:
        print("Usage: python noise_cleaner.py <input_audio_root> <output_root> [prop_decrease]")
        print("\nUses automatic stationary noise estimation (no profiles needed)")
        print("prop_decrease: 0.0-1.0 (default 1.0 = maximum reduction)")
        sys.exit(1)
    
    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    prop_decrease = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    logging.info("="*80)
    logging.info("UNIVERSAL NOISE CLEANER (Stationary Noise Estimation)")
    logging.info("="*80)
    logging.info(f"Input:          {input_root}")
    logging.info(f"Output:         {output_root}")
    logging.info(f"Prop Decrease:  {prop_decrease}")
    logging.info("="*80)
    
    # Initialize cleaner
    cleaner = UniversalNoiseCleaner(prop_decrease=prop_decrease)
    
    # Find all wav files
    audio_files = list(input_root.rglob("*.wav"))
    logging.info(f"Found {len(audio_files)} wav files to process")
    
    # Process each file
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, audio_path in enumerate(audio_files, 1):
        # Maintain folder structure
        rel_path = audio_path.relative_to(input_root)
        output_path = output_root / rel_path.parent / f"{audio_path.stem}_cleaned.wav"
        
        # Skip if already processed
        if output_path.exists():
            logging.info(f"[{idx}/{len(audio_files)}] ✓ SKIP (exists): {audio_path.name}")
            skip_count += 1
            continue
        
        logging.info(f"[{idx}/{len(audio_files)}] Processing: {audio_path.name}")
        
        success = cleaner.clean_audio(audio_path, output_path)
        
        if success:
            success_count += 1
            logging.info(f"  ✓ Cleaned → {output_path.name}")
        else:
            error_count += 1
            logging.error(f"  ✗ Failed to clean")
    
    # Summary
    logging.info("="*80)
    logging.info("CLEANING SUMMARY")
    logging.info("="*80)
    logging.info(f"Success: {success_count} files")
    logging.info(f"Skipped: {skip_count} files")
    logging.info(f"Errors:  {error_count} files")
    logging.info("="*80)

if __name__ == "__main__":
    main()
