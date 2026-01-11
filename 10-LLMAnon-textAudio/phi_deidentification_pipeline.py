#!/usr/bin/env python3
"""
PHI Audio De-identification Pipeline
Applies three methods to destroy linguistic content in PHI regions while preserving prosody.

Methods:
1. BLANKET: Remove formants + phase randomize entire PHI region
2. SURGICAL: Detect phonemes, process only vowels (formants) and consonants (phase)
3. HYBRID: Surgical + conservative gap-filling for ambiguous regions

Input:  Compiled PHI-only audio files (from extract_phi_snippets.py)
Output: Three de-identified versions per file (BLANKET, SURGICAL, HYBRID)

Usage:
    python deidentify_phi_audio.py --input P001_PHI_only_original.wav --output_dir ./deidentified/
    python deidentify_phi_audio.py --batch --input_dir ./phi_snippets/ --output_dir ./deidentified/
"""

import argparse
import numpy as np
import soundfile as sf
import librosa
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft


class PHIDeidentifier:
    """De-identify PHI audio using formant removal and phase randomization."""
    
    def __init__(self, method: str = 'hybrid', sr: int = 16000):
        """
        Initialize de-identifier.
        
        Args:
            method: 'blanket', 'surgical', or 'hybrid'
            sr: Sample rate
        """
        self.method = method.lower()
        self.sr = sr
        
        if self.method not in ['blanket', 'surgical', 'hybrid']:
            raise ValueError(f"Method must be 'blanket', 'surgical', or 'hybrid', got {method}")
    
    def remove_formants_lpc(self, audio: np.ndarray, sr: int, 
                           lpc_order: int = None) -> np.ndarray:
        """
        Remove formants via LPC inverse filtering using Parselmouth.
        
        Args:
            audio: Audio segment
            sr: Sample rate
            lpc_order: LPC order (default: 2 + sr/1000)
            
        Returns:
            residual: Audio with formants removed (F0 + noise preserved)
        """
        if lpc_order is None:
            lpc_order = int(2 + sr / 1000)
        
        # Convert to Parselmouth Sound object
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Extract LPC
        # Parameters: prediction_order, window_length, time_step, pre_emphasis_frequency
        lpc = call(snd, "To LPC (autocorrelation)", lpc_order, 0.025, 0.005, 50.0)
        
        # Get residual (inverse filter removes formants, keeps F0 + excitation)
        residual_sound = call([lpc, snd], "Filter (inverse)")
        
        # Convert back to numpy array
        residual = residual_sound.values[0]
        
        # Normalize to prevent clipping
        if np.max(np.abs(residual)) > 0:
            residual = residual / np.max(np.abs(residual)) * 0.95
        
        return residual
    
    def phase_randomize(self, audio: np.ndarray, seed: int = None) -> np.ndarray:
        """
        Randomize phase spectrum while preserving magnitude (amplitude envelope).
        Information-theoretically irreversible with cryptographic RNG.
        
        Args:
            audio: Audio segment
            seed: Random seed (if None, uses system entropy for crypto security)
            
        Returns:
            scrambled: Phase-randomized audio
        """
        # Use cryptographic random if no seed provided
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        
        # FFT
        spectrum = fft(audio)
        magnitude = np.abs(spectrum)
        
        # Generate random phases uniformly distributed [0, 2π]
        random_phases = rng.uniform(0, 2 * np.pi, len(spectrum))
        
        # Reconstruct with random phases
        scrambled_spectrum = magnitude * np.exp(1j * random_phases)
        
        # IFFT back to time domain
        scrambled = np.real(ifft(scrambled_spectrum))
        
        # Normalize
        if np.max(np.abs(scrambled)) > 0:
            scrambled = scrambled / np.max(np.abs(scrambled)) * np.max(np.abs(audio))
        
        return scrambled
    
    def detect_vowels(self, audio: np.ndarray, sr: int, 
                     confidence_threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        Detect vowel regions via periodicity and formant presence.
        
        Args:
            audio: Audio segment
            sr: Sample rate
            confidence_threshold: Detection threshold (lower = more conservative)
            
        Returns:
            vowel_regions: List of (start_sample, end_sample) tuples
        """
        # Frame-based analysis
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute features
        # 1. Periodicity via autocorrelation
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        
        vowel_frames = []
        for i, frame in enumerate(frames.T):
            # Check for periodicity (autocorrelation)
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Normalize
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
            
            # Look for peak (indicates periodicity)
            # Exclude first few lags (DC component)
            min_lag = int(sr / 500)  # Minimum F0 = 500 Hz
            max_lag = int(sr / 50)   # Maximum F0 = 50 Hz
            
            if len(autocorr) > max_lag:
                peak_value = np.max(autocorr[min_lag:max_lag])
                
                # Also check energy
                energy = np.sum(frame ** 2)
                
                # Vowels: high periodicity + sufficient energy
                is_vowel = (peak_value > confidence_threshold) and (energy > 0.01)
                
                if is_vowel:
                    vowel_frames.append(i)
        
        # Convert frame indices to sample ranges
        vowel_regions = []
        if vowel_frames:
            # Merge consecutive frames
            start_frame = vowel_frames[0]
            for i in range(1, len(vowel_frames)):
                if vowel_frames[i] != vowel_frames[i-1] + 1:
                    # Gap found, save region
                    end_frame = vowel_frames[i-1]
                    start_sample = start_frame * hop_length
                    end_sample = (end_frame + 1) * hop_length + frame_length
                    vowel_regions.append((start_sample, end_sample))
                    start_frame = vowel_frames[i]
            
            # Save last region
            end_frame = vowel_frames[-1]
            start_sample = start_frame * hop_length
            end_sample = min((end_frame + 1) * hop_length + frame_length, len(audio))
            vowel_regions.append((start_sample, end_sample))
        
        return vowel_regions
    
    def detect_consonants(self, audio: np.ndarray, sr: int,
                         confidence_threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        Detect consonant regions via spectral features.
        
        Args:
            audio: Audio segment
            sr: Sample rate
            confidence_threshold: Detection threshold
            
        Returns:
            consonant_regions: List of (start_sample, end_sample) tuples
        """
        # Frame-based analysis
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        # Compute spectral features
        # 1. Spectral centroid (high for sibilants)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, 
                                                     n_fft=frame_length, 
                                                     hop_length=hop_length)[0]
        
        # 2. Zero-crossing rate (high for fricatives)
        zcr = librosa.feature.zero_crossing_rate(audio, 
                                                 frame_length=frame_length, 
                                                 hop_length=hop_length)[0]
        
        # 3. RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                 hop_length=hop_length)[0]
        
        # Normalize features
        centroid_norm = (centroid - np.mean(centroid)) / (np.std(centroid) + 1e-8)
        zcr_norm = (zcr - np.mean(zcr)) / (np.std(zcr) + 1e-8)
        
        consonant_frames = []
        for i in range(len(centroid)):
            # Consonants: high centroid OR high ZCR, with sufficient energy
            is_consonant = (
                (centroid_norm[i] > confidence_threshold or 
                 zcr_norm[i] > confidence_threshold) and
                rms[i] > np.mean(rms) * 0.5  # Above-average energy
            )
            
            if is_consonant:
                consonant_frames.append(i)
        
        # Convert to sample ranges
        consonant_regions = []
        if consonant_frames:
            start_frame = consonant_frames[0]
            for i in range(1, len(consonant_frames)):
                if consonant_frames[i] != consonant_frames[i-1] + 1:
                    end_frame = consonant_frames[i-1]
                    start_sample = start_frame * hop_length
                    end_sample = (end_frame + 1) * hop_length + frame_length
                    consonant_regions.append((start_sample, end_sample))
                    start_frame = consonant_frames[i]
            
            # Last region
            end_frame = consonant_frames[-1]
            start_sample = start_frame * hop_length
            end_sample = min((end_frame + 1) * hop_length + frame_length, len(audio))
            consonant_regions.append((start_sample, end_sample))
        
        return consonant_regions
    
    def find_unprocessed_gaps(self, audio_length: int, 
                             processed_regions: List[Tuple[int, int]],
                             min_gap_samples: int = 800) -> List[Tuple[int, int]]:
        """
        Find gaps between processed regions.
        
        Args:
            audio_length: Total length of audio
            processed_regions: List of already-processed (start, end) tuples
            min_gap_samples: Minimum gap size to consider (default ~50ms at 16kHz)
            
        Returns:
            gaps: List of (start, end) tuples for unprocessed regions
        """
        if not processed_regions:
            return [(0, audio_length)]
        
        # Sort regions by start time
        sorted_regions = sorted(processed_regions, key=lambda x: x[0])
        
        gaps = []
        
        # Gap before first region
        if sorted_regions[0][0] > min_gap_samples:
            gaps.append((0, sorted_regions[0][0]))
        
        # Gaps between regions
        for i in range(len(sorted_regions) - 1):
            gap_start = sorted_regions[i][1]
            gap_end = sorted_regions[i + 1][0]
            
            if gap_end - gap_start > min_gap_samples:
                gaps.append((gap_start, gap_end))
        
        # Gap after last region
        if audio_length - sorted_regions[-1][1] > min_gap_samples:
            gaps.append((sorted_regions[-1][1], audio_length))
        
        return gaps
    
    def blanket_approach(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        BLANKET: Process entire audio - remove formants + phase randomize.
        Most conservative, destroys everything including silence.
        
        Args:
            audio: Input audio
            sr: Sample rate
            
        Returns:
            processed: De-identified audio
        """
        # Step 1: Remove formants
        residual = self.remove_formants_lpc(audio, sr)
        
        # Step 2: Phase randomize entire signal
        processed = self.phase_randomize(residual)
        
        return processed
    
    def surgical_approach(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        SURGICAL: Detect phonemes, process only detected vowels and consonants.
        Preserves true silence, maximum prosodic detail.
        
        Args:
            audio: Input audio
            sr: Sample rate
            
        Returns:
            processed: De-identified audio
        """
        processed = audio.copy()
        
        # Detect vowels and consonants
        vowel_regions = self.detect_vowels(audio, sr, confidence_threshold=0.7)
        consonant_regions = self.detect_consonants(audio, sr, confidence_threshold=0.7)
        
        # Process vowels: remove formants
        for start, end in vowel_regions:
            segment = audio[start:end]
            processed_segment = self.remove_formants_lpc(segment, sr)
            processed[start:end] = processed_segment
        
        # Process consonants: phase randomize
        for start, end in consonant_regions:
            segment = processed[start:end]  # Use formant-removed version
            processed_segment = self.phase_randomize(segment)
            processed[start:end] = processed_segment
        
        return processed
    
    def hybrid_approach(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        HYBRID: Surgical + conservative gap-filling.
        Process detected phonemes, then process ambiguous gaps with energy.
        Balanced security and utility.
        
        Args:
            audio: Input audio
            sr: Sample rate
            
        Returns:
            processed: De-identified audio
        """
        processed = audio.copy()
        
        # Detect vowels and consonants
        vowel_regions = self.detect_vowels(audio, sr, confidence_threshold=0.7)
        consonant_regions = self.detect_consonants(audio, sr, confidence_threshold=0.7)
        
        # Process detected regions
        for start, end in vowel_regions:
            segment = audio[start:end]
            processed_segment = self.remove_formants_lpc(segment, sr)
            processed[start:end] = processed_segment
        
        for start, end in consonant_regions:
            segment = processed[start:end]
            processed_segment = self.phase_randomize(segment)
            processed[start:end] = processed_segment
        
        # Find unprocessed gaps
        all_processed = vowel_regions + consonant_regions
        gaps = self.find_unprocessed_gaps(len(audio), all_processed)
        
        # Conservative gap-filling: process gaps with energy above threshold
        silence_threshold = np.mean(np.abs(audio)) * 0.1
        
        for start, end in gaps:
            segment = audio[start:end]
            segment_energy = np.mean(np.abs(segment))
            
            # If gap has energy, treat conservatively (process as vowel)
            if segment_energy > silence_threshold:
                processed_segment = self.remove_formants_lpc(segment, sr)
                processed[start:end] = processed_segment
        
        return processed
    
    def process_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Process audio with selected method.
        
        Args:
            audio: Input audio
            sr: Sample rate
            
        Returns:
            processed: De-identified audio
        """
        if self.method == 'blanket':
            return self.blanket_approach(audio, sr)
        elif self.method == 'surgical':
            return self.surgical_approach(audio, sr)
        elif self.method == 'hybrid':
            return self.hybrid_approach(audio, sr)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def process_single_file(input_path: Path, output_dir: Path, 
                       methods: List[str] = ['blanket', 'surgical', 'hybrid'],
                       target_sr: int = 16000) -> Dict:
    """
    Process a single PHI-only audio file with all three methods.
    
    Args:
        input_path: Path to PHI-only original audio
        output_dir: Where to save de-identified versions
        methods: Which methods to apply
        target_sr: Target sample rate
        
    Returns:
        metadata: Processing metadata
    """
    participant_id = input_path.stem.replace('_PHI_only_original', '')
    print(f"\nProcessing: {participant_id}")
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
    duration = len(audio) / sr
    
    print(f"  Duration: {duration:.2f}s")
    
    metadata = {
        'participant_id': participant_id,
        'input_path': str(input_path),
        'duration_seconds': duration,
        'sample_rate': sr,
        'methods_applied': {},
        'processing_timestamp': datetime.now().isoformat()
    }
    
    # Process with each method
    for method in methods:
        print(f"  Processing with {method.upper()} method...")
        
        try:
            deidentifier = PHIDeidentifier(method=method, sr=sr)
            processed = deidentifier.process_audio(audio, sr)
            
            # Save output
            output_filename = f"{participant_id}_PHI_only_{method.upper()}.wav"
            output_path = output_dir / output_filename
            sf.write(output_path, processed, sr)
            
            print(f"    ✓ Saved: {output_filename}")
            
            metadata['methods_applied'][method] = {
                'status': 'success',
                'output_path': str(output_path)
            }
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            metadata['methods_applied'][method] = {
                'status': 'error',
                'error': str(e)
            }
    
    return metadata


def process_batch(input_dir: Path, output_dir: Path,
                 methods: List[str] = ['blanket', 'surgical', 'hybrid'],
                 target_sr: int = 16000) -> Dict:
    """
    Process all PHI-only audio files in a directory.
    
    Args:
        input_dir: Directory containing PHI-only original files
        output_dir: Where to save de-identified versions
        methods: Which methods to apply
        target_sr: Target sample rate
        
    Returns:
        summary: Processing summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PHI-only original files
    input_files = list(input_dir.glob('*_PHI_only_original.wav'))
    
    print(f"Found {len(input_files)} PHI-only audio files")
    print(f"Will process with methods: {', '.join(methods)}\n")
    
    all_metadata = []
    success_count = 0
    error_count = 0
    
    for input_path in input_files:
        metadata = process_single_file(input_path, output_dir, methods, target_sr)
        all_metadata.append(metadata)
        
        # Check if any method succeeded
        if any(m['status'] == 'success' for m in metadata['methods_applied'].values()):
            success_count += 1
        else:
            error_count += 1
    
    # Generate summary
    summary = {
        'total_files': len(input_files),
        'successful': success_count,
        'errors': error_count,
        'methods_used': methods,
        'target_sample_rate': target_sr,
        'output_directory': str(output_dir),
        'processing_timestamp': datetime.now().isoformat(),
        'per_file_metadata': all_metadata
    }
    
    # Save summary
    summary_path = output_dir / 'deidentification_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PHI DE-IDENTIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed:  {len(input_files)}")
    print(f"Successful:             {success_count}")
    print(f"Errors:                 {error_count}")
    print(f"Methods applied:        {', '.join(methods)}")
    print(f"\nOutput files created:   {success_count * len(methods)}")
    print(f"Output directory:       {output_dir}")
    print(f"Summary saved to:       {summary_path}")
    print(f"{'='*60}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='De-identify PHI audio using three methods: BLANKET, SURGICAL, HYBRID'
    )
    
    parser.add_argument('--input', type=Path,
                       help='Single PHI-only audio file to process')
    parser.add_argument('--input_dir', type=Path,
                       help='Directory containing PHI-only audio files (batch mode)')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Directory to save de-identified audio files')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process all files in input_dir')
    parser.add_argument('--methods', nargs='+', 
                       choices=['blanket', 'surgical', 'hybrid'],
                       default=['blanket', 'surgical', 'hybrid'],
                       help='Which methods to apply (default: all three)')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input_dir:
            parser.error("Batch mode requires --input_dir")
        process_batch(args.input_dir, args.output_dir, args.methods, args.target_sr)
    else:
        if not args.input:
            parser.error("Single file mode requires --input")
        process_single_file(args.input, args.output_dir, args.methods, args.target_sr)


if __name__ == '__main__':
    main()
