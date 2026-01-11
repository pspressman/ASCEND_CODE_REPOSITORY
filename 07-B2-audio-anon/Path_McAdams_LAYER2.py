#!/usr/bin/env python3
"""
Layered McAdams Anonymization - Second Pass

Applies a SECOND McAdams transformation with per-recording randomized
coefficients on top of already-anonymized audio.

Mathematical basis:
    First pass:  φ' = φ^0.8 (fixed, already applied)
    Second pass: φ'' = (φ')^α₂ where α₂ ~ U(0.9375, 1.125)
    Net effect:  φ'' = φ^(0.8 × α₂) = φ^[0.75, 0.90]

Citation:
    Tayebi Arasteh et al. 2024 - validated U(0.75, 0.90) for pathological speech.
    "Coefficients above 0.90 minimally affect anonymization, while those
    below 0.75 begin to degrade audio quality" for clinical populations.

Security rationale:
    - Per-recording randomization defeats cross-session linkage
    - Each recording has unique second-layer coefficient
    - Effective range [0.75, 0.90] optimized for pathological/clinical speech

Usage:
    python Path_McAdams_LAYER2.py \
        --input-dir /path/to/already_anonymized_audio \
        --output-dir /path/to/double_anonymized_output

Overnight batch processing notes:
    - Progress saved to checkpoint file every 100 files
    - Can resume from checkpoint if interrupted
    - Logs all coefficients for audit trail
    - Estimates remaining time

Author: AUSPEX (Claude) for Peter Pressman / Syntopic Systems
Date: December 2024
"""

import os
import random
import json
import sys
import time
import soundfile as sf
from pathlib import Path
import numpy as np
import scipy
import scipy.io.wavfile
import librosa
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse


class LayeredMcAdamsAnonymizer:
    """
    Second-layer McAdams anonymization for defense in depth.
    
    Applies randomized McAdams on top of already-anonymized audio.
    """
    
    def __init__(self, input_dir, output_dir,
                 mc_coeff_min=0.9375, mc_coeff_max=1.125,
                 first_layer_coeff=0.8,
                 checkpoint_interval=100):
        """
        Initialize the layered anonymizer.
        
        Args:
            input_dir: Directory containing already-anonymized WAV files
            output_dir: Directory for double-anonymized output
            mc_coeff_min: Minimum second-layer coefficient (default: 0.9375)
            mc_coeff_max: Maximum second-layer coefficient (default: 1.125)
            first_layer_coeff: Coefficient used in first pass (default: 0.8)
            checkpoint_interval: Save progress every N files (default: 100)
        
        Note on coefficient range:
            Second-layer range [0.9375, 1.125] is chosen so that:
            effective = first_layer × second_layer = 0.8 × [0.9375, 1.125] = [0.75, 0.90]
            
            This lands in the validated range from Tayebi Arasteh 2024,
            optimized for pathological speech preservation.
            
            Coefficients > 1.0 partially UNDO the first-pass contraction,
            which is intentional to land back in the valid range.
        """
        self.first_layer_coeff = first_layer_coeff
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mc_coeff_min = mc_coeff_min
        self.mc_coeff_max = mc_coeff_max
        self.checkpoint_interval = checkpoint_interval
        
        # Calculate effective range for logging
        self.effective_min = self.first_layer_coeff * self.mc_coeff_min
        self.effective_max = self.first_layer_coeff * self.mc_coeff_max
        
        # Audit trail
        self.coefficient_log = {}
        self.errors_log = []
        
        # Random seed for reproducibility
        self.random_seed = int(datetime.now().timestamp())
        random.seed(self.random_seed)
        
        # Checkpoint file
        self.checkpoint_path = self.output_dir / "_checkpoint.json"
        self.processed_files = set()
        
        # Load existing checkpoint if resuming
        self._load_checkpoint()
        
    def _load_checkpoint(self):
        """Load checkpoint if exists (for resuming interrupted runs)."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                self.processed_files = set(checkpoint.get('processed_files', []))
                self.coefficient_log = checkpoint.get('coefficient_log', {})
                self.random_seed = checkpoint.get('random_seed', self.random_seed)
                random.seed(self.random_seed)
                # Advance random state to match processed count
                for _ in range(len(self.processed_files)):
                    random.random()
                print(f"Resuming from checkpoint: {len(self.processed_files)} files already processed")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                self.processed_files = set()
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint = {
            'processed_files': list(self.processed_files),
            'coefficient_log': self.coefficient_log,
            'random_seed': self.random_seed,
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
    
    def sample_coefficient(self) -> float:
        """Sample random coefficient from uniform distribution."""
        return random.uniform(self.mc_coeff_min, self.mc_coeff_max)
    
    def single_anonymize(self, utterance, sr, output_path,
                         winLengthinms=20, shiftLengthinms=10,
                         lp_order=20, mcadams=0.8):
        """
        Apply McAdams anonymization to a single utterance.
        
        Adapted from VoicePrivacy Challenge Baseline B2.
        """
        eps = np.finfo(np.float32).eps
        utterance = utterance + eps

        # Simulation parameters
        winlen = np.floor(winLengthinms * 0.001 * sr).astype(int)
        shift = np.floor(shiftLengthinms * 0.001 * sr).astype(int)
        length_sig = len(utterance)

        # FFT processing parameters
        NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
        wPR = np.hanning(winlen)
        K = np.sum(wPR) / shift
        win = np.sqrt(wPR / K)
        Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int)

        # Check for minimum viable length
        if Nframes < 2:
            # File too short for processing, copy as-is with warning
            scipy.io.wavfile.write(output_path, sr, np.float32(utterance))
            return False  # Indicate no transformation applied

        sig_rec = np.zeros([length_sig])

        for m in np.arange(1, Nframes):
            index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))
            
            if len(index) < lp_order + 1:
                continue
                
            frame = utterance[index] * win
            
            try:
                a_lpc = librosa.lpc(frame + eps, order=lp_order)
            except Exception:
                continue
                
            poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
            ind_imag = np.where(np.isreal(poles) == False)[0]
            
            if len(ind_imag) == 0:
                # No imaginary poles, use original frame
                outindex = np.arange(m * shift, m * shift + len(frame))
                if outindex[-1] < length_sig:
                    sig_rec[outindex] = sig_rec[outindex] + frame
                continue
                
            ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]

            new_angles = np.angle(poles[ind_imag_con]) ** mcadams

            new_angles[np.where(new_angles >= np.pi)] = np.pi
            new_angles[np.where(new_angles <= 0)] = 0

            new_poles = poles.copy()
            for k in np.arange(np.size(ind_imag_con)):
                new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
                if ind_imag_con[k] + 1 < len(new_poles):
                    new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])

            a_lpc_new = np.real(np.poly(new_poles))
            res = scipy.signal.lfilter(a_lpc, np.array(1), frame)
            frame_rec = scipy.signal.lfilter(np.array([1]), a_lpc_new, res)
            frame_rec = frame_rec * win

            outindex = np.arange(m * shift, m * shift + len(frame_rec))
            if outindex[-1] < length_sig:
                sig_rec[outindex] = sig_rec[outindex] + frame_rec
        
        # Normalize
        max_val = np.max(np.abs(sig_rec))
        if max_val > 0:
            sig_rec = sig_rec / max_val
        
        scipy.io.wavfile.write(output_path, sr, np.float32(sig_rec))
        return True

    def process_directory(self):
        """
        Process all WAV files with second-layer randomized McAdams.
        
        Features:
        - Resumes from checkpoint if interrupted
        - Saves progress periodically
        - Estimates remaining time
        - Logs all coefficients
        """
        
        # Find all WAV files
        wav_files = sorted(list(self.input_dir.rglob("*.wav")))
        total_files = len(wav_files)
        
        # Filter out already processed files
        files_to_process = [
            f for f in wav_files
            if str(f.relative_to(self.input_dir)) not in self.processed_files
        ]
        
        already_done = total_files - len(files_to_process)
        
        print("="*70)
        print("LAYERED McADAMS ANONYMIZATION - SECOND PASS")
        print("="*70)
        print(f"Total files: {total_files}")
        print(f"Already processed: {already_done}")
        print(f"Remaining: {len(files_to_process)}")
        print(f"First layer coefficient: {self.first_layer_coeff}")
        print(f"Second layer range: [{self.mc_coeff_min}, {self.mc_coeff_max}]")
        print(f"Effective range: [{self.effective_min:.3f}, {self.effective_max:.3f}]")
        print(f"Target range (VoicePrivacy/Tayebi Arasteh): [0.5, 0.9]")
        print(f"Random seed: {self.random_seed}")
        print(f"Checkpoint interval: every {self.checkpoint_interval} files")
        print("="*70)
        print()
        
        if len(files_to_process) == 0:
            print("All files already processed!")
            return
        
        start_time = time.time()
        processed_count = 0
        
        for wav_file in tqdm(files_to_process, desc="Layer 2 anonymizing"):
            relative_path = str(wav_file.relative_to(self.input_dir))
            
            try:
                # Load audio
                utterance, sr = sf.read(str(wav_file))
                
                # Force to mono if stereo
                if len(utterance.shape) > 1:
                    utterance = utterance.mean(axis=1)
                
                # Create output path maintaining directory structure
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Sample random coefficient for THIS recording
                mcadams_coef = self.sample_coefficient()
                effective_coef = self.first_layer_coeff * mcadams_coef  # Combined with first pass
                
                # Apply second-layer McAdams
                success = self.single_anonymize(
                    utterance=utterance,
                    sr=sr,
                    output_path=str(output_path),
                    mcadams=mcadams_coef
                )
                
                # Log the coefficient
                self.coefficient_log[relative_path] = {
                    'layer2_coefficient': mcadams_coef,
                    'effective_coefficient': effective_coef,
                    'transformation_applied': success,
                    'timestamp': datetime.now().isoformat(),
                }
                
                self.processed_files.add(relative_path)
                processed_count += 1
                
                # Periodic checkpoint
                if processed_count % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    remaining = len(files_to_process) - processed_count
                    eta = remaining / rate if rate > 0 else 0
                    eta_str = str(timedelta(seconds=int(eta)))
                    tqdm.write(f"  Checkpoint saved. ETA: {eta_str}")
                
            except Exception as e:
                self.errors_log.append({
                    'file': relative_path,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })
                tqdm.write(f"  Error: {wav_file.name}: {e}")
                continue
        
        # Final save
        self._save_final_logs()
        
        # Summary
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print()
        print("="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Files processed this run: {processed_count}")
        print(f"Total files processed: {len(self.processed_files)}")
        print(f"Errors: {len(self.errors_log)}")
        print(f"Elapsed time: {elapsed_str}")
        print()
        print("Output files:")
        print(f"  Audio: {self.output_dir}")
        print(f"  Coefficient log: {self.output_dir / 'layer2_coefficient_log.json'}")
        if self.errors_log:
            print(f"  Error log: {self.output_dir / 'layer2_errors.json'}")
        print()
        print("SECURITY REMINDER:")
        print("  The coefficient log contains anonymization keys.")
        print("  Store it securely, separate from the audio files.")
    
    def _save_final_logs(self):
        """Save final audit logs."""
        
        # Coefficient log
        log_path = self.output_dir / "layer2_coefficient_log.json"
        log_data = {
            'metadata': {
                'description': 'Second-layer McAdams coefficients for defense-in-depth anonymization',
                'first_layer_coefficient': self.first_layer_coeff,
                'second_layer_range': [self.mc_coeff_min, self.mc_coeff_max],
                'effective_range': [self.effective_min, self.effective_max],
                'target_range_reference': '[0.5, 0.9] per VoicePrivacy 2024 and Tayebi Arasteh 2024',
                'random_seed': self.random_seed,
                'processing_completed': datetime.now().isoformat(),
                'total_files': len(self.coefficient_log),
            },
            'coefficients': self.coefficient_log
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Error log (if any)
        if self.errors_log:
            error_path = self.output_dir / "layer2_errors.json"
            with open(error_path, 'w') as f:
                json.dump(self.errors_log, f, indent=2)
        
        # Remove checkpoint (processing complete)
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='Layered McAdams Anonymization - Second Pass',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python Path_McAdams_LAYER2.py \\
        --input-dir /Volumes/ASCEND_ANON/McAdams \\
        --output-dir /Volumes/ASCEND_ANON/McAdams_Layer2

This applies a second randomized McAdams transformation on top of
already-anonymized audio, providing defense-in-depth protection.

Coefficient math:
    First pass (already applied): 0.8
    Second pass range: [0.9375, 1.125]
    Effective range: 0.8 × [0.9375, 1.125] = [0.75, 0.90]

Citation: Tayebi Arasteh et al. 2024 - validated for pathological speech.
"Coefficients above 0.90 minimally affect anonymization, while those
below 0.75 begin to degrade audio quality."

Note: Second-layer coefficients > 1.0 partially UNDO the first-pass
contraction, which is intentional to land in the valid range.
        """
    )
    
    parser.add_argument('--input-dir', required=True, type=Path,
                        help='Input directory (already-anonymized audio)')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Output directory for double-anonymized audio')
    parser.add_argument('--mc-min', type=float, default=0.9375,
                        help='Minimum second-layer coefficient (default: 0.9375)')
    parser.add_argument('--mc-max', type=float, default=1.125,
                        help='Maximum second-layer coefficient (default: 1.125)')
    parser.add_argument('--first-layer', type=float, default=0.8,
                        help='Coefficient used in first pass (default: 0.8)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N files (default: 100)')
    
    args = parser.parse_args()
    
    # Verify input directory exists
    if not args.input_dir.exists():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Warn if output same as input
    if args.input_dir.resolve() == args.output_dir.resolve():
        print("ERROR: Output directory cannot be same as input directory!")
        print("       This would overwrite the first-layer anonymized files.")
        sys.exit(1)
    
    anonymizer = LayeredMcAdamsAnonymizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mc_coeff_min=args.mc_min,
        mc_coeff_max=args.mc_max,
        first_layer_coeff=args.first_layer,
        checkpoint_interval=args.checkpoint_interval
    )
    
    anonymizer.process_directory()


if __name__ == '__main__':
    main()
