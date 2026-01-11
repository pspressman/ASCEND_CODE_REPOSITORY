#!/usr/bin/env python3
"""
Quick Runner for LLMAnon Round 2 Redaction
Pre-configured with your paths - just run it!
"""

import subprocess
import sys
from pathlib import Path

# Your paths (adjust if needed)
PATHS = {
    'csv': r"C:\LocalData\LLMAnon\LLM-Anon-reviewed.csv",
    'transcripts': r"C:\LocalData\ASCEND_PHI\DeID\CliniDeID_organized copy",
    'text_not_found_batch1': r"C:\LocalData\LLMAnon\Batch1_CSA_Timestamps\text_not_found_Batch1.txt",
    'text_not_found_batch2': r"C:\LocalData\LLMAnon\Batch2_Focused_Timestamps\text_not_found_Batch2.txt",
    'output': r"C:\LocalData\LLMAnon\Round2_Transcripts",
    'model': 'llama3.2',
    'ollama_url': 'http://localhost:11434'
}

def main():
    print("="*60)
    print("LLMANON ROUND 2 - QUICK RUNNER")
    print("="*60)
    print("\nConfigured paths:")
    for key, value in PATHS.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    response = input("\nProceed with these settings? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Aborted.")
        return
    
    # Build command
    cmd = [
        sys.executable,
        'llmanon_round2_redactor.py',
        '--csv', PATHS['csv'],
        '--transcripts', PATHS['transcripts'],
        '--text_not_found_batch1', PATHS['text_not_found_batch1'],
        '--text_not_found_batch2', PATHS['text_not_found_batch2'],
        '--output', PATHS['output'],
        '--model', PATHS['model'],
        '--ollama-url', PATHS['ollama_url']
    ]
    
    print("\nLaunching redaction pipeline...\n")
    
    # Run the script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running script: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
