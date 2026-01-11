#!/usr/bin/env python3
"""
Add '_clean' suffix to wav_filename column for processing cleaned audio files.

Usage:
    python add_clean_suffix_to_csv.py input.csv output.csv
"""

import sys
import csv

def add_clean_suffix(input_csv, output_csv):
    """
    Read CSV and add '_cleaned' suffix to wav_filename before .wav extension.
    
    Example:
        "anon_5809437_file.wav" -> "anon_5809437_file_cleaned.wav"
    """
    with open(input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        rows = []
        for row in reader:
            # Modify wav_filename: insert '_cleaned' before '.wav'
            if 'wav_filename' in row:
                wav_filename = row['wav_filename']
                if wav_filename.lower().endswith('.wav'):
                    # Insert '_cleaned' before .wav extension
                    base = wav_filename[:-4]  # Remove .wav
                    row['wav_filename'] = f"{base}_cleaned.wav"
            
            rows.append(row)
    
    # Write modified CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Processed {len(rows)} rows")
    print(f"✓ Output saved to: {output_csv}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python add_clean_suffix_to_csv.py input.csv output.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    add_clean_suffix(input_csv, output_csv)
