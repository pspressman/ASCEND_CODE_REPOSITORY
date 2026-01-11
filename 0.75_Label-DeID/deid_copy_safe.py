#!/usr/bin/env python3
"""
SAFE De-identification Script - COPY VERSION
Copies and renames files to de-identified versions while preserving ALL originals.
Maintains complete folder structure in output directory.

CRITICAL: This script COPIES files, never modifies or deletes originals.
"""

import os
import pandas as pd
import logging
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
key_file = "/path/to/user/Desktop/deid_key.csv"

# Input folders to search (originals will NOT be modified)
folders_to_search = [
    "/path/to/volumes/Databackup2025/ClinWavFiles_Anon_unparsed",
    # Add more folder paths as needed
]

# Output base directory - de-identified copies go here
output_base_dir = "/path/to/volumes/Databackup2025/DeidentifiedData"

# Logs
log_file_debug = "/path/to/user/MyDevelopment/Logs/FileDeIDCopyLog.txt"
log_file_unmatched = "/path/to/user/MyDevelopment/Logs/UnmatchedFilesLog.csv"
log_file_summary = "/path/to/user/MyDevelopment/Logs/DeIDCopySummary.txt"

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    filename=log_file_debug,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# ============================================================================
# LOAD KEY FILE
# ============================================================================

try:
    key_df = pd.read_csv(key_file)
    key_df["mrn"] = key_df["mrn"].astype(str).str.zfill(4)
    
    # Create lookup dictionaries
    mrn_to_uid = dict(zip(key_df["mrn"], key_df["UID"]))
    mrn_to_date_shift = dict(zip(key_df["mrn"], key_df["date_shift_days"]))
    
    logging.info(f"Successfully loaded key file with {len(mrn_to_uid)} patients.")
    print(f"✓ Loaded {len(mrn_to_uid)} patients from key file")
    
except Exception as e:
    logging.error(f"Failed to load key file: {e}")
    print(f"✗ ERROR: {e}")
    raise SystemExit("Key file could not be loaded. Check log for details.")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_mrn_and_date(filename):
    """
    Extract MRN and date from filename like '1234-01-15-22.txt' or '1234-7-15-2022.txt'
    Returns: (mrn, date_string, full_date) or (None, None, None) if no match
    """
    # Pattern: MRN-MM-DD-YY or MRN-MM-DD-YYYY
    match = re.match(r"(\d+)-(\d{1,2})-(\d{1,2})-(\d{2,4})", filename)
    if match:
        mrn = match.group(1).zfill(4)
        month = match.group(2).zfill(2)
        day = match.group(3).zfill(2)
        year = match.group(4)
        
        # Handle 2-digit vs 4-digit years
        if len(year) == 2:
            if int(year) <= 30:
                full_year = "20" + year
            else:
                full_year = "19" + year
        else:
            full_year = year
            
        date_string = f"{month}-{day}-{year}"
        return mrn, date_string, f"{month}/{day}/{full_year}"
    
    return None, None, None

def shift_date(date_str, shift_days):
    """
    Shift a date by the specified number of days.
    Input: date_str in MM/DD/YYYY format
    Output: shifted date in MM-DD-YY format
    """
    try:
        original_date = datetime.strptime(date_str, "%m/%d/%Y")
        shifted_date = original_date + timedelta(days=int(shift_days))
        return shifted_date.strftime("%m-%d-%y")
    except Exception as e:
        logging.error(f"Error shifting date {date_str} by {shift_days} days: {e}")
        return None

def get_output_path(source_path, source_base, output_base):
    """
    Mirror the folder structure from source to output.
    
    Args:
        source_path: Full path to source file
        source_base: Base directory of source files
        output_base: Base directory for output files
    
    Returns:
        Output directory path (file path will be constructed separately)
    """
    # Get relative path from source base
    rel_path = os.path.relpath(os.path.dirname(source_path), source_base)
    
    # Create corresponding path in output base
    if rel_path == '.':
        output_dir = output_base
    else:
        output_dir = os.path.join(output_base, rel_path)
    
    return output_dir

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def copy_and_deid_files(source_folder, output_base, mrn_to_uid, mrn_to_date_shift):
    """
    Copy and de-identify files from source to output, maintaining folder structure.
    
    IMPORTANT: This function NEVER modifies source files!
    """
    stats = {
        'copied': 0,
        'errors': 0,
        'unmatched': [],
        'skipped': 0
    }
    
    # Create base output directory
    os.makedirs(output_base, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing: {source_folder}")
    print(f"Output to:  {output_base}")
    print(f"{'='*70}\n")
    
    for root, dirs, files in os.walk(source_folder):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            original_name, ext = os.path.splitext(file)
            source_path = os.path.join(root, file)
            
            # Extract MRN and date
            mrn, date_string, full_date = extract_mrn_and_date(original_name)
            
            if mrn is None or date_string is None:
                stats['unmatched'].append({
                    'file': source_path,
                    'reason': 'Could not extract MRN/date from filename'
                })
                logging.warning(f"Could not extract MRN/date from: {file}")
                continue
            
            # Look up patient
            if mrn not in mrn_to_uid:
                stats['unmatched'].append({
                    'file': source_path,
                    'reason': f'MRN {mrn} not found in key file'
                })
                logging.warning(f"MRN {mrn} not found in key file: {file}")
                continue
            
            # Get UID and date shift
            uid = mrn_to_uid[mrn]
            date_shift_days = mrn_to_date_shift[mrn]
            
            # Shift the date
            shifted_date = shift_date(full_date, date_shift_days)
            if shifted_date is None:
                stats['errors'] += 1
                logging.error(f"Failed to shift date for {file}")
                continue
            
            # Create new de-identified filename
            new_filename = f"{uid}-{shifted_date}{ext}"
            
            # Get output directory (maintains folder structure)
            output_dir = get_output_path(source_path, source_folder, output_base)
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, new_filename)
            
            # Handle duplicates (shouldn't happen with proper date shifting)
            if os.path.exists(output_path):
                counter = 1
                while os.path.exists(output_path):
                    counter_filename = f"{uid}-{shifted_date}_{counter:03d}{ext}"
                    output_path = os.path.join(output_dir, counter_filename)
                    counter += 1
                new_filename = os.path.basename(output_path)
                logging.warning(f"Duplicate detected, using: {new_filename}")
            
            # COPY the file (NEVER modify original!)
            try:
                shutil.copy2(source_path, output_path)
                stats['copied'] += 1
                
                logging.info(f"Copied: {file} -> {new_filename}")
                
                # Progress indicator every 10 files
                if stats['copied'] % 10 == 0:
                    print(f"  Progress: {stats['copied']} files copied...", end='\r')
                
            except Exception as e:
                stats['errors'] += 1
                logging.error(f"Error copying {source_path}: {e}")
                print(f"✗ Error copying {file}: {e}")
    
    return stats

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("SAFE DE-IDENTIFICATION (COPY MODE)")
    print("Originals will NOT be modified")
    print("="*70)
    
    # Verify output base directory
    print(f"\nOutput directory: {output_base_dir}")
    
    # Check available space
    try:
        stat = os.statvfs(output_base_dir.split('/')[1:3][0])  # Get disk stats
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"Available space: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print("⚠️  WARNING: Less than 10 GB free space!")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted by user.")
                return
    except:
        print("⚠️  Could not check free space")
    
    # Create output base
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each source folder
    all_stats = {
        'total_copied': 0,
        'total_errors': 0,
        'total_unmatched': [],
        'folders_processed': []
    }
    
    start_time = datetime.now()
    
    for source_folder in folders_to_search:
        if not os.path.exists(source_folder):
            print(f"⚠️  Folder not found: {source_folder}")
            continue
        
        try:
            # Create output subdirectory based on source folder name
            folder_name = os.path.basename(source_folder)
            output_dest = os.path.join(output_base_dir, folder_name)
            
            stats = copy_and_deid_files(source_folder, output_dest, 
                                       mrn_to_uid, mrn_to_date_shift)
            
            all_stats['total_copied'] += stats['copied']
            all_stats['total_errors'] += stats['errors']
            all_stats['total_unmatched'].extend(stats['unmatched'])
            all_stats['folders_processed'].append({
                'source': source_folder,
                'output': output_dest,
                'copied': stats['copied'],
                'errors': stats['errors'],
                'unmatched': len(stats['unmatched'])
            })
            
            print(f"\n✓ Completed: {source_folder}")
            print(f"  Files copied: {stats['copied']}")
            print(f"  Errors: {stats['errors']}")
            print(f"  Unmatched: {len(stats['unmatched'])}")
            
        except Exception as e:
            logging.error(f"Error processing folder {source_folder}: {e}")
            print(f"✗ Error processing {source_folder}: {e}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Save unmatched files log
    if all_stats['total_unmatched']:
        try:
            unmatched_df = pd.DataFrame(all_stats['total_unmatched'])
            unmatched_df.to_csv(log_file_unmatched, index=False)
            print(f"\n⚠️  Unmatched files logged to: {log_file_unmatched}")
        except Exception as e:
            logging.error(f"Error logging unmatched files: {e}")
    
    # Print final summary
    print("\n" + "="*70)
    print("DE-IDENTIFICATION COMPLETE (COPY MODE)")
    print("="*70)
    print(f"Total files copied:     {all_stats['total_copied']}")
    print(f"Total errors:           {all_stats['total_errors']}")
    print(f"Total unmatched:        {len(all_stats['total_unmatched'])}")
    print(f"Time elapsed:           {duration}")
    print(f"\nOriginals preserved at: {', '.join(folders_to_search)}")
    print(f"De-identified copies:   {output_base_dir}")
    print("="*70)
    
    # Save summary to file
    try:
        with open(log_file_summary, 'w') as f:
            f.write("DE-IDENTIFICATION SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Processing date: {datetime.now()}\n")
            f.write(f"Duration: {duration}\n\n")
            f.write(f"Total files copied: {all_stats['total_copied']}\n")
            f.write(f"Total errors: {all_stats['total_errors']}\n")
            f.write(f"Total unmatched: {len(all_stats['total_unmatched'])}\n\n")
            
            f.write("Folders processed:\n")
            for folder_stat in all_stats['folders_processed']:
                f.write(f"\nSource: {folder_stat['source']}\n")
                f.write(f"Output: {folder_stat['output']}\n")
                f.write(f"  Copied: {folder_stat['copied']}\n")
                f.write(f"  Errors: {folder_stat['errors']}\n")
                f.write(f"  Unmatched: {folder_stat['unmatched']}\n")
        
        print(f"\nSummary saved to: {log_file_summary}")
    except Exception as e:
        logging.error(f"Error saving summary: {e}")
    
    print("\n✓ All originals remain untouched!")
    print("✓ De-identified copies ready for use.\n")

if __name__ == '__main__':
    main()
