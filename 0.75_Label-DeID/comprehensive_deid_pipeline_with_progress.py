#!/usr/bin/env python3
"""
SIMPLE FIND-AND-REPLACE DE-IDENTIFICATION WITH DATE SHIFTING

Does two things:
1. Find MRNs → Replace with UIDs
2. Find dates in context with MRN → Shift by that patient's date_shift_days

Each patient has their own date shift value.
"""

import os
import pandas as pd
import shutil
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

key_file = "/path/to/user/Desktop/deid_key.csv"
source_dir = "/path/to/volumes/Databackup2025/DELIVERY/ASCEND_AP_Label_DeID"
output_dir = "/path/to/volumes/Databackup2025/DELIVERY/ASCEND_FULL_DEID"

# File types to scan content
scan_content_extensions = ['.csv', '.txt', '.json', '.md', '.log']

log_file = "/path/to/user/MyDevelopment/Logs/SimpleDeID_Log.txt"

# ============================================================================
# SETUP
# ============================================================================

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Load key file
print("Loading key file...")
key_df = pd.read_csv(key_file)
key_df["mrn"] = key_df["mrn"].astype(str)
key_df["UID"] = key_df["UID"].astype(str)

# Create mappings
mrn_to_uid = dict(zip(key_df["mrn"], key_df["UID"]))
mrn_to_date_shift = dict(zip(key_df["mrn"], key_df["date_shift_days"]))

# Sort MRNs by length (longest first) to avoid partial replacements
mrns_sorted = sorted(mrn_to_uid.keys(), key=len, reverse=True)

print(f"Loaded {len(mrn_to_uid)} MRN → UID mappings")

# ============================================================================
# DATE FUNCTIONS
# ============================================================================

def shift_date(date_str: str, shift_days: int) -> str:
    """
    Shift a date by specified number of days.
    Handles formats: MM-DD-YY, M-D-YY, MM-DD-YYYY, M-D-YYYY
    Returns shifted date in same format as input.
    """
    try:
        # Try to parse various formats
        for fmt in ["%m-%d-%y", "%m-%d-%Y", "%-m-%-d-%y", "%-m-%-d-%Y"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                shifted = date_obj + timedelta(days=int(shift_days))
                
                # Return in same format as input
                if len(date_str.split('-')[2]) == 2:  # 2-digit year
                    return shifted.strftime("%m-%d-%y")
                else:  # 4-digit year
                    return shifted.strftime("%m-%d-%Y")
            except:
                continue
        
        # If none worked, try manual parsing
        parts = date_str.split('-')
        if len(parts) == 3:
            month, day, year = parts
            if len(year) == 2:
                full_year = f"20{year}" if int(year) <= 30 else f"19{year}"
            else:
                full_year = year
            
            date_obj = datetime(int(full_year), int(month), int(day))
            shifted = date_obj + timedelta(days=int(shift_days))
            
            if len(year) == 2:
                return shifted.strftime("%m-%d-%y")
            else:
                return shifted.strftime("%m-%d-%Y")
    
    except Exception as e:
        logging.warning(f"Could not shift date {date_str}: {e}")
        return date_str
    
    return date_str

# ============================================================================
# REPLACEMENT FUNCTIONS
# ============================================================================

def find_mrn_in_text(text: str) -> str:
    """Find the first MRN that appears in text"""
    for mrn in mrns_sorted:  # Check longest first
        if mrn in text:
            return mrn
    return None

def replace_in_text(text: str, context_mrn: str = None) -> tuple[str, int]:
    """
    Replace MRNs and dates in text.
    
    Args:
        text: The text to process
        context_mrn: If known (e.g., from filename), which patient's dates to shift
    
    Returns:
        (new_text, number_of_replacements)
    """
    result = text
    count = 0
    
    # If no context MRN provided, try to find one in the text
    if not context_mrn:
        context_mrn = find_mrn_in_text(text)
    
    # Step 1: Replace MRNs with UIDs
    for mrn in mrns_sorted:  # Longest first
        uid = mrn_to_uid[mrn]
        if mrn in result:
            occurrences = result.count(mrn)
            result = result.replace(mrn, uid)
            count += occurrences
    
    # Step 2: Shift dates if we have a context MRN
    if context_mrn and context_mrn in mrn_to_date_shift:
        date_shift = mrn_to_date_shift[context_mrn]
        
        # Find standalone dates: MM-DD-YY or M-D-YY (with word boundaries)
        # Pattern matches dates like "10-22-21" but not "1234-10-22-21"
        date_pattern = r'\b(\d{1,2}-\d{1,2}-\d{2,4})\b'
        
        def replace_date(match):
            nonlocal count
            original_date = match.group(1)
            # Verify it looks like a date (month 1-12, day 1-31)
            parts = original_date.split('-')
            if len(parts) == 3:
                try:
                    month, day = int(parts[0]), int(parts[1])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        shifted = shift_date(original_date, date_shift)
                        if shifted != original_date:
                            count += 1
                        return shifted
                except:
                    pass
            return original_date
        
        result = re.sub(date_pattern, replace_date, result)
    
    return result, count

def replace_in_filename(filename: str) -> tuple[str, str, int]:
    """
    Replace MRNs and dates in filename.
    Returns: (new_filename, context_mrn, num_replacements)
    """
    result = filename
    context_mrn = None
    count = 0
    
    # Find MRN in filename (for date shifting context)
    for mrn in mrns_sorted:
        if mrn in result:
            context_mrn = mrn
            break
    
    # Replace MRNs with UIDs
    for mrn in mrns_sorted:
        uid = mrn_to_uid[mrn]
        if mrn in result:
            result = result.replace(mrn, uid)
            count += 1
    
    # Shift dates if we found an MRN
    if context_mrn and context_mrn in mrn_to_date_shift:
        date_shift = mrn_to_date_shift[context_mrn]
        
        # Find dates in filename
        date_pattern = r'\b(\d{1,2}-\d{1,2}-\d{2,4})\b'
        
        def replace_date(match):
            nonlocal count
            original_date = match.group(1)
            parts = original_date.split('-')
            if len(parts) == 3:
                try:
                    month, day = int(parts[0]), int(parts[1])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        shifted = shift_date(original_date, date_shift)
                        if shifted != original_date:
                            count += 1
                        return shifted
                except:
                    pass
            return original_date
        
        result = re.sub(date_pattern, replace_date, result)
    
    return result, context_mrn, count

def process_csv(source_path: str, output_path: str) -> dict:
    """
    Process CSV file row by row.
    Each row may have a different MRN, so dates shift differently per row.
    """
    try:
        df = pd.read_csv(source_path, dtype=str)
        total_replacements = 0
        
        # Process each row
        for idx in df.index:
            # Find MRN in this row (for date shifting context)
            row_text = ' '.join(str(df.at[idx, col]) for col in df.columns if pd.notna(df.at[idx, col]))
            row_mrn = find_mrn_in_text(row_text)
            
            # Process each cell in this row
            for col in df.columns:
                if pd.notna(df.at[idx, col]):
                    original = str(df.at[idx, col])
                    replaced, count = replace_in_text(original, context_mrn=row_mrn)
                    if replaced != original:
                        df.at[idx, col] = replaced
                        total_replacements += count
        
        # Save de-identified CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return {'status': 'success', 'replacements': total_replacements}
    
    except Exception as e:
        logging.error(f"Error processing CSV {source_path}: {e}")
        return {'status': 'error', 'error': str(e)}

def process_text_file(source_path: str, output_path: str, context_mrn: str = None) -> dict:
    """Process text file"""
    try:
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        new_content, replacements = replace_in_text(content, context_mrn=context_mrn)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return {'status': 'success', 'replacements': replacements}
    
    except Exception as e:
        logging.error(f"Error processing text file {source_path}: {e}")
        return {'status': 'error', 'error': str(e)}

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_file(source_path: str, relative_path: str) -> dict:
    """Process a single file"""
    result = {'status': 'success', 'replacements': 0}
    
    # Get filename parts
    dirname = os.path.dirname(relative_path)
    filename = os.path.basename(source_path)
    name, ext = os.path.splitext(filename)
    
    # Replace MRNs and dates in filename
    new_filename, context_mrn, filename_replacements = replace_in_filename(filename)
    result['replacements'] += filename_replacements
    
    # Build output path
    output_subdir = os.path.join(output_dir, dirname)
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, new_filename)
    
    # Process file content if applicable
    if ext.lower() == '.csv':
        # CSV files: process row by row (each row may have different MRN/date shift)
        content_result = process_csv(source_path, output_path)
        result['replacements'] += content_result.get('replacements', 0)
        if content_result['status'] == 'error':
            result['status'] = 'error'
            result['error'] = content_result['error']
    
    elif ext.lower() in scan_content_extensions:
        # Text files: use MRN from filename as context
        content_result = process_text_file(source_path, output_path, context_mrn=context_mrn)
        result['replacements'] += content_result.get('replacements', 0)
        if content_result['status'] == 'error':
            result['status'] = 'error'
            result['error'] = content_result['error']
    
    else:
        # Other files: just copy (possibly with renamed filename)
        shutil.copy2(source_path, output_path)
    
    if result['replacements'] > 0 or new_filename != filename:
        logging.info(f"{filename} → {new_filename} ({result['replacements']} replacements)")
    
    return result

def main():
    print("\n" + "="*70)
    print("FIND-AND-REPLACE DE-IDENTIFICATION WITH DATE SHIFTING")
    print("="*70)
    print(f"\nSource: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"\nWill replace {len(mrn_to_uid)} MRNs with UIDs")
    print("Will shift dates using patient-specific date_shift_days")
    print("="*70 + "\n")
    
    # First pass: count total files
    print("Scanning directory to count files...")
    all_files = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in files:
            if filename.startswith('.'):
                continue
            
            source_path = os.path.join(root, filename)
            relative_path = os.path.relpath(source_path, source_dir)
            all_files.append((source_path, relative_path))
    
    total_files = len(all_files)
    print(f"Found {total_files} files to process\n")
    
    stats = {
        'total_files': 0,
        'total_replacements': 0,
        'errors': 0
    }
    
    # Process files with progress bar
    with tqdm(total=total_files, 
              desc="Processing files",
              unit="file",
              ncols=100,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for source_path, relative_path in all_files:
            filename = os.path.basename(source_path)
            
            # Update progress bar description with current file
            pbar.set_postfix_str(f"Current: {filename[:40]}...")
            
            stats['total_files'] += 1
            result = process_file(source_path, relative_path)
            
            if result['status'] == 'error':
                stats['errors'] += 1
            
            stats['total_replacements'] += result['replacements']
            
            # Update progress bar
            pbar.update(1)
    
    # Summary
    print("\n" + "="*70)
    print("DE-IDENTIFICATION COMPLETE")
    print("="*70)
    print(f"Total files processed:           {stats['total_files']}")
    print(f"Total replacements:              {stats['total_replacements']}")
    print(f"  (MRNs + dates combined)")
    print(f"Errors:                          {stats['errors']}")
    print(f"\nOriginals preserved in:          {source_dir}")
    print(f"De-identified copies in:         {output_dir}")
    print(f"Log file:                        {log_file}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
