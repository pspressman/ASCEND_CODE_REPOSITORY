#!/usr/bin/env python3
"""
LLMAnon to PHI Timestamps Bridge Script
Converts LLMAnon CSV output to PHI timestamp format for audio redaction (Round 2).

This script:
1. Reads human-reviewed LLMAnon CSV (REVIEW='Y' rows only)
2. Matches flagged text to WhisperX JSON word timestamps
3. Generates timestamp CSV for phi_inplace_deidentifier_MASTER_CSV.py
4. Automatically skips CliniDeID markers (already handled in Round 1)

Author: ASCEND Team
Date: October 31, 2025
"""

import os
import json
import csv
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ============================================================================
# CONFIGURATION - BATCH 2 (FOCUSED)
# ============================================================================

# Input CSV from LLMAnon (same for both batches)
LLMANON_CSV_PATH = r"C:\LocalData\LLMAnon\LLM-Anon-reviewed.csv"

# Search directories - BATCH 2: Focused WordTimings
JSON_SEARCH_DIRS = [
    r"C:\LocalData\ASCEND_PHI\DeID\focused_audio_anon\WordTimings_focused",
]

WAV_SEARCH_DIRS = [
    r"Z:\ASCEND_ANON\McAdams\CleanedSplicedAudio",
    r"Z:\ASCEND_ANON\McAdams\OriginalSplicedAudio",
]

# Output directory - BATCH 2
OUTPUT_DIR = r"C:\LocalData\LLMAnon\Batch2_Focused_Timestamps"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "phi_timestamps_Batch2_Focused.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, f"bridge_Batch2_Focused_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
TEXT_NOT_FOUND_LOG = os.path.join(OUTPUT_DIR, "text_not_found_Batch2.txt")
CLINIDEID_MARKERS_LOG = os.path.join(OUTPUT_DIR, "clinideid_markers_skipped_Batch2.txt")
MISSING_FILES_LOG = os.path.join(OUTPUT_DIR, "missing_files_Batch2.txt")

# Processing parameters
TIMESTAMP_BUFFER = 0.15  # seconds to add/subtract from timestamps

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure comprehensive logging."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# SUBSTRING FILE MATCHING
# ============================================================================

def find_file_by_substring(search_id: str, file_index: Dict[str, Path], 
                          file_type: str = "file") -> Optional[Path]:
    """
    Find file by searching for search_id as substring in indexed filenames.
    
    The FILE-basename in CSV (like "5746613-1-18-18") should appear somewhere
    in the actual filename (like "5746613-1-18-18.deid.piiCategoryTag.json").
    
    Args:
        search_id: The core ID to search for (from CSV FILE-basename column)
        file_index: Dictionary of actual_filename -> Path
        file_type: Type of file for logging (e.g., "JSON", "WAV")
        
    Returns:
        Path to matching file, or None if not found
    """
    # Search for any filename that contains the search_id
    matches = []
    for filename, file_path in file_index.items():
        if search_id in filename:
            matches.append((filename, file_path))
    
    if not matches:
        return None
    
    if len(matches) == 1:
        logger.debug(f"  Found {file_type} match: {search_id} in {matches[0][0]}")
        return matches[0][1]
    
    # Multiple matches - prefer shortest (most specific)
    matches_sorted = sorted(matches, key=lambda x: len(x[0]))
    logger.debug(f"  Found {len(matches)} {file_type} matches for '{search_id}', using shortest: {matches_sorted[0][0]}")
    return matches_sorted[0][1]


# ============================================================================
# CLINIDEID MARKER DETECTION
# ============================================================================

def is_clinideid_marker(text: str) -> bool:
    """
    Check if text is a CliniDeID marker like [***NAME***], [***ADDRESS***], etc.
    These are already redacted in Round 1 and won't be in the JSON.
    
    Args:
        text: Text to check
        
    Returns:
        True if this is a CliniDeID marker
    """
    if not text:
        return False
    
    text = text.strip()
    
    # Pattern: [***SOMETHING***]
    pattern = r'^\[\*\*\*[A-Z][A-Z\s]+\*\*\*\]$'
    
    if re.match(pattern, text):
        return True
    
    # Also check for variations like [**NAME**] or [***name***]
    alt_pattern = r'^\[\*+[A-Za-z\s]+\*+\]$'
    
    return bool(re.match(alt_pattern, text))

# ============================================================================
# FUZZY FILE MATCHING
# ============================================================================

def find_file_by_fuzzy_match(normalized_id: str, file_index: Dict[str, Path], 
                              file_type: str = "file") -> Optional[Path]:
    """
    Find file using fuzzy matching when exact match fails.
    
    Strategy:
    1. Try exact match first
    2. Try "contains" match - find any indexed filename that contains the normalized_id
    3. If multiple matches, prefer shortest (most specific) match
    
    Args:
        normalized_id: Normalized participant ID to search for
        file_index: Dictionary of normalized IDs to file paths
        file_type: Type of file for logging (e.g., "JSON", "WAV")
        
    Returns:
        Path to matching file, or None if not found
    """
    # Strategy 1: Exact match
    if normalized_id in file_index:
        logger.debug(f"  Found {file_type} by exact match: {normalized_id}")
        return file_index[normalized_id]
    
    # Strategy 2: "Contains" match - the normalized_id is a substring of the indexed key
    # Example: normalized_id="5746613-1-18-18" matches indexed key="5746613-1-18-18_extra"
    matches = []
    for indexed_id, file_path in file_index.items():
        if normalized_id in indexed_id or indexed_id in normalized_id:
            matches.append((indexed_id, file_path))
    
    if not matches:
        logger.debug(f"  No {file_type} match found for: {normalized_id}")
        return None
    
    if len(matches) == 1:
        logger.debug(f"  Found {file_type} by fuzzy match: {normalized_id} -> {matches[0][0]}")
        return matches[0][1]
    
    # Multiple matches - prefer shortest (most specific)
    matches_sorted = sorted(matches, key=lambda x: len(x[0]))
    logger.debug(f"  Found {len(matches)} {file_type} fuzzy matches for {normalized_id}")
    logger.debug(f"  Using shortest match: {matches_sorted[0][0]}")
    return matches_sorted[0][1]


# ============================================================================
# FILENAME NORMALIZATION (FOR MATCHING ONLY)
# ============================================================================

def normalize_filename(filename: str) -> str:
    """
    Normalize filename to extract core participant ID.
    Used ONLY for matching - actual filenames used in output CSV.
    
    Strips:
    - Extensions: .wav, .json, .txt, .deid.piiCategoryTag (compound extensions first!)
    - Prefixes: anon_, anonymized_, deidentified_
    - Suffixes: _clean, _cleaned, _processed, _transcript, _deidentified
    
    Args:
        filename: Original filename
        
    Returns:
        Normalized participant ID
    """
    # Strip extensions - CRITICAL: Check compound/longer extensions FIRST
    # Order matters! Longer extensions before shorter ones
    extensions = [
        '.deid.piiCategoryTag.json',
        '.deid.piiCategoryTag',
        '.json',
        '.wav',
        '.txt'
    ]
    
    for ext in extensions:
        if filename.lower().endswith(ext.lower()):
            filename = filename[:-len(ext)]
            break  # Only strip once
    
    # Strip prefixes (case-insensitive)
    for prefix in ['anon_', 'anonymized_', 'deidentified_']:
        if filename.lower().startswith(prefix.lower()):
            filename = filename[len(prefix):]
            break
    
    # Strip suffixes (case-insensitive)
    for suffix in ['_clean', '_cleaned', '_processed', '_transcript', '_deidentified']:
        if filename.lower().endswith(suffix.lower()):
            filename = filename[:-len(suffix)]
            break
    
    return filename

# ============================================================================
# FILE INDEX BUILDERS
# ============================================================================

def build_json_index(search_dirs: List[str]) -> Dict[str, Path]:
    """
    Build index of all JSON files using ACTUAL filenames as keys.
    Uses recursive search.
    NO normalization - we'll do substring matching later.
    
    Args:
        search_dirs: List of directories to search
        
    Returns:
        Dictionary mapping ACTUAL filename (basename) to JSON file path
    """
    logger.info("Building JSON file index...")
    json_index = {}
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            logger.warning(f"JSON search directory does not exist: {search_dir}")
            continue
        
        logger.info(f"  Searching: {search_dir}")
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = Path(root) / file
                    
                    # Use ACTUAL filename as key (no normalization!)
                    actual_filename = file
                    
                    if actual_filename in json_index:
                        logger.debug(f"  Duplicate JSON file: {actual_filename}")
                        logger.debug(f"    Existing: {json_index[actual_filename]}")
                        logger.debug(f"    New: {json_path}")
                    
                    json_index[actual_filename] = json_path
    
    logger.info(f"  Indexed {len(json_index)} JSON files")
    return json_index


def build_wav_index(search_dirs: List[str]) -> Dict[str, Path]:
    """
    Build index of all WAV files using ACTUAL filenames as keys.
    Uses recursive search.
    NO normalization - we'll do substring matching later.
    IMPORTANT: Stores actual Path objects for exact filename retrieval.
    
    Args:
        search_dirs: List of directories to search
        
    Returns:
        Dictionary mapping ACTUAL filename (basename) to WAV file path
    """
    logger.info("Building WAV file index...")
    wav_index = {}
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            logger.warning(f"WAV search directory does not exist: {search_dir}")
            continue
        
        logger.info(f"  Searching: {search_dir}")
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_path = Path(root) / file
                    
                    # Use ACTUAL filename as key (no normalization!)
                    actual_filename = file
                    
                    if actual_filename in wav_index:
                        logger.debug(f"  Duplicate WAV file: {actual_filename}")
                        logger.debug(f"    Existing: {wav_index[actual_filename]}")
                        logger.debug(f"    New: {wav_path}")
                    
                    wav_index[actual_filename] = wav_path
    
    logger.info(f"  Indexed {len(wav_index)} WAV files")
    return wav_index

# ============================================================================
# JSON PROCESSING
# ============================================================================

def load_whisperx_json(json_file: Path) -> Optional[Dict]:
    """
    Load WhisperX JSON transcript with word-level timestamps.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary, or None on error
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON: {json_file.name}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON {json_file}: {e}")
        return None

# ============================================================================
# MULTI-WORD PHRASE MATCHING
# ============================================================================

def find_phrase_timestamps(phrase: str, json_data: Dict, participant_id: str) -> List[Tuple[float, float]]:
    """
    Find all occurrences of a multi-word phrase in WhisperX JSON.
    Handles both single words and multi-word phrases.
    Strips punctuation from both search phrase and JSON words.
    
    Handles TWO JSON structures:
    1. Top-level 'words' array (newer WhisperX format)
    2. 'segments' containing 'words' arrays (older format)
    
    Args:
        phrase: The text phrase to search for (can be multi-word)
        json_data: WhisperX JSON data
        participant_id: Participant ID for logging
        
    Returns:
        List of (start, end) timestamp tuples with buffer applied
    """
    timestamps = []
    
    if not json_data or not phrase:
        return timestamps
    
    # Normalize phrase: strip punctuation, lowercase, split into words
    phrase_normalized = phrase.strip().strip('.,!?\'"').lower()
    search_words = phrase_normalized.split()
    
    if not search_words:
        return timestamps
    
    logger.debug(f"Searching for phrase: '{phrase}' → {search_words}")
    
    # Collect all word_info dictionaries regardless of structure
    word_list = []
    
    # Structure 1: Top-level 'words' array (CURRENT WhisperX format)
    if 'words' in json_data and isinstance(json_data['words'], list):
        word_list = json_data['words']
        logger.debug(f"Using top-level 'words' structure for {participant_id}")
    
    # Structure 2: 'segments' containing 'words' arrays (LEGACY format)
    elif 'segments' in json_data and isinstance(json_data['segments'], list):
        for segment in json_data['segments']:
            if 'words' in segment and isinstance(segment['words'], list):
                word_list.extend(segment['words'])
        logger.debug(f"Using 'segments' → 'words' structure for {participant_id}")
    
    else:
        logger.warning(f"Invalid JSON structure for {participant_id} - no 'words' or 'segments' found")
        return timestamps
    
    # Normalize all words in JSON
    normalized_word_list = []
    for word_info in word_list:
        word = word_info.get('word', '').strip()
        word_normalized = word.strip('.,!?\'"').lower()
        normalized_word_list.append({
            'normalized': word_normalized,
            'original': word,
            'start': word_info.get('start'),
            'end': word_info.get('end')
        })
    
    # Search for consecutive word matches
    num_search_words = len(search_words)
    
    for i in range(len(normalized_word_list) - num_search_words + 1):
        # Check if the next N words match our search phrase
        match = True
        for j in range(num_search_words):
            if normalized_word_list[i + j]['normalized'] != search_words[j]:
                match = False
                break
        
        if match:
            # Found a match! Extract timestamps from first and last word
            first_word = normalized_word_list[i]
            last_word = normalized_word_list[i + num_search_words - 1]
            
            start = first_word['start']
            end = last_word['end']
            
            if start is not None and end is not None:
                # Apply buffer: ±0.15 seconds
                buffered_start = max(0, start - TIMESTAMP_BUFFER)
                buffered_end = end + TIMESTAMP_BUFFER
                
                timestamps.append((buffered_start, buffered_end))
                
                matched_words = ' '.join([normalized_word_list[i + k]['original'] 
                                         for k in range(num_search_words)])
                logger.debug(f"Found '{phrase}' (matched '{matched_words}') at [{start:.3f}, {end:.3f}] "
                           f"→ buffered [{buffered_start:.3f}, {buffered_end:.3f}]")
    
    return timestamps

# ============================================================================
# LLMANON CSV PROCESSING
# ============================================================================

def load_llmanon_csv(csv_path: str) -> List[Dict]:
    """
    Load LLMAnon CSV and extract all rows.
    All rows have been human-reviewed (REVIEW='DONE'), so we process everything.
    
    Required columns: FILE-basename, TEXT
    Optional columns: RISK, CATEGORY, SUGGESTED_REPLACEMENT
    
    Args:
        csv_path: Path to LLMAnon CSV file
        
    Returns:
        List of dictionaries with items to process
    """
    logger.info(f"Loading LLMAnon CSV: {csv_path}")
    
    items = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Validate required columns
            if 'FILE-basename' not in reader.fieldnames or 'TEXT' not in reader.fieldnames:
                logger.error("Required columns missing from CSV: FILE-basename, TEXT")
                logger.error(f"Available columns: {reader.fieldnames}")
                return []
            
            for row_num, row in enumerate(reader, 2):  # Start at 2 (header is row 1)
                # Check for required data
                file_basename = row.get('FILE-basename', '').strip()
                text = row.get('TEXT', '').strip()
                
                if not file_basename or not text:
                    logger.warning(f"Row {row_num}: Missing FILE-basename or TEXT - skipping")
                    continue
                
                # Extract optional fields (default to 'UNKNOWN' if missing)
                risk = row.get('RISK', 'UNKNOWN').strip() if row.get('RISK', '').strip() else 'UNKNOWN'
                category = row.get('CATEGORY', 'UNKNOWN').strip() if row.get('CATEGORY', '').strip() else 'UNKNOWN'
                suggested_replacement = row.get('SUGGESTED_REPLACEMENT', '').strip()
                
                items.append({
                    'row_num': row_num,
                    'file_basename': file_basename,
                    'text': text,
                    'risk': risk,
                    'category': category,
                    'suggested_replacement': suggested_replacement
                })
        
        logger.info(f"Loaded {len(items)} items from CSV (all reviewed)")
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return []
    
    return items

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_llmanon_to_timestamps(llmanon_csv: str, json_dirs: List[str], 
                                  wav_dirs: List[str], output_csv: str) -> Dict:
    """
    Main processing function: converts LLMAnon CSV to PHI timestamps CSV.
    
    Args:
        llmanon_csv: Path to LLMAnon CSV
        json_dirs: Directories to search for JSON files
        wav_dirs: Directories to search for WAV files
        output_csv: Output CSV file path
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info("="*80)
    logger.info("LLMANON TO PHI TIMESTAMPS BRIDGE")
    logger.info("="*80)
    
    stats = {
        'items_in_csv': 0,
        'clinideid_markers_skipped': 0,
        'items_processed': 0,
        'json_files_found': 0,
        'json_files_missing': 0,
        'wav_files_found': 0,
        'wav_files_missing': 0,
        'text_found_in_json': 0,
        'text_not_found_in_json': 0,
        'total_timestamps_generated': 0
    }
    
    # Build file indexes
    json_index = build_json_index(json_dirs)
    wav_index = build_wav_index(wav_dirs)
    
    if not json_index:
        logger.error("No JSON files found. Cannot proceed.")
        return stats
    
    if not wav_index:
        logger.error("No WAV files found. Cannot proceed.")
        return stats
    
    # Load LLMAnon CSV
    items = load_llmanon_csv(llmanon_csv)
    stats['items_in_csv'] = len(items)
    
    if not items:
        logger.error("No items found in CSV. Exiting.")
        return stats
    
    # Prepare output
    csv_rows = []
    text_not_found = []
    clinideid_markers = []
    missing_files = []
    
    # Process each item
    for item in items:
        row_num = item['row_num']
        file_basename = item['file_basename']
        text = item['text']
        risk = item['risk']
        category = item['category']
        
        logger.info(f"\nRow {row_num}: Processing '{text}' from {file_basename}")
        
        # Check if this is a CliniDeID marker
        if is_clinideid_marker(text):
            logger.info(f"  ✓ CliniDeID marker detected - skipping (already handled in Round 1)")
            clinideid_markers.append(f"Row {row_num}: {text} (from {file_basename})")
            stats['clinideid_markers_skipped'] += 1
            continue
        
        # Use file_basename directly for substring search (no normalization needed!)
        search_id = file_basename
        logger.info(f"  Search ID: {search_id}")
        
        # Find matching JSON file using substring search
        json_file = find_file_by_substring(search_id, json_index, "JSON")
        if not json_file:
            logger.warning(f"  ✗ JSON file not found containing '{search_id}'")
            missing_files.append(f"Row {row_num}: JSON not found containing {file_basename}")
            stats['json_files_missing'] += 1
            continue
        
        logger.info(f"  ✓ Found JSON: {json_file.name}")
        stats['json_files_found'] += 1
        
        # Find matching WAV file using substring search
        wav_file = find_file_by_substring(search_id, wav_index, "WAV")
        if not wav_file:
            logger.warning(f"  ✗ WAV file not found containing '{search_id}'")
            missing_files.append(f"Row {row_num}: WAV not found containing {file_basename}")
            stats['wav_files_missing'] += 1
            continue
        
        logger.info(f"  ✓ Found WAV: {wav_file.name}")
        stats['wav_files_found'] += 1
        
        # Load JSON data
        json_data = load_whisperx_json(json_file)
        if not json_data:
            logger.error(f"  ✗ Failed to load JSON - skipping")
            stats['json_files_missing'] += 1
            continue
        
        # Find timestamps for this text
        timestamps = find_phrase_timestamps(text, json_data, search_id)
        
        if timestamps:
            logger.info(f"  ✓ Found {len(timestamps)} occurrence(s) in JSON")
            stats['text_found_in_json'] += 1
            stats['total_timestamps_generated'] += len(timestamps)
            
            # Add each timestamp as a separate CSV row
            for start, end in timestamps:
                csv_rows.append({
                    'participant_id': search_id,
                    'start_time': f"{start:.3f}",
                    'end_time': f"{end:.3f}",
                    'phi_word': text,
                    'wav_filename': wav_file.name,  # EXACT filename as on disk
                    'json_source': json_file.name,
                    'risk_level': risk,
                    'category': category
                })
        else:
            logger.warning(f"  ✗ Text not found in JSON")
            text_not_found.append(f"Row {row_num}: '{text}' not found in {file_basename}")
            stats['text_not_found_in_json'] += 1
        
        stats['items_processed'] += 1
    
    # Write CSV output
    if csv_rows:
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['participant_id', 'start_time', 'end_time', 'phi_word', 
                             'wav_filename', 'json_source', 'risk_level', 'category']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            logger.info(f"\n✓ Successfully wrote {len(csv_rows)} timestamp rows to: {output_csv}")
        except Exception as e:
            logger.error(f"Error writing CSV: {e}")
    else:
        logger.warning("No timestamp data to write to CSV")
    
    # Write logs
    if text_not_found:
        try:
            with open(TEXT_NOT_FOUND_LOG, 'w', encoding='utf-8') as f:
                f.write("Text Not Found in JSON Transcripts\n")
                f.write("="*80 + "\n\n")
                f.write("These items were flagged by LLMAnon but could not be found in the JSON.\n")
                f.write("This may indicate:\n")
                f.write("  - LLM was overzealous / context-based identification\n")
                f.write("  - Text differs between transcript versions\n")
                f.write("  - Paraphrased or summarized content\n\n")
                for item in text_not_found:
                    f.write(f"{item}\n")
            
            logger.info(f"✓ Wrote {len(text_not_found)} not-found items to: {TEXT_NOT_FOUND_LOG}")
        except Exception as e:
            logger.error(f"Error writing text-not-found log: {e}")
    
    if clinideid_markers:
        try:
            with open(CLINIDEID_MARKERS_LOG, 'w', encoding='utf-8') as f:
                f.write("CliniDeID Markers Skipped (Already Handled in Round 1)\n")
                f.write("="*80 + "\n\n")
                f.write("These are markers from CliniDeID output like [***NAME***].\n")
                f.write("They are already redacted in Round 1 audio and don't need timestamps.\n\n")
                for item in clinideid_markers:
                    f.write(f"{item}\n")
            
            logger.info(f"✓ Wrote {len(clinideid_markers)} CliniDeID markers to: {CLINIDEID_MARKERS_LOG}")
        except Exception as e:
            logger.error(f"Error writing CliniDeID markers log: {e}")
    
    if missing_files:
        try:
            with open(MISSING_FILES_LOG, 'w', encoding='utf-8') as f:
                f.write("Missing JSON or WAV Files\n")
                f.write("="*80 + "\n\n")
                for item in missing_files:
                    f.write(f"{item}\n")
            
            logger.info(f"✓ Wrote {len(missing_files)} missing files to: {MISSING_FILES_LOG}")
        except Exception as e:
            logger.error(f"Error writing missing files log: {e}")
    
    return stats


def print_summary(stats: Dict):
    """Print a formatted summary of processing results."""
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Items in CSV:                  {stats['items_in_csv']}")
    logger.info(f"CliniDeID markers skipped:     {stats['clinideid_markers_skipped']}")
    logger.info(f"Items processed:               {stats['items_processed']}")
    logger.info(f"JSON files found:              {stats['json_files_found']}")
    logger.info(f"JSON files missing:            {stats['json_files_missing']}")
    logger.info(f"WAV files found:               {stats['wav_files_found']}")
    logger.info(f"WAV files missing:             {stats['wav_files_missing']}")
    logger.info(f"Text found in JSON:            {stats['text_found_in_json']}")
    logger.info(f"Text NOT found in JSON:        {stats['text_not_found_in_json']}")
    logger.info(f"Total timestamps generated:    {stats['total_timestamps_generated']}")
    logger.info("="*80)
    
    if stats['total_timestamps_generated'] > 0:
        logger.info(f"\n✓ SUCCESS: Timestamp CSV ready at: {OUTPUT_CSV}")
        logger.info(f"\nNext step: Run phi_inplace_deidentifier_MASTER_CSV.py with this CSV")
    else:
        logger.error(f"\n✗ WARNING: No timestamps generated. Check logs.")
    
    logger.info(f"\nFull log: {LOG_FILE}")
    if stats['text_not_found_in_json'] > 0:
        logger.info(f"Text not found log: {TEXT_NOT_FOUND_LOG}")
    if stats['clinideid_markers_skipped'] > 0:
        logger.info(f"CliniDeID markers log: {CLINIDEID_MARKERS_LOG}")
    if stats['json_files_missing'] > 0 or stats['wav_files_missing'] > 0:
        logger.info(f"Missing files log: {MISSING_FILES_LOG}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    logger.info(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate input CSV
    if not os.path.exists(LLMANON_CSV_PATH):
        logger.error(f"LLMAnon CSV not found: {LLMANON_CSV_PATH}")
        logger.error("Please update LLMANON_CSV_PATH in the script configuration.")
        return
    
    # Validate search directories
    valid_json_dirs = [d for d in JSON_SEARCH_DIRS if os.path.exists(d)]
    if not valid_json_dirs:
        logger.error("No valid JSON search directories found.")
        logger.error("Please update JSON_SEARCH_DIRS in the script configuration.")
        return
    
    valid_wav_dirs = [d for d in WAV_SEARCH_DIRS if os.path.exists(d)]
    if not valid_wav_dirs:
        logger.error("No valid WAV search directories found.")
        logger.error("Please update WAV_SEARCH_DIRS in the script configuration.")
        return
    
    logger.info(f"LLMAnon CSV: {LLMANON_CSV_PATH}")
    logger.info(f"Searching for JSON files in {len(valid_json_dirs)} directories")
    logger.info(f"Searching for WAV files in {len(valid_wav_dirs)} directories")
    
    # Process all items
    stats = process_llmanon_to_timestamps(
        llmanon_csv=LLMANON_CSV_PATH,
        json_dirs=valid_json_dirs,
        wav_dirs=valid_wav_dirs,
        output_csv=OUTPUT_CSV
    )
    
    # Print summary
    print_summary(stats)
    
    # Calculate runtime
    end_time = datetime.now()
    runtime = end_time - start_time
    logger.info(f"\nTotal runtime: {runtime}")
    logger.info(f"Script completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
