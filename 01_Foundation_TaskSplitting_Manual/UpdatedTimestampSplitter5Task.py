#!/usr/bin/env python3
"""
Audio and Transcript Segmentation by Task Type
Segments recordings into task-specific audio and transcript files based on timestamps.

Usage:
    python segment_audio_by_task.py --test-mode --test-files test_list.txt
    python segment_audio_by_task.py --full
"""

import os
import re
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

# Audio processing
try:
    from pydub import AudioSegment
    AUDIO_LIBRARY = "pydub"
except ImportError:
    try:
        import soundfile as sf
        import numpy as np
        AUDIO_LIBRARY = "soundfile"
    except ImportError:
        raise ImportError("Please install either pydub (pip install pydub) or soundfile (pip install soundfile)")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TASK_TYPES = [
    "SpontSpeech",
    "MotorSpeechEval", 
    "PicnicDescription",
    "GrandfatherPassage",
    "ConflictConv"
]

CSV_COLUMNS = {
    "SpontSpeech": ("SpontSpeechStart", "SpontSpeechStop"),
    "MotorSpeechEval": ("MotorSpeechEvalStart", "MotorSpeechEvalStop"),
    "PicnicDescription": ("PicnicDescriptionStart", "PicnicDescriptionStop"),
    "GrandfatherPassage": ("GrandfatherPassageStart", "GrandfatherPassageStop"),
    "ConflictConv": ("ConflictConvStart", "ConflictConvStop")
}


def normalize_timestamp(time_str: str) -> str:
    """
    Convert any timestamp format to MM:SS
    Handles:
        - Seconds as numbers: "1201.334" -> "20:01"
        - Just seconds: "45" -> "00:45"
        - Already formatted: "2:13" -> "02:13"
    """
    if not time_str or str(time_str).strip() in ['', 'nan', 'NaN']:
        return None
        
    time_str = str(time_str).strip()
    
    # If it contains a colon, it's already MM:SS format
    if ':' in time_str:
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return f"{minutes:02d}:{seconds:02d}"
        except ValueError:
            logger.warning(f"Could not parse timestamp: {time_str}")
            return None
    
    # Otherwise, treat as total seconds (float or int)
    try:
        total_seconds = float(time_str)
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    except ValueError:
        logger.warning(f"Could not parse timestamp: {time_str}")
        return None


def timestamp_to_milliseconds(timestamp: str) -> int:
    """Convert MM:SS timestamp to milliseconds"""
    if not timestamp:
        return 0
    parts = timestamp.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    return (minutes * 60 + seconds) * 1000


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert MM:SS timestamp to seconds"""
    if not timestamp:
        return 0.0
    parts = timestamp.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    return minutes * 60.0 + seconds


def parse_transcript_line(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a transcript line with timestamp.
    Returns (start_time, end_time, text) or None
    
    Example: "[00:01 - 00:02] All right, thank you."
    Returns: ("00:01", "00:02", "All right, thank you.")
    """
    match = re.match(r'\[(\d+:\d+)\s*-\s*(\d+:\d+)\]\s*(.*)', line)
    if match:
        start = normalize_timestamp(match.group(1))
        end = normalize_timestamp(match.group(2))
        text = match.group(3)
        return (start, end, text)
    return None


def find_matching_files(base_dir: Path, transcripts_dir: Path, base_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find matching WAV and transcript files for a base identifier.
    Returns (wav_path, transcript_path) or (None, None)
    """
    # Try exact match first
    wav_path = base_dir / f"{base_id}.wav"
    transcript_path = transcripts_dir / f"{base_id}_transcript.txt"
    
    if wav_path.exists() and transcript_path.exists():
        return (wav_path, transcript_path)
    
    # Try fuzzy matching
    wav_files = list(base_dir.glob("**/*.wav"))
    transcript_files = list(transcripts_dir.glob("**/*_transcript.txt"))
    
    # Find WAV
    wav_match = None
    for wav_file in wav_files:
        if base_id in wav_file.stem:
            wav_match = wav_file
            break
    
    # Find transcript
    transcript_match = None
    for transcript_file in transcript_files:
        transcript_base = transcript_file.stem.replace("_transcript", "")
        if base_id in transcript_base:
            transcript_match = transcript_file
            break
    
    return (wav_match, transcript_match)


def load_timestamp_csv(csv_path: Path) -> List[Dict]:
    """Load and parse the timestamp CSV"""
    records = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Get file identifier (should be first column)
            file_id = row.get('FILE', row.get('file', row.get('Filename', None)))
            
            if not file_id:
                # Try first column
                file_id = list(row.values())[0] if row else None
            
            if not file_id or file_id.strip() == '':
                continue
            
            # Remove any .wav or _transcript.txt extensions
            base_id = file_id.replace('.wav', '').replace('_transcript.txt', '').strip()
            
            record = {'base_id': base_id, 'tasks': {}}
            
            # Parse task timestamps
            for task_type, (start_col, stop_col) in CSV_COLUMNS.items():
                start_time = normalize_timestamp(row.get(start_col, ''))
                stop_time = normalize_timestamp(row.get(stop_col, ''))
                
                if start_time and stop_time:
                    record['tasks'][task_type] = {
                        'start': start_time,
                        'stop': stop_time
                    }
            
            records.append(record)
    
    logger.info(f"Loaded {len(records)} records from CSV")
    return records


def segment_audio(audio_path: Path, start_time: str, stop_time: str, output_path: Path) -> bool:
    """
    Segment audio file from start_time to stop_time.
    Returns True if successful.
    """
    try:
        if AUDIO_LIBRARY == "pydub":
            # Use pydub
            audio = AudioSegment.from_wav(str(audio_path))
            start_ms = timestamp_to_milliseconds(start_time)
            stop_ms = timestamp_to_milliseconds(stop_time)
            
            segment = audio[start_ms:stop_ms]
            segment.export(str(output_path), format="wav")
            
        else:  # soundfile
            # Use soundfile
            data, samplerate = sf.read(str(audio_path))
            start_sample = int(timestamp_to_seconds(start_time) * samplerate)
            stop_sample = int(timestamp_to_seconds(stop_time) * samplerate)
            
            segment = data[start_sample:stop_sample]
            sf.write(str(output_path), segment, samplerate)
        
        return True
        
    except Exception as e:
        logger.error(f"Error segmenting audio {audio_path}: {e}")
        return False


def segment_transcript(transcript_path: Path, start_time: str, stop_time: str, output_path: Path) -> bool:
    """
    Segment transcript file - keep only lines within time range.
    Returns True if successful.
    """
    try:
        start_seconds = timestamp_to_seconds(start_time)
        stop_seconds = timestamp_to_seconds(stop_time)
        
        output_lines = []
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = parse_transcript_line(line)
                if parsed:
                    line_start, line_end, text = parsed
                    line_start_seconds = timestamp_to_seconds(line_start)
                    
                    # Include line if its start time falls within task range
                    if start_seconds <= line_start_seconds <= stop_seconds:
                        # Keep original timestamp format
                        output_lines.append(line)
        
        # Write segmented transcript
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        
        return True
        
    except Exception as e:
        logger.error(f"Error segmenting transcript {transcript_path}: {e}")
        return False


def create_output_structure(base_dir: Path) -> None:
    """Create the output directory structure"""
    
    # Create segmented directories
    for task_type in TASK_TYPES:
        task_dir = base_dir / "segmented" / task_type
        (task_dir / "wav").mkdir(parents=True, exist_ok=True)
        (task_dir / "transcripts" / "original_cut").mkdir(parents=True, exist_ok=True)
    
    # Create reports directory
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)
    
    logger.info("Created output directory structure")


def process_file(record: Dict, wav_dir: Path, transcript_dir: Path, output_base: Path, stats: Dict) -> None:
    """Process a single file - segment audio and transcripts for all tasks"""
    
    base_id = record['base_id']
    logger.info(f"Processing: {base_id}")
    
    # Find matching files
    wav_path, transcript_path = find_matching_files(wav_dir, transcript_dir, base_id)
    
    if not wav_path:
        logger.warning(f"WAV file not found for: {base_id}")
        stats['missing_wav'].append(base_id)
        return
    
    if not transcript_path:
        logger.warning(f"Transcript not found for: {base_id}")
        stats['missing_transcript'].append(base_id)
        return
    
    logger.info(f"  Found WAV: {wav_path.name}")
    logger.info(f"  Found transcript: {transcript_path.name}")
    
    # Process each task
    for task_type, task_data in record['tasks'].items():
        start_time = task_data['start']
        stop_time = task_data['stop']
        
        logger.info(f"  Segmenting {task_type}: {start_time} - {stop_time}")
        
        # Output paths
        wav_output = output_base / "segmented" / task_type / "wav" / f"{base_id}_{task_type}.wav"
        transcript_output = output_base / "segmented" / task_type / "transcripts" / "original_cut" / f"{base_id}_{task_type}_transcript.txt"
        
        # Segment audio
        if segment_audio(wav_path, start_time, stop_time, wav_output):
            logger.info(f"    ✓ Audio segmented: {wav_output.name}")
            stats['segments_created'][task_type] = stats['segments_created'].get(task_type, 0) + 1
        else:
            logger.error(f"    ✗ Audio segmentation failed")
            stats['errors'].append(f"{base_id}_{task_type}_audio")
        
        # Segment transcript
        if segment_transcript(transcript_path, start_time, stop_time, transcript_output):
            logger.info(f"    ✓ Transcript segmented: {transcript_output.name}")
        else:
            logger.error(f"    ✗ Transcript segmentation failed")
            stats['errors'].append(f"{base_id}_{task_type}_transcript")
    
    stats['files_processed'].append(base_id)


def generate_report(stats: Dict, output_dir: Path) -> None:
    """Generate validation report"""
    
    report_path = output_dir / "reports" / "segmentation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AUDIO AND TRANSCRIPT SEGMENTATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Files processed: {len(stats['files_processed'])}\n")
        f.write(f"Missing WAV files: {len(stats['missing_wav'])}\n")
        f.write(f"Missing transcripts: {len(stats['missing_transcript'])}\n")
        f.write(f"Errors: {len(stats['errors'])}\n\n")
        
        f.write("Segments created by task type:\n")
        for task_type in TASK_TYPES:
            count = stats['segments_created'].get(task_type, 0)
            f.write(f"  {task_type}: {count}\n")
        
        if stats['missing_wav']:
            f.write("\nMissing WAV files:\n")
            for base_id in stats['missing_wav']:
                f.write(f"  - {base_id}\n")
        
        if stats['missing_transcript']:
            f.write("\nMissing transcripts:\n")
            for base_id in stats['missing_transcript']:
                f.write(f"  - {base_id}\n")
        
        if stats['errors']:
            f.write("\nErrors:\n")
            for error in stats['errors']:
                f.write(f"  - {error}\n")
    
    logger.info(f"Report generated: {report_path}")
    
    # Also save JSON version
    json_path = output_dir / "reports" / "segmentation_report.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"JSON report: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment audio and transcripts by task type")
    parser.add_argument("--csv", default="/path/to/user/Desktop/CLIN-Batch2-tasktimes.csv",
                       help="Path to CSV with task timestamps")
    parser.add_argument("--wav-dir", default="/path/to/user/Desktop/ClinicWavsToProcess",
                       help="Directory containing WAV files")
    parser.add_argument("--transcript-dir", default="/path/to/user/Desktop/ClinicWavsToProcess/TranscriptsToReview",
                       help="Directory containing transcripts")
    parser.add_argument("--output-dir", default="/path/to/user/Desktop/ClinicWavsToProcess",
                       help="Base output directory")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test-mode", action="store_true",
                      help="Test mode - process only files in test list")
    group.add_argument("--full", action="store_true",
                      help="Full mode - process all files in CSV")
    
    parser.add_argument("--test-files", 
                       help="Path to text file with base IDs for test mode (one per line)")
    
    args = parser.parse_args()
    
    # Validate paths
    csv_path = Path(args.csv)
    wav_dir = Path(args.wav_dir)
    transcript_dir = Path(args.transcript_dir)
    output_dir = Path(args.output_dir)
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    if not wav_dir.exists():
        logger.error(f"WAV directory not found: {wav_dir}")
        return
    
    if not transcript_dir.exists():
        logger.error(f"Transcript directory not found: {transcript_dir}")
        return
    
    # Create output structure
    create_output_structure(output_dir)
    
    # Load CSV
    logger.info(f"Loading timestamps from: {csv_path}")
    records = load_timestamp_csv(csv_path)
    
    # Filter for test mode if requested
    if args.test_mode:
        if not args.test_files:
            logger.error("--test-files required for test mode")
            return
        
        test_files_path = Path(args.test_files)
        if not test_files_path.exists():
            logger.error(f"Test files list not found: {test_files_path}")
            return
        
        # Load test file IDs
        with open(test_files_path, 'r') as f:
            test_ids = set(line.strip() for line in f if line.strip())
        
        logger.info(f"Test mode: Processing {len(test_ids)} files")
        records = [r for r in records if r['base_id'] in test_ids]
        
        if len(records) == 0:
            logger.error("No matching records found for test IDs")
            return
    
    # Initialize statistics
    stats = {
        'files_processed': [],
        'missing_wav': [],
        'missing_transcript': [],
        'segments_created': {},
        'errors': []
    }
    
    # Process files
    logger.info(f"Processing {len(records)} files...")
    for record in records:
        process_file(record, wav_dir, transcript_dir, output_dir, stats)
    
    # Generate report
    generate_report(stats, output_dir)
    
    # Summary
    logger.info("=" * 80)
    logger.info("SEGMENTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Files processed: {len(stats['files_processed'])}")
    logger.info(f"Total segments created: {sum(stats['segments_created'].values())}")
    logger.info(f"Errors: {len(stats['errors'])}")
    logger.info(f"Report: {output_dir / 'reports' / 'segmentation_report.txt'}")


if __name__ == "__main__":
    main()
