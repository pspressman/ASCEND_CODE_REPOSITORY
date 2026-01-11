# LLMAnon to Timestamps Bridge - Round 2 Anonymization

## Overview

This script bridges the gap between **LLMAnon output** (human-reviewed CSV of additional identifying content) and the **audio de-identification pipeline** for Round 2 anonymization.

**What it does:**
1. Reads LLMAnon CSV (all rows - already human-reviewed)
2. Matches flagged text to WhisperX JSON word-level timestamps
3. Generates timestamp CSV in the format needed by `phi_inplace_deidentifier_MASTER_CSV.py`
4. Automatically skips CliniDeID markers (already handled in Round 1)
5. Handles multi-word phrase matching with punctuation stripping
6. **Handles MIXED JSON formats** (critical - both old and new WhisperX structures)

## The Complete Round 2 Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Round 1 (Already Complete)                                   │
│ ─────────────────────────                                    │
│ CliniDeID → detectedPII → timestamps → audio redaction       │
│ Original transcripts → CliniDeID → redacted transcripts      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Round 2 (This Workflow)                                       │
│ ───────────────────────                                      │
│                                                               │
│  AUDIO PATH:                                                  │
│  1. LLMAnon CSV → [THIS SCRIPT] → PHI timestamps CSV         │
│  2. Round 1 audio + timestamps → Round 2 audio (further)     │
│                                                               │
│  TRANSCRIPT PATH (parallel):                                 │
│  1. Round 1 transcripts + LLMAnon CSV → Round 2 transcripts  │
│     (uses transcript_dual_redactor.py)                       │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Input Files Required

1. **LLMAnon CSV** (human-reviewed)
   - Columns: `REVIEW, FILE-basename, FILE, RISK, CATEGORY, TEXT, REASONING, SUGGESTED_REPLACEMENT`
   - All rows already reviewed (REVIEW='DONE'), so all are processed
   - Must have: `FILE-basename`, `TEXT` (minimum)
   - Optional: `RISK`, `CATEGORY`, `SUGGESTED_REPLACEMENT` (defaults to 'UNKNOWN' if missing)
   
2. **WhisperX JSON files** (word-level timestamps)
   - Located in searchable directories
   - Must contain `words` array with timestamps
   
3. **Round 1 redacted audio files** (from CliniDeID pass)
   - WAV format
   - Already have CliniDeID redactions applied

### Python Dependencies

```bash
# Standard library only - no additional packages needed!
```

## Configuration

### Step 1: Edit the Configuration Section

Open `LLMAnon_to_timestamps_bridge.py` and update these paths:

```python
# Input CSV from LLMAnon (after human review)
LLMANON_CSV_PATH = r"C:\path\to\LLMAnon_REVIEW_CONSOLIDATED.csv"

# Search directories for JSON transcripts
JSON_SEARCH_DIRS = [
    r"C:\LocalData\ASCEND_PHI\DeID\CSA_WordStamps",
]

# Search directories for Round 1 audio (already redacted from CliniDeID)
WAV_SEARCH_DIRS = [
    r"Z:\ASCEND_ANON\McAdams\ROUND1_REDACTED_AUDIO",
]

# Output directory
OUTPUT_DIR = r"C:\LocalData\ASCEND_PHI\DeID\Round2"
```

### Step 2: Verify Directory Structure

Make sure your directories exist and contain files:

```
JSON_SEARCH_DIRS/
├── participant_001.json
├── participant_002.json
└── ...

WAV_SEARCH_DIRS/
├── anon_participant_001_clean.wav
├── anon_participant_002_clean.wav
└── ...
```

## Usage

### Step 1: Generate Timestamps (Bridge Script)

```bash
python LLMAnon_to_timestamps_bridge.py
```

**Output:**
- `phi_timestamps_from_LLMAnon.csv` - Timestamp CSV for deidentifier
- `llmanon_bridge_YYYYMMDD_HHMMSS.log` - Full processing log
- `llmanon_text_not_found.txt` - Text not found in JSON (expected for some)
- `clinideid_markers_skipped.txt` - Markers already handled in Round 1
- `missing_files.txt` - Missing JSON or WAV files

### Step 2: Apply Audio Redactions

```bash
python phi_inplace_deidentifier_MASTER_CSV.py \
    --audio_dir "Z:\ASCEND_ANON\McAdams\ROUND1_REDACTED_AUDIO" \
    --master_csv "C:\LocalData\ASCEND_PHI\DeID\Round2\phi_timestamps_from_LLMAnon.csv" \
    --output_dir "Z:\ASCEND_ANON\McAdams\ROUND2_REDACTED_AUDIO" \
    --method hybrid
```

### Step 3 (Optional): Redact Transcripts in Parallel

In a separate terminal:

```bash
python transcript_dual_redactor.py \
    "C:\path\to\LLMAnon_REVIEW_CONSOLIDATED.csv" \
    "C:\path\to\ROUND1_REDACTED_TRANSCRIPTS" \
    --output "C:\path\to\ROUND2_REDACTED_TRANSCRIPTS"
```

## Output Format

### Generated Timestamp CSV

The script generates `phi_timestamps_from_LLMAnon.csv` with these columns:

```csv
participant_id,start_time,end_time,phi_word,wav_filename,json_source,risk_level,category
5809437,104.634,105.014,"won national award",anon_5809437_clean.wav,5809437.json,HIGH,BIOGRAPHICAL
5809437,87.250,87.890,"35 years",anon_5809437_clean.wav,5809437.json,MEDIUM,BIOGRAPHICAL
```

**Key features:**
- One row per timestamp match (multiple if text appears multiple times)
- `wav_filename` is the **EXACT** filename as found on disk
- Timestamps include ±0.15s buffer for clean audio cuts
- `phi_word` preserves original text from LLMAnon CSV

## How It Works

### 🎯 Critical Feature: Mixed JSON Format Support

**This was EXTREMELY painful to debug and is essential for your data!**

Your WhisperX JSON files have **TWO different structures**:

**Format 1 (Newer):** Top-level `words` array
```json
{
  "words": [
    {"word": "hello", "start": 0.5, "end": 0.8},
    {"word": "world", "start": 0.9, "end": 1.2}
  ]
}
```

**Format 2 (Legacy):** `segments` containing `words` arrays
```json
{
  "segments": [
    {
      "words": [
        {"word": "hello", "start": 0.5, "end": 0.8},
        {"word": "world", "start": 0.9, "end": 1.2}
      ]
    }
  ]
}
```

**The script automatically detects and handles BOTH.** This is non-negotiable for your dataset!

### CliniDeID Marker Detection

The script automatically detects and **skips** CliniDeID markers like:
- `[***NAME***]`
- `[***ADDRESS***]`
- `[***DATE***]`

**Why?** These are already redacted in Round 1 audio. The JSON transcripts are from **before** CliniDeID ran, so these markers won't exist in the JSON. This is expected and correct!

### Multi-Word Phrase Matching

The script handles both single words and multi-word phrases:

**Example:** `"won national award"`
1. Strips punctuation from both search phrase and JSON words
2. Splits phrase into: `["won", "national", "award"]`
3. Searches for **consecutive** word matches in JSON
4. Returns ALL occurrences with timestamps

### File Matching Logic

**Normalization for matching:**
```python
"anon_participant_001_clean.wav" → "participant_001"
"participant_001.json" → "participant_001"
"participant_001_transcript.txt" → "participant_001"
```

**BUT** - the actual WAV filename in the output CSV is preserved exactly as found:
```csv
wav_filename: anon_participant_001_clean.wav  ← Exact filename
```

## Expected Outcomes

### Text Not Found in JSON

**This is normal and expected!** Common reasons:

1. **LLM Overzealousness**: Context-based identification that isn't literal text
2. **Paraphrasing**: Text differs between transcript versions
3. **Summarization**: LLM identified summarized content
4. **Context Leak**: LLM used context across multiple files

**Action:** Review `llmanon_text_not_found.txt` - most will be false positives from LLM.

### CliniDeID Markers Skipped

**Also normal!** These are from Round 1 and are already handled.

**Example entries in `clinideid_markers_skipped.txt`:**
```
Row 45: [***NAME***] (from 5809437)
Row 67: [***ADDRESS***] (from 5809438)
```

## Testing Strategy

### Before Running Full Dataset

**Test on 3-5 files first:**

1. Create a test CSV with just 3-5 participants
2. Run the bridge script
3. Verify timestamps generated correctly
4. Run deidentifier on 1-2 audio files
5. Listen to output - confirm additional redactions applied
6. If good → scale to full dataset

### Quality Checks

```bash
# Check output CSV has rows
wc -l phi_timestamps_from_LLMAnon.csv

# Check unique files
cut -d',' -f5 phi_timestamps_from_LLMAnon.csv | sort -u | wc -l

# Check for errors in log
grep "ERROR" llmanon_bridge_*.log
```

## Troubleshooting

### No timestamps generated

**Check:**
1. Does LLMAnon CSV have any `REVIEW='Y'` rows?
2. Are JSON_SEARCH_DIRS and WAV_SEARCH_DIRS correct?
3. Do the JSON files exist?
4. Check `missing_files.txt` for what's missing

### Text not found but should be

**Check:**
1. Punctuation differences? Script strips punctuation automatically
2. Capitalization? Script lowercases for matching
3. Multi-word phrase? Script handles this automatically
4. Check the actual JSON file to see if text exists

### WAV filename mismatch

**Check:**
1. Is the WAV filename in the CSV exact?
2. Run deidentifier with `--audio_dir` pointing to correct location
3. Deidentifier builds its own index with recursive search

## Performance

**Expected runtime:**
- Small dataset (10 files): ~10 seconds
- Medium dataset (100 files): ~1-2 minutes  
- Large dataset (1000+ files): ~10-15 minutes

**Most time spent on:**
- Building file indexes (recursive search)
- Multi-word phrase matching in JSON

## File Organization

```
OUTPUT_DIR/
├── phi_timestamps_from_LLMAnon.csv          ← MAIN OUTPUT (feed to deidentifier)
├── llmanon_bridge_YYYYMMDD_HHMMSS.log       ← Full processing log
├── llmanon_text_not_found.txt               ← Expected mismatches
├── clinideid_markers_skipped.txt            ← Already handled (Round 1)
└── missing_files.txt                        ← Missing JSON/WAV files
```

## Integration with Audio Deidentifier

The output CSV is **directly compatible** with `phi_inplace_deidentifier_MASTER_CSV.py`:

```bash
python phi_inplace_deidentifier_MASTER_CSV.py \
    --audio_dir <ROUND1_AUDIO_DIR> \
    --master_csv phi_timestamps_from_LLMAnon.csv \
    --output_dir <ROUND2_AUDIO_DIR> \
    --method hybrid
```

**The deidentifier will:**
1. Load all timestamps from the CSV
2. Group by WAV filename
3. Apply redactions to Round 1 audio files
4. Output Round 2 audio with **additional** redactions

## Parallel Transcript Processing

While audio processes, you can simultaneously redact transcripts:

```bash
# Terminal 1: Audio redaction
python phi_inplace_deidentifier_MASTER_CSV.py --audio_dir ... --master_csv ...

# Terminal 2: Transcript redaction (parallel)
python transcript_dual_redactor.py LLMAnon_CSV.csv transcripts/ --output ...
```

**Both use the same LLMAnon CSV** - no coordination needed!

## Architecture Decisions

### Why Skip CliniDeID Markers?

Round 1 already redacted them. The JSON is from **before** CliniDeID, so these markers won't exist there. This is correct behavior!

### Why Multi-Word Phrase Support?

LLMAnon flags things like:
- "won national award" (biographical)
- "35 years experience" (duration)
- "regional medical center" (location)

These need to be matched as **single units**, not individual words.

### Why Exact WAV Filenames?

The deidentifier builds its own file index and needs exact filenames to find files. Normalization is **only** for matching during the bridge step.

## Support

**If you encounter issues:**

1. Check log files in OUTPUT_DIR
2. Verify all paths in configuration section
3. Test on small dataset first
4. Review `missing_files.txt` for what's not being found

## Related Scripts

- `phi_inplace_deidentifier_MASTER_CSV.py` - Audio redaction (Round 2)
- `transcript_dual_redactor.py` - Transcript redaction (Round 2)
- `generate_phi_timestamps_WAV_BASED2.py` - Round 1 timestamp generation

---

**Generated for ASCEND Audio Anonymization Pipeline**  
**Version:** 2.0 (Round 2 - LLMAnon Integration)  
**Date:** October 31, 2025
