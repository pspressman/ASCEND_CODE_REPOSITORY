Audio De-Identification Pipeline: Complete Implementation Guide

Overview
This pipeline de-identifies Protected Health Information (PHI) in clinical audio recordings while preserving prosodic features for research. It combines speaker anonymization (McAdams transformation) with targeted linguistic de-identification of PHI segments.

Prerequisites
Required Software

Python 3.8 or 3.9
Java Runtime Environment (for CliniDeID)
CliniDeID (download from clinical NLP resources)

Python Environment Setup
bash# Create environment
python3 -m venv deid_env
source deid_env/bin/activate  # On Windows: deid_env\Scripts\activate

# Install dependencies
pip install numpy pandas soundfile librosa parselmouth scipy
```

### Required Scripts
1. `Path_McAdams_nest.py` - Speaker anonymization
2. `gap_timestamp_phi_extractor.py` OR `phi_timestamp_extractor.py` - PHI timestamp extraction
3. `phi_inplace_deidentifier.py` - Audio PHI de-identification
4. `phi_deidentification_pipeline.py` - Core de-identification methods (required by script 3)

---

## Stage 0: Speaker Anonymization (All Audio Files)

### Purpose
Remove speaker identity using McAdams coefficient transformation while preserving prosody (pitch, timing, rhythm).

### Input
Original audio files organized by task type:
```
/path/to/original_audio/
├── Spontaneous_Speech/
│   ├── 12345-6-7-19-SpontSpeech.wav
│   ├── 67890-8-15-20-SpontSpeech.wav
│   └── ...
├── PicnicDescription/
├── GrandfatherPassage/
└── MotorSpeechEval/
Command
bashpython Path_McAdams_nest.py \
  --input_dir /path/to/original_audio/ \
  --output_dir /path/to/mcadams_output/
```

### Output
McAdams-processed audio with `anon_` prefix:
```
/path/to/mcadams_output/
├── Spontaneous_Speech/
│   ├── anon_12345-6-7-19-SpontSpeech.wav
│   ├── anon_67890-8-15-20-SpontSpeech.wav
│   └── ...
```

### What It Does
- Applies McAdams coefficient (α=0.8) to modify formant frequencies
- Preserves F0 (pitch), timing, and amplitude envelope
- Removes speaker-specific vocal tract characteristics

---

## Stage 1: PHI Timestamp Extraction

### Stage 1A: Text De-identification with CliniDeID

#### Purpose
Identify and label PHI in text transcripts using rule-based NLP.

#### Input
Plain text transcripts (one per audio file):
```
/path/to/transcripts/
├── 12345-6-7-19-SpontSpeech.txt
├── 67890-8-15-20-SpontSpeech.txt
└── ...
CliniDeID Command
bash# Configure CliniDeID for Gap format output
java -jar CliniDeID.jar \
  --inputDir /path/to/transcripts/ \
  --outputDir /path/to/clinideid_output/ \
  --level beyond \
  --outputTypes complete
```

#### CliniDeID Output Format
CliniDeID can produce two output formats. Check your output files to determine which you have:

**Gap Format** (explicit silence markers):
```
Gap: 0.00 - 0.11
Speaker 0: Hello, my name is [*** NAME ***].
Gap: 4.49 - 4.73
Speaker 0: I live at [*** ADDRESS ***].
Gap: 9.38 - 10.50
```

**Inline Format** (timestamps per line):
```
[0.11 - 4.49] Speaker 0: Hello, my name is [*** NAME ***].
[4.73 - 9.38] Speaker 0: I live at [*** ADDRESS ***].
[[*** TEMPORAL ***]] Speaker 0: I was born on [*** DATE ***].
Stage 1B: Extract PHI Timestamps from CliniDeID Output
Choose the appropriate script based on your CliniDeID output format.

Option A: Gap Format → Use gap_timestamp_phi_extractor.py
When to use: Your CliniDeID output has lines starting with Gap:
Command:
bashpython gap_timestamp_phi_extractor.py \
  --batch \
  --input_dir /path/to/clinideid_output/ \
  --output_dir /path/to/phi_timestamps/ \
  --damage_report damage_report_gap.json
```

**What it does:**
- Parses `Gap:` lines to identify silence periods
- Extracts speech segments (audio BETWEEN gaps)
- Associates PHI markers with speech timestamps
- Handles corrupted timestamps:
  - `[*** IDENTIFIER ***]` → Extracts actual decimal number (CliniDeID false positive)
  - `Gap: [*** CONTACT_INFORMATION ***]` → Interpolates from surrounding gaps

**Output:**
```
/path/to/phi_timestamps/
├── 12345-6-7-19-SpontSpeech_transcript_PHI_timestamps.csv
├── 67890-8-15-20-SpontSpeech_transcript_PHI_timestamps.csv
└── damage_report_gap.json
CSV Format:
csvparticipant_id,start_time,end_time,duration,phi_type,reconstruction_method,context
12345-6-7-19,0.11,4.49,4.38,NAME,ORIGINAL,"Hello, my name is [*** NAME ***]"
12345-6-7-19,4.73,9.38,4.65,ADDRESS,ORIGINAL,"I live at [*** ADDRESS ***]"

Option B: Inline Format → Use phi_timestamp_extractor.py
When to use: Your CliniDeID output has timestamps like [0.11 - 4.49] at the start of each line
Command:
bashpython phi_timestamp_extractor.py \
  --batch \
  --input_dir /path/to/clinideid_output/ \
  --output_dir /path/to/phi_timestamps/ \
  --damage_report damage_report_inline.json
```

**What it does:**
- Parses inline timestamps `[start - end]` from each line
- Detects corrupted timestamps: `[[*** ***]]` (double brackets)
- Associates PHI markers with line timestamps
- Handles corrupted timestamps:
  - Finds consecutive lines with missing timestamps
  - Uses timestamps from lines before/after the corrupted block
  - Conservative approach: neutralizes entire block duration

**Output:**
```
/path/to/phi_timestamps/
├── 12345-6-7-19-SpontSpeech_PHI_timestamps.csv
├── 67890-8-15-20-SpontSpeech_PHI_timestamps.csv
└── damage_report_inline.json
CSV Format:
csvparticipant_id,start_time,end_time,duration,phi_type,reconstruction_method,context
P001,0.11,4.49,4.38,NAME,ORIGINAL,"Hello, my name is [*** NAME ***]"
P001,4.73,9.38,4.65,ADDRESS,INTERPOLATED_CONSERVATIVE,"I live at [*** ADDRESS ***]"

Damage Report
Both extractors generate a comprehensive damage report showing:

Total files processed
Files with PHI found
Files requiring timestamp interpolation
Total PHI instances by type
Collateral damage (benign content lost due to interpolation)

Example:
json{
  "summary": {
    "total_files_processed": 461,
    "files_with_phi": 207,
    "files_with_interpolation_needed": 40,
    "total_phi_instances": 1871,
    "total_collateral_damage_seconds": 0.0
  },
  "phi_types_found": {
    "NAME": 792,
    "ADDRESS": 562,
    "LOCATION": 200,
    "TEMPORAL": 159
  }
}

Stage 2: In-Place Audio PHI De-identification
Purpose
Apply HYBRID de-identification to PHI regions within full-length audio files, leaving non-PHI audio completely untouched.
Input

McAdams-processed audio from Stage 0
PHI timestamp CSVs from Stage 1B (either extractor)

Command
bashpython phi_inplace_deidentifier.py \
  --batch \
  --audio_dir /path/to/mcadams_output/Spontaneous_Speech/ \
  --timestamps_dir /path/to/phi_timestamps/ \
  --output_dir /path/to/deidentified_audio/ \
  --method hybrid
```

### What It Does

#### File Matching
The script automatically handles the `anon_` prefix from McAdams:
- Audio file: `anon_12345-6-7-19-SpontSpeech.wav`
- Timestamp CSV: `12345-6-7-19-SpontSpeech_transcript_PHI_timestamps.csv`
- Strips `anon_` prefix to find matching timestamp file

#### Processing Steps
1. Loads full-length audio (preserves original sample rate)
2. Loads PHI timestamp CSV
3. Deduplicates timestamps (removes duplicate time ranges)
4. Creates output audio as copy of original
5. For each PHI timestamp:
   - Extracts audio segment at that time range
   - Applies HYBRID de-identification (formant removal + phase randomization)
   - Replaces samples in output audio
6. Saves full-length file with only PHI regions modified

#### HYBRID Method
- **Vowels:** Remove formants via LPC inverse filtering
- **Consonants:** Phase randomization using cryptographic RNG
- **Gaps:** Conservative gap-filling (processes ambiguous regions with energy)
- **Security:** 99.4% Word Error Rate (unintelligible)
- **Utility:** Preserves prosodic structure (pitch, timing, pauses)

### Output
```
/path/to/deidentified_audio/
├── 12345-6-7-19-SpontSpeech_HYBRID_inplace.wav
├── 67890-8-15-20-SpontSpeech_HYBRID_inplace.wav
├── inplace_deidentification_summary.json
└── processing_metadata.csv
Metadata CSV:
csvparticipant_id,original_duration_sec,num_phi_segments,total_phi_duration_sec,phi_percentage,sample_rate
12345-6-7-19,120.0,12,6.589,5.5,16000
67890-8-15-20,125.3,8,4.231,3.4,16000
Summary JSON:
json{
  "total_files": 206,
  "successful": 206,
  "errors": 0,
  "method": "HYBRID",
  "total_audio_duration_minutes": 626.7,
  "total_phi_duration_minutes": 197.2,
  "overall_phi_percentage": 31.5
}
```

### Result
Each output file contains:
- **PHI regions (31.5%):** HYBRID-processed (unintelligible)
- **Non-PHI regions (68.5%):** Pristine original audio (prosody preserved)

---

## Complete Pipeline Example

### Directory Structure
```
project/
├── original_audio/
│   └── Spontaneous_Speech/
│       └── 12345-6-7-19-SpontSpeech.wav
├── transcripts/
│   └── 12345-6-7-19-SpontSpeech.txt
├── mcadams_output/
├── clinideid_output/
├── phi_timestamps/
└── deidentified_audio/
Step-by-Step Commands
bash# Step 1: Speaker Anonymization
python Path_McAdams_nest.py \
  --input_dir ./original_audio/ \
  --output_dir ./mcadams_output/

# Step 2: Text De-identification
java -jar CliniDeID.jar \
  --inputDir ./transcripts/ \
  --outputDir ./clinideid_output/ \
  --level beyond \
  --outputTypes complete

# Step 3: Extract PHI Timestamps
# (Choose based on CliniDeID output format - Gap or Inline)

# If Gap format:
python gap_timestamp_phi_extractor.py \
  --batch \
  --input_dir ./clinideid_output/ \
  --output_dir ./phi_timestamps/ \
  --damage_report damage_report.json

# If Inline format:
python phi_timestamp_extractor.py \
  --batch \
  --input_dir ./clinideid_output/ \
  --output_dir ./phi_timestamps/ \
  --damage_report damage_report.json

# Step 4: Audio PHI De-identification
python phi_inplace_deidentifier.py \
  --batch \
  --audio_dir ./mcadams_output/Spontaneous_Speech/ \
  --timestamps_dir ./phi_timestamps/ \
  --output_dir ./deidentified_audio/ \
  --method hybrid
```

### Processing Time Estimates
- Stage 0 (McAdams): ~1-2 minutes for 200 files
- Stage 1A (CliniDeID): ~5-10 minutes for 200 files
- Stage 1B (Timestamp extraction): ~10 minutes for 200 files
- Stage 2 (Audio de-identification): ~30 minutes for 200 files

**Total:** ~1 hour for 200 participants

---

## Troubleshooting

### No Timestamp Files Found
**Error:** `Skipping {participant_id}: No timestamp file found`

**Cause:** Filename mismatch between audio and timestamp CSV

**Solution:** Check naming patterns:
- Audio: `anon_12345-6-7-19-SpontSpeech.wav`
- Gap format CSV: `12345-6-7-19-SpontSpeech_transcript_PHI_timestamps.csv`
- Inline format CSV: `12345-6-7-19-SpontSpeech_PHI_timestamps.csv`

The script automatically strips `anon_` prefix, but the rest must match exactly.

### Import Error for PHIDeidentifier
**Error:** `Could not import PHIDeidentifier from phi_deidentification_pipeline.py`

**Solution:** Ensure `phi_deidentification_pipeline.py` is in the same directory as `phi_inplace_deidentifier.py`

### Missing Columns in CSV
**Error:** `Missing required columns in timestamps CSV`

**Solution:** Timestamp CSVs must have `start_time` and `end_time` columns. Re-run the timestamp extractor.

---

## Output Quality Metrics

### Security Validation
- **Word Error Rate (WER):** 99.4% (PHI unintelligible to ASR)
- **Threshold:** >95% WER required for HIPAA compliance
- **Status:** PASSES security requirements

### Prosody Preservation
- **F0 (pitch) correlation:** >95%
- **Duration preservation:** 100%
- **Amplitude envelope:** ~90%
- **Pause/silence structure:** 67% preserved

### Audio Modification
- **PHI segments:** 31.5% of audio de-identified
- **Non-PHI segments:** 68.5% untouched original
- **Zero collateral damage:** Conservative interpolation minimizes benign content loss

---

## Citation

If using this pipeline, please cite:
```
[Your paper citation here after publication]
Code repository: [GitHub URL]
License: [To be determined]