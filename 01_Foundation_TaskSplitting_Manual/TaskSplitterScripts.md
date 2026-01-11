Task Splitter Scripts
CSA Clinical/Research (4 tasks)
Script: UpdatedCSATaskSplitter.py

NOTE-- Python 3.8

Tasks:

SpontSpeech
MotorSpeechEval
PicnicDescription
GrandfatherPassage

Extended Protocol (5 tasks)
Script: UpdatedTimestampSplitter5Task.py
Tasks: All 4 above + ConflictConv

Computer
Manual workflow - can run on any machine with Python
Typically: LOCUS (where transcripts are generated)

Input Files
1. CSV with Task Timestamps
Required columns:

FILE (or file, Filename) - base identifier
{Task}Start - start time in MM:SS or seconds
{Task}Stop - stop time in MM:SS or seconds

Example CSV:
csvFILE,SpontSpeechStart,SpontSpeechStop,MotorSpeechEvalStart,MotorSpeechEvalStop
CSA_001_baseline,00:45,03:12,04:30,08:15
CSA_002_baseline,1:20,4:05,5:30,9:45
```

**Timestamp formats supported:**
- `MM:SS` → "02:13"
- `M:SS` → "2:13" (auto-padded)
- Seconds as integer → "45" (converts to "00:45")
- Seconds as float → "1201.334" (converts to "20:01")

### 2. Audio Files
- **Location:** `--wav-dir` directory
- **Format:** `.wav`
- **Naming:** Must match CSV `FILE` column (with or without `.wav` extension)

### 3. Transcript Files
- **Location:** `--transcript-dir` directory
- **Format:** `{base_id}_transcript.txt`
- **Must contain:** Timestamped lines like `[00:45 - 00:48] Text here`

---

## Output Structure
```
output_dir/
├── segmented/
│   ├── SpontSpeech/
│   │   ├── wav/
│   │   │   └── {base_id}_SpontSpeech.wav
│   │   └── transcripts/
│   │       └── original_cut/
│   │           └── {base_id}_SpontSpeech_transcript.txt
│   ├── MotorSpeechEval/
│   ├── PicnicDescription/
│   ├── GrandfatherPassage/
│   └── ConflictConv/  (if using 5-task version)
└── reports/
    ├── segmentation_report.txt
    └── segmentation_report.json

Usage
Test Mode (Recommended First)
Process only specific files to validate timestamps:
bashpython UpdatedCSATaskSplitter.py \
  --test-mode \
  --test-files test_list.txt \
  --csv /path/to/timestamps.csv \
  --wav-dir /path/to/wav/files \
  --transcript-dir /path/to/transcripts \
  --output-dir /path/to/output
```

**test_list.txt example:**
```
CSA_001_baseline
CSA_002_baseline
CSA_003_baseline
Full Processing
Process all files in CSV:
bashpython UpdatedCSATaskSplitter.py \
  --full \
  --csv /path/to/timestamps.csv \
  --wav-dir /path/to/wav/files \
  --transcript-dir /path/to/transcripts \
  --output-dir /path/to/output
For 5-Task Protocol
Simply swap the script:
bashpython UpdatedTimestampSplitter5Task.py --full --csv ...

Dependencies
bashpip install pydub  # Preferred
# OR
pip install soundfile numpy  # Alternative
Note: pydub requires ffmpeg:
bashbrew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
```

---

## How It Works

### 1. CSV Parsing
- Reads timestamp CSV
- Normalizes all timestamps to MM:SS format
- Handles multiple timestamp formats automatically

### 2. File Matching
- Attempts exact match: `{base_id}.wav` and `{base_id}_transcript.txt`
- Falls back to fuzzy matching if exact fails
- Searches recursively in subdirectories

### 3. Audio Segmentation
- Converts MM:SS timestamps to milliseconds
- Extracts audio segment using `pydub` or `soundfile`
- Saves as task-specific WAV file

### 4. Transcript Segmentation
- Parses transcript lines with regex: `\[(\d+:\d+) - (\d+:\d+)\] (.*)`
- Keeps only lines where start time falls within task range
- Preserves original timestamp format

### 5. Validation
- Generates detailed reports
- Tracks missing files, errors, segment counts
- Outputs both TXT and JSON reports

---

## Pipeline Integration

### Feeds These Steps:
- **04-11:** Track A acoustic features (openSMILE, Parselmouth, Librosa)
- **12-13:** Track A NLP features
- **14-15:** Track A video features (if video exists)
- **16-17:** Track A manual ratings
- **18-26:** Track B de-identification pipeline

### Critical Path:
```
Step 01 (Task Splitting)
    ↓
Step 02 outputs (WhisperX on segmented tasks)
    ↓
Step 04-13 (Feature extraction)
    ↓
CSAforProton/TranscriptFeatures/
CSAforProton/AcousticFeatures/

Troubleshooting
"WAV file not found"

Check CSV FILE column matches WAV filename (without .wav)
Verify --wav-dir path is correct
Check for typos in filenames

"Transcript not found"

Ensure transcripts end with _transcript.txt
Verify --transcript-dir path
Run transcript generation first (Step 02 or 32/33)

"No segments generated"

Verify CSV timestamps are not empty
Check timestamp format in CSV
Ensure start < stop for all tasks

Empty transcript segments

Check transcript timestamps overlap with task range
Verify transcript format: [MM:SS - MM:SS] Text
Review original transcript quality


Status Tracking
Step 01 - Task Splitting

 Timestamp CSV created/validated
 Test mode run successfully
 Output structure verified
 Sample segments spot-checked
 Full processing completed
 Reports reviewed for errors
 Outputs ready for Step 02/04+


Next Steps
After successful task splitting:

Step 02: Run WhisperX on segmented tasks for refined transcripts
Step 03: Run Pyannote for speaker diarization (if needed)
Step 04+: Begin Track A feature extraction pipeline
