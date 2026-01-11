Track D: Alternative ASR Methods

Note-- Python 3.8, but Faster-Whisper requires different transformers than next optimized audio extraction and transcript generation, so create new environment or prepare for conflict and re-install.  

Purpose
Generate timestamped transcripts using alternative ASR engines (OpenAI Whisper, Faster-Whisper) for comparison with WhisperX baseline and to provide initial transcripts for manual timestamp annotation.
Pipeline Position

Runs independently or before Step 01 (Task Splitting)
Generates transcripts that can be reviewed to create timestamp CSVs
Alternative to Step 02 (WhisperX Foundation) for Track D comparison analysis


Step 32: OpenAI Whisper Full Visit (VOX)
Computer: VOX
Script: openAI-Whisper-ASR.py
Input

Audio files: /Volumes/Databackup2025/conversational speech 18-0456/Participant Data and Forms/**/*.wav
Recursively searches for all WAV files
Excludes: Files in folders containing "follow" and "up" (case-insensitive)

Output

Location: ~/Desktop/CSA_Research_OpenAIWhisperTranscript/
Files: {original_filename}_transcript.txt
Format: Timestamped segments in MM:SS format
Preserves: Original directory structure

Dependencies
bashpip install openai-whisper tqdm
Usage
bashpython openAI-Whisper-ASR.py
```

**Configuration (edit in script):**
- `audio_root`: Source directory for WAV files
- `output_root`: Destination for transcripts
- Model: `base` (can change to `small`, `medium`, `large`)

### Output Format
```
Transcript: filename.wav
Source: relative/path/to/file.wav
Model: OpenAI Whisper (base)
================================================================================

[00:01 - 00:03] All right, thank you.
[00:04 - 00:07] Let's begin with the spontaneous speech task.

Step 33: Faster-Whisper Full Visit (LOCUS)
Computer: LOCUS
Script: fasterWhisperASR.py
Input

Audio files: /Volumes/Databackup2025/conversational speech 18-0456/Participant Data and Forms/**/*.wav
Recursively searches all subdirectories

Output

Location: ~/Desktop/CSA_Research_FasterWhisperTranscript/
Files: {original_filename}_transcript.txt
Format: Timestamped segments in MM:SS format
Preserves: Directory structure

Dependencies
bashpip install faster-whisper tqdm
Usage
bashpython fasterWhisperASR.py
```

**Configuration (edit in script):**
- `audio_root`: Source directory
- `output_root`: Destination directory
- Model: `base` (faster-whisper optimized)
- VAD: Enabled with 500ms min silence

### Key Differences from OpenAI Whisper
- **Faster inference** using CTranslate2
- **Built-in VAD** filtering
- **Lower memory** footprint
- Better for **batch processing**

### Output Format
```
Transcript: filename.wav
Source: relative/path/to/file.wav
================================================================================

[00:01 - 00:03] All right, thank you.
[00:04 - 00:07] Let's begin with the spontaneous speech task.

Comparison Notes
FeatureOpenAI WhisperFaster-WhisperSpeedModerate2-4x fasterMemoryHigherLowerVADExternalBuilt-inModel SizeStandardOptimizedUse CaseBaseline referenceProduction batches

Pipeline Integration
For Track D Analysis (Steps 34-36)

Run both ASR methods on same audio
Compare transcripts with WhisperX (Step 02 output)
Feed alternative transcripts to NLP1 (Step 35) and NLP2 (Step 36)

For Task Timestamp Creation (Step 01 prep)

Run either ASR method on full recordings
Review transcripts manually
Note task start/stop times in MM:SS format
Create CSV with timestamps for task splitting


Status Tracking
Step 32 - OpenAI Whisper

 VOX environment configured
 Dependencies installed
 Audio paths verified
 Test run on 3 files
 Full batch processing
 Output validated

Step 33 - Faster-Whisper

 LOCUS environment configured
 Dependencies installed
 Audio paths verified
 Test run on 3 files
 Full batch processing
 Output validated
 Speed comparison documented


Step 01: Foundation Task Splitting (Manual)
Purpose
Segment full-length audio recordings and transcripts into task-specific clips based on manually annotated timestamps. This creates the foundation inputs for all downstream Track A and Track B processing.
Pipeline Position

First step in the pipeline (after initial ASR generation)
Feeds: All Track A steps (04-17) and Track B de-identification (18-26)
Prerequisites:

Full-length WAV files
WhisperX transcripts (from Step 02) OR alternative ASR transcripts
CSV with task timestamps
