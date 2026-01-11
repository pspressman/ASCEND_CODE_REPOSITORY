# ASCEND Code Repository

**Automated Speech Comparison Engine for Neurocognitive Detection**

Speech and language biomarker extraction pipeline for dementia differential diagnosis (AD, FTD, LBD).

## Overview

This repository contains the complete audio processing and de-identification pipeline developed for the ASCEND project. The codebase supports multimodal feature extraction from clinical speech recordings while maintaining HIPAA compliance through comprehensive de-identification.

## Pipeline Components

| Stage | Purpose | Key Scripts |
|-------|---------|-------------|
| **Task Segmentation** | Split recordings by task type | `UpdatedCSATaskSplitter.py` |
| **Transcription** | ASR with timestamps | WhisperX, Vosk, Faster-Whisper |
| **Acoustic Features** | Prosody, voice quality | OpenSMILE, Librosa, Parselmouth |
| **NLP Features** | Linguistic analysis (3 tiers) | `optimized-CLE-CSA-R.py`, `tier2_fixed_redux_final.py`, `tier3_postid_nexus.py` |
| **Text De-ID** | PHI removal from transcripts | CliniDeID + LLM anonymization |
| **Voice Anonymization** | Speaker de-identification | McAdams coefficient transformation |
| **Video Processing** | Facial feature extraction | OpenFace integration |

## Directory Structure

```
├── 0.75_Label-DeID/          # De-identification pipeline
├── 01_Foundation_TaskSplitting_Manual/  # Task segmentation
├── 01.5_cleaning/            # Audio denoising
├── 02_Baseline_Extraction/   # OpenSMILE + WhisperX
├── 03-AltExtraction/         # Librosa, Parselmouth, alternative ASR
├── 04-NLP/                   # NLP feature extraction (Tiers 1-3)
├── 05-Video_Processing/      # OpenFace video processing
├── 07-B2-audio-anon/         # McAdams voice anonymization
├── 10-LLMAnon-textAudio/     # LLM-based text anonymization
└── 11_Utilities_FileManagement/  # File utilities
```

## Key Features

### Voice Anonymization (McAdams)

Implements VoicePrivacy Challenge Baseline B2 methodology with **per-recording randomized coefficients** (0.75-0.90 range per Tayebi Arasteh 2024). This defeats cross-session linkage attacks that would be possible with a fixed coefficient.

```python
# Correct: randomized per recording
coefficient = random.uniform(0.75, 0.90)

# WRONG: fixed coefficient exposes entire corpus if one recording is compromised
coefficient = 0.8  # DO NOT USE
```

### NLP Feature Tiers

| Tier | Script | Requires PHI | GPU |
|------|--------|--------------|-----|
| Tier 1 | `optimized-CLE-CSA-R.py` | No | Yes |
| Tier 2 | `tier2_fixed_redux_final.py` | **Yes** | No |
| Tier 3 | `tier3_postid_nexus.py` | No | No |

**Important:** Tier 2 extracts specificity features (proper nouns, named entities) and must run BEFORE text de-identification.

## Dependencies

Third-party tools (not included, install separately):

| Tool | Purpose | Source |
|------|---------|--------|
| OpenSMILE | Acoustic features | https://audeering.github.io/opensmile/ |
| WhisperX | Timestamped ASR | https://github.com/m-bain/whisperX |
| Pyannote | Speaker diarization | https://github.com/pyannote/pyannote-audio |
| OpenFace | Facial features | https://github.com/TadasBaltrusaitis/OpenFace |
| Vosk | Offline ASR | https://alphacephei.com/vosk/ |
| Faster-Whisper | Fast Whisper | https://github.com/SYSTRAN/faster-whisper |

Python dependencies vary by script. Key packages: `librosa`, `parselmouth-praat`, `opensmile`, `torch`, `transformers`, `duckdb`.

## Setup

1. Clone the repository
2. Install required third-party tools
3. Create a HuggingFace token for Pyannote access (gated model)
4. Update paths in scripts to match your local environment
5. Set up Python environment with required packages

```bash
# Example for acoustic extraction
conda create -n ascend python=3.10
conda activate ascend
pip install librosa parselmouth-praat opensmile pandas numpy
```

## Usage Notes

- All hardcoded paths have been replaced with placeholders (`/path/to/...`)
- Store HuggingFace tokens in environment variables, never in code
- Coefficient logs from McAdams anonymization are PHI-adjacent - store securely
- Use `rglob` for file discovery (directory structures vary in depth)

## Pipeline Execution Order

For complete de-identification workflow:

1. Audio Transcription (WhisperX) → timestamps + text
2. NLP Tier 2 (PRE-DeID) → specificity features
3. Text De-ID (CliniDeID) → rule-based PHI removal
4. LLM Anonymization → context-dependent PHI
5. NLP Tiers 1 & 3 (POST-DeID) → remaining features
6. McAdams Anonymization → voice de-identification
7. Acoustic Feature Extraction → final features

## Citation

If you use this code, please cite:

```
Pressman, P. (2025). ASCEND: Automated Speech Comparison Engine for 
Neurocognitive Detection. https://github.com/pspressman/ASCEND_CODE_REPOSITORY
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

Peter Pressman  
Syntopic Systems LLC  
OHSU Department of Neurology
Layton Aging and Alzheimer's Disease Center'

---

*Developed with support from NIH K23 award [NIA K23 AG063900]. Research reported in this repository was supported by the National Institute 
on Aging of the National Institutes of Health under Award Number K23AG12345. 
The content is solely the responsibility of the authors and does not necessarily 
represent the official views of the National Institutes of Health.*
