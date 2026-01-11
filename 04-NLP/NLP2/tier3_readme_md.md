# Tier 3 Post-De-identification Feature Extraction

## Overview

**Script**: `tier3_postid_safe.py`

This script extracts **60 linguistic features that SURVIVE de-identification**. These features can be extracted either before OR after running de-identification processes, as they are not affected by synthetic name replacement, date generalization, or location removal.

### Key Advantage ✅
**Safe to run at any time** - these features are preserved through de-identification, so there's no risk of data loss.

---

## Feature Categories Extracted

### 1. Enhanced Disfluency (10 features)
Measures speech production difficulties and repairs:
- **Filled pauses**: um, uh, er, ah, mm, hmm
- **Word repetitions**: Consecutive identical words
- **False starts**: Very short incomplete sentences
- **Self-corrections**: "I mean", "or rather", "actually"
- **Restart markers**: Dashes, ellipses
- **Density metrics**: Normalized by word count

**Why preserved**: Disfluencies are surface-level speech patterns unaffected by entity replacement.

**Clinical relevance**: Elevated in cognitive decline, motor speech disorders, and language processing difficulties.

### 2. Discourse Markers (15 features)
Tracks connectives and organizational language:

**Temporal markers**:
- then, next, after, before, while, when, later, earlier, finally

**Causal connectives**:
- because, since, so, therefore, thus, hence, consequently

**Contrastive markers**:
- but, however, although, though, yet, nevertheless, instead

**Additive markers**:
- and, also, moreover, furthermore, additionally, plus

**Clarification markers**:
- for example, for instance, such as, like, that is

**Why preserved**: Discourse markers are functional words that remain unchanged during de-identification.

**Clinical relevance**: Reduced discourse marker usage correlates with executive function deficits and discourse coherence problems.

### 3. Narrative Structure (10 features)
Evaluates story grammar and narrative organization:
- **Setting markers**: was, were, had
- **Problem indicators**: problem, difficult, trouble, challenge
- **Resolution markers**: solved, fixed, resolved, ended
- **Emotion/evaluation**: happy, sad, worried, felt
- **Sequence markers**: first, second, then, next, finally
- **Sentence length metrics**: Average sentence length

**Why preserved**: Narrative structure depends on organizational patterns, not specific entities.

**Clinical relevance**: Narrative impairments are sensitive markers of cognitive decline, especially in Alzheimer's disease.

### 4. Enhanced Lexical (15 features)
Advanced vocabulary diversity and sophistication metrics:

**Word length statistics**:
- Average word length
- Maximum word length
- Word length variance

**Lexical diversity**:
- **MATTR** (Moving Average Type-Token Ratio): More stable than simple TTR
- **Honore's Statistic**: Vocabulary richness measure
- **Yule's K**: Lexical diversity index
- Academic/formal word ratio (long words as proxy)

**Why preserved**: Lexical diversity measures are based on word distributions, not specific content.

**Clinical relevance**: Lexical diversity declines in dementia and correlates with cognitive reserve.

### 5. Lightweight Graph Features (10 features)
Network analysis of word co-occurrence:
- **Node count**: Unique words in graph
- **Edge count**: Word-to-word connections
- **Graph density**: Connectivity ratio
- **Average degree**: Mean connections per word
- **Clustering coefficient**: Local connectivity measure
- **Connected components**: Separate subgraphs

**Why preserved**: Graph structure depends on word relationships, not entity identities.

**Clinical relevance**: Graph metrics capture semantic network organization, which degrades in cognitive decline.

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 500MB for outputs
- **CPU**: Any modern CPU (no GPU required)

### Tested Platforms
- ✅ Synology DS923+ (NEXUS)
- ✅ Linux servers
- ✅ macOS (Intel and Apple Silicon)
- ✅ Windows (with Python 3.8+)

### Python Dependencies
```
nltk>=3.8
textblob>=0.17.1
pandas>=1.5.0
networkx>=2.8
```

**Important**: Does NOT require:
- ❌ spaCy (too memory-intensive for Synology)
- ❌ GPU acceleration
- ❌ TensorFlow/PyTorch
- ❌ Large language models

---

## Installation

### On Synology NAS (NEXUS)

1. **SSH into Synology**:
```bash
ssh admin@10.0.0.49
```

2. **Activate virtual environment**:
```bash
source /volume1/python_install/nexus_env/bin/activate
export NLTK_DATA=/volume1/python_install/nltk_data
```

3. **Install dependencies** (if not already installed):
```bash
pip install nltk textblob pandas networkx
```

4. **Download NLTK data**:
```bash
python3.9 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

5. **Transfer script to NEXUS**:
```bash
# From your local machine:
scp tier3_postid_safe.py admin@10.0.0.49:/volume1/scripts/
```

### On Other Systems (Linux/Mac/Windows)

1. **Create virtual environment**:
```bash
python3 -m venv nlp_env
source nlp_env/bin/activate  # Linux/Mac
# OR
nlp_env\Scripts\activate  # Windows
```

2. **Install dependencies**:
```bash
pip install nltk textblob pandas networkx
```

3. **Download NLTK data**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

---

## Usage

### Basic Usage

```bash
python3.9 tier3_postid_safe.py \
  --transcripts /volume1/video_analysis \
  --output /volume1/linguistic_features/tier3_output
```

### Can Run on De-identified Data

```bash
# After de-identification
python3.9 tier3_postid_safe.py \
  --transcripts /volume1/deidentified_transcripts \
  --output /volume1/linguistic_features/tier3_output_deidentified
```

### Command-Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--transcripts` | ✅ Yes | Base directory containing transcript files | `/volume1/video_analysis` |
| `--output` | ✅ Yes | Directory for output CSV and logs | `/volume1/linguistic_features/tier3_output` |

### What the Script Does

1. **Searches recursively** for all `*_transcript.txt` files
2. **Parses diarization** - separates speech by speaker
3. **Creates multiple rows** per transcript (combined, speaker_0, speaker_1, etc.)
4. **Extracts 60 features** per speaker view
5. **Saves progress** for crash recovery
6. **Outputs CSV** with all features

---

## Input Format

### Transcript Files

**Expected file patterns**:
- `*_transcript.txt` (primary)
- `*.txt` (fallback)

**Filename formats supported**:
- `2-002-8-TaskName-Mac-1-8-24-EM_transcript.txt`
- `2-002-8-GrandfatherPassage-12-19-22-EM_transcript.txt`
- `MRN-m-d-yy-TaskName_transcript.txt`

**Works with**:
- ✅ Original transcripts (pre-de-identification)
- ✅ De-identified transcripts (post-anonymization)
- ✅ Diarized or non-diarized
- ✅ Any language (optimized for English)

---

## Output Format

### Primary Output: `tier3_postid_features.csv`

**Columns**:
- `participant_id`: From filename (or anonymized ID)
- `date`: From filename (YYYY-MM-DD format or anonymized)
- `task_type`: Grandfather, Picnic, Spontaneous, Conversation, Unknown
- `speaker`: combined, speaker_0, speaker_1, etc.
- `word_count`: Total words in this speaker view
- **60 feature columns**: All Tier 3 features

**Example row**:
```csv
participant_id,date,task_type,speaker,word_count,disf_filled_pause_count,disc_temporal_markers,...
2-002-8,2024-01-08,Grandfather,speaker_0,342,8,12,...
2-002-8,2024-01-08,Grandfather,combined,418,10,15,...
```

### Additional Outputs

**`tier3_extraction.log`**: Detailed processing log
- Progress tracking
- Error messages
- Processing statistics
- ETA calculations

**`processing_state_tier3.json`**: Crash recovery state
- List of completed files
- Last update timestamp
- Enables resuming after interruption

---

## Performance Metrics

### Processing Speed (Typical)

| Transcript Length | Processing Time per Speaker View | Features Extracted |
|-------------------|----------------------------------|-------------------|
| 50-100 words | 1-2 seconds | 60 |
| 100-300 words | 2-4 seconds | 60 |
| 300-500 words | 4-6 seconds | 60 |
| 500+ words | 6-10 seconds | 60 |

**Note**: Faster than Tier 2 due to fewer features and simpler computations.

### Full Cohort Estimates

| Cohort Size | Speaker Views | Total Time | Output Size |
|-------------|---------------|------------|-------------|
| 50 transcripts | 100-150 rows | 10-20 min | ~1-3 MB |
| 100 transcripts | 200-300 rows | 20-40 min | ~3-6 MB |
| 300 transcripts | 600-900 rows | 1-2 hours | ~9-18 MB |

**Note**: Times are for Synology DS923+. Faster on more powerful machines.

---

## Crash Recovery

The script automatically saves progress after each transcript. If interrupted:

1. **Simply re-run the same command**
2. Script will:
   - Load `processing_state_tier3.json`
   - Skip already-processed files
   - Continue from where it stopped
3. **No data loss** - all completed extractions are preserved

To **start fresh** (reprocess everything):
```bash
rm /volume1/linguistic_features/tier3_output/processing_state_tier3.json
```

---

## Troubleshooting

### Issue: "No transcript files found"

**Solution**: Check your `--transcripts` path
```bash
# List what's in the directory
ls -la /volume1/video_analysis

# Search for transcript files manually
find /volume1/video_analysis -name "*transcript.txt"
```

### Issue: "ModuleNotFoundError: No module named 'networkx'"

**Solution**: Install networkx
```bash
pip install networkx
```

### Issue: Graph features all return zeros

**Possible causes**:
1. Transcript too short (< 3 words)
2. NetworkX error (check log)
3. All words are non-alphabetic

**Solution**: Check log file for specific error messages

### Issue: MATTR calculation fails

**Cause**: Transcript shorter than 50 words (MATTR window size)

**Expected behavior**: Script falls back to simple TTR for short transcripts

---

## Integration with Full Pipeline

### Workflow Position

```
1. ASR Transcription (WhisperX/Faster-Whisper)
2. Diarization (Pyannote)
3. Tier 2 Pre-DeID Features (must run before de-ID)
4. De-identification (CliniDeID, LLM anonymization)
5. ⭐ THIS SCRIPT (Tier 3 Post-DeID Features) ← YOU ARE HERE
6. Tier 1 Coherence Features (CLINAMEN - GPU-heavy)
7. Analysis & ML modeling
```

### Can Run in Parallel

Since Tier 3 features are preserved:
- ✅ Can run simultaneously with Tier 2
- ✅ Can run before or after de-identification
- ✅ Can run multiple times for verification

### Combining Feature Sets

```python
import pandas as pd

tier1 = pd.read_csv('tier1_coherence_features.csv')
tier2 = pd.read_csv('tier2_preid_features.csv')
tier3 = pd.read_csv('tier3_postid_features.csv')

# Merge on participant_id, date, task_type, speaker
all_features = tier1.merge(tier2, on=['participant_id', 'date', 'task_type', 'speaker'])
all_features = all_features.merge(tier3, on=['participant_id', 'date', 'task_type', 'speaker'])

all_features.to_csv('comprehensive_features.csv', index=False)
```

---

## Expected Output Statistics

### Feature Coverage

After successful extraction on 300 transcripts:
- **Rows**: ~600-900 (2-3 speaker views per transcript)
- **Columns**: ~65 (5 metadata + 60 features)
- **File size**: ~9-18 MB
- **Non-null coverage**: >98% for most features

### Feature Value Ranges (Typical)

| Feature Category | Expected Range | Interpretation |
|------------------|----------------|----------------|
| Filled pause density | 0-5% | Higher = more disfluency |
| Discourse markers | 5-20 per 100 words | Higher = more organized |
| MATTR | 0.6-0.9 | Higher = more lexical diversity |
| Graph density | 0.1-0.4 | Higher = more word repetition |
| Clustering coefficient | 0.0-0.5 | Higher = more local semantic connectivity |

---

## Comparison: Pre vs. Post De-identification

### Verification Study

To quantify de-identification impact:

1. **Extract Tier 3 features from ORIGINAL transcripts**:
```bash
python3.9 tier3_postid_safe.py \
  --transcripts /volume1/original_transcripts \
  --output /volume1/tier3_original
```

2. **Run de-identification** on transcripts

3. **Extract Tier 3 features from DE-IDENTIFIED transcripts**:
```bash
python3.9 tier3_postid_safe.py \
  --transcripts /volume1/deidentified_transcripts \
  --output /volume1/tier3_deidentified
```

4. **Compare**:
```python
import pandas as pd

original = pd.read_csv('/volume1/tier3_original/tier3_postid_features.csv')
deidentified = pd.read_csv('/volume1/tier3_deidentified/tier3_postid_features.csv')

# Merge and compute correlations
merged = original.merge(deidentified, on=['participant_id', 'task_type', 'speaker'], suffixes=('_orig', '_deid'))

for feature in [col for col in original.columns if col not in ['participant_id', 'date', 'task_type', 'speaker']]:
    corr = merged[f'{feature}_orig'].corr(merged[f'{feature}_deid'])
    print(f"{feature}: r={corr:.3f}")
```

**Expected**: Correlations > 0.95 for all features (proving preservation)

---

## Best Practices

### Before Running

1. ✅ Verify all transcripts are complete
2. ✅ Check disk space (need ~50MB per 300 transcripts)
3. ✅ Ensure NLTK data is downloaded
4. ✅ Test on 2-3 transcripts first

### While Running

1. ✅ Monitor log file for errors
2. ✅ Don't interrupt (but safe if you do - has crash recovery!)
3. ✅ Can run in background on NEXUS

### After Running

1. ✅ Verify output CSV row count
2. ✅ Check for high NaN rates (>10% suggests problems)
3. ✅ Spot-check a few rows manually
4. ✅ Compare with Tier 2 row counts (should match)

---

## Validation Checklist

After extraction completes:

```bash
# 1. Check output file exists
ls -lh /volume1/linguistic_features/tier3_output/tier3_postid_features.csv

# 2. Count rows
wc -l /volume1/linguistic_features/tier3_output/tier3_postid_features.csv

# 3. Check log for errors
grep ERROR /volume1/linguistic_features/tier3_output/tier3_extraction.log

# 4. Verify column count (should be ~65)
head -1 /volume1/linguistic_features/tier3_output/tier3_postid_features.csv | tr ',' '\n' | wc -l
```

**Expected results**:
- ✅ File size: 30KB - 20MB (depending on cohort)
- ✅ Row count: 2-3x number of transcripts
- ✅ Column count: ~65
- ✅ No ERROR messages in log

---

## When to Use This Script

### ✅ Use Tier 3 When:

1. **After de-identification**: Extracting features from anonymized data
2. **Parallel processing**: Running alongside Tier 2 to save time
3. **Verification**: Testing that features truly survive de-identification
4. **Low-resource environments**: Need features but don't have GPU for Tier 1
5. **Quick turnaround**: Need linguistic features fast (Tier 3 is fastest)

### ❌ Don't Use Tier 3 For:

1. **Coherence analysis**: Use Tier 1 (CLINAMEN with embeddings)
2. **Specificity metrics**: Use Tier 2 (destroyed by de-identification)
3. **Entity tracking**: Use Tier 2 (needs proper nouns intact)
4. **Episodic memory**: Use Tier 2 (needs temporal/spatial details)

---

## Clinical Applications

### Tier 3 Features Are Useful For:

**Speech fluency assessment**:
- Disfluency features sensitive to motor speech disorders
- Filled pause patterns in Parkinson's disease

**Discourse organization**:
- Discourse marker usage in executive function deficits
- Narrative structure in Alzheimer's disease

**Lexical abilities**:
- Vocabulary diversity (MATTR) in cognitive reserve
- Word-finding difficulties (lexical diversity decline)

**Semantic networks**:
- Graph connectivity in semantic dementia
- Clustering patterns in thought disorder

---

## Citation

If you use this script in research, please cite:

```
Post-De-identification Linguistic Feature Extraction (Tier 3)
ASCEND Project - Speech & Language Biomarkers
Portland VA Medical Center / Oregon Health & Science University
2025
```

---

## Support & Contact

**Issues**: Check log file first (`tier3_extraction.log`)

**Questions**: Review this README and the troubleshooting section

**Script version**: 1.0 (January 2025)

---

## License

Internal research use only. Not for commercial distribution.

---

## Changelog

**Version 1.0** (January 2025)
- Initial release
- 60 post-de-identification features
- Diarization support
- Crash recovery
- NEXUS/Synology optimized
- NetworkX graph features
- MATTR lexical diversity
