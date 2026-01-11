# TIER 2 FEATURE EXTRACTION - SETUP & USAGE GUIDE
## For Locus (MacBook Pro)

## Quick Start

```bash
# 1. Place files in your home directory
cd ~
# Download tier2_fixed_redux_final.py and run_tier2_locus.sh

# 2. Make the script executable
chmod +x ~/run_tier2_locus.sh

# 3. Run it!
~/run_tier2_locus.sh
```

---

## Prerequisites

### 1. Python Dependencies

```bash
# Install required packages
pip3 install pandas numpy nltk spacy textblob

# Download spaCy language model (required for NER)
python3 -m spacy download en_core_web_sm

# Download NLTK data (script will auto-download if missing)
python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 2. MRC Psycholinguistic Database (OPTIONAL but RECOMMENDED)

**Without MRC database:** Script runs fine but all 24 psycholinguistic features will be set to 0.

**With MRC database:** Extract AoA, familiarity, imageability, concreteness, and meaningfulness norms.

```bash
# Download from HuggingFace (140MB)
cd ~
wget https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/resolve/main/mrc_psycholinguistic_database.csv -O mrc_database.csv
```

---

## Directory Structure

### Input (Automatically detected)
```
/Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/
├── Transcripts-alternative/
│   ├── FasterWhisper_Transcripts/
│   │   └── *_transcript.txt files
│   └── Vosk_output2/
│       └── *_transcript.txt files
```

### Output (Automatically created)
```
/Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/
└── Tier2_Features/
    ├── FasterWhisper/
    │   ├── tier2_preid_features.csv
    │   └── tier2_extraction.log
    └── Vosk/
        ├── tier2_preid_features.csv
        └── tier2_extraction.log
```

---

## What Gets Extracted?

### Feature Categories (Total: ~102 features per speaker)

#### 1. Lexical Diversity (10 features)
- **TTR variants**: min, max, mean, std, range across 6 preprocessing combinations
- **MATTR**: Moving Average TTR (windows: 25, 50, 100 tokens)
- **Honore's Statistic**: Accounts for hapax legomena
- **Brunet's Index**: Advanced lexical richness

#### 2. WordNet Semantic Granularity (13 features)
- **Concept depth**: Distribution of hypernym path lengths (bins 2-12)
- **Mean/std**: Average semantic specificity
- **Clinical relevance**: Quantifies specific→general concept shifts in dementia

#### 3. MRC Psycholinguistic Norms (24 features = 6 metrics Ã— 4 word types)
Word types: all content words, nouns, verbs, proper nouns

Metrics extracted:
- **Age of Acquisition (AoA)**: Earlier-learned words preserved in dementia
- **Familiarity**: Subjective frequency ratings
- **Imageability**: Ease of mental image formation
- **Concreteness**: Abstract vs. concrete (⭐ AD shows abstract→concrete shift)
- **Meaningfulness (Colorado)**: Association richness (⭐ semantic network integrity)
- **Meaningfulness (Paivio)**: Alternative meaningfulness measure (⭐ recall performance)

#### 4. Capitalization as Specificity Proxy (4 features)
- **Mid-sentence capitals**: Proper noun density (excluding sentence-initial)
- **All-caps words**: Acronyms and emphasis
- **Title case sequences**: Multi-word proper nouns (e.g., "Mayo Clinic")

⭐ **Novel insight**: These patterns disappear after de-identification but capture semantic specificity!

#### 5. Proper Noun Patterns (8 features)
- Count, density, uniqueness, diversity
- Repetition rate, multi-word sequences
- **Diagnostic relevance**: Naming ability, semantic memory, discourse coherence

#### 6. Temporal Markers (8 features)
What gets REMOVED by Safe Harbor de-identification:
- Date patterns (MM/DD/YYYY)
- Month/day names
- Years (1900-2099)
- Time expressions (HH:MM, AM/PM)
- Relative temporal markers (yesterday, recently)
- Ages >89

#### 7. Spatial Markers (5 features)
What gets REMOVED by Safe Harbor de-identification:
- Street addresses
- City, State patterns
- ZIP codes
- Building/landmark mentions
- Location prepositions (spatial language density)

#### 8. Named Entity Recognition (6 features)
- Person, Location, Organization, Date entity counts
- Total entity count
- Entity density

#### 9. Core Linguistic Features (20 features)
- **Lexical**: Word count, unique words, filler words (um, uh)
- **Sentiment**: Mean, max, min, std of sentence polarity
- **Syntactic**: POS frequencies (8 types), content density

---

## Usage Examples

### Standard Run (Both Transcript Sources)
```bash
~/run_tier2_locus.sh
```

### Custom Locations
Edit `run_tier2_locus.sh` if your paths differ:
```bash
# Around line 21-23
BASE_DIR="/your/custom/path/Audio/NLP"
TRANSCRIPTS_BASE="${BASE_DIR}/Transcripts-alternative"
OUTPUT_BASE="${BASE_DIR}/Tier2_Features"
```

---

## Performance Expectations (on Locus)

### Processing Speed
- **Per transcript**: ~2 seconds (with spaCy NER + MRC lookups)
- **1,000 transcripts**: ~30-40 minutes
- **2,872 transcripts** (your dataset): ~90-120 minutes

### Resource Usage
- **CPU**: High during processing (all cores utilized)
- **Memory**: ~2-3 GB (mostly for MRC database and spaCy model)
- **Disk**: Minimal (<100 MB per output CSV)

---

## Output Files Explained

### tier2_preid_features.csv
**Columns:**
- `participant_id`: Filename without `_transcript.txt`
- `date`: Extracted from filename if present
- `task_type`: Grandfather, Picnic, Spontaneous, Conversation, Unknown
- `speaker`: combined, speaker_1, speaker_2, etc.
- ~102 feature columns

**One row per speaker per transcript** (e.g., conversation with 2 speakers = 3 rows: combined + speaker_1 + speaker_2)

### tier2_extraction.log
**Contains:**
- Total transcripts found
- MRC norms loaded confirmation (~119,000 words)
- Progress updates every 100 files
- Feature category breakdown
- Processing time and errors

---

## Quality Control

### Automatic Filtering
- Minimum 10 words per speaker sample
- Skips invalid transcripts with error logging
- Handles missing timestamps gracefully

### Validation Checks
After running, check the log file:
```bash
# FasterWhisper results
tail -50 /Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/Tier2_Features/FasterWhisper/tier2_extraction.log

# Vosk results
tail -50 /Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/Tier2_Features/Vosk/tier2_extraction.log
```

---

## Troubleshooting

### "ERROR: Missing Python dependencies"
```bash
pip3 install pandas numpy nltk spacy textblob
python3 -m spacy download en_core_web_sm
```

### "spaCy model not found"
```bash
python3 -m spacy download en_core_web_sm
```

### "MRC database not found" (WARNING, not error)
This is fine! Script will run without MRC features (set to 0).

To get MRC features:
```bash
cd ~
wget https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/resolve/main/mrc_psycholinguistic_database.csv -O mrc_database.csv
```

### Slow processing?
- **Expected**: ~2 seconds per transcript
- **Slow if**: First run (spaCy model loading), low memory
- **Solution**: Close other applications, ensure you have 4+ GB free RAM

### "Transcript directory not found"
Update paths in `run_tier2_locus.sh` lines 21-27 to match your system.

---

## HIPAA Compliance ⚠️

### CRITICAL: This script processes IDENTIFIED data!

**The output CSV contains features derived from:**
- Proper nouns (names, places)
- Dates, times, locations
- Named entities

**These features ARE PHI** until merged with de-identified datasets.

### Best Practices

1. **Run Tier 2 on original (identified) transcripts** âœ"
2. **Run de-identification on transcripts** (separate process)
3. **Run Tier 1 & Tier 3 on de-identified transcripts**
4. **Merge all feature CSVs using anonymized participant IDs**
5. **Secure or delete Tier 2 raw outputs** per institutional policy

### Storage
- Keep Tier 2 outputs in secure location
- Use institutional encryption
- Follow data retention policies
- Document data lineage

---

## Next Steps After Tier 2

### 1. Compare with Tier 3 (Post-De-ID)
Tier 3 runs on de-identified transcripts. Compare feature sets to quantify:
- Which features are most affected by de-identification
- Information loss percentage
- Whether de-ID destroys diagnostically relevant features

### 2. Merge with Other Tiers
- **Tier 0**: Acoustic features (pitch, intensity, MFCCs)
- **Tier 1**: Coherence & probability features (NLP1/Clinamen)
- **Tier 2**: Pre-de-ID features (this script)
- **Tier 3**: Post-de-ID features

### 3. Model Training
Use merged features for:
- AD detection models
- Discourse analysis
- Longitudinal tracking
- Information loss studies

---

## Feature Importance for AD Detection

### Top Clinical Markers (from Tier 2)

1. **Age of Acquisition (AoA)**
   - Earlier-learned words preserved longer
   - Shift to early-acquired vocabulary = decline signal

2. **Proper Noun Diversity**
   - Reduced diversity = naming difficulty
   - Repetition patterns = discourse coherence issues

3. **Concreteness**
   - Abstract→concrete shift in AD
   - Quantifiable via MRC norms

4. **Temporal Markers**
   - Reduced temporal references = episodic memory decline
   - Date/time confusion patterns

5. **Semantic Granularity (WordNet)**
   - Specific→general concept shifts
   - "Golden retriever" â†' "dog" â†' "animal"

---

## Citation

If you use this pipeline in publications, cite:

**MRC Database:**
> Wilson, M. (1988). MRC Psycholinguistic Database: Machine-usable dictionary, version 2.00. *Behavior Research Methods, Instruments, & Computers*, 20, 6-10.

**MATTR:**
> Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type-token ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94-100.

**WordNet:**
> Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39-41.

---

## Support

For issues specific to this implementation:
- Check log files first
- Verify all dependencies installed
- Ensure transcript format is correct (`*_transcript.txt`)
- Confirm directory paths are accessible

For questions about:
- **MRC Database**: https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database
- **WordNet**: https://wordnet.princeton.edu/
- **spaCy**: https://spacy.io/

---

## Version

**Tier 2 Script**: tier2_fixed_redux_final.py (October 2025)
- Full MRC integration (6 psycholinguistic metrics)
- Capitalization as specificity proxy
- Robust TTR implementation
- WordNet semantic granularity
- Comprehensive temporal/spatial/entity markers

---

**Happy feature extracting! 🚀**
