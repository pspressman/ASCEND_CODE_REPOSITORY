# TIER 2 FEATURE EXTRACTION - COMPLETE PACKAGE
## Ready to Deploy on Locus (MacBook Pro)

---

## 📦 Package Contents

This package contains everything you need to run Tier 2 Pre-De-Identification Feature Extraction:

### Core Files
1. **tier2_fixed_redux_final.py** - Main feature extraction script
2. **run_tier2_locus.sh** - Automated execution script for both transcript sources
3. **check_tier2_setup.sh** - Pre-flight validation script

### Documentation
4. **TIER2_SETUP_GUIDE.md** - Comprehensive setup and usage guide
5. **TIER2_QUICK_REFERENCE.txt** - One-page command reference card
6. **tier2_readme.md** - Original documentation from developer
7. **THIS FILE** - Deployment instructions

---

## 🚀 Quick Start (3 Steps)

### Step 1: Place Files on Locus
```bash
# Copy all files to your home directory
cd ~
# Place tier2_fixed_redux_final.py here
# Place run_tier2_locus.sh here
# Place check_tier2_setup.sh here

# Make scripts executable
chmod +x ~/run_tier2_locus.sh
chmod +x ~/check_tier2_setup.sh
```

### Step 2: Validate Environment
```bash
# Run pre-flight check
~/check_tier2_setup.sh
```

This will check:
- ✅ Python 3.8+ installed
- ✅ All required packages (pandas, numpy, nltk, spacy, textblob)
- ✅ spaCy model (en_core_web_sm)
- ✅ NLTK data packages
- ✅ Transcript directories accessible
- ✅ Output directory writable
- ⚠️ MRC database (optional but recommended)

### Step 3: Run Extraction
```bash
# Execute Tier 2 pipeline
~/run_tier2_locus.sh
```

**That's it!** The script will:
1. Process all FasterWhisper transcripts
2. Process all Vosk transcripts
3. Generate feature CSVs and logs
4. Print comprehensive summary

---

## 📋 Detailed Setup Instructions

### Prerequisites Installation

#### 1. Install Python Packages
```bash
pip3 install pandas numpy nltk spacy textblob
```

#### 2. Download spaCy Model (REQUIRED)
```bash
python3 -m spacy download en_core_web_sm
```

#### 3. Download NLTK Data (Auto-downloads on first run, but you can pre-download)
```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

#### 4. Download MRC Database (OPTIONAL but recommended for psycholinguistic features)
```bash
cd ~
wget https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/resolve/main/mrc_psycholinguistic_database.csv -O mrc_database.csv
```

**Note:** Without MRC database, script still runs but 24 psycholinguistic features will be 0.

---

## 📂 What Gets Processed

### Input Directories (Automatically Detected)
```
/Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/Transcripts-alternative/
├── FasterWhisper_Transcripts/
│   └── *_transcript.txt files
└── Vosk_output2/
    └── *_transcript.txt files
```

### Output Directories (Automatically Created)
```
/Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/Tier2_Features/
├── FasterWhisper/
│   ├── tier2_preid_features.csv
│   └── tier2_extraction.log
└── Vosk/
    ├── tier2_preid_features.csv
    └── tier2_extraction.log
```

---

## 🎯 Features Extracted (102 per speaker)

### Full Feature Breakdown

#### 1. Lexical Diversity (10 features)
- TTR (Type-Token Ratio) variants: min, max, mean, std, range
- MATTR (Moving Average TTR): windows 25, 50, 100
- Honore's Statistic
- Brunet's Index

#### 2. WordNet Semantic Granularity (13 features)
- Concept depth distribution (bins 2-12)
- Mean and std of semantic specificity
- **Clinical relevance:** Tracks specific→general concept shifts in dementia

#### 3. MRC Psycholinguistic Norms (24 features)
Extracted for 4 word types (all content, nouns, verbs, proper nouns):
- **Age of Acquisition (AoA)**: Earlier-learned words preserved in dementia
- **Familiarity**: Subjective frequency ratings
- **Imageability**: Mental image formation ease
- **Concreteness**: ⭐ Abstract→concrete shift in AD
- **Meaningfulness (Colorado)**: ⭐ Semantic network integrity
- **Meaningfulness (Paivio)**: ⭐ Recall performance marker

#### 4. Capitalization as Specificity Proxy (4 features)
- Mid-sentence capitals (proper noun density)
- All-caps words (acronyms)
- Title case sequences (multi-word proper nouns)
- **Novel insight:** These patterns disappear after de-identification!

#### 5. Proper Noun Patterns (8 features)
- Count, density, uniqueness, diversity
- Repetition rate and patterns
- Multi-word sequences
- **Diagnostic relevance:** Naming ability, discourse coherence

#### 6. Temporal Markers (8 features)
Features that HIPAA Safe Harbor removes:
- Date patterns, month/day names
- Year mentions, time expressions
- Relative temporal markers
- Ages >89

#### 7. Spatial Markers (5 features)
Features that HIPAA Safe Harbor removes:
- Street addresses, city/state patterns
- ZIP codes, landmarks
- Spatial language density

#### 8. Named Entity Recognition (6 features)
- Person, Location, Organization, Date counts
- Total entity count and density

#### 9. Core Linguistic Features (20 features)
- Lexical: word count, unique words, filler words
- Sentiment: mean, max, min, std polarity
- Syntactic: POS frequencies, content density

#### 10. Metadata (4 columns)
- participant_id, date, task_type, speaker

---

## ⏱️ Expected Performance

### On Locus (MacBook Pro)
- **Speed:** ~2 seconds per transcript
- **Memory:** 2-3 GB (MRC database + spaCy model)
- **CPU:** High utilization (all cores)

### Time Estimates
| Transcript Count | Expected Time |
|-----------------|---------------|
| 500 transcripts | ~15-20 min    |
| 1,000 transcripts | ~30-40 min  |
| 2,872 transcripts | ~90-120 min |

---

## 🔍 Validation & Quality Control

### After Running, Check:

#### 1. Log Files
```bash
# FasterWhisper log
tail -50 /Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/Tier2_Features/FasterWhisper/tier2_extraction.log

# Vosk log
tail -50 /Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/Tier2_Features/Vosk/tier2_extraction.log
```

Expected in logs:
- ✅ MRC norms loaded: ~119,000 words
- ✅ Total transcripts found
- ✅ Feature category breakdown
- ✅ Processing time

#### 2. Output Row Counts
```bash
# Should match: transcripts × speakers (usually 2-3 rows per transcript)
wc -l /Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/Tier2_Features/FasterWhisper/tier2_preid_features.csv
```

#### 3. Preview Features
```bash
head -n 5 /Volumes/video_research/ASCEND_PROCESSING/Audio/NLP/Tier2_Features/FasterWhisper/tier2_preid_features.csv | column -t -s,
```

---

## ⚠️ CRITICAL HIPAA WARNING

### This Script Processes IDENTIFIED Data!

**The output CSVs contain features derived from PHI:**
- Proper nouns (names, places, organizations)
- Dates, times, temporal references
- Locations, spatial markers
- Named entities

**⚠️ THESE OUTPUTS ARE PHI** - Treat as protected data!

### Recommended Workflow

1. **✅ Run Tier 2 on identified transcripts** (this script)
2. **🔒 De-identify original transcripts** (separate process)
3. **📊 Run Tier 1/3 on de-identified transcripts**
4. **🔀 Merge all feature tiers** using anonymized participant IDs
5. **🗑️ Secure or delete Tier 2 raw outputs** per institutional policy

---

## 🔧 Troubleshooting

### Common Issues

#### "Python dependencies missing"
```bash
pip3 install pandas numpy nltk spacy textblob
```

#### "spaCy model not found"
```bash
python3 -m spacy download en_core_web_sm
```

#### "MRC database not found" (WARNING only)
- Script runs without MRC, but psycholinguistic features = 0
- Download: `wget https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/resolve/main/mrc_psycholinguistic_database.csv -O ~/mrc_database.csv`

#### "Transcript directory not found"
- Verify paths in `run_tier2_locus.sh` match your system
- Edit lines 21-27 if necessary

#### Processing very slow?
- Close other applications (need 4+ GB free RAM)
- First run slower (loading models)
- Expected: ~2 seconds per transcript

---

## 📊 Next Steps After Tier 2

### 1. Compare with Tier 3 (Post-De-ID Features)
- Run Tier 3 on de-identified transcripts
- Quantify information loss from de-identification
- Identify which features are most affected

### 2. Merge Feature Tiers
Combine all feature extractions:
- **Tier 0:** Acoustic features (pitch, intensity, MFCCs)
- **Tier 1:** Coherence/probability (NLP1/Clinamen)
- **Tier 2:** Pre-de-ID features (this!)
- **Tier 3:** Post-de-ID features

### 3. Model Development
Use merged features for:
- Alzheimer's Disease detection
- Discourse analysis
- Longitudinal tracking
- Information loss studies

---

## 📚 Additional Resources

### Documentation Files
- **TIER2_SETUP_GUIDE.md** - Comprehensive guide with all details
- **TIER2_QUICK_REFERENCE.txt** - One-page command reference
- **tier2_readme.md** - Original developer documentation

### Key Citations

**MRC Database:**
> Wilson, M. (1988). MRC Psycholinguistic Database: Machine-usable dictionary, version 2.00. *Behavior Research Methods, Instruments, & Computers*, 20, 6-10.

**MATTR:**
> Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type-token ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94-100.

**WordNet:**
> Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39-41.

---

## ✅ Deployment Checklist

Use this checklist to ensure successful deployment:

- [ ] All files placed in home directory (~/)
- [ ] Scripts made executable (`chmod +x`)
- [ ] Python 3.8+ installed
- [ ] All Python packages installed (pandas, numpy, nltk, spacy, textblob)
- [ ] spaCy model downloaded (en_core_web_sm)
- [ ] NLTK data downloaded (punkt, wordnet, averaged_perceptron_tagger)
- [ ] MRC database downloaded (optional but recommended)
- [ ] Transcript directories accessible
- [ ] Pre-flight check passed (`~/check_tier2_setup.sh`)
- [ ] Ready to run! (`~/run_tier2_locus.sh`)

---

## 📞 Support

### If You Encounter Issues:

1. **First:** Run `~/check_tier2_setup.sh` to diagnose
2. **Check:** Log files for detailed error messages
3. **Verify:** Transcript format is `*_transcript.txt`
4. **Confirm:** Directory paths are accessible

### For Questions About:
- **MRC Database:** https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database
- **WordNet:** https://wordnet.princeton.edu/
- **spaCy:** https://spacy.io/

---

## 🎉 Ready to Go!

Your Tier 2 Pre-De-Identification Feature Extraction pipeline is ready to deploy on Locus!

**Final steps:**
```bash
# 1. Validate setup
~/check_tier2_setup.sh

# 2. If all checks pass, run extraction
~/run_tier2_locus.sh
```

**Expected output:**
- Two feature CSVs (FasterWhisper + Vosk)
- Two log files with processing details
- ~90-120 minutes total processing time for ~3,000 transcripts

---

**Version:** October 2025  
**Script:** tier2_fixed_redux_final.py  
**Platform:** macOS (Locus - MacBook Pro)  
**Status:** Production Ready ✅
