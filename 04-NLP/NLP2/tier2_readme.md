# TIER 2: Pre-De-Identification Feature Extractor

## Overview

The Tier 2 Pre-De-Identification Feature Extractor captures linguistic and psycholinguistic features from speech transcripts that will be **destroyed or altered by HIPAA Safe Harbor de-identification**. This extractor runs on the original, identified transcripts to preserve critical information about proper nouns, temporal references, spatial markers, and semantic specificity before these identifiers are removed.

## Purpose & Scientific Rationale

### The De-Identification Problem

HIPAA Safe Harbor compliance requires removing 18 categories of identifiers, including:
- All proper nouns (names, places, organizations)
- Geographic subdivisions smaller than state
- All dates (except year) and ages over 89
- Device identifiers and serial numbers

While necessary for privacy, this process destroys linguistically informative features that may be diagnostically relevant for detecting cognitive impairment, particularly:
- **Semantic specificity** (captured via proper noun usage and capitalization patterns)
- **Temporal reasoning** (date/time references, temporal markers)
- **Spatial cognition** (location descriptions, geographic specificity)
- **Lexical richness** (measured through proper noun diversity and patterns)

### The Two-Tier Solution

**TIER 1 (NLP1/Clinamen):** Extracts coherence and probability-based features that survive de-identification
**TIER 2 (This script):** Extracts features that will be destroyed by de-identification

This approach allows us to:
1. Preserve maximum diagnostic information before de-identification
2. Maintain HIPAA compliance by de-identifying the original transcripts
3. Compare pre-de-ID vs. post-de-ID feature sets to quantify information loss

---

## Features Extracted

### 1. Robust Lexical Diversity (Cohen et al. methodology)

**Type-Token Ratio (TTR) Variants** - 5 features
- Calculated across 6 different preprocessing combinations
- Captures: min, max, mean, std, range of TTR values
- Accounts for case normalization, punctuation, contractions, and repetition handling

**Moving Average TTR (MATTR)** - 3 features
- Window sizes: 25, 50, 100 tokens
- More stable than simple TTR for varying text lengths

**Honore's Statistic & Brunet's Index** - 2 features
- Advanced lexical richness measures accounting for hapax legomena
- Sensitive to vocabulary sophistication

### 2. WordNet Semantic Granularity - 13 features

**Concept Hierarchy Position**
- Measures semantic specificity via hypernym path length to root concept
- Distribution bins from depth 2-12
- Mean and standard deviation of granularity scores

**Why This Matters:** Cognitive decline often shows shifts from specific to general concepts (e.g., "golden retriever" → "dog" → "animal"). WordNet depth quantifies this.

### 3. MRC Psycholinguistic Norms - 24 features

Extracted from the MRC Psycholinguistic Database (150,000 words) across 4 word types (all content words, nouns, verbs, proper nouns):

**Age of Acquisition (AoA)**
- Earlier-learned words are better preserved in dementia
- Critical marker of semantic memory integrity

**Familiarity**
- Subjective frequency ratings
- Correlates with word accessibility

**Imageability**
- How easily a mental image can be formed
- High imageability words are more resilient to cognitive decline

**Concreteness** ⭐ UNIQUE TO MRC
- Abstract vs. concrete word usage
- AD shows abstract→concrete shift

**Meaningfulness (Colorado Norms)** ⭐ UNIQUE TO MRC
- Association richness (how many associations generated)
- Reflects semantic network integrity

**Meaningfulness (Paivio Norms)** ⭐ UNIQUE TO MRC
- Alternative meaningfulness measure
- Correlated with recall performance

### 4. Capitalization as Specificity Proxy - 4 features

**Novel Insight:** Capitalization patterns indicate semantic specificity before de-identification removes proper nouns.

- **Mid-sentence capitals:** Count and rate of capitalized words (excluding sentence-initial)
- **All-caps words:** Acronyms and emphasis markers
- **Title case sequences:** Multi-word proper noun phrases (e.g., "Mayo Clinic")

**Why This Matters:** These patterns disappear after de-identification but capture specificity/generality of language.

### 5. Proper Noun Patterns - 8 features

- Count, density, uniqueness, diversity
- Repetition patterns and rates
- Multi-word proper noun sequences
- Maximum sequence length

**Diagnostic Relevance:** Proper noun usage reflects:
- Naming ability preservation
- Semantic memory access
- Discourse coherence (appropriate re-introduction vs. inappropriate repetition)

### 6. Temporal Markers - 8 features

**Removed by Safe Harbor:**
- Exact dates (MM/DD/YYYY patterns)
- Month and day names
- Year mentions (1900-2099)
- Time expressions (HH:MM, AM/PM)
- Relative temporal markers (yesterday, recently)
- Ages over 89

**Diagnostic Relevance:** Temporal reasoning and episodic memory integrity

### 7. Spatial Markers - 5 features

**Removed by Safe Harbor:**
- Street addresses
- City, State patterns
- ZIP codes
- Building/landmark mentions
- Location prepositions (spatial language density)

**Diagnostic Relevance:** Spatial cognition, geographic memory, navigation-related language

### 8. Named Entity Recognition - 6 features

Using spaCy NER:
- Person, Location, Organization, Date counts
- Total entity count
- Entity density

**Diagnostic Relevance:** Discourse informativeness, specificity of communication

### 9. Core Linguistic Features - 20 features

**Lexical:** Word count, unique words, filler words (um, uh, etc.)

**Sentiment:** Mean, max, min, std of sentence-level sentiment polarity

**Syntactic (POS):** Frequencies of 8 parts of speech (with proper nouns separated from common nouns), content density

---

## Technical Details

### Dependencies

```
Python 3.8+
pandas
numpy
nltk (punkt, wordnet, averaged_perceptron_tagger)
spacy (en_core_web_sm)
textblob
```

### Input Format

**Transcripts:** Plain text files ending in `*_transcript.txt`

**Supported formats:**
```
[timestamp] Speaker X: text
Speaker X: text
```

**Directory structure:** Any depth of nested folders supported via recursive search (`rglob`)

### Output Format

**CSV file:** `tier2_preid_features.csv`

**Columns:**
- `participant_id`: Full filename (without `_transcript.txt`)
- `date`: Extracted from filename (if present)
- `task_type`: Grandfather, Picnic, Spontaneous, Conversation, or Unknown
- `speaker`: combined, speaker_1, speaker_2, etc.
- ~110-120 feature columns

### Processing Pipeline

1. **Transcript Discovery:** Recursive search for all `*_transcript.txt` files
2. **Speaker Separation:** Extracts text per speaker + combined text
3. **Feature Extraction:** Runs all feature extractors on each speaker's text
4. **Quality Control:** Filters out samples with <10 words
5. **Output:** Single CSV with one row per speaker per transcript

### Performance

- **Speed:** ~2 seconds per transcript (including spaCy NER and MRC norm lookups)
- **Memory:** ~2GB RAM for loading MRC norms dictionary
- **Scalability:** Tested on 2,872 transcripts (~1 hour runtime)

---

## Usage

### Basic Usage

```bash
python tier2_fixed_redux_final.py \
    --transcripts /path/to/transcripts \
    --output /path/to/output \
    --mrc-norms mrc_database.csv
```

### Arguments

- `--transcripts`: Base directory containing transcript files (searches recursively)
- `--output`: Directory for output CSV and log file
- `--mrc-norms`: Path to MRC Psycholinguistic Database CSV file

### MRC Database

**Download:** https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database

```bash
wget https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/resolve/main/mrc_psycholinguistic_database.csv -O mrc_database.csv
```

**Note:** Script runs without MRC norms but sets all psycholinguistic features to 0.

---

## Scientific Applications

### 1. Alzheimer's Disease Detection

**Key Features:**
- AoA (early words preserved), concreteness (abstract→concrete shift)
- Proper noun diversity (naming ability)
- Temporal markers (episodic memory)
- Semantic granularity (concept generalization)

### 2. Information Loss Quantification

Compare Tier 2 features to post-de-identification features to quantify:
- Which features are most affected by de-identification
- Whether de-ID destroys diagnostically relevant information
- Optimal feature selection for de-identified datasets

### 3. Discourse Analysis

- Proper noun patterns reveal discourse coherence
- Capitalization captures referential specificity
- Named entity density measures informativeness

### 4. Multimodal Integration

Tier 2 features complement:
- Acoustic features (Tier 0)
- Coherence/probability features (Tier 1/NLP1)
- Post-de-ID linguistic features (Tier 3)

---

## Important Notes

### HIPAA Compliance

⚠️ **This script processes IDENTIFIED data.** Outputs contain features derived from identifiers and should be treated as PHI until merged with de-identified datasets and properly secured.

**Best Practice:**
1. Run Tier 2 on identified transcripts
2. De-identify original transcripts
3. Run Tier 1/3 on de-identified transcripts
4. Merge feature sets using anonymized participant IDs
5. Delete or secure Tier 2 outputs per institutional policy

### Limitations

- **spaCy NER required:** Install via `python -m spacy download en_core_web_sm`
- **MRC coverage:** ~119,000 words; rare/technical terms may not have norms
- **Language:** English only (TextBlob POS tagging, WordNet)
- **Transcript quality:** Assumes relatively clean ASR output

### Quality Control

- Minimum 10 words per speaker sample
- Error logging for failed transcripts
- Progress tracking every 100 files
- Comprehensive feature category summaries in log

---

## Output Validation

### Expected Feature Counts

- **Lexical Diversity:** 10 features
- **WordNet Granularity:** 13 features  
- **MRC Norms:** 24 features (6 metrics × 4 word types)
- **Capitalization:** 4 features
- **Proper Nouns:** 8 features
- **Temporal Markers:** 8 features
- **Spatial Markers:** 5 features
- **Named Entities:** 6 features
- **Core Linguistic:** 20 features
- **Metadata:** 4 columns

**Total:** ~102 columns

### Log File Contents

- MRC norms loaded: Should show ~119,000 words
- Transcripts found: Total count
- Progress updates: Every 100 files
- Feature category breakdown
- Total processing time
- Output file location

---

## Citation & Acknowledgments

### Psycholinguistic Databases

**MRC Psycholinguistic Database:**
- Wilson, M. (1988). MRC Psycholinguistic Database: Machine-usable dictionary, version 2.00. *Behavior Research Methods, Instruments, & Computers*, 20, 6-10.
- Dataset: StephanAkkerman (2024). HuggingFace Datasets.

**Methodological References:**
- **Robust TTR:** Cohen, K. B., et al. (2020). Type-token ratio: A stable measure of lexical diversity.
- **MATTR:** Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type-token ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94-100.

### Tools

- **spaCy:** Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
- **NLTK:** Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media Inc.
- **WordNet:** Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39-41.

---

## Version History

- **v1.0 (October 2025):** Initial production release
  - Full MRC norm integration (6 psycholinguistic metrics)
  - Capitalization as specificity proxy
  - Robust TTR implementation
  - WordNet semantic granularity
  - Comprehensive temporal/spatial/entity markers

---

## Contact & Support

For questions, bug reports, or feature requests related to this specific implementation, consult project documentation or create an issue in the project repository.

For questions about the underlying psycholinguistic databases:
- **MRC Database:** See original Wilson (1988) paper
- **WordNet:** https://wordnet.princeton.edu/

---

## License

This script is part of the ASCEND (Automated Speech Comparison Engine for Neurocognitive Detection) project. Consult project-level licensing for terms of use.

**Psycholinguistic databases have their own licenses:**
- MRC Database: MIT License (via HuggingFace)
- WordNet: WordNet License (free for research)
