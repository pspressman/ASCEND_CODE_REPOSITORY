# TIER 2 OUTPUT CSV SCHEMA
## tier2_preid_features.csv

---

## File Format

- **Format:** CSV (Comma-Separated Values)
- **Encoding:** UTF-8
- **Header:** Yes (first row)
- **Rows:** One per speaker per transcript
- **Typical structure:** 2-3 rows per transcript (combined + individual speakers)

---

## Column Structure (~106 columns)

### METADATA COLUMNS (4)

| Column Name | Type | Description | Example |
|------------|------|-------------|---------|
| participant_id | string | Full filename without `_transcript.txt` | `2-001-8-GrandfatherPassage_clean` |
| date | string | Extracted from filename (YYYY-MM-DD format) | `2024-03-15` |
| task_type | string | Inferred from filename | `Grandfather`, `Picnic`, `Spontaneous`, `Conversation`, `Unknown` |
| speaker | string | Speaker identifier | `combined`, `speaker_1`, `speaker_2` |

---

### FEATURE COLUMNS (~102)

#### 1. LEXICAL DIVERSITY (10 columns)

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| ttr_min | float | 0.0-1.0 | Minimum Type-Token Ratio across preprocessing variants |
| ttr_max | float | 0.0-1.0 | Maximum Type-Token Ratio across preprocessing variants |
| ttr_mean | float | 0.0-1.0 | Mean Type-Token Ratio across preprocessing variants |
| ttr_std | float | 0.0-1.0 | Standard deviation of TTR across variants |
| ttr_range | float | 0.0-1.0 | Range (max-min) of TTR across variants |
| mattr_25 | float | 0.0-1.0 | Moving Average TTR with window=25 |
| mattr_50 | float | 0.0-1.0 | Moving Average TTR with window=50 |
| mattr_100 | float | 0.0-1.0 | Moving Average TTR with window=100 |
| honore_statistic | float | 0.0-∞ | Honore's R statistic (accounts for hapax legomena) |
| brunet_index | float | 0.0-∞ | Brunet's W index (lexical richness) |

**Clinical Relevance:** Lower diversity scores may indicate reduced vocabulary access in cognitive decline.

---

#### 2. WORDNET SEMANTIC GRANULARITY (13 columns)

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| wordnet_depth_2 | float | 0.0-1.0 | Proportion of words at depth 2 (very general) |
| wordnet_depth_3 | float | 0.0-1.0 | Proportion of words at depth 3 |
| wordnet_depth_4 | float | 0.0-1.0 | Proportion of words at depth 4 |
| wordnet_depth_5 | float | 0.0-1.0 | Proportion of words at depth 5 (mid-level) |
| wordnet_depth_6 | float | 0.0-1.0 | Proportion of words at depth 6 |
| wordnet_depth_7 | float | 0.0-1.0 | Proportion of words at depth 7 |
| wordnet_depth_8 | float | 0.0-1.0 | Proportion of words at depth 8 |
| wordnet_depth_9 | float | 0.0-1.0 | Proportion of words at depth 9 |
| wordnet_depth_10 | float | 0.0-1.0 | Proportion of words at depth 10 |
| wordnet_depth_11 | float | 0.0-1.0 | Proportion of words at depth 11 |
| wordnet_depth_12 | float | 0.0-1.0 | Proportion of words at depth 12+ (very specific) |
| wordnet_mean_depth | float | 2.0-12.0 | Mean semantic depth |
| wordnet_std_depth | float | 0.0-∞ | Standard deviation of semantic depth |

**Clinical Relevance:** AD shows shift toward lower depths (generalâ†specific). Higher mean depth = more specific concepts.

---

#### 3. MRC PSYCHOLINGUISTIC NORMS (24 columns)

**Format:** 6 metrics × 4 word types = 24 columns

**Word Types:**
- `all`: All content words (nouns, verbs, adjectives, adverbs, proper nouns)
- `noun`: Common nouns only
- `verb`: Verbs only
- `propn`: Proper nouns only

**Metrics:**

| Metric Suffix | Full Name | Range | Description |
|--------------|-----------|-------|-------------|
| _aoa | Age of Acquisition | 100-700 | When word is typically learned (lower = earlier) |
| _fam | Familiarity | 100-700 | Subjective frequency rating |
| _img | Imageability | 100-700 | Ease of mental image formation |
| _conc | Concreteness | 100-700 | Abstract vs. concrete (higher = more concrete) |
| _meanc | Meaningfulness (Colorado) | 100-700 | Association richness |
| _meanp | Meaningfulness (Paivio) | 100-700 | Alternative meaningfulness measure |

**Column Names (24 total):**
```
mrc_all_aoa, mrc_all_fam, mrc_all_img, mrc_all_conc, mrc_all_meanc, mrc_all_meanp
mrc_noun_aoa, mrc_noun_fam, mrc_noun_img, mrc_noun_conc, mrc_noun_meanc, mrc_noun_meanp
mrc_verb_aoa, mrc_verb_fam, mrc_verb_img, mrc_verb_conc, mrc_verb_meanc, mrc_verb_meanp
mrc_propn_aoa, mrc_propn_fam, mrc_propn_img, mrc_propn_conc, mrc_propn_meanc, mrc_propn_meanp
```

**Clinical Relevance:**
- **AoA:** Earlier-learned words preserved in dementia (expect LOWER values in AD)
- **Concreteness:** AD shows abstract→concrete shift (expect HIGHER values in AD)
- **Meaningfulness:** Reflects semantic network integrity (expect LOWER in AD)

**Note:** If MRC database not available, all 24 columns = 0.0

---

#### 4. CAPITALIZATION AS SPECIFICITY PROXY (4 columns)

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| cap_mid_sentence_count | int | 0-∞ | Count of mid-sentence capitalized words (excluding sentence-initial) |
| cap_mid_sentence_rate | float | 0.0-1.0 | Rate per 100 words |
| cap_all_caps_count | int | 0-∞ | Count of fully capitalized words (e.g., "NASA", "HIPAA") |
| cap_title_case_sequences | int | 0-∞ | Count of multi-word title case sequences (e.g., "Mayo Clinic") |

**Clinical Relevance:** ⭐ Novel insight - these patterns disappear after de-identification but capture semantic specificity before removal of proper nouns.

---

#### 5. PROPER NOUN PATTERNS (8 columns)

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| propn_count | int | 0-∞ | Total proper noun count |
| propn_density | float | 0.0-1.0 | Proper nouns per 100 words |
| propn_unique | int | 0-∞ | Number of unique proper nouns |
| propn_diversity | float | 0.0-1.0 | Unique / total ratio |
| propn_repetition_count | int | 0-∞ | Count of repeated proper nouns |
| propn_repetition_rate | float | 0.0-1.0 | Repetition rate (repeated / total) |
| propn_multiword_sequences | int | 0-∞ | Multi-word proper noun sequences (e.g., "John Smith") |
| propn_max_sequence_length | int | 0-∞ | Maximum words in a single proper noun sequence |

**Clinical Relevance:** 
- Lower diversity → naming difficulty
- Higher repetition → discourse coherence issues
- Fewer multiword sequences → reduced specificity

---

#### 6. TEMPORAL MARKERS (8 columns)

**⚠️ REMOVED BY SAFE HARBOR DE-IDENTIFICATION**

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| temporal_date_patterns | int | 0-∞ | Date formats (MM/DD/YYYY, etc.) |
| temporal_month_names | int | 0-∞ | Month names (January, Feb, etc.) |
| temporal_day_names | int | 0-∞ | Day names (Monday, Tuesday, etc.) |
| temporal_year_mentions | int | 0-∞ | Year mentions (1900-2099) |
| temporal_time_expressions | int | 0-∞ | Time formats (HH:MM, AM/PM) |
| temporal_relative_markers | int | 0-∞ | Relative terms (yesterday, recently, last week) |
| temporal_age_over_89 | int | 0-∞ | Mentions of ages >89 (removed by Safe Harbor) |
| temporal_total_density | float | 0.0-1.0 | Total temporal markers per 100 words |

**Clinical Relevance:** Reduced temporal markers may indicate episodic memory decline. Compare with post-de-ID to measure information loss.

---

#### 7. SPATIAL MARKERS (5 columns)

**⚠️ REMOVED BY SAFE HARBOR DE-IDENTIFICATION**

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| spatial_addresses | int | 0-∞ | Street addresses |
| spatial_city_state_patterns | int | 0-∞ | City, State patterns |
| spatial_zip_codes | int | 0-∞ | ZIP code patterns |
| spatial_location_mentions | int | 0-∞ | Building/landmark mentions |
| spatial_preposition_density | float | 0.0-1.0 | Spatial prepositions per 100 words (in, at, near, etc.) |

**Clinical Relevance:** Spatial cognition, geographic memory, navigation-related language. Compare with post-de-ID features.

---

#### 8. NAMED ENTITY RECOGNITION (6 columns)

**Using spaCy NER**

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| ner_person_count | int | 0-∞ | PERSON entities |
| ner_location_count | int | 0-∞ | GPE, LOC entities |
| ner_organization_count | int | 0-∞ | ORG entities |
| ner_date_count | int | 0-∞ | DATE entities |
| ner_total_entities | int | 0-∞ | Total named entities |
| ner_entity_density | float | 0.0-1.0 | Entities per 100 words |

**Clinical Relevance:** Entity density measures discourse informativeness and specificity.

---

#### 9. CORE LINGUISTIC FEATURES (20 columns)

**Lexical (4 columns):**

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| word_count | int | 0-∞ | Total words |
| unique_words | int | 0-∞ | Unique word count |
| filler_word_count | int | 0-∞ | Filler words (um, uh, like, you know) |
| filler_word_rate | float | 0.0-1.0 | Filler words per 100 words |

**Sentiment (4 columns):**

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| sentiment_mean | float | -1.0 to 1.0 | Mean sentence polarity |
| sentiment_max | float | -1.0 to 1.0 | Maximum sentence polarity |
| sentiment_min | float | -1.0 to 1.0 | Minimum sentence polarity |
| sentiment_std | float | 0.0-∞ | Standard deviation of polarity |

**Syntactic - POS Tags (8 columns):**

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| pos_noun_freq | float | 0.0-1.0 | Common noun frequency (excludes proper nouns) |
| pos_propn_freq | float | 0.0-1.0 | Proper noun frequency |
| pos_verb_freq | float | 0.0-1.0 | Verb frequency |
| pos_adj_freq | float | 0.0-1.0 | Adjective frequency |
| pos_adv_freq | float | 0.0-1.0 | Adverb frequency |
| pos_pron_freq | float | 0.0-1.0 | Pronoun frequency |
| pos_det_freq | float | 0.0-1.0 | Determiner frequency |
| pos_prep_freq | float | 0.0-1.0 | Preposition frequency |

**Syntactic - Derived (4 columns):**

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| content_density | float | 0.0-1.0 | Content words / total words |
| noun_verb_ratio | float | 0.0-∞ | Nouns / verbs (if verbs > 0) |
| pronoun_noun_ratio | float | 0.0-∞ | Pronouns / nouns (if nouns > 0) |
| function_word_density | float | 0.0-1.0 | Function words / total words |

---

## Data Types Summary

- **int**: Integer counts (≥ 0)
- **float**: Decimal values
- **string**: Text values

---

## Missing Data Handling

- **MRC features without database:** 0.0
- **Division by zero:** 0.0 (e.g., ratios when denominator = 0)
- **Empty text:** All features = 0
- **Insufficient tokens (<10 words):** Row excluded

---

## Row Structure Examples

### Example 1: Grandfather Passage (Single Speaker)
```
participant_id,date,task_type,speaker,word_count,unique_words,...
2-001-8-GrandfatherPassage_clean,2024-03-15,Grandfather,combined,119,87,...
```

### Example 2: Conversation (Two Speakers)
```
participant_id,date,task_type,speaker,word_count,unique_words,...
2-001-8-Conversation_clean,2024-03-15,Conversation,combined,342,156,...
2-001-8-Conversation_clean,2024-03-15,Conversation,speaker_1,198,103,...
2-001-8-Conversation_clean,2024-03-15,Conversation,speaker_2,144,89,...
```

---

## Validation Checks

### Expected Column Count
- **With metadata:** 4 metadata + ~102 features = ~106 columns
- Exact count may vary slightly based on feature implementation

### Expected Row Count
- **Per transcript:** 
  - Single-speaker tasks (Grandfather, Picnic): 1 row (combined only)
  - Multi-speaker tasks (Conversation): 3+ rows (combined + individual speakers)

### Data Quality Checks
```python
import pandas as pd

# Load CSV
df = pd.read_csv('tier2_preid_features.csv')

# Check dimensions
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# Check for missing values
print(df.isnull().sum().sum(), "missing values")

# Check value ranges
print("TTR range:", df['ttr_mean'].min(), "-", df['ttr_mean'].max())
print("Word count range:", df['word_count'].min(), "-", df['word_count'].max())

# Check speakers
print("Speakers:", df['speaker'].unique())
```

---

## Usage in Analysis

### Loading for Analysis
```python
import pandas as pd

# Load features
df = pd.read_csv('tier2_preid_features.csv')

# Filter by speaker type
combined_only = df[df['speaker'] == 'combined']
speaker1_only = df[df['speaker'] == 'speaker_1']

# Filter by task type
grandfather = df[df['task_type'] == 'Grandfather']
conversations = df[df['task_type'] == 'Conversation']
```

### Merging with Other Tiers
```python
# Merge Tier 1 (post-de-ID) and Tier 2 (pre-de-ID)
tier1 = pd.read_csv('tier1_features.csv')
tier2 = pd.read_csv('tier2_preid_features.csv')

# Merge on participant_id and speaker
merged = pd.merge(tier1, tier2, 
                  on=['participant_id', 'speaker'], 
                  suffixes=('_postdeid', '_preid'))
```

### Feature Selection
```python
# Select MRC features only
mrc_cols = [col for col in df.columns if col.startswith('mrc_')]
mrc_df = df[['participant_id', 'speaker'] + mrc_cols]

# Select temporal markers (will be removed by de-ID)
temporal_cols = [col for col in df.columns if col.startswith('temporal_')]
temporal_df = df[['participant_id', 'speaker'] + temporal_cols]
```

---

## Comparison with Post-De-ID (Tier 3)

### Expected Differences After De-Identification

**These features will be AFFECTED:**
- All proper noun features (propn_*)
- All temporal markers (temporal_*)
- All spatial markers (spatial_*)
- Named entity counts (ner_*)
- Capitalization patterns (cap_*)
- MRC proper noun features (mrc_propn_*)
- WordNet depth (may shift toward general concepts)

**These features should be PRESERVED:**
- TTR and MATTR (lexical diversity)
- MRC features for non-proper words (mrc_all_*, mrc_noun_*, mrc_verb_*)
- POS frequencies (excluding proper nouns)
- Sentiment features
- Core lexical features

### Quantifying Information Loss
```python
# Compare feature means pre- vs post-de-ID
tier2_means = tier2[temporal_cols].mean()
tier3_means = tier3[temporal_cols].mean()
loss = ((tier2_means - tier3_means) / tier2_means * 100)
print("Temporal marker loss:", loss.mean(), "%")
```

---

## File Size Estimates

- **Small dataset (100 transcripts):** ~1-2 MB
- **Medium dataset (1,000 transcripts):** ~10-20 MB
- **Large dataset (10,000 transcripts):** ~100-200 MB

---

## Version History

- **v1.0 (October 2025):** Initial schema
  - 102 features + 4 metadata columns
  - Full MRC integration (24 features)
  - Capitalization proxy (4 features)
  - Comprehensive temporal/spatial markers

---

## Notes

1. **PHI Warning:** This CSV contains features derived from PHI. Treat as protected data.
2. **Feature Stability:** Feature names and counts may vary slightly across script versions.
3. **Row Filtering:** Rows with <10 words per speaker are excluded automatically.
4. **Encoding:** Always use UTF-8 when reading/writing this CSV.

---

**Schema Version:** 1.0  
**Last Updated:** October 2025  
**Corresponds to:** tier2_fixed_redux_final.py
