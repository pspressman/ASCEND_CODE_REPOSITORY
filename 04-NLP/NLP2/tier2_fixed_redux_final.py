#!/usr/bin/env python3
"""
TIER 2: PRE-DE-IDENTIFICATION FEATURE EXTRACTOR
Final production version with NO PLACEHOLDERS

Features extracted that will be DESTROYED by de-identification:
1. Robust TTR (multiple type/token definitions per Cohen et al.)
2. MATTR (Moving Average TTR), Honore's statistic
3. WordNet semantic granularity (proper noun-specific paths)
4. MRC psycholinguistic norms (AoA, familiarity, imageability, concreteness, meaningfulness)
5. Proper noun patterns (count, density, repetition)
6. Capitalization as specificity proxy
7. Temporal markers (dates, times, months, days)
8. Spatial markers (addresses, locations, cities)
9. Named entities (people, places, organizations)
10. Lexical/sentiment/syntactic features adapted from feature_extraction_origbu.py

NO coherence/probability features (already in NLP1/Clinamen)
"""

import os
import re
import math
import time
import logging
import argparse
import warnings
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

import nltk
import spacy
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class Tier2PreDeIDExtractor:
    """Extract features that will be destroyed by Safe Harbor de-identification"""
    
    def __init__(self, transcript_base_dir: str, output_dir: str, 
                 mrc_norms_path: Optional[str] = None):
        self.transcript_base_dir = Path(transcript_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Load spaCy for NER
        self.logger.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load MRC norms if available
        self.mrc_norms = None
        if mrc_norms_path and Path(mrc_norms_path).exists():
            self.logger.info(f"Loading MRC norms from {mrc_norms_path}")
            self.mrc_norms = self.load_mrc_norms(mrc_norms_path)
        else:
            self.logger.warning("MRC norms not provided. Psycholinguistic features will be skipped.")
        
        self.logger.info("="*80)
        self.logger.info("TIER 2 PRE-DE-ID EXTRACTOR - FINAL VERSION")
        self.logger.info(f"Transcripts: {self.transcript_base_dir}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("="*80)
    
    def setup_logging(self):
        log_file = self.output_dir / "tier2_extraction.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_mrc_norms(self, path: str) -> Dict:
        """Load MRC psycholinguistic norms from CSV file"""
        try:
            df = pd.read_csv(path)
            norms = {}
            
            for _, row in df.iterrows():
                word = str(row.get('Word', '')).lower()
                if not word or pd.isna(word):
                    continue
                
                norms[word] = {
                    'aoa': float(row.get('Age of Acquisition Rating', 0)) if pd.notna(row.get('Age of Acquisition Rating')) else 0,
                    'familiarity': float(row.get('Familiarity', 0)) if pd.notna(row.get('Familiarity')) else 0,
                    'imageability': float(row.get('Imageability', 0)) if pd.notna(row.get('Imageability')) else 0,
                    'concreteness': float(row.get('Concreteness', 0)) if pd.notna(row.get('Concreteness')) else 0,
                    'meaningfulness_colorado': float(row.get('Meaningfulness: Coloradao Norms', 0)) if pd.notna(row.get('Meaningfulness: Coloradao Norms')) else 0,
                    'meaningfulness_paivio': float(row.get('Meaningfulness: Pavio Norms', 0)) if pd.notna(row.get('Meaningfulness: Pavio Norms')) else 0,
                    'brown_verbal_freq': float(row.get('Brown Verbal Frequency', 0)) if pd.notna(row.get('Brown Verbal Frequency')) else 0,
                }
            
            self.logger.info(f"Loaded MRC norms: {len(norms)} words")
            return norms
            
        except Exception as e:
            self.logger.error(f"Error loading MRC norms: {e}")
            return None
    
    # =========================================================================
    # FILE I/O
    # =========================================================================
    
    def find_all_transcripts(self) -> List[Path]:
        self.logger.info("Searching for transcripts...")
        transcripts = list(self.transcript_base_dir.rglob("*_transcript.txt"))
        self.logger.info(f"Found {len(transcripts)} transcript files")
        return sorted(transcripts)
    
    def extract_participant_info(self, filename: str) -> Tuple[str, str, str]:
        """Use FULL filename as participant_id, extract task type"""
        participant_id = filename.replace('_transcript.txt', '').replace('.txt', '')
        
        name_lower = filename.lower()
        if 'grandfather' in name_lower or 'gfp' in name_lower:
            task_type = 'Grandfather'
        elif 'picnic' in name_lower:
            task_type = 'Picnic'
        elif 'spontaneous' in name_lower or 'spont' in name_lower:
            task_type = 'Spontaneous'
        elif 'conversation' in name_lower:
            task_type = 'Conversation'
        else:
            task_type = 'Unknown'
        
        return participant_id, 'unknown', task_type
    
    def parse_transcript(self, file_path: Path) -> Dict[str, str]:
        """Parse transcript, handling both timestamp formats"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        speaker_texts = {}
        all_text = []
        
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith('gap'):
                continue
            
            # Format 1: [timestamp] Speaker X: text
            match1 = re.match(r'\[[\d\.\s\-]+\]\s*Speaker\s+(\d+|Unknown):\s*(.+)', line)
            if match1:
                speaker_num = match1.group(1)
                text = match1.group(2).strip()
                if text:
                    speaker_key = f'speaker_{speaker_num}'
                    if speaker_key not in speaker_texts:
                        speaker_texts[speaker_key] = []
                    speaker_texts[speaker_key].append(text)
                    all_text.append(text)
                continue
            
            # Format 2: Speaker X: text (no brackets)
            match2 = re.match(r'Speaker\s+(\d+|Unknown):\s*(.+)', line)
            if match2:
                speaker_num = match2.group(1)
                text = match2.group(2).strip()
                if text:
                    speaker_key = f'speaker_{speaker_num}'
                    if speaker_key not in speaker_texts:
                        speaker_texts[speaker_key] = []
                    speaker_texts[speaker_key].append(text)
                    all_text.append(text)
        
        result = {'combined': ' '.join(all_text)}
        for speaker_key, text_list in speaker_texts.items():
            result[speaker_key] = ' '.join(text_list)
        
        return result
    
    # =========================================================================
    # ROBUST TTR (following Cohen et al. paper)
    # =========================================================================
    
    def get_tokens_with_preprocessing(self, text: str, 
                                     normalize_case: bool = True,
                                     remove_punct: bool = False,
                                     expand_contractions: bool = False,
                                     remove_repeats: bool = False) -> List[str]:
        """Get tokens with specific preprocessing"""
        tokens = word_tokenize(text)
        
        if expand_contractions:
            # Simple contraction expansion
            contractions_map = {
                "n't": "not", "'re": "are", "'ve": "have",
                "'ll": "will", "'d": "would", "'m": "am"
            }
            expanded = []
            for token in tokens:
                found = False
                for contraction, expansion in contractions_map.items():
                    if token.endswith(contraction):
                        base = token[:-len(contraction)]
                        expanded.extend([base, expansion])
                        found = True
                        break
                if not found:
                    expanded.append(token)
            tokens = expanded
        
        if remove_punct:
            tokens = [t for t in tokens if t.isalnum()]
        
        if normalize_case:
            tokens = [t.lower() for t in tokens]
        
        if remove_repeats:
            # Remove consecutive duplicates
            filtered = []
            prev = None
            for token in tokens:
                if token != prev:
                    filtered.append(token)
                prev = token
            tokens = filtered
        
        return tokens
    
    def calculate_ttr_variants(self, text: str) -> Dict:
        """Calculate TTR across multiple type/token definitions (Cohen et al.)"""
        features = {}
        
        # Define preprocessing combinations
        preprocessing_configs = [
            {'normalize_case': False, 'remove_punct': False, 'expand_contractions': False, 'remove_repeats': False},
            {'normalize_case': True, 'remove_punct': False, 'expand_contractions': False, 'remove_repeats': False},
            {'normalize_case': True, 'remove_punct': True, 'expand_contractions': False, 'remove_repeats': False},
            {'normalize_case': True, 'remove_punct': False, 'expand_contractions': True, 'remove_repeats': False},
            {'normalize_case': True, 'remove_punct': True, 'expand_contractions': True, 'remove_repeats': False},
            {'normalize_case': True, 'remove_punct': True, 'expand_contractions': True, 'remove_repeats': True},
        ]
        
        ttr_values = []
        for config in preprocessing_configs:
            tokens = self.get_tokens_with_preprocessing(text, **config)
            if len(tokens) > 0:
                types = len(set(tokens))
                ttr = types / len(tokens)
                ttr_values.append(ttr)
        
        # Robust TTR statistics
        if ttr_values:
            features['ttr_min'] = np.min(ttr_values)
            features['ttr_max'] = np.max(ttr_values)
            features['ttr_mean'] = np.mean(ttr_values)
            features['ttr_std'] = np.std(ttr_values)
            features['ttr_range'] = np.max(ttr_values) - np.min(ttr_values)
        else:
            features['ttr_min'] = 0
            features['ttr_max'] = 0
            features['ttr_mean'] = 0
            features['ttr_std'] = 0
            features['ttr_range'] = 0
        
        return features
    
    def calculate_mattr(self, text: str, window_size: int = 50) -> float:
        """Moving Average Type-Token Ratio (Covington & McFall 2010)"""
        tokens = word_tokenize(text.lower())
        
        if len(tokens) < window_size:
            # If text shorter than window, return simple TTR
            return len(set(tokens)) / len(tokens) if tokens else 0
        
        ttr_values = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i+window_size]
            types = len(set(window))
            ttr = types / window_size
            ttr_values.append(ttr)
        
        return np.mean(ttr_values) if ttr_values else 0
    
    def calculate_honore_statistic(self, text: str) -> float:
        """Honore's statistic for lexical richness"""
        tokens = word_tokenize(text.lower())
        
        if not tokens:
            return 0
        
        N = len(tokens)  # Total tokens
        V = len(set(tokens))  # Total types
        
        # Count words appearing exactly once
        word_freq = Counter(tokens)
        V1 = sum(1 for count in word_freq.values() if count == 1)
        
        if V1 == V:
            return 0
        
        try:
            honore = 100 * math.log(N) / (1 - (V1 / V))
        except (ValueError, ZeroDivisionError):
            honore = 0
        
        return honore
    
    def lexical_diversity_features(self, text: str) -> Dict:
        """All lexical diversity features"""
        features = {}
        
        # Robust TTR
        features.update(self.calculate_ttr_variants(text))
        
        # MATTR with multiple window sizes
        for window in [25, 50, 100]:
            features[f'mattr_w{window}'] = self.calculate_mattr(text, window)
        
        # Honore's statistic
        features['honore_statistic'] = self.calculate_honore_statistic(text)
        
        # Brunet's Index (from feature_extraction_origbu.py)
        tokens = word_tokenize(text.lower())
        types = len(set(tokens))
        if tokens and types > 0:
            features['brunets_index'] = types ** (-0.165) * math.log(len(tokens))
        else:
            features['brunets_index'] = 0
        
        return features
    
    # =========================================================================
    # WORDNET SEMANTIC GRANULARITY
    # =========================================================================
    
    def get_wordnet_granularity(self, word: str) -> Optional[int]:
        """Get WordNet granularity (shortest path to 'entity')"""
        synsets = wn.synsets(word, pos=wn.NOUN)
        
        if not synsets:
            return None
        
        # Find shortest path to entity for all synsets
        min_depth = float('inf')
        for synset in synsets:
            try:
                # Get hypernym paths (paths to root)
                paths = synset.hypernym_paths()
                for path in paths:
                    if len(path) < min_depth:
                        min_depth = len(path)
            except:
                continue
        
        return min_depth if min_depth != float('inf') else None
    
    def wordnet_granularity_features(self, text: str) -> Dict:
        """WordNet-based semantic granularity distribution"""
        features = {}
        
        # Get nouns via POS tagging
        blob = TextBlob(text)
        nouns = [word.lower() for word, tag in blob.tags if tag in ['NN', 'NNS']]
        
        if not nouns:
            for i in range(2, 13):
                features[f'granularity_bin_{i}'] = 0
            features['granularity_mean'] = 0
            features['granularity_std'] = 0
            return features
        
        # Calculate granularity for each noun
        granularities = []
        for noun in nouns:
            gran = self.get_wordnet_granularity(noun)
            if gran is not None:
                granularities.append(gran)
        
        # Create histogram bins (2-12)
        bins = {i: 0 for i in range(2, 13)}
        for gran in granularities:
            if gran >= 12:
                bins[12] += 1
            elif gran >= 2:
                bins[gran] += 1
        
        # Normalize by total nouns
        total_nouns = len(nouns)
        for i in range(2, 13):
            features[f'granularity_bin_{i}'] = bins[i] / total_nouns if total_nouns > 0 else 0
        
        # Statistics
        if granularities:
            features['granularity_mean'] = np.mean(granularities)
            features['granularity_std'] = np.std(granularities)
        else:
            features['granularity_mean'] = 0
            features['granularity_std'] = 0
        
        return features
    
    # =========================================================================
    # MRC PSYCHOLINGUISTIC NORMS
    # =========================================================================
    
    def psycholinguistic_norms_features(self, text: str) -> Dict:
        """Extract MRC psycholinguistic norm features"""
        features = {}
        
        if self.mrc_norms is None:
            for metric in ['aoa', 'familiarity', 'imageability', 'concreteness',
                           'meaningfulness_colorado', 'meaningfulness_paivio']:
                features[f'{metric}_mean_all'] = 0
                features[f'{metric}_mean_nouns'] = 0
                features[f'{metric}_mean_verbs'] = 0
                features[f'{metric}_mean_proper_nouns'] = 0
            return features
        
        # Get POS tags
        blob = TextBlob(text)
        
        # Separate word types
        all_content_words = []
        nouns = []
        verbs = []
        proper_nouns = []
        
        for word, tag in blob.tags:
            word_lower = word.lower()
            if tag in ['NN', 'NNS']:
                nouns.append(word_lower)
                all_content_words.append(word_lower)
            elif tag in ['NNP', 'NNPS']:
                proper_nouns.append(word_lower)
                all_content_words.append(word_lower)
            elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                verbs.append(word_lower)
                all_content_words.append(word_lower)
        
        # Calculate norms for each word type
        for metric in ['aoa', 'familiarity', 'imageability', 'concreteness',
                       'meaningfulness_colorado', 'meaningfulness_paivio']:
            # All content words
            values = [self.mrc_norms[w][metric] for w in all_content_words
                     if w in self.mrc_norms and self.mrc_norms[w][metric] > 0]
            features[f'{metric}_mean_all'] = np.mean(values) if values else 0
            
            # Nouns
            values = [self.mrc_norms[w][metric] for w in nouns
                     if w in self.mrc_norms and self.mrc_norms[w][metric] > 0]
            features[f'{metric}_mean_nouns'] = np.mean(values) if values else 0
            
            # Verbs
            values = [self.mrc_norms[w][metric] for w in verbs
                     if w in self.mrc_norms and self.mrc_norms[w][metric] > 0]
            features[f'{metric}_mean_verbs'] = np.mean(values) if values else 0
            
            # Proper nouns
            values = [self.mrc_norms[w][metric] for w in proper_nouns
                     if w in self.mrc_norms and self.mrc_norms[w][metric] > 0]
            features[f'{metric}_mean_proper_nouns'] = np.mean(values) if values else 0
        
        return features
    
    # =========================================================================
    # CAPITALIZATION AS SPECIFICITY PROXY
    # =========================================================================
    
    def capitalization_specificity_features(self, text: str) -> Dict:
        """Capitalization patterns as specificity proxy"""
        features = {}
        
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        if not words:
            return {
                'mid_sentence_capitals': 0,
                'mid_sentence_capital_rate': 0,
                'all_caps_words': 0,
                'title_case_sequences': 0
            }
        
        # Non-sentence-initial words
        non_initial_words = []
        for sent in sentences:
            sent_words = word_tokenize(sent)
            if len(sent_words) > 1:
                non_initial_words.extend(sent_words[1:])
        
        # Mid-sentence capitalized words (proper nouns, etc.)
        capitalized_non_initial = sum(1 for w in non_initial_words
                                       if w and w[0].isupper() and not w.isupper())
        features['mid_sentence_capitals'] = capitalized_non_initial
        features['mid_sentence_capital_rate'] = (capitalized_non_initial / len(non_initial_words)
                                                  if non_initial_words else 0)
        
        # All-caps words (acronyms, emphasis)
        all_caps = sum(1 for w in words if len(w) > 1 and w.isupper())
        features['all_caps_words'] = all_caps
        
        # Title Case Sequences (multiple capitals in a row = specific entities)
        title_sequences = 0
        consecutive_caps = 0
        for w in non_initial_words:
            if w and w[0].isupper():
                consecutive_caps += 1
            else:
                if consecutive_caps >= 2:
                    title_sequences += 1
                consecutive_caps = 0
        features['title_case_sequences'] = title_sequences
        
        return features
    
    # =========================================================================
    # PROPER NOUN PATTERNS (PRE-DE-ID SPECIFIC)
    # =========================================================================
    
    def proper_noun_features(self, text: str) -> Dict:
        """Proper noun patterns that will be destroyed by de-identification"""
        features = {}
        
        blob = TextBlob(text)
        words = word_tokenize(text.lower())
        
        # Get proper nouns
        proper_nouns = [word for word, tag in blob.tags if tag in ['NNP', 'NNPS']]
        proper_nouns_lower = [w.lower() for w in proper_nouns]
        
        features['proper_noun_count'] = len(proper_nouns)
        features['proper_noun_density'] = len(proper_nouns) / len(words) * 100 if words else 0
        features['proper_noun_unique'] = len(set(proper_nouns_lower))
        features['proper_noun_diversity'] = len(set(proper_nouns_lower)) / len(proper_nouns) if proper_nouns else 0
        
        # Repetition patterns
        pn_counter = Counter(proper_nouns_lower)
        features['proper_noun_repeated'] = sum(1 for count in pn_counter.values() if count > 1)
        features['proper_noun_repetition_rate'] = sum(count - 1 for count in pn_counter.values()) / len(proper_nouns) if proper_nouns else 0
        
        # Multi-word proper noun sequences
        consecutive_pn = 0
        max_consecutive = 0
        current_consecutive = 0
        
        for word, tag in blob.tags:
            if tag in ['NNP', 'NNPS']:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                if current_consecutive >= 2:
                    consecutive_pn += 1
                current_consecutive = 0
        
        features['proper_noun_sequences'] = consecutive_pn
        features['proper_noun_max_sequence'] = max_consecutive
        
        return features
    
    # =========================================================================
    # TEMPORAL MARKERS (PRE-DE-ID SPECIFIC)
    # =========================================================================
    
    def temporal_markers_features(self, text: str) -> Dict:
        """Temporal references that will be removed by Safe Harbor"""
        features = {}
        
        text_lower = text.lower()
        
        # Exact dates (MM/DD/YYYY, DD-MM-YYYY, etc.)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        ]
        features['temporal_exact_dates'] = sum(len(re.findall(p, text)) for p in date_patterns)
        
        # Month names
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december',
                 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec']
        features['temporal_month_mentions'] = sum(text_lower.count(month) for month in months)
        
        # Day names
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
               'mon', 'tue', 'tues', 'wed', 'thu', 'thur', 'thurs', 'fri', 'sat', 'sun']
        features['temporal_day_mentions'] = sum(text_lower.count(day) for day in days)
        
        # Year mentions (1900-2099)
        features['temporal_year_mentions'] = len(re.findall(r'\b(19|20)\d{2}\b', text))
        
        # Time expressions (HH:MM, X AM/PM)
        features['temporal_time_expressions'] = len(re.findall(r'\d{1,2}:\d{2}', text))
        features['temporal_ampm_mentions'] = len(re.findall(r'\d{1,2}\s*(am|pm|a\.m\.|p\.m\.)', text_lower))
        
        # Relative temporal markers (also removed as potentially identifying)
        relative_markers = ['yesterday', 'tomorrow', 'today', 'last week', 'next month', 
                          'last year', 'recently', 'lately']
        features['temporal_relative_markers'] = sum(text_lower.count(marker) for marker in relative_markers)
        
        # Specific ages (over 89 per Safe Harbor)
        ages_over_89 = re.findall(r'\b(9\d|[1-9]\d{2,})\s*years?\s*old\b', text_lower)
        features['temporal_age_over_89'] = len([age for age in ages_over_89 if int(age.split()[0]) > 89])
        
        return features
    
    # =========================================================================
    # SPATIAL MARKERS (PRE-DE-ID SPECIFIC)
    # =========================================================================
    
    def spatial_markers_features(self, text: str) -> Dict:
        """Spatial references removed by Safe Harbor (subdivisions smaller than state)"""
        features = {}
        
        text_lower = text.lower()
        
        # Street addresses
        address_patterns = [
            r'\d+\s+\w+\s+(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|court|ct|place|pl)',
            r'\b\d+\s+[A-Z][a-z]+\s+(Street|Avenue|Road|Boulevard|Drive|Lane|Way|Court|Place)\b'
        ]
        features['spatial_street_addresses'] = sum(len(re.findall(p, text, re.IGNORECASE)) for p in address_patterns)
        
        # City, State patterns
        features['spatial_city_state'] = len(re.findall(r'[A-Z][a-z]+,\s*[A-Z]{2}\b', text))
        
        # ZIP codes
        features['spatial_zip_codes'] = len(re.findall(r'\b\d{5}(-\d{4})?\b', text))
        
        # Building/landmark mentions
        landmarks = ['hospital', 'school', 'university', 'building', 'center', 'mall', 
                    'church', 'temple', 'mosque', 'library', 'park']
        features['spatial_landmark_mentions'] = sum(text_lower.count(landmark) for landmark in landmarks)
        
        # Location prepositions (indicators of spatial specificity)
        location_preps = ['at', 'in', 'on', 'near', 'by', 'inside', 'outside', 'beside', 'behind']
        features['spatial_location_prepositions'] = sum(text_lower.count(f' {prep} ') for prep in location_preps)
        
        return features
    
    # =========================================================================
    # NAMED ENTITIES (PRE-DE-ID SPECIFIC)
    # =========================================================================
    
    def named_entity_features(self, text: str) -> Dict:
        """Named entities using spaCy NER"""
        features = {}
        
        if self.nlp is None:
            features['ne_person_count'] = 0
            features['ne_location_count'] = 0
            features['ne_organization_count'] = 0
            features['ne_date_count'] = 0
            features['ne_total_count'] = 0
            features['ne_density'] = 0
            return features
        
        doc = self.nlp(text)
        
        entity_counts = Counter([ent.label_ for ent in doc.ents])
        
        features['ne_person_count'] = entity_counts.get('PERSON', 0)
        features['ne_location_count'] = entity_counts.get('GPE', 0) + entity_counts.get('LOC', 0)
        features['ne_organization_count'] = entity_counts.get('ORG', 0)
        features['ne_date_count'] = entity_counts.get('DATE', 0)
        features['ne_total_count'] = len(doc.ents)
        
        # Entity density
        words = word_tokenize(text)
        features['ne_density'] = len(doc.ents) / len(words) * 100 if words else 0
        
        return features
    
    # =========================================================================
    # FEATURES FROM feature_extraction_origbu.py (ADAPTED)
    # =========================================================================
    
    def lexical_features(self, text: str) -> Dict:
        """Lexical features from feature_extraction_origbu.py"""
        features = {}
        
        words = word_tokenize(text)
        words = [w for w in words if w.strip()]
        
        features['word_count'] = len(words)
        features['unique_word_count'] = len(set([w.lower() for w in words]))
        
        # Filler words
        fillers = ['um', 'uhm', 'ah', 'uh', 'hm', 'huh', 'er', 'mm']
        filler_count = sum(1 for w in words if w.lower() in fillers)
        features['filler_word_count'] = filler_count
        features['filler_word_freq'] = filler_count / len(words) if words else 0
        
        return features
    
    def sentiment_features(self, text: str) -> Dict:
        """Sentiment features from feature_extraction_origbu.py"""
        features = {}
        
        blob = TextBlob(text)
        sentences = blob.sentences
        
        if sentences:
            sentiments = [s.sentiment.polarity for s in sentences]
            features['sentiment_mean'] = np.mean(sentiments)
            features['sentiment_max'] = np.max(sentiments)
            features['sentiment_min'] = np.min(sentiments)
            features['sentiment_std'] = np.std(sentiments)
        else:
            features['sentiment_mean'] = 0
            features['sentiment_max'] = 0
            features['sentiment_min'] = 0
            features['sentiment_std'] = 0
        
        return features
    
    def syntactic_features(self, text: str) -> Dict:
        """Syntactic/POS features from feature_extraction_origbu.py
        WITH proper nouns separated"""
        features = {}
        
        words = word_tokenize(text)
        wc = len([w for w in words if w.strip()])
        
        if wc == 0:
            return {f'pos_{k}': 0 for k in ['common_nouns_freq', 'proper_nouns_freq', 'verbs_freq', 
                   'adjectives_freq', 'adverbs_freq', 'pronouns_freq', 'determiners_freq', 
                   'prepositions_freq', 'content_density']}
        
        blob = TextBlob(text)
        
        pos_counts = {
            'common_nouns': 0,
            'proper_nouns': 0,
            'verbs': 0,
            'adjectives': 0,
            'adverbs': 0,
            'pronouns': 0,
            'determiners': 0,
            'prepositions': 0,
        }
        
        for word, tag in blob.tags:
            if tag in ['NN', 'NNS']:
                pos_counts['common_nouns'] += 1
            elif tag in ['NNP', 'NNPS']:
                pos_counts['proper_nouns'] += 1
            elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                pos_counts['verbs'] += 1
            elif tag in ['JJ', 'JJR', 'JJS']:
                pos_counts['adjectives'] += 1
            elif tag in ['RB', 'RBR', 'RBS']:
                pos_counts['adverbs'] += 1
            elif tag in ['PRP', 'PRP$']:
                pos_counts['pronouns'] += 1
            elif tag == 'DT':
                pos_counts['determiners'] += 1
            elif tag == 'IN':
                pos_counts['prepositions'] += 1
        
        # Frequencies
        for key, count in pos_counts.items():
            features[f'pos_{key}_freq'] = count / wc
        
        # Content density
        content_words = (pos_counts['common_nouns'] + pos_counts['proper_nouns'] + 
                        pos_counts['verbs'] + pos_counts['adjectives'] + pos_counts['adverbs'])
        features['pos_content_density'] = content_words / wc
        
        return features
    
    # =========================================================================
    # MAIN EXTRACTION
    # =========================================================================
    
    def extract_all_features(self, text: str, participant_id: str, 
                           date: str, task_type: str, speaker: str) -> Optional[Dict]:
        """Extract all Tier 2 features"""
        words = word_tokenize(text)
        if len(words) < 10:
            return None
        
        features = {
            'participant_id': participant_id,
            'date': date,
            'task_type': task_type,
            'speaker': speaker,
        }
        
        # Core features
        features.update(self.lexical_features(text))
        features.update(self.sentiment_features(text))
        features.update(self.syntactic_features(text))
        
        # Lexical diversity (robust)
        features.update(self.lexical_diversity_features(text))
        
        # WordNet granularity
        features.update(self.wordnet_granularity_features(text))
        
        # MRC psycholinguistic norms
        features.update(self.psycholinguistic_norms_features(text))
        
        # Pre-de-id specific
        features.update(self.capitalization_specificity_features(text))
        features.update(self.proper_noun_features(text))
        features.update(self.temporal_markers_features(text))
        features.update(self.spatial_markers_features(text))
        features.update(self.named_entity_features(text))
        
        return features
    
    def process_all_transcripts(self):
        """Main processing loop"""
        all_transcripts = self.find_all_transcripts()
        
        if len(all_transcripts) == 0:
            self.logger.error("No transcript files found!")
            return
        
        all_features = []
        start_time = time.time()
        
        for idx, file_path in enumerate(all_transcripts, 1):
            try:
                participant_id, date, task_type = self.extract_participant_info(file_path.name)
                speaker_texts = self.parse_transcript(file_path)
                
                for speaker_key, text in speaker_texts.items():
                    features = self.extract_all_features(text, participant_id, date, task_type, speaker_key)
                    if features:
                        all_features.append(features)
                
                if idx % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = (len(all_transcripts) - idx) * avg_time
                    self.logger.info(f"[{idx}/{len(all_transcripts)}] {len(all_features)} rows, ETA: {remaining/3600:.1f}h")
                
            except Exception as e:
                self.logger.error(f"ERROR on {file_path.name}: {e}")
                continue
        
        if all_features:
            df = pd.DataFrame(all_features)
            output_file = self.output_dir / "tier2_preid_features.csv"
            df.to_csv(output_file, index=False)
            
            total_time = time.time() - start_time
            
            self.logger.info("\n" + "="*80)
            self.logger.info("TIER 2 EXTRACTION COMPLETE!")
            self.logger.info(f"Rows: {len(df)}")
            self.logger.info(f"Columns: {len(df.columns)}")
            self.logger.info(f"Time: {total_time/60:.1f} minutes")
            self.logger.info(f"Output: {output_file}")
            self.logger.info("="*80)
            
            # Show feature categories
            self.logger.info("\nFEATURE CATEGORIES:")
            categories = {
                'Lexical diversity': [c for c in df.columns if 'ttr' in c or 'mattr' in c or 'honore' in c or 'brunet' in c],
                'WordNet granularity': [c for c in df.columns if 'granularity' in c],
                'MRC norms': [c for c in df.columns if any(x in c for x in ['aoa', 'familiarity', 'imageability', 'concreteness', 'meaningfulness'])],
                'Capitalization': [c for c in df.columns if 'capital' in c or 'title_case' in c],
                'Proper nouns': [c for c in df.columns if 'proper_noun' in c],
                'Temporal': [c for c in df.columns if 'temporal' in c],
                'Spatial': [c for c in df.columns if 'spatial' in c],
                'Named entities': [c for c in df.columns if 'ne_' in c],
                'Sentiment': [c for c in df.columns if 'sentiment' in c],
                'POS': [c for c in df.columns if 'pos_' in c],
            }
            
            for cat, cols in categories.items():
                if cols:
                    self.logger.info(f"  {cat}: {len(cols)} features")
        else:
            self.logger.error("No features extracted!")


def main():
    parser = argparse.ArgumentParser(description='Tier 2 Pre-De-ID Feature Extraction')
    parser.add_argument('--transcripts', required=True, help='Base directory with transcripts')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--mrc-norms', help='Path to MRC norms CSV file', default=None)
    args = parser.parse_args()
    
    extractor = Tier2PreDeIDExtractor(args.transcripts, args.output, args.mrc_norms)
    extractor.process_all_transcripts()


if __name__ == "__main__":
    main()
