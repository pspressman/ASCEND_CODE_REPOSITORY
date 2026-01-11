#!/usr/bin/env python3
"""
Comprehensive Linguistic Feature Extraction for Clinamen
Optimized for parallel execution with OpenPose

Directory Structure:
    CSA-research-Xsection-transcripts&OutputOriginal/
      â”œâ”€â”€ Mac_Audacity_10MinuteConvo_withoutCoordinatorSpeech/
      â”œâ”€â”€ Mac_Audacity_Grandfather_Passage/
      â”œâ”€â”€ Mac_Audacity_Picnic_Description/
      â”œâ”€â”€ Mac_Audacity_Spontaneous_Speech/
      â”œâ”€â”€ PC_10MinuteConversations_WithoutCoordinatorSpeech/
      â”œâ”€â”€ PC_Audacity_Grandfather_Passage/
      â”œâ”€â”€ PC_Audacity_Picnic_Description/
      â””â”€â”€ PC_Audacity_Spontaneous_Speech/

File Format: 2-002-8-TaskName-[Mac]-M-D-YY-INITIALS_transcript.txt
Examples:
    - 2-002-8-SpontaneousSpeech-Mac-1-8-24-EM_transcript.txt
    - 2-002-8-GrandfatherPassage-12-19-22-EM_transcript.txt
    - 2-002-8-10MinuteConversationWithoutCoordinatorSpeech-1-19-24-EM_transcript.txt

Diarization Architecture:
    - Each transcript is parsed for speaker labels (Speaker 0, Speaker 1, etc.)
    - Features extracted separately for:
        * 'combined': All speakers together (baseline/non-diarized)
        * 'speaker_0': Usually participant speech only
        * 'speaker_1': Usually investigator speech only
        * Additional speakers if present
    - Output CSV contains multiple rows per transcript (one per speaker view)
    - This enables ASCEND comparison: diarized vs non-diarized performance

Features Extracted:
- All original coherence features (GloVe, USE, BERT, ELMo)
- Lexical, syntactic, sentiment features
- Named entities and specificity metrics
- Sentence/word probability features

Resource Management:
- Batch processing with configurable batch size
- Automatic garbage collection
- Resume capability for crash recovery
- Progress tracking and ETA
"""

import sys
import os
import time
import re
import json
import math
import nltk
import torch
import string
import itertools
import numpy as np
import contractions
import pandas as pd
import networkx as nx
from scipy import stats
import tensorflow as tf
from scipy import spatial
from nltk.util import ngrams
import tensorflow_hub as hub
from textblob import TextBlob
from scipy.stats import linregress
from statistics import mean, stdev
from torch.nn import functional as F
try:
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    KeyedVectors = None
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import gc
import spacy

import warnings
warnings.filterwarnings('ignore')

# Ensure TensorFlow doesn't hog all GPU memory (for OpenPose compatibility)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


class ComprehensiveLinguisticExtractor:
    """
    Comprehensive linguistic feature extraction
    Optimized for parallel execution with OpenPose on Clinamen
    """
    
    def __init__(self, transcript_base_dir: str, output_dir: str, batch_size: int = 3):
        """
        Initialize extractor with all models
        
        Args:
            transcript_base_dir: Base directory containing task folders (GFPOutput, etc.)
            output_dir: Output directory for features
            batch_size: Process this many transcripts before saving (reduced for OpenPose parallel)
        """
        self.transcript_base_dir = Path(transcript_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_files = None  # Set by main() if in test mode
        
        # Task folders
        self.task_folders = {
            'Mac_10MinConvo': self.transcript_base_dir / 'Mac_Audacity_10MinuteConvo_withoutCoordinatorSpeech',
            'Mac_Grandfather': self.transcript_base_dir / 'Mac_Audacity_Grandfather_Passage',
            'Mac_Picnic': self.transcript_base_dir / 'Mac_Audacity_Picnic_Description',
            'Mac_Spontaneous': self.transcript_base_dir / 'Mac_Audacity_Spontaneous_Speech',
            'PC_10MinConvo': self.transcript_base_dir / 'PC_10MinuteConversations_WithoutCoordinatorSpeech',
            'PC_Grandfather': self.transcript_base_dir / 'PC_Audacity_Grandfather_Passage',
            'PC_Picnic': self.transcript_base_dir / 'PC_Audacity_Picnic_Description',
            'PC_Spontaneous': self.transcript_base_dir / 'PC_Audacity_Spontaneous_Speech'
        }
        
        # Setup logging
        self.setup_logging()
        
        # Load processing state
        self.processed_files = self.load_processing_state()
        
        # Load all embedding models
        self.logger.info("Loading embedding models (this will take several minutes)...")
        self.logger.info("Running in parallel with OpenPose - using conservative memory settings")
        self.embeddings = {}
        self.load_all_embeddings()
        
        # Load spaCy for named entities
        self.logger.info("Loading spaCy for named entity recognition...")
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except:
            self.logger.warning("spaCy large model not found. Installing...")
            os.system("python -m spacy download en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")
        
        self.logger.info("All models loaded successfully")
        self.logger.info(f"Processing transcripts from: {self.transcript_base_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
            """Setup comprehensive logging"""
            log_file = self.output_dir / "comprehensive_extraction.log"
            
            # Create handlers with immediate flushing capability
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            # Configure logger
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            self.logger = logger
    def flush_logs(self):
        """Force flush all log handlers to disk immediately"""
        for handler in self.logger.handlers:
            handler.flush()

    def load_processing_state(self) -> set:
        """Load set of already processed files"""
        state_file = self.output_dir / "processing_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                return set(state.get('processed_files', []))
        return set()
    
    def save_processing_state(self, filename: str):
        """Save processing state for crash recovery"""
        self.processed_files.add(filename)
        state_file = self.output_dir / "processing_state.json"
        with open(state_file, 'w') as f:
            json.dump({
                'processed_files': list(self.processed_files),
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
    

    def load_all_embeddings(self):
        """Load all embedding models"""
        # GloVe 2024 Dolma
        self.logger.info("Loading GloVe 2024 Dolma embeddings...")
        glove_path = "embeddings/glove.2024.dolma.300d.txt"
        if not os.path.exists(glove_path):
            self.logger.error(f"GloVe file not found: {glove_path}")
            self.logger.error("Expected location: embeddings/glove.2024.dolma.300d.txt")
            raise FileNotFoundError(f"GloVe embeddings required: {glove_path}")
        
        glove_embeddings = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = [float(x) for x in values[1:]]
                glove_embeddings[word] = vector
        self.embeddings['GloVe'] = glove_embeddings
        self.logger.info(f"GloVe loaded: {len(glove_embeddings)} words")
        
        # Word2Vec (optional - skip if not present)
        # self.logger.info("Loading Word2Vec embeddings...")
        # if not GENSIM_AVAILABLE:
        #     self.logger.warning("gensim not installed - Word2Vec unavailable")
        #     self.embeddings['W2V'] = None
        # else:
        #     w2v_path = "embeddings/GoogleNews-vectors-negative300.bin"
        #     if os.path.exists(w2v_path):
        #         self.embeddings['W2V'] = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        #         self.logger.info("Word2Vec loaded")
        #      else:
        #         self.logger.warning(f"Word2Vec not found: {w2v_path} - skipping")
        #         self.embeddings['W2V'] = None
        
        # DCP embeddings (optional)
        # self.logger.info("Loading DCP embeddings...")
        # dcp_path = "embeddings/EN-wform.w.5.cbow.neg10.400.subsmpl.txt"
        # if os.path.exists(dcp_path):
        #     dcp_embeddings = {}
        #     with open(dcp_path, 'r') as f:
        #         for line in f:
        #             tokens = line.split('\t')
        #             tokens[-1] = tokens[-1].strip()
        #             word = tokens[0]
        #             vector = [float(x) for x in tokens[1:-1]]
        #             dcp_embeddings[word] = vector
        #     self.embeddings['DCP'] = dcp_embeddings
        #     self.logger.info(f"DCP loaded: {len(dcp_embeddings)} words")
        # else:
        #     self.logger.warning(f"DCP not found: {dcp_path} - skipping")
        #     self.embeddings['DCP'] = None
        
        # Universal Sentence Encoder
        self.logger.info("Loading Universal Sentence Encoder...")
        use_module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.embeddings['USE'] = hub.load(use_module_url)
        self.logger.info("USE loaded")
        
        # ELMo
        self.logger.info("Loading ELMo...")
        elmo_model_url = "https://tfhub.dev/google/elmo/2"
        self.embeddings['ELMo'] = hub.load(elmo_model_url).signatures["default"]
        self.logger.info("ELMo loaded")
        
        # BERT
        self.logger.info("Loading BERT...")
        self.embeddings['BERT'] = {
            'model': BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True),
            'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased')
        }
        self.embeddings['BERT']['model'].eval()
        self.logger.info("BERT loaded")
        
        # BERT for MLM (word probabilities)
        # self.logger.info("Loading BERT MLM...")
        # self.bert_mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
        # self.bert_mlm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # BERT for sentence probabilities
        self.bert_lm_model = BertForMaskedLM.from_pretrained('bert-large-cased').eval()
        self.bert_lm_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        
        # GPT-2
        self.logger.info("Loading GPT-2...")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.logger.info("GPT-2 loaded")
    
    # =========================================================================
    # TRANSCRIPT LOADING
    # =========================================================================
    
    def extract_participant_info(self, filename: str, folder_name: str) -> Tuple[str, str, str]:
        """
        Extract participant info from filename
        Handles multiple formats:
        - Full: 2-002-8-TaskName-[Mac]-M-D-YY-INITIALS_transcript.txt
        - Short: 2-013-8-10Minute_transcript.txt (missing date)
        
        Returns:
            (participant_id, date, task_type)
        """
        # Remove _transcript.txt suffix
        name = filename.replace('_transcript.txt', '')
        
        # Split on hyphens
        parts = name.split('-')
        
        # Participant ID is always first 3 parts: "2-002-8"
        if len(parts) >= 3:
            participant_id = f"{parts[0]}-{parts[1]}-{parts[2]}"
        else:
            self.logger.warning(f"Could not parse participant ID from: {filename}")
            return filename.replace('_transcript.txt', ''), 'unknown', 'unknown'
        
        # Determine task type from folder name as fallback
        folder_lower = folder_name.lower()
        if '10minute' in folder_lower or 'conversation' in folder_lower:
            task_type_fallback = '10MinConversation'
        elif 'grandfather' in folder_lower:
            task_type_fallback = 'Grandfather'
        elif 'picnic' in folder_lower:
            task_type_fallback = 'Picnic'
        elif 'spontaneous' in folder_lower:
            task_type_fallback = 'Spontaneous'
        else:
            task_type_fallback = 'Unknown'
        
        # Try to parse date if enough parts exist
        if len(parts) >= 7:
            # Full format with date
            task_name = parts[3]
            
            # Date depends on whether "Mac" is present
            if parts[4] == 'Mac':
                # Mac files: 2-002-8-TaskName-Mac-M-D-YY-INITIALS
                month = parts[5]
                day = parts[6]
                year = parts[7]
            else:
                # PC files: 2-002-8-TaskName-M-D-YY-INITIALS
                month = parts[4]
                day = parts[5]
                year = parts[6]
            
            date_str = f"20{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            # Determine task type from task_name
            task_name_lower = task_name.lower()
            if 'grandfather' in task_name_lower:
                task_type = 'Grandfather'
            elif 'picnic' in task_name_lower:
                task_type = 'Picnic'
            elif 'spontaneous' in task_name_lower:
                task_type = 'Spontaneous'
            elif '10minute' in task_name_lower or 'conversation' in task_name_lower:
                task_type = '10MinConversation'
            else:
                task_type = task_type_fallback
            
            return participant_id, date_str, task_type
        
        elif len(parts) >= 4:
            # Short format without date: 2-013-8-10Minute_transcript.txt
            # Use task name from filename if available
            task_name = parts[3]
            task_name_lower = task_name.lower()
            
            if 'grandfather' in task_name_lower:
                task_type = 'Grandfather'
            elif 'picnic' in task_name_lower:
                task_type = 'Picnic'
            elif 'spontaneous' in task_name_lower:
                task_type = 'Spontaneous'
            elif '10minute' in task_name_lower or 'conversation' in task_name_lower:
                task_type = '10MinConversation'
            else:
                task_type = task_type_fallback
            
            # No date available - use placeholder
            date_str = 'unknown'
            
            self.logger.warning(f"No date in filename: {filename}, using 'unknown'")
            return participant_id, date_str, task_type
        
        else:
            self.logger.warning(f"Could not parse filename: {filename}")
            return participant_id, 'unknown', task_type_fallback

    def find_all_transcripts(self) -> List[Tuple[Path, str, str, str]]:
        """
        Find all transcript files across all task folders
        
        Returns:
            List of (file_path, participant_id, date, task_type)
        """
        all_transcripts = []
        
        for task_name, task_folder in self.task_folders.items():
            if not task_folder.exists():
                self.logger.warning(f"Task folder not found: {task_folder}")
                continue
            
            self.logger.info(f"Scanning {task_name} folder: {task_folder}")
            
            # Find all .txt files
            txt_files = list(task_folder.glob("*.txt"))
            self.logger.info(f"  Found {len(txt_files)} files")
            
            for file_path in txt_files:
                participant_id, date, task_type = self.extract_participant_info(file_path.name, task_name)
                all_transcripts.append((file_path, participant_id, date, task_type))
        
        self.logger.info(f"\nTotal transcripts found: {len(all_transcripts)}")
        return all_transcripts
    
    def load_transcript(self, file_path: Path) -> str:
        """Load transcript text from file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return text.strip()
    
    def parse_diarized_transcript(self, file_path: Path) -> Dict[str, str]:
        """
        Parse diarized transcript and separate by speaker
        
        Format:
            Gap: 6.918 - 7.459
            Speaker 0: You wish to know all about my grandfather.
            55.443 - 56.647
            Speaker 1: Oh my goodness.
        
        Returns:
            {
                'combined': 'all text concatenated',
                'speaker_0': 'speaker 0 text only',
                'speaker_1': 'speaker 1 text only',
                'speaker_2': 'speaker 2 text only' (if present),
                ...
            }
        """
        text = self.load_transcript(file_path)
        lines = text.split('\n')
        
        # Dictionary to store text by speaker
        speaker_texts = {}
        all_text = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip timestamp lines (various formats)
            # Patterns: "Gap: 6.918 - 7.459", "55.443 - 56.647", "[6.918 - 7.459]", etc.
            if re.match(r'^(\[)?(\w+:?\s*)?[\d\.]+ *[-â€“] *[\d\.]+', line):
                continue
            
            # Check if line starts with "Speaker N:"
            speaker_match = re.match(r'Speaker (\d+):\s*(.*)', line, re.IGNORECASE)
            
            if speaker_match:
                speaker_num = speaker_match.group(1)
                speaker_text = speaker_match.group(2).strip()
                
                # Only process if there's actual text (not just label)
                if speaker_text:
                    # Store by speaker
                    speaker_key = f'speaker_{speaker_num}'
                    if speaker_key not in speaker_texts:
                        speaker_texts[speaker_key] = []
                    speaker_texts[speaker_key].append(speaker_text)
                    
                    # Add to combined text
                    all_text.append(speaker_text)
        
        # Concatenate all texts
        result = {
            'combined': ' '.join(all_text)
        }
        
        # Add individual speakers
        for speaker_key, texts in speaker_texts.items():
            result[speaker_key] = ' '.join(texts)
        
        return result
    
    # =========================================================================
    # FEATURE EXTRACTION (Original Features)
    # =========================================================================
    
    def lexical_features(self, text: str) -> Dict:
        """Extract lexical features"""
        words = [x for x in text.split() if x != '']
        wc = len(words)
        types = len(set(words))
        
        features = {
            'participant_wc': wc,
            'participant_types': types,
            'participant_type_token_ratio': types / wc if wc > 0 else 0,
            'participant_brunets_index': types**(-0.165) * math.log(wc) if wc > 0 else 0
        }
        
        # Filled pauses
        ums_ahs = sum([1 for word in words if word.lower() in ['um', 'uhm', 'ah', 'uh', 'hm', 'huh']])
        features['participant_ums_or_ahs_count'] = ums_ahs
        features['participant_ums_or_ahs_freq'] = ums_ahs / wc if wc > 0 else 0
        
        return features
    
    def sentiment_features(self, text: str) -> Dict:
        """Extract sentiment features"""
        blob = TextBlob(text)
        all_sentiments = [sentence.sentiment.polarity for sentence in blob.sentences]
        
        if len(all_sentiments) == 0:
            return {
                'participant_mean_sentiment': 0,
                'participant_max_sentiment': 0,
                'participant_min_sentiment': 0,
                'participant_stdv_sentiment': 0
            }
        
        return {
            'participant_mean_sentiment': mean(all_sentiments),
            'participant_max_sentiment': max(all_sentiments),
            'participant_min_sentiment': min(all_sentiments),
            'participant_stdv_sentiment': stdev(all_sentiments) if len(all_sentiments) > 1 else 0
        }
    
    def syntactic_features(self, text: str, wc: int) -> Dict:
        """Extract syntactic/POS features"""
        blob = TextBlob(text)
        
        # Count POS tags
        pos_counts = {
            'nouns': 0, 'determiners': 0, 'prepositions': 0,
            'base_verbs': 0, 'pasttense_verbs': 0, 'gerund_verbs': 0,
            'pastpart_verbs': 0, 'non3rd_verbs': 0, '3rd_verbs': 0,
            'tos': 0, 'adverbs': 0, 'adjectives': 0, 'modals': 0,
            'coord_conj': 0, 'cardinals': 0, 'particles': 0,
            'personal_pronouns': 0, 'wh_adverbs': 0, 'poss_pronouns': 0,
            'wh_determiners': 0, 'predeterminers': 0, 'interjections': 0,
            'existential_there': 0, 'wh_pronouns': 0
        }
        
        for _, tag in blob.tags:
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                pos_counts['nouns'] += 1
            elif tag == 'DT':
                pos_counts['determiners'] += 1
            elif tag == 'IN':
                pos_counts['prepositions'] += 1
            elif tag == 'VB':
                pos_counts['base_verbs'] += 1
            elif tag == 'VBD':
                pos_counts['pasttense_verbs'] += 1
            elif tag == 'VBG':
                pos_counts['gerund_verbs'] += 1
            elif tag == 'VBN':
                pos_counts['pastpart_verbs'] += 1
            elif tag == 'VBP':
                pos_counts['non3rd_verbs'] += 1
            elif tag == 'VBZ':
                pos_counts['3rd_verbs'] += 1
            elif tag == 'TO':
                pos_counts['tos'] += 1
            elif tag in ['RB', 'RBR', 'RBS']:
                pos_counts['adverbs'] += 1
            elif tag in ['JJ', 'JJR', 'JJS']:
                pos_counts['adjectives'] += 1
            elif tag == 'MD':
                pos_counts['modals'] += 1
            elif tag == 'CC':
                pos_counts['coord_conj'] += 1
            elif tag == 'RP':
                pos_counts['particles'] += 1
            elif tag == 'CD':
                pos_counts['cardinals'] += 1
            elif tag == 'PRP':
                pos_counts['personal_pronouns'] += 1
            elif tag == 'WRB':
                pos_counts['wh_adverbs'] += 1
            elif tag == 'PRP$':
                pos_counts['poss_pronouns'] += 1
            elif tag == 'WDT':
                pos_counts['wh_determiners'] += 1
            elif tag == 'PDT':
                pos_counts['predeterminers'] += 1
            elif tag == 'UH':
                pos_counts['interjections'] += 1
            elif tag == 'EX':
                pos_counts['existential_there'] += 1
            elif tag in ['WP', 'WP$']:
                pos_counts['wh_pronouns'] += 1
        
        # Calculate frequencies
        total_verbs = sum([pos_counts[k] for k in pos_counts if 'verb' in k])
        
        features = {}
        for key, count in pos_counts.items():
            features[f'participant_{key}_freq'] = count / wc if wc > 0 else 0
        
        features['participant_TOTAL_verb_freq'] = total_verbs / wc if wc > 0 else 0
        
        # Content density
        content_words = total_verbs + pos_counts['nouns'] + pos_counts['adjectives'] + pos_counts['adverbs']
        features['participant_content_density'] = content_words / wc if wc > 0 else 0
        
        return features
    
    # =========================================================================
    # COHERENCE FEATURES
    # =========================================================================
    
    def vector_sum(self, vectors):
        """Sum vectors"""
        n = len(vectors)
        if n == 0:
            return None
        d = len(vectors[0])
        s = np.zeros(d)
        for vector in vectors:
            s = s + np.array(vector)
        return list(s)
    
    def embed(self, text, embedding_type, model, tokenizer=None):
        """Embed text with specified embedding"""
        if embedding_type == 'USE_inter':
            return model([text])[0]
        
        elif embedding_type == 'USE_intra':
            embeddings = []
            for word in text.split():
                embeddings.append(model([word])[0])
            return embeddings
        
        elif embedding_type == 'ELMo_inter':
            return model(tf.constant([text]))["default"].numpy()[0]
        
        elif embedding_type == 'ELMo_intra':
            embeddings_tensor = model(tf.constant([text]))
            word_embeddings = embeddings_tensor['word_emb'][0]
            word_embeddings_unpacked = [x.numpy() for x in tf.unstack(word_embeddings)]
            return word_embeddings_unpacked
        
        elif embedding_type == 'BERT_inter':
            tokenized_text = tokenizer.encode(text)
            input_ids = torch.tensor(tokenized_text).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids)
            last_hidden_state = outputs[0]
            this_batch = last_hidden_state[0]
            cls_vector = this_batch[0].detach().numpy()
            return cls_vector
        
        elif embedding_type == 'BERT_intra':
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            with torch.no_grad():
                outputs = model(tokens_tensor)
                hidden_states = outputs[2]
                token_embeddings = torch.stack(hidden_states, dim=0)
                token_embeddings = torch.squeeze(token_embeddings, dim=1)
                token_embeddings = token_embeddings.permute(1, 0, 2)
                token_vecs_sum = []
                for token in token_embeddings[1:-1]:
                    sum_vec = torch.sum(token[-4:], dim=0)
                    token_vecs_sum.append(sum_vec)
            return token_vecs_sum
        
        elif embedding_type in ['DCP_inter', 'W2V_inter', 'GloVe_inter']:
            words = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
            vecs = []
            for word in words:
                if embedding_type == 'GloVe_inter' and word in model:
                    vecs.append(model[word])
                elif embedding_type == 'W2V_inter' and model and word in model.key_to_index:
                    vecs.append(model[word])
                elif embedding_type == 'DCP_inter' and model and word in model:
                    vecs.append(model[word])
            if len(vecs) == 0:
                return None
            return self.vector_sum(vecs)
        
        elif embedding_type in ['DCP_intra', 'W2V_intra', 'GloVe_intra']:
            words = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
            vecs = []
            for word in words:
                if embedding_type == 'GloVe_intra' and word in model:
                    vecs.append(model[word])
                elif embedding_type == 'W2V_intra' and model and word in model.key_to_index:
                    vecs.append(model[word])
                elif embedding_type == 'DCP_intra' and model and word in model:
                    vecs.append(model[word])
            if len(vecs) == 0:
                return None
            return vecs
        
        return None
    
    def get_intra_window_cosines(self, text, embedding_type, embedding_model, tokenizer=None):
        """Get intra-window cosines"""
        all_embeddings = self.embed(text, embedding_type + '_intra', embedding_model, tokenizer)
        
        if not all_embeddings or len(all_embeddings) < 2:
            return None
        
        all_cosines = []
        for a, b in itertools.combinations(all_embeddings, 2):
            cos = 1 - spatial.distance.cosine(a, b)
            all_cosines.append(cos)
        
        return all_cosines
    
    def get_inter_window_cosine(self, text1, text2, embedding_type, embedding_model, tokenizer=None):
        """Get inter-window cosine"""
        e1 = self.embed(text1, embedding_type + '_inter', embedding_model, tokenizer)
        e2 = self.embed(text2, embedding_type + '_inter', embedding_model, tokenizer)
        
        if e1 is None or e2 is None:
            return None
        
        return 1 - spatial.distance.cosine(e1, e2)
    
    def get_ngrams(self, text, n):
        """Get n-grams"""
        n_grams = ngrams(word_tokenize(text), n)
        return [' '.join(grams) for grams in n_grams]
    
    def get_slope(self, nums):
        """Get slope"""
        if len(nums) <= 1:
            return np.nan
        x = range(len(nums))
        y = nums
        slope, _, _, _, _ = linregress(x, y)
        return slope
    
    def coherence_features_single_embedding(self, text: str, embedding_name: str) -> Dict:
        """Extract coherence features for a single embedding"""
        features = {}
        
        embedding_model = self.embeddings.get(embedding_name)
        if embedding_model is None:
            self.logger.warning(f"Embedding {embedding_name} not available, skipping")
            return features
        
        tokenizer = None
        if embedding_name == 'BERT':
            embedding_model_obj = embedding_model['model']
            tokenizer = embedding_model['tokenizer']
        else:
            embedding_model_obj = embedding_model
        
        # Prepare text
        text_lower = text.lower().replace(',', '')
        no_contractions = contractions.fix(text_lower)
        
        # Intra-window coherence
        for window in [3, 5, 'sentence']:
            if window == 'sentence':
                n_grams = sent_tokenize(no_contractions)
            else:
                n_grams = self.get_ngrams(no_contractions.replace('.', '').replace('?', ''), window)
            
            cosines = []
            for i in range(len(n_grams)):
                if embedding_name == 'BERT':
                    res_list = self.get_intra_window_cosines(n_grams[i], embedding_name, embedding_model_obj, tokenizer)
                else:
                    res_list = self.get_intra_window_cosines(n_grams[i], embedding_name, embedding_model_obj)
                
                if res_list:
                    cosines.append(np.array(res_list).mean())
            
            if len(cosines) > 0:
                features[f'mean_intrawindow_coherence_{window}_{embedding_name}'] = np.array(cosines).mean()
                features[f'std_intrawindow_coherence_{window}_{embedding_name}'] = np.array(cosines).std()
                features[f'min_intrawindow_coherence_{window}_{embedding_name}'] = min(cosines)
                features[f'max_intrawindow_coherence_{window}_{embedding_name}'] = max(cosines)
            else:
                features[f'mean_intrawindow_coherence_{window}_{embedding_name}'] = np.nan
                features[f'std_intrawindow_coherence_{window}_{embedding_name}'] = np.nan
                features[f'min_intrawindow_coherence_{window}_{embedding_name}'] = np.nan
                features[f'max_intrawindow_coherence_{window}_{embedding_name}'] = np.nan
        
        # Inter-window coherence
        for window in [3, 5, 'sentence']:
            if window == 'sentence':
                n_grams = sent_tokenize(no_contractions)
                cosines = []
                for i in range(len(n_grams) - 1):
                    if embedding_name == 'BERT':
                        res = self.get_inter_window_cosine(n_grams[i], n_grams[i+1], embedding_name, embedding_model_obj, tokenizer)
                    else:
                        res = self.get_inter_window_cosine(n_grams[i], n_grams[i+1], embedding_name, embedding_model_obj)
                    if res is not None:
                        cosines.append(res)
            else:
                n_grams = self.get_ngrams(no_contractions.replace('.', '').replace('?', ''), window)
                cosines = []
                for i in range(len(n_grams) - window):
                    if embedding_name == 'BERT':
                        res = self.get_inter_window_cosine(n_grams[i], n_grams[i+window], embedding_name, embedding_model_obj, tokenizer)
                    else:
                        res = self.get_inter_window_cosine(n_grams[i], n_grams[i+window], embedding_name, embedding_model_obj)
                    if res is not None:
                        cosines.append(res)
            
            if len(cosines) > 0:
                features[f'mean_coherence_{window}_{embedding_name}_interwindow'] = np.array(cosines).mean()
                features[f'std_coherence_{window}_{embedding_name}_interwindow'] = np.array(cosines).std()
                features[f'min_coherence_{window}_{embedding_name}_interwindow'] = min(cosines)
                features[f'max_coherence_{window}_{embedding_name}_interwindow'] = max(cosines)
            else:
                features[f'mean_coherence_{window}_{embedding_name}_interwindow'] = np.nan
                features[f'std_coherence_{window}_{embedding_name}_interwindow'] = np.nan
                features[f'min_coherence_{window}_{embedding_name}_interwindow'] = np.nan
                features[f'max_coherence_{window}_{embedding_name}_interwindow'] = np.nan
            
            # Tangentiality
            if window == 'sentence':
                slope_cosines = []
                for i in range(len(n_grams) - 1):
                    if embedding_name == 'BERT':
                        res = self.get_inter_window_cosine(n_grams[0], n_grams[i+1], embedding_name, embedding_model_obj, tokenizer)
                    else:
                        res = self.get_inter_window_cosine(n_grams[0], n_grams[i+1], embedding_name, embedding_model_obj)
                    if res is not None:
                        slope_cosines.append(res)
                features[f'tangentiality_{window}_{embedding_name}_interwindow'] = self.get_slope(slope_cosines)
            else:
                slope_cosines = []
                for i in range(len(n_grams) - window):
                    if embedding_name == 'BERT':
                        res = self.get_inter_window_cosine(n_grams[0], n_grams[i+window], embedding_name, embedding_model_obj, tokenizer)
                    else:
                        res = self.get_inter_window_cosine(n_grams[0], n_grams[i+window], embedding_name, embedding_model_obj)
                    if res is not None:
                        slope_cosines.append(res)
                features[f'tangentiality_{window}_{embedding_name}_interwindow'] = self.get_slope(slope_cosines)
        
        return features
    
    # =========================================================================
    # SENTENCE PROBABILITY FEATURES
    # =========================================================================
    
    def get_bert_sentence_score(self, sentence: str) -> float:
        """Get BERT sentence score"""
        try:
            tokenize_input = self.bert_lm_tokenizer.tokenize(sentence)
            tokenize_input = ["[CLS]"] + tokenize_input + ["[SEP]"]
            tensor_input = torch.tensor([self.bert_lm_tokenizer.convert_tokens_to_ids(tokenize_input)])
            with torch.no_grad():
                loss = self.bert_lm_model(tensor_input, labels=tensor_input)[0]
            return np.exp(loss.detach().numpy())
        except:
            return np.nan
    
    def get_gpt2_sentence_score(self, sentence: str) -> float:
        """Get GPT-2 sentence score"""
        try:
            tokenize_input = self.gpt2_tokenizer.encode(sentence)
            tensor_input = torch.tensor([tokenize_input])
            with torch.no_grad():
                loss = self.gpt2_model(tensor_input, labels=tensor_input)[0]
            return np.exp(loss.detach().numpy())
        except:
            return np.nan
    
    def sentence_probability_features(self, text: str) -> Dict:
        """Extract sentence probability features"""
        sentences = sent_tokenize(text.lower())
        sentences = [s for s in sentences if len(s.split()) > 2]
        
        if len(sentences) == 0:
            return self._empty_sentence_prob_features()
        
        bert_scores = []
        gpt2_scores = []
        
        for sent in sentences:
            bert_score = self.get_bert_sentence_score(sent)
            if not np.isnan(bert_score):
                bert_scores.append(bert_score)
            
            gpt2_score = self.get_gpt2_sentence_score(sent)
            if not np.isnan(gpt2_score):
                gpt2_scores.append(gpt2_score)
        
        features = {}
        
        for model_name, scores in [('BERT', bert_scores), ('GPT2', gpt2_scores)]:
            if len(scores) > 0:
                features[f'mean_sentence_probability_{model_name}'] = np.mean(scores)
                features[f'min_sentence_probability_{model_name}'] = np.min(scores)
                features[f'max_sentence_probability_{model_name}'] = np.max(scores)
                features[f'stdv_sentence_probability_{model_name}'] = np.std(scores)
                features[f'firstquartile_sentence_probability_{model_name}'] = np.percentile(scores, 25)
                features[f'median_sentence_probability_{model_name}'] = np.percentile(scores, 50)
                features[f'thirdquartile_sentence_probability_{model_name}'] = np.percentile(scores, 75)
            else:
                for stat in ['mean', 'min', 'max', 'stdv', 'firstquartile', 'median', 'thirdquartile']:
                    features[f'{stat}_sentence_probability_{model_name}'] = np.nan
        
        return features
    
    def _empty_sentence_prob_features(self) -> Dict:
        """Empty sentence prob features"""
        features = {}
        for model in ['BERT', 'GPT2']:
            for stat in ['mean', 'min', 'max', 'stdv', 'firstquartile', 'median', 'thirdquartile']:
                features[f'{stat}_sentence_probability_{model}'] = np.nan
        return features
    
    # =========================================================================
    # NEW FEATURES: Named Entities and Specificity
    # =========================================================================
    
    def named_entity_features(self, text: str) -> Dict:
        """Extract named entity features using spaCy"""
        doc = self.nlp(text)
        
        person_entities = [e for e in doc.ents if e.label_ == 'PERSON']
        location_entities = [e for e in doc.ents if e.label_ in ['GPE', 'LOC']]
        org_entities = [e for e in doc.ents if e.label_ == 'ORG']
        date_entities = [e for e in doc.ents if e.label_ == 'DATE']
        time_entities = [e for e in doc.ents if e.label_ == 'TIME']
        
        wc = len(text.split())
        
        features = {
            'ne_person_count': len(person_entities),
            'ne_location_count': len(location_entities),
            'ne_organization_count': len(org_entities),
            'ne_date_count': len(date_entities),
            'ne_time_count': len(time_entities),
            'ne_total_count': len(doc.ents),
            
            'ne_person_unique': len(set([e.text.lower() for e in person_entities])),
            'ne_location_unique': len(set([e.text.lower() for e in location_entities])),
            
            'ne_person_density': len(person_entities) / wc * 100 if wc > 0 else 0,
            'ne_location_density': len(location_entities) / wc * 100 if wc > 0 else 0,
        }
        
        return features
    
    def specificity_features(self, text: str) -> Dict:
        """Extract specificity features"""
        doc = self.nlp(text)
        
        date_entities = [e for e in doc.ents if e.label_ == 'DATE']
        exact_dates = sum([1 for e in date_entities if re.search(r'\d{1,2}[/-]\d{1,2}', e.text.lower())])
        vague_temporal = sum([1 for word in ['sometime', 'recently', 'once'] if word in text.lower()])
        
        location_entities = [e for e in doc.ents if e.label_ in ['GPE', 'LOC']]
        vague_locations = sum([1 for word in ['somewhere', 'that place'] if word in text.lower()])
        
        features = {
            'specificity_exact_dates': exact_dates,
            'specificity_vague_temporal': vague_temporal,
            'specificity_location_count': len(location_entities),
            'specificity_vague_locations': vague_locations,
        }
        
        return features
    
    # =========================================================================
    # MAIN EXTRACTION
    # =========================================================================
    
    def extract_all_features(self, text: str, participant_id: str, date: str, task_type: str, speaker: str) -> Dict:
        """Extract all features from transcript"""
        
        # Safety check for text length
        word_count = len(text.split())
        if word_count > 2000:
            self.logger.warning(f"  Text too long ({word_count} words), truncating to 2000")
            text = ' '.join(text.split()[:2000])
        elif word_count < 10:
            self.logger.warning(f"  Text too short ({word_count} words), returning minimal features")
            return {
                'participant_id': participant_id,
                'date': date,
                'task_type': task_type,
                'speaker': speaker,
                'participant_wc': word_count
            }
        
        features = {
            'participant_id': participant_id,
            'date': date,
            'task_type': task_type,
            'speaker': speaker
        }
            
        self.logger.info(f"  Lexical features...")
        features.update(self.lexical_features(text))
        
        self.logger.info(f"  Sentiment features...")
        features.update(self.sentiment_features(text))
        
        self.logger.info(f"  Syntactic features...")
        wc = features['participant_wc']
        features.update(self.syntactic_features(text, wc))
        
        self.logger.info(f"  Named entity features...")
        features.update(self.named_entity_features(text))
        
        self.logger.info(f"  Specificity features...")
        features.update(self.specificity_features(text))
        
        # Coherence features (prioritize GloVe, USE, BERT)
        # Coherence features (prioritize GloVe, USE, BERT)
        for emb_name in ['GloVe', 'USE', 'BERT', 'ELMo']:
            self.logger.info(f"  Coherence features ({emb_name})...")
            self.flush_logs()  # Force write to disk NOW
            features.update(self.coherence_features_single_embedding(text, emb_name))
            self.logger.info(f"  Completed coherence features ({emb_name})")
            self.flush_logs()  # Force write to disk NOW
            gc.collect()
        
        self.logger.info(f"  Sentence probability features...")
        features.update(self.sentence_probability_features(text))
        
        return features
    
    def process_all_transcripts(self):
        """Main processing loop"""
        all_transcripts = self.find_all_transcripts()
        
        if len(all_transcripts) == 0:
            self.logger.error("No transcript files found!")
            return
        
        # Apply max_files limit if set
        if self.max_files:
            all_transcripts = all_transcripts[:self.max_files]
            self.logger.info(f"TEST/LIMITED MODE: Processing only {len(all_transcripts)} transcripts")
        
        all_features = []
        batch_count = 0
        start_time = time.time()
        
        for idx, (file_path, participant_id, date, task_type) in enumerate(all_transcripts, 1):
            if str(file_path) in self.processed_files:
                self.logger.info(f"Skipping already processed: {file_path.name}")
                continue
            
            try:
                self.logger.info(f"\nProcessing {idx}/{len(all_transcripts)}: {file_path.name}")
                self.logger.info(f"  Participant: {participant_id}, Date: {date}, Task: {task_type}")
                
                # Parse diarized transcript - get text for each speaker
                speaker_texts = self.parse_diarized_transcript(file_path)
                self.logger.info(f"  Found speakers: {list(speaker_texts.keys())}")
                
                # Extract features for each speaker view
                for speaker_key, text in speaker_texts.items():
                    if not text or len(text.strip()) == 0:
                        self.logger.warning(f"  Skipping {speaker_key}: empty text")
                        continue
                    
                    self.logger.info(f"  Extracting features for {speaker_key} ({len(text.split())} words)...")
                    features = self.extract_all_features(text, participant_id, date, task_type, speaker_key)
                    all_features.append(features)
                    
                    # Force cleanup between speaker views
                    gc.collect()
                    torch.cuda.empty_cache()
                    tf.keras.backend.clear_session()
                    self.logger.info(f"  Completed {speaker_key}, GPU memory cleared")
                
                self.save_processing_state(str(file_path))
                
                # Every 20 transcripts, reload TensorFlow models to prevent memory accumulation
                if idx % 20 == 0 and idx > 0:
                    self.logger.info("Reloading TensorFlow models to clear GPU memory...")
                    
                    # Clear TensorFlow session
                    tf.keras.backend.clear_session()
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Reload USE
                    self.logger.info("  Reloading USE...")
                    self.embeddings['USE'] = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                    
                    # Reload ELMo
                    self.logger.info("  Reloading ELMo...")
                    self.embeddings['ELMo'] = hub.load("https://tfhub.dev/google/elmo/2").signatures["default"]
                    
                    self.logger.info("  TensorFlow models reloaded")
                
                if len(all_features) >= self.batch_size:
                    self.save_batch(all_features, batch_count)
                    all_features = []
                    batch_count += 1
                    gc.collect()
                    torch.cuda.empty_cache()
                    tf.keras.backend.clear_session()
                
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (len(all_transcripts) - idx) * avg_time
                self.logger.info(f"Progress: {idx}/{len(all_transcripts)} ({idx/len(all_transcripts)*100:.1f}%) - ETA: {remaining/3600:.1f}h")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        if all_features:
            self.save_batch(all_features, batch_count)
        
        self.consolidate_batches()
        
        total_time = time.time() - start_time
        self.logger.info(f"\n{'='*80}")
        if self.max_files:
            self.logger.info(f"TEST/LIMITED RUN COMPLETE!")
            self.logger.info(f"Processed {len(all_transcripts)} transcripts (limited mode)")
        else:
            self.logger.info(f"FULL EXTRACTION COMPLETE!")
            self.logger.info(f"Processed all {len(all_transcripts)} transcripts")
        self.logger.info(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
        self.logger.info(f"Output: {self.output_dir / 'comprehensive_linguistic_features.csv'}")
        self.logger.info(f"{'='*80}")
        
        # For test runs, show comprehensive diagnostics
        if self.max_files and self.max_files <= 10:
            try:
                output_file = self.output_dir / "comprehensive_linguistic_features.csv"
                if output_file.exists():
                    df = pd.read_csv(output_file)
                    self.logger.info("\n" + "="*80)
                    self.logger.info("ðŸ“Š COMPREHENSIVE TEST DIAGNOSTICS")
                    self.logger.info("="*80)
                    
                    # Basic info
                    self.logger.info(f"\nâœ“ Rows processed: {len(df)}")
                    self.logger.info(f"âœ“ Total columns: {len(df.columns)}")
                    self.logger.info(f"âœ“ Participants: {df['participant_id'].unique().tolist()}")
                    self.logger.info(f"âœ“ Task types: {df['task_type'].unique().tolist()}")
                    self.logger.info(f"âœ“ Speakers: {df['speaker'].unique().tolist()}")
                    
                    # Show speaker breakdown
                    self.logger.info(f"\nðŸ“Š DIARIZATION BREAKDOWN:")
                    for speaker in df['speaker'].unique():
                        count = len(df[df['speaker'] == speaker])
                        self.logger.info(f"   {speaker}: {count} rows")
                    
                    # Feature category breakdown
                    self.logger.info("\nðŸ“‹ FEATURE CATEGORIES EXTRACTED:")
                    
                    feature_categories = {
                        'Metadata': ['participant_id', 'date', 'task_type', 'speaker'],
                        'Lexical': [c for c in df.columns if 'participant_wc' in c or 'type' in c or 'brunet' in c or 'ums' in c],
                        'Sentiment': [c for c in df.columns if 'sentiment' in c],
                        'Syntactic/POS': [c for c in df.columns if any(x in c for x in ['_freq', 'verb', 'noun', 'pronoun', 'adjective', 'adverb', 'content_density'])],
                        'Named Entities': [c for c in df.columns if c.startswith('ne_')],
                        'Specificity': [c for c in df.columns if c.startswith('specificity_')],
                        'GloVe Coherence': [c for c in df.columns if 'GloVe' in c],
                        'USE Coherence': [c for c in df.columns if 'USE' in c],
                        'BERT Coherence': [c for c in df.columns if 'BERT' in c and 'sentence_probability' not in c],
                        'ELMo Coherence': [c for c in df.columns if 'ELMo' in c],
                        'W2V Coherence': [c for c in df.columns if 'W2V' in c],
                        'DCP Coherence': [c for c in df.columns if 'DCP' in c],
                        'Sentence Probability': [c for c in df.columns if 'sentence_probability' in c]
                    }
                    
                    for category, cols in feature_categories.items():
                        if cols:
                            nan_count = df[cols].isna().sum().sum()
                            total_values = len(df) * len(cols)
                            coverage = 100 * (total_values - nan_count) / total_values if total_values > 0 else 0
                            self.logger.info(f"   {category}: {len(cols)} features, {coverage:.1f}% coverage")
                        else:
                            self.logger.info(f"   {category}: 0 features âš ï¸")
                    
                    # Check for completely empty feature categories
                    self.logger.info("\nðŸ” FEATURE EXTRACTION STATUS:")
                    issues_found = False
                    
                    for category, cols in feature_categories.items():
                        if not cols and category not in ['Metadata', 'W2V Coherence', 'DCP Coherence']:
                            self.logger.warning(f"   âš ï¸ {category}: NO FEATURES EXTRACTED - Check dependencies!")
                            issues_found = True
                        elif cols:
                            # Check if all values are NaN for this category
                            if df[cols].isna().all().all():
                                self.logger.warning(f"   âš ï¸ {category}: All values are NaN - Extraction may have failed!")
                                issues_found = True
                            else:
                                self.logger.info(f"   âœ“ {category}: Extracted successfully")
                    
                    # Summary
                    if not issues_found:
                        self.logger.info("\nâœ… ALL FEATURE CATEGORIES EXTRACTED SUCCESSFULLY!")
                        self.logger.info("   Ready for full run on all transcripts.")
                    else:
                        self.logger.warning("\nâš ï¸ SOME FEATURES FAILED TO EXTRACT")
                        self.logger.warning("   Review warnings above before full run.")
                    
                    # Show actual feature names by category
                    self.logger.info("\nðŸ“ DETAILED FEATURE LIST BY CATEGORY:")
                    for category, cols in feature_categories.items():
                        if cols and len(cols) <= 20:  # Show all if â‰¤20
                            self.logger.info(f"\n   {category} ({len(cols)} features):")
                            for col in cols:
                                self.logger.info(f"     - {col}")
                        elif cols:  # Show first 10 and last 5 if >20
                            self.logger.info(f"\n   {category} ({len(cols)} features):")
                            self.logger.info("     First 10:")
                            for col in cols[:10]:
                                self.logger.info(f"       - {col}")
                            self.logger.info(f"     ... ({len(cols)-15} more) ...")
                            self.logger.info("     Last 5:")
                            for col in cols[-5:]:
                                self.logger.info(f"       - {col}")
                    
                    self.logger.info("\n" + "="*80)
                    
            except Exception as e:
                self.logger.error(f"Could not display test diagnostics: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
    
    def save_batch(self, features: List[Dict], batch_num: int):
        """Save batch of features"""
        df = pd.DataFrame(features)
        batch_file = self.output_dir / f"features_batch_{batch_num}.csv"
        df.to_csv(batch_file, index=False)
        self.logger.info(f"Saved batch {batch_num}: {len(features)} transcripts")
    
    def consolidate_batches(self):
        """Consolidate all batch files"""
        batch_files = list(self.output_dir.glob("features_batch_*.csv"))
        
        if not batch_files:
            self.logger.warning("No batch files to consolidate")
            return
        
        dfs = [pd.read_csv(f) for f in sorted(batch_files)]
        combined_df = pd.concat(dfs, ignore_index=True)
        
        output_file = self.output_dir / "comprehensive_linguistic_features.csv"
        combined_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Consolidated {len(batch_files)} batches")
        self.logger.info(f"Total rows (including all speaker views): {len(combined_df)}")
        self.logger.info(f"Total features: {len(combined_df.columns) - 4}")  # -4 for metadata columns
        
        for batch_file in batch_files:
            batch_file.unlink()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Linguistic Feature Extraction")
    parser.add_argument('--transcripts', required=True, 
                       help='Base directory containing task folders (e.g., Batch1-SplicedAudioTranscripts)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=3, 
                       help='Batch size (reduced for OpenPose parallel)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only first 3 transcripts')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Process at most this many files (for testing)')
    
    args = parser.parse_args()
    
    extractor = ComprehensiveLinguisticExtractor(
        transcript_base_dir=args.transcripts,
        output_dir=args.output,
        batch_size=args.batch_size
    )
    
    # Set test mode if requested
    if args.test:
        extractor.logger.info("=" * 80)
        extractor.logger.info("TEST MODE: Processing only first 3 transcripts")
        extractor.logger.info("=" * 80)
        extractor.max_files = 3
    elif args.max_files:
        extractor.logger.info("=" * 80)
        extractor.logger.info(f"LIMITED MODE: Processing only first {args.max_files} transcripts")
        extractor.logger.info("=" * 80)
        extractor.max_files = args.max_files
    else:
        extractor.max_files = None
    
    extractor.process_all_transcripts()


if __name__ == "__main__":
    main()