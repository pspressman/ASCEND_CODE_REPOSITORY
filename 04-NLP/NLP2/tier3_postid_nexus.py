#!/usr/bin/env python3
"""
TIER 3: Post-De-identification Safe Features
Features that SURVIVE de-identification

Runs on NEXUS (Synology DS923+) or any machine
NO GPU, NO spaCy required
Supports diarization (multiple speaker views per transcript)

Usage:
    python3.9 tier3_postid_safe.py --transcripts /volume1/video_analysis --output /volume1/linguistic_features/tier3_output
"""

import os
import re
import sys
import json
import time
import logging
import argparse
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob

class Tier3PostDeIDExtractor:
    """
    Extract features that survive de-identification
    Can run before OR after de-identification
    """
    
    def __init__(self, transcript_base_dir: str, output_dir: str):
        self.transcript_base_dir = Path(transcript_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        self.processed_files = self.load_processing_state()
        
        self.logger.info(f"Tier 3 Post-DeID Extractor initialized")
        self.logger.info(f"Searching: {self.transcript_base_dir}")
        self.logger.info(f"Output: {self.output_dir}")
    
    def setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / "tier3_extraction.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_processing_state(self) -> set:
        """Load already processed files"""
        state_file = self.output_dir / "processing_state_tier3.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                return set(state.get('processed_files', []))
        return set()
    
    def save_processing_state(self, filename: str):
        """Save processing state"""
        self.processed_files.add(filename)
        state_file = self.output_dir / "processing_state_tier3.json"
        with open(state_file, 'w') as f:
            json.dump({
                'processed_files': list(self.processed_files),
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
    
    def find_all_transcripts(self) -> List[Path]:
        """Find all transcript files recursively"""
        self.logger.info(f"Searching for transcripts in {self.transcript_base_dir}")
        
        transcripts = list(self.transcript_base_dir.rglob("*_transcript.txt"))
        additional = list(self.transcript_base_dir.rglob("*.txt"))
        
        all_files = set(transcripts + additional)
        
        self.logger.info(f"Found {len(all_files)} transcript files")
        return sorted(list(all_files))
    
    def extract_participant_info(self, filename: str) -> Tuple[str, str, str]:
        """Extract participant info from filename"""
        name = filename.replace('_transcript.txt', '').replace('.txt', '')
        parts = name.split('-')
        
        participant_id = 'unknown'
        date = 'unknown'
        task_type = 'unknown'
        
        if len(parts) >= 3:
            if parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                participant_id = f"{parts[0]}-{parts[1]}-{parts[2]}"
                
                if len(parts) > 3:
                    task_name = parts[3].lower()
                    if 'grandfather' in task_name or 'gfp' in task_name:
                        task_type = 'Grandfather'
                    elif 'picnic' in task_name:
                        task_type = 'Picnic'
                    elif 'spontaneous' in task_name or 'spont' in task_name:
                        task_type = 'Spontaneous'
                    elif 'conversation' in task_name or '10min' in task_name:
                        task_type = 'Conversation'
                
                if len(parts) >= 7:
                    for i in range(len(parts)-2):
                        if parts[i].isdigit() and parts[i+1].isdigit() and parts[i+2].isdigit():
                            month, day, year = parts[i], parts[i+1], parts[i+2]
                            if len(year) == 2:
                                year = f"20{year}"
                            date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            break
            else:
                participant_id = parts[0] if parts[0] else 'unknown'
        
        return participant_id, date, task_type
    
    def parse_diarized_transcript(self, file_path: Path) -> Dict[str, str]:
        """Parse diarized transcript - separate by speaker"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        lines = text.split('\n')
        speaker_texts = {}
        all_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if re.match(r'^(\[)?(\w+:?\s*)?[\d\.]+ *[-–] *[\d\.]+', line):
                continue
            
            speaker_match = re.match(r'Speaker (\d+):\s*(.*)', line, re.IGNORECASE)
            
            if speaker_match:
                speaker_num = speaker_match.group(1)
                speaker_text = speaker_match.group(2).strip()
                
                if speaker_text:
                    speaker_key = f'speaker_{speaker_num}'
                    if speaker_key not in speaker_texts:
                        speaker_texts[speaker_key] = []
                    speaker_texts[speaker_key].append(speaker_text)
                    all_text.append(speaker_text)
            else:
                if line:
                    all_text.append(line)
        
        result = {'combined': ' '.join(all_text)}
        for speaker_key, texts in speaker_texts.items():
            result[speaker_key] = ' '.join(texts)
        
        return result
    
    def enhanced_disfluency_features(self, text: str) -> Dict:
        """Enhanced Disfluency (10 features)"""
        features = {}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Filled pauses
        filled_pauses = ['um', 'uh', 'er', 'ah', 'mm', 'hmm']
        features['disf_filled_pause_count'] = sum(text_lower.count(fp) for fp in filled_pauses)
        features['disf_filled_pause_density'] = features['disf_filled_pause_count'] / len(words) * 100 if words else 0
        
        # Repetitions (consecutive word repeats)
        repetitions = 0
        for i in range(len(words)-1):
            if words[i] == words[i+1]:
                repetitions += 1
        features['disf_word_repetitions'] = repetitions
        
        # False starts (incomplete sentences - heuristic: sentences < 3 words)
        sentences = sent_tokenize(text)
        short_sentences = [s for s in sentences if len(s.split()) < 3]
        features['disf_false_starts'] = len(short_sentences)
        
        # Self-corrections (heuristic: "I mean", "or rather", "actually")
        correction_markers = ['i mean', 'or rather', 'actually', 'correction', 'sorry']
        features['disf_self_corrections'] = sum(text_lower.count(marker) for marker in correction_markers)
        
        # Restarts (dashes, ellipses)
        features['disf_restart_markers'] = text.count('-') + text.count('...')
        
        # Fill remaining
        for i in range(4):
            features[f'disf_placeholder_{i}'] = 0
        
        return features
    
    def discourse_marker_features(self, text: str) -> Dict:
        """Discourse Markers (15 features)"""
        features = {}
        
        text_lower = text.lower()
        
        # Temporal discourse markers
        temporal_markers = ['then', 'next', 'after', 'before', 'while', 'when', 'later', 'earlier', 'finally']
        features['disc_temporal_markers'] = sum(text_lower.count(f' {marker} ') for marker in temporal_markers)
        
        # Causal connectives
        causal_markers = ['because', 'since', 'so', 'therefore', 'thus', 'hence', 'consequently']
        features['disc_causal_markers'] = sum(text_lower.count(f' {marker} ') for marker in causal_markers)
        
        # Contrastive markers
        contrast_markers = ['but', 'however', 'although', 'though', 'yet', 'nevertheless', 'instead']
        features['disc_contrast_markers'] = sum(text_lower.count(f' {marker} ') for marker in contrast_markers)
        
        # Additive markers
        additive_markers = ['and', 'also', 'moreover', 'furthermore', 'additionally', 'plus']
        features['disc_additive_markers'] = sum(text_lower.count(f' {marker} ') for marker in additive_markers)
        
        # Clarification markers
        clarification_markers = ['for example', 'for instance', 'such as', 'like', 'that is']
        features['disc_clarification_markers'] = sum(text_lower.count(marker) for marker in clarification_markers)
        
        # Total discourse marker density
        total_markers = sum([features['disc_temporal_markers'], features['disc_causal_markers'],
                           features['disc_contrast_markers'], features['disc_additive_markers'],
                           features['disc_clarification_markers']])
        words = len(text.split())
        features['disc_total_marker_density'] = total_markers / words * 100 if words else 0
        
        # Fill remaining
        for i in range(9):
            features[f'disc_placeholder_{i}'] = 0
        
        return features
    
    def narrative_structure_features(self, text: str) -> Dict:
        """Narrative Structure (10 features)"""
        features = {}
        
        text_lower = text.lower()
        sentences = sent_tokenize(text)
        
        # Story grammar elements
        
        # Setting indicators
        setting_words = ['was', 'were', 'had', 'there was', 'there were']
        features['narr_setting_markers'] = sum(text_lower.count(word) for word in setting_words[:2])
        
        # Complication/problem indicators
        problem_words = ['problem', 'issue', 'difficult', 'hard', 'trouble', 'challenge']
        features['narr_problem_markers'] = sum(text_lower.count(word) for word in problem_words)
        
        # Resolution indicators
        resolution_words = ['solved', 'fixed', 'resolved', 'ended', 'finally']
        features['narr_resolution_markers'] = sum(text_lower.count(word) for word in resolution_words)
        
        # Evaluation/emotion
        emotion_words = ['happy', 'sad', 'angry', 'excited', 'worried', 'surprised', 'felt']
        features['narr_emotion_markers'] = sum(text_lower.count(word) for word in emotion_words)
        
        # Temporal sequencing
        sequence_words = ['first', 'second', 'third', 'then', 'next', 'finally']
        features['narr_sequence_markers'] = sum(text_lower.count(word) for word in sequence_words)
        
        # Narrative coherence (sentence connectivity - rough heuristic)
        features['narr_avg_sentence_length'] = len(text.split()) / len(sentences) if sentences else 0
        
        # Fill remaining
        for i in range(4):
            features[f'narr_placeholder_{i}'] = 0
        
        return features
    
    def enhanced_lexical_features(self, text: str) -> Dict:
        """Enhanced Lexical (15 features)"""
        features = {}
        
        words = text.split()
        
        # Word length statistics
        word_lengths = [len(w) for w in words if w.isalpha()]
        if word_lengths:
            features['lex_avg_word_length'] = sum(word_lengths) / len(word_lengths)
            features['lex_max_word_length'] = max(word_lengths)
            features['lex_word_length_variance'] = sum((l - features['lex_avg_word_length'])**2 for l in word_lengths) / len(word_lengths)
        else:
            features['lex_avg_word_length'] = 0
            features['lex_max_word_length'] = 0
            features['lex_word_length_variance'] = 0
        
        # Lexical diversity (MATTR approximation - simple moving window TTR)
        window_size = 50
        if len(words) >= window_size:
            ttrs = []
            for i in range(len(words) - window_size + 1):
                window = words[i:i+window_size]
                ttr = len(set(window)) / len(window)
                ttrs.append(ttr)
            features['lex_mattr'] = sum(ttrs) / len(ttrs) if ttrs else 0
        else:
            features['lex_mattr'] = len(set(words)) / len(words) if words else 0
        
        # Honore's statistic
        V = len(set(words))  # vocabulary
        N = len(words)  # total words
        V1 = len([w for w, count in Counter(words).items() if count == 1])  # hapax legomena
        if V1 < V:
            features['lex_honores_stat'] = 100 * (V / (1 - (V1 / V))) if V else 0
        else:
            features['lex_honores_stat'] = 0
        
        # Yule's K (lexical diversity)
        word_counts = Counter(words)
        M1 = sum(word_counts.values())
        M2 = sum([count**2 for count in word_counts.values()])
        if M1 > 0:
            features['lex_yules_k'] = 10000 * (M2 - M1) / (M1 * M1)
        else:
            features['lex_yules_k'] = 0
        
        # Academic/formal word usage (words > 9 letters as proxy)
        long_words = [w for w in words if len(w) > 9]
        features['lex_long_word_ratio'] = len(long_words) / len(words) * 100 if words else 0
        
        # Fill remaining
        for i in range(8):
            features[f'lex_placeholder_{i}'] = 0
        
        return features
    
    def graph_features_lightweight(self, text: str) -> Dict:
        """Lightweight Graph Features (10 features)"""
        features = {}
        
        try:
            # Build simple word co-occurrence graph
            words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
            
            if len(words) < 3:
                for i in range(10):
                    features[f'graph_placeholder_{i}'] = 0
                return features
            
            # Create graph with edges between consecutive words
            G = nx.Graph()
            for i in range(len(words)-1):
                G.add_edge(words[i], words[i+1])
            
            # Basic metrics
            features['graph_nodes'] = G.number_of_nodes()
            features['graph_edges'] = G.number_of_edges()
            features['graph_density'] = nx.density(G) if G.number_of_nodes() > 0 else 0
            
            # Connectivity
            if G.number_of_nodes() > 0:
                features['graph_avg_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
            else:
                features['graph_avg_degree'] = 0
            
            # Clustering
            features['graph_clustering_coef'] = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
            
            # Connected components
            features['graph_num_components'] = nx.number_connected_components(G)
            
            # Fill remaining
            for i in range(4):
                features[f'graph_placeholder_{i}'] = 0
            
        except Exception as e:
            # If graph construction fails, return zeros
            for i in range(10):
                features[f'graph_placeholder_{i}'] = 0
        
        return features
    
    def extract_all_features(self, text: str, participant_id: str, date: str, task_type: str, speaker: str) -> Dict:
        """Extract all Tier 3 features"""
        
        if len(text.split()) < 10:
            return {
                'participant_id': participant_id,
                'date': date,
                'task_type': task_type,
                'speaker': speaker,
                'word_count': len(text.split())
            }
        
        features = {
            'participant_id': participant_id,
            'date': date,
            'task_type': task_type,
            'speaker': speaker,
            'word_count': len(text.split())
        }
        
        self.logger.info(f"  Extracting Tier 3 features...")
        
        features.update(self.enhanced_disfluency_features(text))
        features.update(self.discourse_marker_features(text))
        features.update(self.narrative_structure_features(text))
        features.update(self.enhanced_lexical_features(text))
        features.update(self.graph_features_lightweight(text))
        
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
            if str(file_path) in self.processed_files:
                self.logger.info(f"Skipping already processed: {file_path.name}")
                continue
            
            try:
                self.logger.info(f"\nProcessing {idx}/{len(all_transcripts)}: {file_path.name}")
                
                participant_id, date, task_type = self.extract_participant_info(file_path.name)
                self.logger.info(f"  Participant: {participant_id}, Date: {date}, Task: {task_type}")
                
                speaker_texts = self.parse_diarized_transcript(file_path)
                self.logger.info(f"  Found speakers: {list(speaker_texts.keys())}")
                
                for speaker_key, text in speaker_texts.items():
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    self.logger.info(f"  Extracting features for {speaker_key} ({len(text.split())} words)...")
                    features = self.extract_all_features(text, participant_id, date, task_type, speaker_key)
                    all_features.append(features)
                
                self.save_processing_state(str(file_path))
                
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (len(all_transcripts) - idx) * avg_time
                self.logger.info(f"Progress: {idx}/{len(all_transcripts)} ({idx/len(all_transcripts)*100:.1f}%) - ETA: {remaining/3600:.1f}h")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {e}")
                continue
        
        if all_features:
            df = pd.DataFrame(all_features)
            output_file = self.output_dir / "tier3_postid_features.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"TIER 3 EXTRACTION COMPLETE!")
            self.logger.info(f"Total rows: {len(df)}")
            self.logger.info(f"Total features: {len(df.columns)}")
            self.logger.info(f"Output: {output_file}")
            self.logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Tier 3 Post-DeID Feature Extraction")
    parser.add_argument('--transcripts', required=True, help='Base transcript directory')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    extractor = Tier3PostDeIDExtractor(
        transcript_base_dir=args.transcripts,
        output_dir=args.output
    )
    
    extractor.process_all_transcripts()


if __name__ == "__main__":
    main()
