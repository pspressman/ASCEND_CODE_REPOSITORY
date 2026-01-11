#!/usr/bin/env python3
"""
NEXUS Batch Information-Theoretic Analyzer
Processes 14,000+ CliniDeID labeled transcript files for LLM-based anonymization review

Adapted for:
- NEXUS storage paths
- CliniDeID labeled file patterns (*.deid.piiCategoryTag.txt, notes without _transcript)
- Qwen 2.5 14B model (better reasoning)
- Large-scale batch processing with checkpointing
- Consolidated CSV output for human review

Usage:
    python IT_nexus_batch_analyzer.py --input /path/to/volumes/video_research/ForNEXUS/CliniDeID_LLM_Ready/ \
                                       --output /path/to/volumes/video_research/ForNEXUS/LLM_Analysis_Output/ \
                                       --batch-size 50
"""

import requests
import json
import os
from pathlib import Path
from datetime import datetime
import argparse
import time
import csv

def sanitize_filename(filename: str) -> str:
    """
    Remove or replace invalid filename characters for Windows compatibility.
    Handles: < > : " | ? * , &
    """
    # Replace problematic characters with underscores
    invalid_chars = '<>:"|?*,&'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Remove any duplicate underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    # Remove trailing underscores
    filename = filename.strip('_')
    return filename

class NEXUSBatchAnalyzer:
    
    # Model configuration - using qwen2.5-14b for better reasoning
    DEFAULT_MODEL = 'qwen2.5:14b'
    
    def __init__(self, ollama_url="http://localhost:11434", model_name=None):
        self.ollama_url = ollama_url
        self.model_name = model_name or self.DEFAULT_MODEL
        self.verify_connection()
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'errors': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'total_redactions': 0
        }
    
    def verify_connection(self):
        """Check Ollama connection and model availability"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = [m['name'] for m in response.json().get('models', [])]
            
            if self.model_name not in models:
                print(f"[ERROR] Model {self.model_name} not found")
                print(f"\nTo install, run:")
                print(f"  ollama pull {self.model_name}")
                print(f"\nAvailable models on your system:")
                for m in models[:10]:
                    print(f"  - {m}")
                exit(1)
            
            print(f"[OK] Connected to Ollama")
            print(f"[INFO] Using: {self.model_name}")
            print()
        except Exception as e:
            print(f"[ERROR] Ollama not running or not accessible")
            print(f"   Error: {e}")
            print(f"\nStart Ollama with: ollama serve")
            exit(1)
    
    def create_analysis_prompt(self):
        """Create the information-theoretic analysis prompt"""
        return """You are analyzing conversation transcripts that have already been processed by CliniDeID (names, dates, places removed). Your task is to identify MINIMAL additional redactions needed based on POPULATION THRESHOLDS.

CORE PRINCIPLE: Flag details only when they describe fewer than ~20,000 people in the US, especially when details COMBINE to narrow the population.

ANALYSIS PROCESS:

1. IDENTIFY each factual detail and estimate population:
   - VERY_LOW specificity: >10 million people (e.g., "teacher", "has diabetes")
   - LOW specificity: 1-10 million (e.g., "nurse", "worked in hospital")
   - MEDIUM specificity: 100k-1 million (e.g., "drama teacher", "20 year tenure")
   - HIGH specificity: 10k-100k (e.g., "ran quality improvement program")
   - VERY_HIGH specificity: <10k (e.g., "won national award", "astronaut")

2. IDENTIFY COMPOUNDING EFFECTS:
   - Temporal details that narrow cohorts (specific durations, sequences)
   - Sequential trajectories (unique career paths)
   - Relational details (connections to specific others)
   - Verifiable events (things that "got in the paper", awards, public incidents)
   - Geographic + occupational + temporal combinations

3. CALCULATE COMBINED POPULATION:
   - Multiply probabilities: teacher (3M) Ã— 35 years (0.2%) Ã— small town (0.1%) = 600 people âš ï¸
   - Flag when combination drops below ~20,000 people

4. RECOMMEND MINIMAL REMOVAL:
   - Start with VERY_HIGH specificity items
   - Include specific numbers/durations that narrow populations
   - Include publicly verifiable events
   - Include unique sequential trajectories
   - PRESERVE common details (diagnoses, medications, generic roles)

DO NOT FLAG:
- Common diagnoses (Alzheimer's, diabetes, depression) - millions have these
- Common medications - hundreds of thousands take them
- Generic occupations without modifiers (teacher, nurse, accountant)
- State-level geography (already removed by CliniDeID)
- Common ages â‰¤89
- Generic family status (married, has children)
- Common activities (golf, church, reading)
- CliniDeID labels like [*** NAME ***], [*** DATE ***] - already redacted

DO FLAG:
- Specific tenure lengths that narrow populations (e.g., "35 years", "worked there since it opened")
- Publicly verifiable events ("got in the paper", "won award", "was on TV")
- Rare occupational combinations ("astronaut who became a surgeon")
- Unique trajectories ("started in ER, got MBA, ran quality improvement")
- Small population indicators ("everyone in town knew", "only one in the county")
- Rare achievements/recognition
- Connections to named/specific third parties (if CliniDeID missed them)

RESPONSE FORMAT (JSON):
{
  "conversational_segments": [
    {
      "segment_number": 1,
      "original_text": "exact text from transcript",
      "line_number": 42,
      "details_identified": [
        {
          "detail": "taught drama",
          "population_estimate": 200000,
          "specificity": "MEDIUM-LOW",
          "reasoning": "Drama teachers are common in schools"
        },
        {
          "detail": "35 years",
          "population_estimate": 40000,
          "specificity": "HIGH",
          "reasoning": "Very long tenure, narrows to <1% of teachers"
        }
      ],
      "combined_population_estimate": 8000,
      "risk_level": "HIGH",
      "recommended_removals": [
        {
          "text_to_remove": "35 years",
          "reason": "Specific duration narrows population significantly",
          "replacement_suggestion": "remove entirely or replace with 'many years'"
        }
      ],
      "proposed_redacted_version": "Mom taught drama and directed school musicals.",
      "population_after_redaction": 200000
    }
  ],
  "overall_summary": {
    "total_segments_analyzed": 0,
    "high_risk_segments": 0,
    "medium_risk_segments": 0,
    "low_risk_segments": 0,
    "total_recommended_redactions": 0,
    "overall_recommendation": "REDACTIONS_NEEDED|MINIMAL_CHANGES|ACCEPTABLE_AS_IS"
  }
}

TRANSCRIPT TO ANALYZE:
"""
    
    def chunk_text(self, text, chunk_size=3000):
        """Split text into manageable chunks"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        current_line_num = 0
        chunk_metadata = []
        
        for i, line in enumerate(lines):
            line_size = len(line)
            if current_size + line_size > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                chunk_metadata.append({
                    'start_line': current_line_num,
                    'end_line': i - 1
                })
                current_chunk = [line]
                current_size = line_size
                current_line_num = i
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            chunk_metadata.append({
                'start_line': current_line_num,
                'end_line': len(lines) - 1
            })
        
        return chunks, chunk_metadata
    
    def analyze_chunk(self, chunk_text, chunk_num, total_chunks):
        """Analyze a single chunk of text"""
        try:
            prompt = self.create_analysis_prompt() + "\n\n" + chunk_text
            
            print(f"      Analyzing chunk {chunk_num}/{total_chunks}...", end=" ", flush=True)
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=300
            )
            
            if response.status_code != 200:
                print("[FAIL]")
                return {"error": f"HTTP {response.status_code}"}
            
            response_data = response.json()
            analysis_text = response_data.get('response', '')
            
            if not analysis_text.strip():
                print("[FAIL]")
                return {"error": "Empty response from model"}
            
            try:
                analysis = json.loads(analysis_text)
                print("[OK]")
                return analysis
            except json.JSONDecodeError as e:
                print("[JSON ERROR]")
                return {
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": analysis_text[:500]
                }
                
        except requests.exceptions.Timeout:
            print("[TIMEOUT]")
            return {"error": "Request timeout"}
        except requests.exceptions.RequestException as e:
            print(f"[NETWORK ERROR]")
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            print(f"[ERROR]")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def analyze_file(self, file_path):
        """Analyze a single transcript file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks, metadata = self.chunk_text(content)
            print(f"      Processing {len(chunks)} chunk(s)")
            
            all_segments = []
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks, 1):
                chunk_analysis = self.analyze_chunk(chunk, i, len(chunks))
                
                if 'error' in chunk_analysis:
                    return {
                        'error': chunk_analysis['error'],
                        'file': str(file_path),
                        'chunk': i
                    }
                
                segments = chunk_analysis.get('conversational_segments', [])
                
                # Validate segments - skip any that are strings instead of dicts
                valid_segments = []
                for seg in segments:
                    if isinstance(seg, dict):
                        valid_segments.append(seg)
                    else:
                        print(f"      [WARN] Skipping malformed segment (type: {type(seg).__name__})")
                
                all_segments.extend(valid_segments)
                chunk_summaries.append(chunk_analysis.get('overall_summary', {}))
            
            # Aggregate summaries
            total_high = sum(s.get('high_risk_segments', 0) for s in chunk_summaries)
            total_medium = sum(s.get('medium_risk_segments', 0) for s in chunk_summaries)
            total_low = sum(s.get('low_risk_segments', 0) for s in chunk_summaries)
            total_redactions = sum(s.get('total_recommended_redactions', 0) for s in chunk_summaries)
            
            return {
                'source_file': str(file_path),
                'processed_at': datetime.now().isoformat(),
                'model_used': self.model_name,
                'chunks_processed': len(chunks),
                'conversational_segments': all_segments,
                'overall_summary': {
                    'total_segments_analyzed': len(all_segments),
                    'high_risk_segments': total_high,
                    'medium_risk_segments': total_medium,
                    'low_risk_segments': total_low,
                    'total_recommended_redactions': total_redactions,
                    'overall_recommendation': (
                        'REDACTIONS_NEEDED' if total_high > 0 else
                        'MINIMAL_CHANGES' if total_medium > 0 else
                        'ACCEPTABLE_AS_IS'
                    )
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'file': str(file_path)
            }
    
    def find_transcript_files(self, input_dir):
        """Find all CliniDeID labeled transcript files"""
        input_path = Path(input_dir)
        
        # Pattern: *.deid.piiCategoryTag.txt, but NOT *_notes.deid.piiCategoryTag.txt
        all_deid_files = list(input_path.glob("*.deid.piiCategoryTag.txt"))
        
        # Filter out notes files (those without _transcript in name are notes)
        transcript_files = [
            f for f in all_deid_files 
            if '_transcript.deid.piiCategoryTag.txt' in f.name
        ]
        
        return sorted(transcript_files)
    
    def process_batch(self, input_dir, output_dir, batch_size=50):
        """Process a batch of files with checkpointing"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all transcript files
        all_files = self.find_transcript_files(input_dir)
        print(f"[INFO] Found {len(all_files)} transcript files")
        
        if not all_files:
            print("[ERROR] No transcript files found matching pattern: *.deid.piiCategoryTag.txt")
            print("   (excluding *_notes.deid.piiCategoryTag.txt)")
            return
        
        # Get already processed files
        tracker_file = output_path / '.processed_files.txt'
        processed_files = set()
        if tracker_file.exists():
            with open(tracker_file, 'r', encoding='utf-8') as f:
                processed_files = set(line.strip() for line in f)
        
        # Extract just filenames from tracker (handles Mac vs Windows path differences)
        processed_filenames = set()
        for line in processed_files:
            try:
                processed_filenames.add(Path(line).name)
            except:
                pass  # Skip any malformed lines
        
        print(f"   Tracker file has {len(processed_files)} completed files")
        
        # Compare by filename only, not full path
        remaining_files = [f for f in all_files if f.name not in processed_filenames]
        
        print(f"[OK] Already processed: {len(processed_files)}")
        print(f"[INFO] Remaining: {len(remaining_files)}")
        print()
        
        if not remaining_files:
            print("[DONE] All files already processed!")
            print("\nGenerating consolidated CSV...")
            self.generate_consolidated_csv(output_dir)
            return
        
        # Take next batch
        batch_to_process = remaining_files[:batch_size]
        self.stats['total_files'] = len(all_files)
        
        print(f"[BATCH] Processing batch of {len(batch_to_process)} files")
        print(f"   ({len(remaining_files) - len(batch_to_process)} will remain after this batch)")
        print()
        
        # Process each file
        for i, file_path in enumerate(batch_to_process, 1):
            # Check if JSON already exists - if so, skip it
            sanitized_stem = sanitize_filename(file_path.stem)
            json_file = output_path / f"{sanitized_stem}_analysis.json"
            if json_file.exists():
                print(f"[SKIP] {file_path.name} - output already exists")
                continue
                
            print(f"\n{'='*70}")
            print(f"[FILE {i}/{len(batch_to_process)}] {file_path.name}")
            print(f"{'='*70}")
            
            start_time = time.time()
            analysis = self.analyze_file(file_path)
            elapsed = time.time() - start_time
            
            if 'error' in analysis:
                print(f"   [ERROR] {analysis['error']}")
                self.stats['errors'] += 1
                # Sanitize filename for error file
                error_filename = sanitize_filename(file_path.stem) + "_ERROR.json"
                error_file = output_path / error_filename
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2)
                # Still mark as processed even on error to skip it next time
                with open(tracker_file, 'a', encoding='utf-8') as f:
                    f.write(f"{file_path}\n")
                continue
            
            # Save JSON - SANITIZE THE FILENAME HERE
            sanitized_stem = sanitize_filename(file_path.stem)
            json_file = output_path / f"{sanitized_stem}_analysis.json"
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2)
            except OSError as e:
                # If filename still has issues, use a fallback name
                print(f"      [WARN] Filename issue, using fallback: {e}")
                json_file = output_path / f"file_{hash(str(file_path)) % 1000000}_analysis.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2)
            
            # Update stats
            summary = analysis.get('overall_summary', {})
            self.stats['processed'] += 1
            self.stats['high_risk'] += summary.get('high_risk_segments', 0)
            self.stats['medium_risk'] += summary.get('medium_risk_segments', 0)
            self.stats['total_redactions'] += summary.get('total_recommended_redactions', 0)
            
            # Mark as processed (write filename only for cross-platform compatibility)
            with open(tracker_file, 'a', encoding='utf-8') as f:
                f.write(f"{file_path.name}\n")
            
            print(f"   [OK] Complete in {elapsed:.1f}s")
            print(f"      High risk: {summary.get('high_risk_segments', 0)}")
            print(f"      Medium risk: {summary.get('medium_risk_segments', 0)}")
            print(f"      Redactions: {summary.get('total_recommended_redactions', 0)}")
        
        # Save progress stats
        stats_file = output_path / 'processing_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                **self.stats,
                'last_update': datetime.now().isoformat(),
                'remaining': len(remaining_files) - len(batch_to_process)
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"[BATCH COMPLETE]")
        print(f"{'='*70}")
        print(f"   Processed this batch: {len(batch_to_process)}")
        print(f"   Total processed: {len(processed_files) + len(batch_to_process)}")
        print(f"   Remaining: {len(remaining_files) - len(batch_to_process)}")
        
        if len(remaining_files) > len(batch_to_process):
            print(f"\n[INFO] {len(remaining_files) - len(batch_to_process)} files still remaining")
            print(f"   Run script again to process next batch")
        else:
            print(f"\n[DONE] ALL FILES PROCESSED!")
            print(f"\nGenerating consolidated CSV for review...")
            self.generate_consolidated_csv(output_path)
    
    def generate_consolidated_csv(self, output_dir):
        """Generate single CSV from all JSON analysis files for human review"""
        output_path = Path(output_dir)
        analysis_files = sorted(list(output_path.glob("*_analysis.json")))
        
        if not analysis_files:
            print("[ERROR] No analysis files found to consolidate")
            return
        
        print(f"\n[INFO] Consolidating {len(analysis_files)} analysis files into CSV...")
        
        all_rows = []
        
        for file_path in analysis_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'error' in data:
                    continue
                
                file_name = Path(data.get('source_file', file_path.stem)).name
                segments = data.get('conversational_segments', [])
                
                for segment in segments:
                    risk_level = segment.get('risk_level', 'UNKNOWN')
                    removals = segment.get('recommended_removals', [])
                    
                    if removals:
                        for removal in removals:
                            all_rows.append({
                                'REVIEW': '',  # Empty for human to mark Y/M/N
                                'FILE': file_name,
                                'RISK': risk_level,
                                'CATEGORY': 'BIOGRAPHICAL_IDENTIFIER',
                                'TEXT': removal.get('text_to_remove', ''),
                                'REASONING': removal.get('reason', ''),
                                'SUGGESTED_REPLACEMENT': removal.get('replacement_suggestion', '')
                            })
                            
            except Exception as e:
                print(f"[WARN] Error processing {file_path.name}: {e}")
                continue
        
        if not all_rows:
            print("[WARN] No redactions recommended across all files")
            return
        
        # Sort by FILE, then RISK (HIGH first)
        risk_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2, 'UNKNOWN': 3}
        all_rows.sort(key=lambda x: (x['FILE'], risk_order.get(x['RISK'], 4)))
        
        # Write consolidated CSV
        csv_file = output_path / 'LLM_REVIEW_CONSOLIDATED.csv'
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['REVIEW', 'FILE', 'RISK', 'CATEGORY', 'TEXT', 'REASONING', 'SUGGESTED_REPLACEMENT']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\n[OK] CONSOLIDATED CSV CREATED")
        print(f"   Location: {csv_file}")
        print(f"   Total rows: {len(all_rows)}")
        print(f"   Files analyzed: {len(analysis_files)}")
        
        # Summary stats
        high_risk = sum(1 for row in all_rows if row['RISK'] == 'HIGH')
        medium_risk = sum(1 for row in all_rows if row['RISK'] == 'MEDIUM')
        
        print(f"\n[SUMMARY] REVIEW SUMMARY:")
        print(f"   HIGH risk items: {high_risk}")
        print(f"   MEDIUM risk items: {medium_risk}")
        print(f"   LOW/other items: {len(all_rows) - high_risk - medium_risk}")
        
        print(f"\n[INFO] REVIEW INSTRUCTIONS:")
        print(f"   1. Open CSV in Excel/LibreOffice")
        print(f"   2. Mark REVIEW column: Y (identifying), M (maybe), blank (false positive)")
        print(f"   3. Focus on HIGH risk items first")
        print(f"   4. Save as: LLM_REVIEW_CONSOLIDATED_REVIEWED.csv")


def main():
    parser = argparse.ArgumentParser(
        description='NEXUS Batch Information-Theoretic Analyzer for CliniDeID labeled files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process batch of 50 files (default)
  python IT_nexus_batch_analyzer.py --input /path/to/volumes/video_research/ForNEXUS/CliniDeID_LLM_Ready/ \\
                                      --output /path/to/volumes/video_research/ForNEXUS/LLM_Analysis_Output/
  
  # Larger batch
  python IT_nexus_batch_analyzer.py --input /path/to/volumes/video_research/ForNEXUS/CliniDeID_LLM_Ready/ \\
                                      --output /path/to/volumes/video_research/ForNEXUS/LLM_Analysis_Output/ \\
                                      --batch-size 100
  
  # Custom model (default is qwen2.5:14b)
  python IT_nexus_batch_analyzer.py --input /path/to/volumes/video_research/ForNEXUS/CliniDeID_LLM_Ready/ \\
                                      --output /path/to/volumes/video_research/ForNEXUS/LLM_Analysis_Output/ \\
                                      --model llama3.1:70b
        """
    )
    
    parser.add_argument('--input', required=True,
                       help='Input directory with CliniDeID labeled files')
    parser.add_argument('--output', required=True,
                       help='Output directory for analysis results (on NEXUS)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of files to process per run (default: 50)')
    parser.add_argument('--model', default='qwen2.5:14b',
                       help='Ollama model to use (default: qwen2.5:14b)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Input directory not found: {args.input}")
        exit(1)
    
    print("="*70)
    print("NEXUS BATCH INFORMATION-THEORETIC ANALYZER")
    print("LLM-Based Anonymization Review for CliniDeID Labeled Transcripts")
    print("="*70)
    print()
    
    analyzer = NEXUSBatchAnalyzer(model_name=args.model)
    analyzer.process_batch(args.input, args.output, args.batch_size)

if __name__ == "__main__":
    main()