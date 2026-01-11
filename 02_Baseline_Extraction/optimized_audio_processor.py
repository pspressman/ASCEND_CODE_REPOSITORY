import os
import sys
import whisperx
import gc 
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
import opensmile
import pandas as pd
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation, Segment
import logging
import time
import traceback
import re
import json

# Set OpenSMILE environment
os.environ['OPENSMILE_ROOT'] = '/path/to/user/opensmile'
os.environ['LD_LIBRARY_PATH'] = '/path/to/user/opensmile/progsrc/smileapi:' + os.environ.get('LD_LIBRARY_PATH', '')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_speaker_number(speaker_label):
    match = re.search(r'\d+', speaker_label)
    return int(match.group()) if match else 0

def is_already_processed(audio_file, output_subdir):
    """Check if file is completely processed"""
    base_name = audio_file.stem
    
    core_files = [
        output_subdir / f"{base_name}_transcript.txt",
        output_subdir / f"{base_name}_non_speech_segments.csv",
        output_subdir / f"{base_name}.rttm"
    ]
    
    if not all(f.exists() and f.stat().st_size > 0 for f in core_files):
        return False
    
    speaker_files = list(output_subdir.glob(f"{base_name}_speaker*_features.csv"))
    return len(speaker_files) > 0 and all(f.stat().st_size > 0 for f in speaker_files)

def check_partial_processing(audio_file, output_subdir):
    """Check what's already done to enable resume"""
    base_name = audio_file.stem
    
    transcript_done = (output_subdir / f"{base_name}_transcript_raw.json").exists()
    
    rttm_file = output_subdir / f"{base_name}.rttm"
    diarization_done = rttm_file.exists() and rttm_file.stat().st_size > 0
    
    existing_speakers = set()
    for speaker_file in output_subdir.glob(f"{base_name}_speaker*_features.csv"):
        if speaker_file.stat().st_size > 0:
            match = re.search(r'speaker(\d+)_features\.csv', speaker_file.name)
            if match:
                existing_speakers.add(int(match.group(1)))
    
    non_speech_done = (output_subdir / f"{base_name}_non_speech_segments.csv").exists()
    final_transcript_done = (output_subdir / f"{base_name}_transcript.txt").exists()
    
    return {
        'transcript_done': transcript_done,
        'diarization_done': diarization_done,
        'existing_speakers': existing_speakers,
        'non_speech_done': non_speech_done,
        'final_transcript_done': final_transcript_done,
        'features_done': is_already_processed(audio_file, output_subdir)
    }

class AudioProcessor:
    """Reusable processor that loads models once"""
    
    def __init__(self, device, batch_size, compute_type, hf_token):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.hf_token = hf_token
        
        logging.info("Loading WhisperX model (once)...")
        try:
            self.whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        except ValueError as e:
            if "unsupported device mps" in str(e):
                logging.warning("MPS not supported, falling back to CPU")
                self.device = "cpu"
                self.whisper_model = whisperx.load_model("large-v2", "cpu", compute_type=compute_type)
            else:
                raise
        
        logging.info("Loading diarization pipeline (once)...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=hf_token
        )
        self.diarization_pipeline.to(torch.device(self.device))
        
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        logging.info("All models loaded. Ready to process files.")
    
    def process_file(self, audio_file, output_subdir):
        """Process single audio file with resume capability"""
        base_name = audio_file.stem
        status = check_partial_processing(audio_file, output_subdir)
        
        logging.info(f"Resume status:")
        logging.info(f"  Transcript: {'✓ Done' if status['transcript_done'] else '→ Need to do'}")
        logging.info(f"  Diarization: {'✓ Done' if status['diarization_done'] else '→ Need to do'}")
        if status['existing_speakers']:
            logging.info(f"  Features: ✓ Speakers {sorted(status['existing_speakers'])} done")
        logging.info(f"  Outputs: {'✓ Done' if (status['non_speech_done'] and status['final_transcript_done']) else '→ Need to do'}")
        
        start_time = time.time()
        
        logging.info(f"Loading audio: {audio_file.name}")
        audio_array = whisperx.load_audio(str(audio_file))
        waveform, sample_rate = torchaudio.load(str(audio_file))
        
        try:
            # STEP 1: Transcription
            if status['transcript_done']:
                logging.info("  ✓ Transcript exists, loading...")
                with open(output_subdir / f"{base_name}_transcript_raw.json") as f:
                    result = json.load(f)
            else:
                logging.info("  → Transcribing...")
                result = self.whisper_model.transcribe(audio_array, batch_size=self.batch_size)
                
                logging.info("  → Aligning...")
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], 
                    device=self.device
                )
                result = whisperx.align(
                    result["segments"], model_a, metadata, 
                    audio_array, self.device, 
                    return_char_alignments=False
                )
                
                del model_a
                del metadata
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                with open(output_subdir / f"{base_name}_transcript_raw.json", 'w') as f:
                    json.dump(result, f)
                logging.info(f"  ✓ Transcription complete: {len(result['segments'])} segments")
            
            # STEP 2: Diarization
            if status['diarization_done']:
                logging.info("  ✓ Diarization exists, loading from RTTM...")
                diarization = Annotation()
                
                rttm_file = output_subdir / f"{base_name}.rttm"
                with open(rttm_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            start = float(parts[3])
                            duration = float(parts[4])
                            speaker = parts[7]
                            diarization[Segment(start, start + duration)] = speaker
                
                logging.info(f"  ✓ Loaded diarization: {len(set(diarization.labels()))} speakers")
            else:
                logging.info("  → Diarizing...")
                with ProgressHook() as hook:
                    diarization = self.diarization_pipeline(
                        {"waveform": waveform, "sample_rate": sample_rate},
                        min_speakers=2,
                        max_speakers=5,
                        hook=hook
                    )
                
                with open(output_subdir / f"{base_name}.rttm", "w") as f:
                    diarization.write_rttm(f)
                
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                logging.info(f"  ✓ Diarization complete: {len(set(diarization.labels()))} speakers")
            
            # STEP 3: Assign speakers
            logging.info("  → Assigning speakers to words...")
            diarize_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_num = extract_speaker_number(speaker)
                diarize_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker_num
                })
            
            diarize_df = pd.DataFrame(diarize_segments)
            result = whisperx.assign_word_speakers(diarize_df, result)
            
            # STEP 4: Process segments
            logging.info("  → Analyzing turn-taking...")
            processed_segments = []
            for i, segment in enumerate(result['segments']):
                current_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": segment.get("speaker"),
                    "type": "speech"
                }
                
                if i < len(result['segments']) - 1:
                    next_segment = result['segments'][i+1]
                    
                    if current_segment["end"] > next_segment["start"]:
                        overlap = {
                            "start": next_segment["start"],
                            "end": min(current_segment["end"], next_segment["end"]),
                            "type": "overlap",
                            "speakers": [current_segment["speaker"], next_segment.get("speaker")],
                            "controller": next_segment.get("speaker")
                        }
                        processed_segments.append(overlap)
                        current_segment["end"] = next_segment["start"]
                    
                    elif current_segment["end"] < next_segment["start"]:
                        if current_segment["speaker"] == next_segment.get("speaker"):
                            pause = {
                                "start": current_segment["end"],
                                "end": next_segment["start"],
                                "type": "pause",
                                "speaker": current_segment["speaker"]
                            }
                            processed_segments.append(pause)
                        else:
                            gap = {
                                "start": current_segment["end"],
                                "end": next_segment["start"],
                                "type": "gap",
                                "controller": next_segment.get("speaker")
                            }
                            processed_segments.append(gap)
                
                processed_segments.append(current_segment)
            
            # STEP 5: Extract features
            logging.info("  → Extracting acoustic features...")
            speaker_segments = {}
            non_speech_segments = []
            
            for segment in processed_segments:
                if segment['type'] == 'speech':
                    speaker = segment['speaker']
                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append(segment)
                else:
                    non_speech_segments.append(segment)
            
            for speaker, segments in speaker_segments.items():
                if speaker in status['existing_speakers']:
                    logging.info(f"    ✓ Speaker {speaker}: Already done, skipping")
                    continue
                
                logging.info(f"    Speaker {speaker}: {len(segments)} segments")
                
                feature_list = []
                for segment in segments:
                    start_sample = int(segment['start'] * sample_rate)
                    end_sample = int(segment['end'] * sample_rate)
                    
                    if start_sample >= end_sample:
                        continue
                    
                    audio_segment = waveform[:, start_sample:end_sample]
                    
                    try:
                        features = self.smile.process_signal(
                            audio_segment.numpy().flatten(), 
                            sample_rate
                        )
                        feature_list.append(features)
                    except Exception as e:
                        logging.warning(f"    Failed to extract features for segment: {e}")
                        continue
                
                if feature_list:
                    aggregated = self.aggregate_features(feature_list)
                    output_file = output_subdir / f"{base_name}_speaker{speaker}_features.csv"
                    aggregated.to_csv(output_file, index=False)
                    logging.info(f"    ✓ Saved speaker {speaker} features")
                    
                    del feature_list
                    del aggregated
            
            gc.collect()
            
            # STEP 6: Save outputs
            if not status['non_speech_done']:
                pd.DataFrame(non_speech_segments).to_csv(
                    output_subdir / f"{base_name}_non_speech_segments.csv",
                    index=False
                )
                logging.info("  ✓ Saved non-speech segments")
            
            if not status['final_transcript_done']:
                with open(output_subdir / f"{base_name}_transcript.txt", "w") as f:
                    for segment in processed_segments:
                        if segment['type'] == 'speech':
                            f.write(f"Speaker {segment['speaker']}: {segment['text']}\n")
                        else:
                            f.write(f"{segment['type'].capitalize()}: {segment['start']:.2f} - {segment['end']:.2f}\n")
                logging.info("  ✓ Saved final transcript")
            
            elapsed = time.time() - start_time
            logging.info(f"✓ COMPLETE: {audio_file.name} ({elapsed:.1f}s)")
        
        finally:
            try:
                del audio_array
                del waveform
            except:
                pass
            
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
    
    def aggregate_features(self, feature_list):
        """Aggregate list of feature dataframes"""
        df = pd.concat(feature_list)
        
        aggregated = df.agg(['mean', 'median', 'std', 'min', 'max'])
        aggregated = aggregated.T.add_suffix('_value').T
        
        cov = df.std() / df.mean().replace(0, np.nan)
        aggregated.loc['cov'] = cov.fillna(0)
        
        q75 = df.quantile(0.75)
        q25 = df.quantile(0.25)
        aggregated.loc['iqr'] = q75 - q25
        
        return aggregated.T.reset_index().melt(id_vars='index')

def main():
    start_time = time.time()
    logging.info("="*80)
    logging.info("OPTIMIZED AUDIO PROCESSOR")
    logging.info("="*80)
    
    if len(sys.argv) >= 3:
        audio_root = sys.argv[1]
        output_root = sys.argv[2]
    else:
        audio_root = "/Volumes/Databackup2025/Data/CSAND/Clinic Recordings/"
        output_root = "/path/to/user/Desktop/CompleteClinicAudioOutput"
    
    logging.info(f"Input:  {audio_root}")
    logging.info(f"Output: {output_root}")
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logging.info(f"Device: {device}")
    
    batch_size = 16
    compute_type = "float32"
    hf_token = ""
    
    audio_root_path = Path(audio_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(audio_root_path.rglob("*.wav"))
    logging.info(f"Found {len(audio_files)} audio files")
    
    files_to_process = []
    for audio_file in audio_files:
        rel_path = audio_file.relative_to(audio_root_path)
        output_subdir = output_root_path / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        if is_already_processed(audio_file, output_subdir):
            logging.info(f"✓ SKIP (complete): {audio_file.name}")
        else:
            files_to_process.append((audio_file, output_subdir))
    
    logging.info(f"Files to process: {len(files_to_process)}")
    logging.info(f"Files already done: {len(audio_files) - len(files_to_process)}")
    
    if not files_to_process:
        logging.info("No files to process. Exiting.")
        return
    
    logging.info("="*80)
    logging.info("INITIALIZING MODELS")
    logging.info("="*80)
    processor = AudioProcessor(device, batch_size, compute_type, hf_token)
    
    logging.info("="*80)
    logging.info("PROCESSING FILES")
    logging.info("="*80)
    
    for idx, (audio_file, output_subdir) in enumerate(files_to_process, 1):
        logging.info(f"\n[{idx}/{len(files_to_process)}] {audio_file.name}")
        try:
            processor.process_file(audio_file, output_subdir)
        except Exception as e:
            logging.error(f"ERROR: {audio_file.name}")
            logging.error(f"  {str(e)}")
            logging.error(traceback.format_exc())
            continue
    
    total_time = time.time() - start_time
    logging.info("="*80)
    logging.info(f"COMPLETE: {len(files_to_process)} files in {total_time/60:.1f} minutes")
    logging.info(f"Average: {total_time/len(files_to_process):.1f}s per file")
    logging.info("="*80)

if __name__ == "__main__":
    main()
