#!/usr/bin/env python3
"""
ASCEND Audio Feature Extraction Pipeline Orchestrator
Sequential autopilot processing with state management and resume capability

Usage:
    python pipeline_orchestrator.py --config pipeline_config.json
    python pipeline_orchestrator.py --config pipeline_config.json --fresh
    python pipeline_orchestrator.py --config pipeline_config.json --start-from stage2_librosa_turn
    python pipeline_orchestrator.py --config pipeline_config.json --dry-run
"""

import sys
import json
import subprocess
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring disabled")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_orchestrator.log')
    ]
)


class PipelineState:
    """Manages pipeline execution state for resume capability"""
    
    def __init__(self, state_file='pipeline_state.json'):
        self.state_file = Path(state_file)
        self.state = self._load_or_create()
    
    def _load_or_create(self) -> Dict:
        """Load existing state or create new"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'pipeline_run_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'stages': {}
            }
    
    def save(self):
        """Save current state to file"""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        logging.debug(f"State saved to {self.state_file}")
    
    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage has been completed"""
        return (stage_name in self.state['stages'] and 
                self.state['stages'][stage_name].get('status') == 'completed')
    
    def mark_stage_started(self, stage_name: str):
        """Mark stage as started"""
        self.state['stages'][stage_name] = {
            'status': 'in_progress',
            'started': datetime.now().isoformat(),
            'completed': None,
            'errors': [],
            'files_processed': 0
        }
        self.save()
    
    def mark_stage_completed(self, stage_name: str, files_processed: int = 0):
        """Mark stage as completed"""
        self.state['stages'][stage_name].update({
            'status': 'completed',
            'completed': datetime.now().isoformat(),
            'files_processed': files_processed
        })
        self.save()
    
    def mark_stage_failed(self, stage_name: str, error: str):
        """Mark stage as failed"""
        self.state['stages'][stage_name].update({
            'status': 'failed',
            'completed': datetime.now().isoformat()
        })
        self.state['stages'][stage_name]['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error
        })
        self.save()
    
    def add_stage_error(self, stage_name: str, error: str):
        """Add error to stage without failing it"""
        if stage_name in self.state['stages']:
            self.state['stages'][stage_name]['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error
            })
            self.save()
    
    def reset(self):
        """Reset state to fresh start"""
        self.state = {
            'pipeline_run_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'stages': {}
        }
        self.save()


class PipelineOrchestrator:
    """Orchestrates sequential execution of audio feature extraction pipeline"""
    
    def __init__(self, config_path: str, dry_run: bool = False, fresh_start: bool = False, start_from: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.dry_run = dry_run
        self.start_from = start_from
        self.state = PipelineState()
        
        if fresh_start:
            logging.info("Fresh start requested - resetting pipeline state")
            self.state.reset()
        
        self._validate_paths()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _validate_paths(self):
        """Validate that all configured paths exist"""
        input_root = Path(self.config['paths']['input_root'])
        if not input_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {input_root}")
        
        # Check code files exist
        code_paths = self.config['paths']['code']
        for name, path in code_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Code path '{name}' does not exist: {path}")
        
        # Check environments exist
        env_paths = self.config['paths']['environments']
        for name, path in env_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Environment '{name}' does not exist: {path}")
    
    def _run_command(self, command: str, env_path: str, stage_name: str) -> bool:
        """Execute command with environment activation"""
        
        # Build full command with environment activation
        if sys.platform == 'darwin':  # macOS
            full_command = f"source {env_path}/bin/activate && {command}"
        else:
            full_command = f"source {env_path}/bin/activate && {command}"
        
        if self.dry_run:
            logging.info(f"[DRY RUN] Would execute: {full_command}")
            return True
        
        logging.info(f"Executing: {command}")
        logging.info(f"Environment: {env_path}")
        
        try:
            result = subprocess.run(
                full_command,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True
            )
            
            # Log output
            if result.stdout:
                logging.info(f"STDOUT:\n{result.stdout}")
            
            if result.stderr:
                logging.warning(f"STDERR:\n{result.stderr}")
            
            if result.returncode != 0:
                error_msg = f"Command failed with return code {result.returncode}"
                logging.error(error_msg)
                self.state.add_stage_error(stage_name, error_msg)
                
                if not self.config['options'].get('continue_on_error', True):
                    raise Exception(error_msg)
                return False
            
            return True
            
        except Exception as e:
            error_msg = f"Exception during command execution: {str(e)}"
            logging.error(error_msg)
            self.state.add_stage_error(stage_name, error_msg)
            
            if not self.config['options'].get('continue_on_error', True):
                raise
            return False
    
    def _check_standardization_needed(self) -> bool:
        """Check if directories need standardization"""
        input_root = Path(self.config['paths']['input_root'])
        
        # Look for non-standardized naming patterns
        non_standard_patterns = [
            'Grandfather Passage',
            'Motor Speech Evaluation',
            'Picnic Description',
            'Spontaneous Speech',
            'Mac_Audacity_10MinConvo_with',
            'PC_10MinuteConversations_'
        ]
        
        for pattern in non_standard_patterns:
            if list(input_root.rglob(f"*{pattern}*")):
                return True
        
        return False
    
    def _find_filtered_directories(self, patterns: List[str]) -> List[Path]:
        """Find directories matching any of the given patterns"""
        input_root = Path(self.config['paths']['input_root'])
        found_dirs = set()
        
        for pattern in patterns:
            dirs = input_root.rglob(f"*{pattern}*")
            for d in dirs:
                if d.is_dir():
                    found_dirs.add(d)
        
        return sorted(list(found_dirs))
    
    def _wait_for_memory(self, required_gb: float = 15.0, check_interval: int = 300):
        """
        Wait until sufficient memory is available before proceeding.
        Polls memory every check_interval seconds.
        
        Args:
            required_gb: Minimum GB of free memory required
            check_interval: Seconds between memory checks (default 5 minutes)
        """
        if not PSUTIL_AVAILABLE:
            logging.warning("⚠️  psutil not available - skipping memory check")
            logging.warning("   Install with: pip install psutil")
            return
        
        if self.dry_run:
            logging.info(f"[DRY RUN] Would wait for {required_gb}GB free memory")
            return
        
        required_bytes = required_gb * 1_000_000_000
        first_check = True
        
        while True:
            mem = psutil.virtual_memory()
            available_gb = mem.available / 1_000_000_000
            used_gb = mem.used / 1_000_000_000
            total_gb = mem.total / 1_000_000_000
            
            if mem.available >= required_bytes:
                logging.info("="*80)
                logging.info(f"✓ MEMORY AVAILABLE: {available_gb:.1f}GB free (required: {required_gb}GB)")
                logging.info(f"  System: {used_gb:.1f}GB used / {total_gb:.1f}GB total ({mem.percent:.1f}% used)")
                logging.info("  Proceeding with Stage 3...")
                logging.info("="*80)
                return
            
            if first_check:
                logging.info("="*80)
                logging.info("⏳ WAITING FOR MEMORY AVAILABILITY")
                logging.info("="*80)
                logging.info(f"Stage 3 requires {required_gb}GB free memory for safe execution")
                logging.info("This prevents crashes from simultaneous memory-intensive processes")
                logging.info("(e.g., WhisperX transcription, background Parselmouth jobs)")
                logging.info("")
                logging.info("Pipeline will automatically resume when memory is available")
                logging.info(f"Checking every {check_interval//60} minutes...")
                logging.info("="*80)
                first_check = False
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            logging.info(f"[{timestamp}] ⏳ Waiting: {available_gb:.1f}GB free / {required_gb:.1f}GB needed")
            logging.info(f"           Memory usage: {mem.percent:.1f}% ({used_gb:.1f}GB / {total_gb:.1f}GB)")
            logging.info(f"           Next check in {check_interval//60} minutes...")
            
            time.sleep(check_interval)
    
    def run_pipeline(self):
        """Execute complete pipeline sequentially"""
        
        logging.info("="*80)
        logging.info("ASCEND AUDIO FEATURE EXTRACTION PIPELINE")
        logging.info("="*80)
        logging.info(f"Pipeline Run ID: {self.state.state['pipeline_run_id']}")
        logging.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        logging.info(f"Input: {self.config['paths']['input_root']}")
        logging.info(f"Output: {self.config['paths']['output_root']}")
        logging.info("="*80)
        
        # Show current state
        self._show_pipeline_status()
        
        # Execute stages
        stages = [
            ('stage0_standardization', self.run_stage0_standardization),
            ('stage1_opensmile', self.run_stage1_opensmile),
            ('stage1_librosa_agg', self.run_stage1_librosa_aggregated),
            ('stage1_parselmouth_agg', self.run_stage1_parselmouth_aggregated),
            ('stage2_librosa_turn', self.run_stage2_librosa_turn),
            ('stage2_parselmouth_turn', self.run_stage2_parselmouth_turn),
            ('stage3_parselmouth_wholefile', self.run_stage3_parselmouth_wholefile),
            ('stage3_mse', self.run_stage3_mse)
        ]
        
        for stage_name, stage_func in stages:
            # Skip if starting from a later stage
            if self.start_from and stage_name != self.start_from and not self.state.is_stage_complete(self.start_from):
                logging.info(f"\nSkipping {stage_name} (starting from {self.start_from})")
                continue
            
            # Skip if already complete
            if self.state.is_stage_complete(stage_name):
                logging.info(f"\n✓ {stage_name} already complete, skipping")
                continue
            
            # Execute stage
            logging.info(f"\n{'='*80}")
            logging.info(f"EXECUTING: {stage_name.upper()}")
            logging.info(f"{'='*80}")
            
            try:
                stage_func()
            except Exception as e:
                logging.error(f"Stage {stage_name} failed: {e}")
                if not self.config['options'].get('continue_on_error', True):
                    raise
        
        # Generate summary
        self.generate_summary()
    
    def _show_pipeline_status(self):
        """Display current pipeline completion status"""
        stages = self.state.state['stages']
        if not stages:
            logging.info("\nNo stages completed yet (fresh start)")
            return
        
        logging.info("\nCurrent Pipeline Status:")
        for stage_name, stage_data in stages.items():
            status = stage_data['status']
            if status == 'completed':
                logging.info(f"  ✓ {stage_name}: COMPLETED")
            elif status == 'failed':
                logging.info(f"  ✗ {stage_name}: FAILED")
            elif status == 'in_progress':
                logging.info(f"  ⋯ {stage_name}: IN PROGRESS")
        logging.info("")
    
    def run_stage0_standardization(self):
        """Stage 0: Directory name standardization"""
        stage_name = 'stage0_standardization'
        
        # Check if standardization needed
        if not self._check_standardization_needed():
            logging.info("Directory names already standardized, skipping")
            self.state.mark_stage_completed(stage_name)
            return
        
        if self.config['options'].get('skip_standardization', False):
            logging.info("Standardization skipped per configuration")
            self.state.mark_stage_completed(stage_name)
            return
        
        self.state.mark_stage_started(stage_name)
        
        standardize_script = Path(__file__).parent / 'standardize_task_directories.py'
        input_root = self.config['paths']['input_root']
        manifest_file = f"standardization_manifest_{self.state.state['pipeline_run_id']}.csv"
        
        command = f"python {standardize_script} --root {input_root} --manifest {manifest_file}"
        
        if self._run_command(command, sys.prefix, stage_name):
            self.state.mark_stage_completed(stage_name)
        else:
            self.state.mark_stage_failed(stage_name, "Standardization failed")
    
    def run_stage1_opensmile(self):
        """Stage 1: openSMILE full-task extraction"""
        stage_name = 'stage1_opensmile'
        self.state.mark_stage_started(stage_name)
        
        script = self.config['paths']['code']['opensmile']
        env = self.config['paths']['environments']['audio_env']
        input_dir = self.config['paths']['input_root']
        output_dir = Path(self.config['paths']['output_root']) / 'opensmile'
        
        command = f"python {script} --input {input_dir} --output {output_dir}"
        
        if self._run_command(command, env, stage_name):
            self.state.mark_stage_completed(stage_name)
        else:
            self.state.mark_stage_failed(stage_name, "openSMILE extraction failed")
    
    def run_stage1_librosa_aggregated(self):
        """Stage 1: Librosa aggregated full-task extraction"""
        stage_name = 'stage1_librosa_agg'
        self.state.mark_stage_started(stage_name)
        
        script = Path(self.config['paths']['code']['librosa_dir']) / 'librosa_per_speaker_aggregated.py'
        env = self.config['paths']['environments']['librosa_env']
        input_dir = self.config['paths']['input_root']
        output_file = Path(self.config['paths']['output_root']) / 'librosa_aggregated_master.csv'
        
        command = f"python {script} --input {input_dir} --output {output_file}"
        
        if self._run_command(command, env, stage_name):
            self.state.mark_stage_completed(stage_name)
        else:
            self.state.mark_stage_failed(stage_name, "Librosa aggregated extraction failed")
    
    def run_stage1_parselmouth_aggregated(self):
        """Stage 1: Parselmouth aggregated full-task extraction"""
        stage_name = 'stage1_parselmouth_agg'
        self.state.mark_stage_started(stage_name)
        
        script = Path(self.config['paths']['code']['parselmouth_dir']) / 'parselmouth_per_speaker_aggregated.py'
        env = self.config['paths']['environments']['ml_env']
        input_dir = self.config['paths']['input_root']
        output_file = Path(self.config['paths']['output_root']) / 'parselmouth_aggregated_master.csv'
        
        command = f"python {script} --input {input_dir} --output {output_file}"
        
        if self._run_command(command, env, stage_name):
            self.state.mark_stage_completed(stage_name)
        else:
            self.state.mark_stage_failed(stage_name, "Parselmouth aggregated extraction failed")
    
    def run_stage2_librosa_turn(self):
        """Stage 2: Librosa turn-by-turn (Conflict + SpontSpeech only)"""
        stage_name = 'stage2_librosa_turn'
        self.state.mark_stage_started(stage_name)
        
        # Find filtered directories
        filtered_dirs = self._find_filtered_directories(['ConflictConv', 'SpontSpeech'])
        logging.info(f"Found {len(filtered_dirs)} directories for turn-by-turn processing")
        
        script = Path(self.config['paths']['code']['librosa_dir']) / 'librosa_per_turn.py'
        env = self.config['paths']['environments']['librosa_env']
        output_dir = Path(self.config['paths']['output_root']) / 'librosa_turn'
        
        # Process each directory
        for dir_path in filtered_dirs:
            logging.info(f"Processing: {dir_path.relative_to(self.config['paths']['input_root'])}")
            command = f"python {script} --input {dir_path} --output {output_dir}"
            self._run_command(command, env, stage_name)
        
        self.state.mark_stage_completed(stage_name, len(filtered_dirs))
    
    def run_stage2_parselmouth_turn(self):
        """Stage 2: Parselmouth turn-by-turn (Conflict + SpontSpeech only)"""
        stage_name = 'stage2_parselmouth_turn'
        self.state.mark_stage_started(stage_name)
        
        # Find filtered directories
        filtered_dirs = self._find_filtered_directories(['ConflictConv', 'SpontSpeech'])
        logging.info(f"Found {len(filtered_dirs)} directories for turn-by-turn processing")
        
        script = Path(self.config['paths']['code']['parselmouth_dir']) / 'parselmouth_per_turn.py'
        env = self.config['paths']['environments']['ml_env']
        output_dir = Path(self.config['paths']['output_root']) / 'parselmouth_turn'
        
        # Process each directory
        for dir_path in filtered_dirs:
            logging.info(f"Processing: {dir_path.relative_to(self.config['paths']['input_root'])}")
            command = f"python {script} --input {dir_path} --output {output_dir}"
            self._run_command(command, env, stage_name)
        
        self.state.mark_stage_completed(stage_name, len(filtered_dirs))
    
    def run_stage3_parselmouth_wholefile(self):
        """Stage 3: Parselmouth 0.1s windows whole-file (non-diarized)"""
        stage_name = 'stage3_parselmouth_wholefile'
        
        # MEMORY GATE: Wait for 15GB free before proceeding
        logging.info("\n" + "="*80)
        logging.info("STAGE 3 PRE-CHECK: MEMORY AVAILABILITY")
        logging.info("="*80)
        self._wait_for_memory(required_gb=15.0, check_interval=300)
        
        self.state.mark_stage_started(stage_name)
        
        script = Path(self.config['paths']['code']['parselmouth_dir']) / 'parselmouth_standalone.py'
        env = self.config['paths']['environments']['ml_env']
        input_dir = self.config['paths']['input_root']
        output_dir = Path(self.config['paths']['output_root']) / 'parselmouth_0.1sec_wholefile'
        
        command = f"python {script} --input {input_dir} --output {output_dir}"
        
        if self._run_command(command, env, stage_name):
            self.state.mark_stage_completed(stage_name)
        else:
            self.state.mark_stage_failed(stage_name, "Parselmouth whole-file extraction failed")
    
    def run_stage3_mse(self):
        """Stage 3: MSE 0.1s windows per-speaker (diarized, MSE/Paralang only)"""
        stage_name = 'stage3_mse'
        
        # MEMORY GATE: Wait for 15GB free before proceeding
        logging.info("\n" + "="*80)
        logging.info("STAGE 3 MSE PRE-CHECK: MEMORY AVAILABILITY")
        logging.info("="*80)
        self._wait_for_memory(required_gb=15.0, check_interval=300)
        
        self.state.mark_stage_started(stage_name)
        
        script = Path(self.config['paths']['code']['parselmouth_dir']) / 'mse_standalone.py'
        env = self.config['paths']['environments']['ml_env']
        input_dir = self.config['paths']['input_root']
        output_dir = Path(self.config['paths']['output_root']) / 'parselmouth_0.1sec_mse'
        
        command = f"python {script} --input {input_dir} --output {output_dir}"
        
        if self._run_command(command, env, stage_name):
            self.state.mark_stage_completed(stage_name)
        else:
            self.state.mark_stage_failed(stage_name, "MSE extraction failed")
    
    def generate_summary(self):
        """Generate final pipeline execution summary"""
        logging.info("\n" + "="*80)
        logging.info("PIPELINE EXECUTION SUMMARY")
        logging.info("="*80)
        
        stages = self.state.state['stages']
        
        total_stages = len(stages)
        completed = sum(1 for s in stages.values() if s['status'] == 'completed')
        failed = sum(1 for s in stages.values() if s['status'] == 'failed')
        
        logging.info(f"Total Stages: {total_stages}")
        logging.info(f"Completed: {completed}")
        logging.info(f"Failed: {failed}")
        logging.info("")
        
        for stage_name, stage_data in stages.items():
            status_symbol = {
                'completed': '✓',
                'failed': '✗',
                'in_progress': '⋯'
            }.get(stage_data['status'], '?')
            
            logging.info(f"{status_symbol} {stage_name}: {stage_data['status'].upper()}")
            
            if stage_data['errors']:
                logging.info(f"  Errors: {len(stage_data['errors'])}")
                for error in stage_data['errors'][:3]:  # Show first 3 errors
                    logging.info(f"    - {error['error'][:100]}")
        
        logging.info("="*80)
        
        if completed == total_stages:
            logging.info("🎉 Pipeline completed successfully!")
        elif failed > 0:
            logging.info("⚠️  Pipeline completed with errors. Check logs for details.")
        else:
            logging.info("ℹ️  Pipeline partially completed. Some stages may be pending.")
        
        logging.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='ASCEND Audio Feature Extraction Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline (autopilot mode, resume if state exists)
  python pipeline_orchestrator.py --config pipeline_config.json
  
  # Fresh start (ignore existing state)
  python pipeline_orchestrator.py --config pipeline_config.json --fresh
  
  # Resume from specific stage
  python pipeline_orchestrator.py --config pipeline_config.json --start-from stage2_librosa_turn
  
  # Dry run (preview without executing)
  python pipeline_orchestrator.py --config pipeline_config.json --dry-run
        """
    )
    
    parser.add_argument('--config', required=True, help='Path to pipeline configuration JSON file')
    parser.add_argument('--dry-run', action='store_true', help='Preview commands without executing')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore existing state)')
    parser.add_argument('--start-from', help='Resume from specific stage (e.g., stage2_librosa_turn)')
    
    args = parser.parse_args()
    
    try:
        orchestrator = PipelineOrchestrator(
            config_path=args.config,
            dry_run=args.dry_run,
            fresh_start=args.fresh,
            start_from=args.start_from
        )
        
        orchestrator.run_pipeline()
        
    except KeyboardInterrupt:
        logging.info("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
