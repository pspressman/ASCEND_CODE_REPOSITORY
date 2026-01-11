#!/usr/bin/env python3
"""
Windows OpenFace Batch Processor
Action Unit extraction for clinical videos
"""

import os
import sys
import json
import time
import subprocess
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import psutil

class WindowsOpenFaceProcessor:
    """
    OpenFace processor for Windows
    Features: Resume capability, resource management, crash recovery
    """
    
    def __init__(self):
        """Initialize processor with configuration"""
        # Windows paths - UPDATE THESE IF NEEDED
        self.openface_exe = Path(r"C:\path\to\user\OneDrive\Desktop\Open Face\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe")
        self.input_dir = Path(r"C:\path\to\user\OneDrive\Desktop\Open Face\converted_videos")
        self.output_dir = Path(r"C:\path\to\user\OneDrive\Desktop\Open Face\openface_results")
        self.state_dir = Path(r"C:\path\to\user\OneDrive\Desktop\Open Face\logs")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.load_configuration()
        
        # Setup logging
        self.setup_logging()
        
        # Processing state
        self.processed_videos = set()
        self.current_processing = {}
        self.shutdown_requested = False
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(self.config['resource_limits'])
        
        # Load processing state
        self.load_processing_state()
        
    def load_configuration(self):
        """Load processing configuration"""
        self.config = {
            "openface_settings": {
                "extract_features": {
                    "action_units": True, 
                    "head_pose": True,
                    "gaze_vectors": True,
                    "facial_landmarks": True
                },
                "quality_settings": {"min_confidence": 0.5},
                "output_format": {"detailed_csv": True}
            },
            "processing_settings": {
                "video_extensions": [".avi", ".AVI", ".mp4", ".MP4"]
            },
            "resource_limits": {"max_cpu_percent": 85, "max_memory_gb": 6},
            "state_management": {"save_frequency_seconds": 30}
        }
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.state_dir / "openface_processor.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('OpenFaceProcessor')
        self.logger.info("="*60)
        self.logger.info("Windows OpenFace processor initialized")
        self.logger.info("="*60)
    
    def load_processing_state(self):
        """Load processing state for resume capability"""
        state_file = self.state_dir / "openface_processing_state.json"
        
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.processed_videos = set(state.get('processed_videos', []))
                    self.current_processing = state.get('current_processing', {})
                self.logger.info(f"Loaded state: {len(self.processed_videos)} videos already completed")
            except Exception as e:
                self.logger.error(f"Error loading state: {e}")
                self.processed_videos = set()
    
    def save_processing_state(self):
        """Save current processing state"""
        state = {
            'processed_videos': list(self.processed_videos),
            'current_processing': self.current_processing,
            'last_update': datetime.now().isoformat(),
            'total_processed': len(self.processed_videos)
        }
        
        state_file = self.state_dir / "openface_processing_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def find_video_files(self) -> List[Tuple[str, Path]]:
        """Find all video files in participant directories"""
        video_files = []
        
        self.logger.info(f"Scanning for videos in: {self.input_dir}")
        
        try:
            # Look for participant directories (2-001 through 2-999 format)
            for participant_dir in sorted(self.input_dir.glob("[2-9]-[0-9][0-9][0-9]")):
                participant_id = participant_dir.name
                
                # Find videos directly in participant folder
                for ext in self.config['processing_settings']['video_extensions']:
                    for video_file in participant_dir.glob(f"*{ext}"):
                        video_files.append((participant_id, video_file))
                        self.logger.info(f"  Found: {participant_id}/{video_file.name}")
            
            # If no participant folders found, look for videos directly
            if not video_files:
                self.logger.info("No participant folders found, searching all subdirectories...")
                for ext in self.config['processing_settings']['video_extensions']:
                    for video_file in self.input_dir.rglob(f"*{ext}"):
                        # Use parent folder name as participant ID
                        participant_id = video_file.parent.name
                        video_files.append((participant_id, video_file))
                        self.logger.info(f"  Found: {participant_id}/{video_file.name}")
        
        except Exception as e:
            self.logger.error(f"Error scanning for videos: {e}")
            
        self.logger.info(f"\nTotal videos found: {len(video_files)}\n")
        return video_files
    
    def create_openface_command(self, video_path: Path, output_dir: Path) -> List[str]:
        """Create OpenFace command for video processing"""
        settings = self.config['openface_settings']
        
        cmd = [
            str(self.openface_exe),
            '-f', str(video_path),
            '-out_dir', str(output_dir)
        ]
        
        # Add feature extraction flags
        if settings['extract_features'].get('action_units', True):
            cmd.append('-aus')
        
        if settings['extract_features'].get('head_pose', True):
            cmd.append('-pose')
        
        if settings['extract_features'].get('gaze_vectors', True):
            cmd.append('-gaze')
        
        if settings['extract_features'].get('facial_landmarks', True):
            cmd.append('-2Dfp')
            cmd.append('-3Dfp')
        
        # Quality settings
        quality = settings['quality_settings']
        if 'min_confidence' in quality:
            cmd.extend(['-q', str(quality['min_confidence'])])
        
        return cmd
    
    def process_video_openface(self, video_path: Path, participant_id: str) -> bool:
        """Process single video with OpenFace"""
        video_name = video_path.stem
        video_key = f"{participant_id}_{video_name}"
        
        # Skip if already processed
        if video_key in self.processed_videos:
            self.logger.info(f"✓ Skipping {video_key} - already completed")
            return True
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {video_key}")
        self.logger.info(f"{'='*60}")
        
        # Create output directory
        try:
            participant_output = self.output_dir / participant_id
            participant_output.mkdir(exist_ok=True)
            video_output = participant_output / video_name
            video_output.mkdir(exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating output directory: {e}")
            return False
        
        # Update current processing state
        self.current_processing[video_key] = {
            'start_time': datetime.now().isoformat(),
            'participant_id': participant_id,
            'video_name': video_name,
            'status': 'processing'
        }
        
        try:
            # Create OpenFace command
            cmd = self.create_openface_command(video_path, video_output)
            
            self.logger.info(f"Running OpenFace...")
            
            # Run OpenFace
            start_time = time.time()
            
            # Set working directory to OpenFace directory for model access
            openface_dir = self.openface_exe.parent
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(openface_dir)
            )
            
            # Monitor process
            while process.poll() is None:
                if self.shutdown_requested:
                    process.terminate()
                    self.logger.info(f"Terminated {video_key} due to shutdown request")
                    return False
                
                # Check resource usage
                if not self.resource_monitor.check_resources():
                    self.logger.warning("Resource limits exceeded, pausing...")
                    time.sleep(10)
                
                time.sleep(2)
            
            # Get results
            stdout, stderr = process.communicate()
            processing_time = time.time() - start_time
            
            if process.returncode == 0:
                self.logger.info(f"✓ OpenFace completed in {processing_time:.1f}s")
                
                # Process and save results
                success = self.process_openface_output(video_output, participant_id, video_name)
                
                if success:
                    # Mark as completed
                    self.processed_videos.add(video_key)
                    self.current_processing.pop(video_key, None)
                    self.save_processing_state()
                    return True
                else:
                    self.logger.error(f"Failed to process OpenFace output for {video_key}")
                    return False
            else:
                self.logger.error(f"✗ OpenFace failed for {video_key}")
                if stderr:
                    self.logger.error(f"Error: {stderr[:500]}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception processing {video_key}: {e}")
            self.current_processing.pop(video_key, None)
            return False
    
    def process_openface_output(self, output_dir: Path, participant_id: str, video_name: str) -> bool:
        """Process and organize OpenFace output files"""
        try:
            # Find OpenFace CSV output
            csv_files = list(output_dir.glob("*.csv"))
            
            if not csv_files:
                self.logger.error(f"No CSV output found in {output_dir}")
                return False
            
            # Process main CSV file
            main_csv = csv_files[0]
            
            # Read and validate
            df = pd.read_csv(main_csv)
            
            # Create standardized output filename
            output_filename = f"{participant_id}_{video_name}_openface_features.csv"
            final_output = output_dir / output_filename
            
            # Save with consistent formatting
            df.to_csv(final_output, index=False)
            
            # Create metadata file
            metadata = {
                'participant_id': participant_id,
                'video_name': video_name,
                'processing_date': datetime.now().isoformat(),
                'openface_version': '2.2.0_windows_binary',
                'total_frames': len(df),
                'features_extracted': list(df.columns),
                'processing_config': self.config['openface_settings']
            }
            
            metadata_file = output_dir / f"{participant_id}_{video_name}_openface_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✓ Processed {len(df)} frames")
            self.logger.info(f"  Output: {final_output}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing OpenFace output: {e}")
            return False
    
    def run_processing_batch(self):
        """Run the main processing batch"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting OpenFace batch processing")
        self.logger.info("="*60 + "\n")
        
        # Verify OpenFace executable exists
        if not self.openface_exe.exists():
            self.logger.error(f"ERROR: OpenFace executable not found at:")
            self.logger.error(f"  {self.openface_exe}")
            print(f"\nERROR: Cannot find OpenFace executable!")
            print(f"Expected location: {self.openface_exe}")
            print("Please check that OpenFace was extracted correctly")
            return
            
        # Test input directory
        if not self.input_dir.exists():
            self.logger.error(f"ERROR: Cannot access input directory:")
            self.logger.error(f"  {self.input_dir}")
            print(f"\nERROR: Cannot find input directory!")
            print(f"Expected location: {self.input_dir}")
            print("Make sure you ran the converter script first")
            return
        
        # Find all videos
        video_files = self.find_video_files()
        
        if not video_files:
            self.logger.error("No video files found!")
            print("\nERROR: No videos found!")
            print("Make sure the converter script completed successfully")
            return
        
        # Filter out already processed
        remaining_videos = [
            (pid, path) for pid, path in video_files
            if f"{pid}_{path.stem}" not in self.processed_videos
        ]
        
        self.logger.info(f"Videos to process: {len(remaining_videos)}")
        self.logger.info(f"Already completed: {len(self.processed_videos)}\n")
        
        if not remaining_videos:
            self.logger.info("All videos already processed!")
            print("\n✓ All videos already processed!")
            return
        
        # Process videos
        successful = 0
        failed = 0
        
        for i, (participant_id, video_path) in enumerate(remaining_videos, 1):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested, stopping processing")
                break
            
            self.logger.info(f"\n[{i}/{len(remaining_videos)}] Processing {participant_id}")
            
            try:
                success = self.process_video_openface(video_path, participant_id)
                
                if success:
                    successful += 1
                    self.logger.info(f"✓ Completed {participant_id}")
                else:
                    failed += 1
                    self.logger.error(f"✗ Failed {participant_id}")
                
                # Save state periodically
                if i % 3 == 0:
                    self.save_processing_state()
                
            except Exception as e:
                failed += 1
                self.logger.error(f"Error processing {participant_id}: {e}")
                continue
        
        # Final state save
        self.save_processing_state()
        
        # Summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PROCESSING COMPLETE")
        self.logger.info(f"Total processed: {len(self.processed_videos)}")
        self.logger.info(f"This session - Successful: {successful}, Failed: {failed}")
        self.logger.info(f"{'='*60}\n")
        
        print(f"\n{'='*60}")
        print(f"✓ OpenFace processing complete!")
        print(f"  Successfully processed: {successful} videos")
        if failed > 0:
            print(f"  Failed: {failed} videos (check logs)")
        print(f"  Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")

class ResourceMonitor:
    """Monitor and enforce resource limits"""
    
    def __init__(self, limits: Dict):
        self.max_cpu_percent = limits.get('max_cpu_percent', 85)
        self.max_memory_gb = limits.get('max_memory_gb', 6)
        
    def check_resources(self) -> bool:
        """Check if current resource usage is within limits"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_gb = psutil.virtual_memory().used / (1024**3)
            
            if cpu_percent > self.max_cpu_percent:
                return False
            if memory_gb > self.max_memory_gb:
                return False
            
            return True
        except:
            return True  # If can't check, assume OK

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("OpenFace Video Processor")
    print("="*60 + "\n")
    
    processor = WindowsOpenFaceProcessor()
    
    try:
        processor.run_processing_batch()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        processor.logger.info("Processing interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        processor.logger.error(f"Fatal error: {e}")
    finally:
        processor.save_processing_state()
        processor.logger.info("Processor shutdown complete")
        print("\nProcessor stopped. Progress has been saved.")
        print("You can run this script again to continue where you left off.\n")
