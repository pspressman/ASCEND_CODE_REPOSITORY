from pathlib import Path
import subprocess
import os
from multiprocessing import Pool
import logging
from datetime import datetime
import psutil
import time
import signal
import pycine
import cv2
import numpy as np

class AdaptiveConverter:
    def __init__(self, input_dir="/Volumes/Backup1/VidTester", 
                 output_dir="/Volumes/Backup1/FastFaceExp-conversion_output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.setup_logging()
        
        # Adjusted for 32GB RAM
        self.current_processes = 4    # Start with 4 cores
        self.min_processes = 2        # Keep minimum at 2
        self.max_processes = 6        # Maximum 6 cores
        
    def setup_logging(self):
        """Setup logging to track conversion process"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            filename=log_dir / f'conversion_{datetime.now():%Y%m%d_%H%M}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def check_system_load(self):
        """Monitor resources with 32GB RAM-appropriate thresholds"""
        cpu_percent = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        
        if cpu_percent > 75 or mem.percent > 85:
            self.current_processes = max(self.min_processes, self.current_processes - 1)
            logging.info(f"Reducing to {self.current_processes} processes due to load")
            time.sleep(5)
            
        elif cpu_percent < 50 and mem.percent < 70:
            if self.current_processes < self.max_processes:
                time.sleep(10)
                if psutil.cpu_percent() < 50:
                    self.current_processes = min(self.max_processes, self.current_processes + 1)
                    logging.info(f"Increasing to {self.current_processes} processes")
            
        return cpu_percent, mem.percent

    def convert_cine_file(self, args):
        """Convert single file using pycine with balanced resource usage"""
        cine_path, participant_id = args
        
        try:
            participant_dir = self.output_dir / participant_id
            participant_dir.mkdir(exist_ok=True, parents=True)
            output_path = participant_dir / f"{cine_path.stem}.avi"
            
            if output_path.exists():
                logging.info(f"Skipping {cine_path.name} - already converted")
                return True
            
            # Check system load before starting
            cpu_percent, mem_percent = self.check_system_load()
            if cpu_percent > 85 or mem_percent > 90:
                logging.info(f"System load too high ({cpu_percent}% CPU, {mem_percent}% RAM). Waiting...")
                time.sleep(60)
                return "retry"
            
            logging.info(f"Starting conversion of {cine_path.name}")
            
            with pycine.Cine(str(cine_path)) as cine:
                # Get video properties
                first_frame = cine.get_frame(0)
                height, width = first_frame.shape[:2]
                total_frames = cine.image_count
                
                logging.info(f"Video properties: {width}x{height}, {total_frames} frames")
                
                # Initialize video writer
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*'FFV1'),
                    1000,  # Phantom camera fps
                    (width, height),
                    isColor=True
                )
                
                # Process frames with system load monitoring
                for frame_idx in range(total_frames):
                    if frame_idx % 1000 == 0:
                        logging.info(f"Processing frame {frame_idx}/{total_frames}")
                        cpu_percent, mem_percent = self.check_system_load()
                        if cpu_percent > 80:
                            time.sleep(2)  # Brief pause if system is stressed
                    
                    frame = cine.get_frame(frame_idx)
                    if len(frame.shape) == 2:  # Convert grayscale to BGR if needed
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    writer.write(frame)
                
                writer.release()
                logging.info(f"Successfully converted {cine_path.name}")
                return True
                
        except Exception as e:
            logging.error(f"Exception processing {cine_path.name}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def process_directory(self):
        """Process all .cine files recursively"""
        conversion_tasks = []

        try:
            # Verify input directory
            if not self.input_dir.exists():
                logging.error(f"Input directory does not exist: {self.input_dir}")
                return False

            # Find all participant folders
            participant_pattern = "[2-9]-[0-9][0-9][0-9]"
            participant_folders = list(self.input_dir.glob(participant_pattern))
            logging.info(f"Found {len(participant_folders)} participant folders")
            
            # Find all .cine files recursively
            for participant_folder in participant_folders:
                logging.info(f"\nExamining participant folder: {participant_folder}")
                
                cine_files = list(participant_folder.rglob("*.[cC][iI][nN][eE]"))
                if cine_files:
                    logging.info(f"Found {len(cine_files)} .cine files in {participant_folder}")
                    for cine_file in cine_files:
                        logging.info(f"Adding task for file: {cine_file}")
                        conversion_tasks.append((cine_file, participant_folder.name))
                else:
                    logging.warning(f"No .cine files found in {participant_folder}")

            total_files = len(conversion_tasks)
            if total_files == 0:
                logging.error("No .cine files found for conversion!")
                return False
                
            logging.info(f"\nPreparing to convert {total_files} files")
            
            # Process files with parallel processing and resource management
            completed = []
            retry_queue = []
            
            while conversion_tasks or retry_queue:
                cpu_percent, mem_percent = self.check_system_load()
                current_tasks = retry_queue + conversion_tasks[:self.current_processes]
                retry_queue = []
                
                with Pool(self.current_processes) as pool:
                    for result in pool.imap_unordered(self.convert_cine_file, current_tasks):
                        if result == "retry":
                            retry_queue.append(current_tasks.pop(0))
                        else:
                            completed.append(result)
                            if conversion_tasks:
                                conversion_tasks.pop(0)
                        
                        logging.info(f"Processed {len(completed)}/{total_files} files")
                
                if cpu_percent > 75:
                    logging.info("System load high, taking 5-minute break")
                    time.sleep(300)
            
            successful = sum(1 for x in completed if x is True)
            logging.info(f"\nConversion complete. Successfully converted {successful}/{total_files} files")
            return successful == total_files
            
        except Exception as e:
            logging.error(f"Error in process_directory: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False

def main():
    converter = AdaptiveConverter()
    success = converter.process_directory()
    
    if success:
        print("All conversions completed successfully")
    else:
        print("Some conversions failed - check logs for details")

if __name__ == "__main__":
    main()