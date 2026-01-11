# Phantom High-Speed Video Processing Pipeline

## Overview
This project consists of two main components for processing high-speed Phantom camera recordings:
1. `adaptive_converter.py` - Converts proprietary .cine files to lossless .avi format
2. `adaptive_analyzer.py` - Performs frame-by-frame facial affect analysis on converted videos

## System Requirements

### Hardware
- Apple Silicon Mac (optimized for M3 Pro with 11-core CPU)
- Recommended minimum 16GB RAM
- SSD with sufficient space for both .cine and .avi files (>3TB recommended)

### Software
- macOS Sonoma or later
- Python 3.9+
- FFmpeg with FFV1 codec support
- OpenCV
- FER (Facial Expression Recognition)

### Python Environment Setup
```bash
# Create and activate virtual environment in MyDevelopment folder
cd ~/MyDevelopment
python -m venv phantom_processing
source phantom_processing/bin/activate

# Install required packages
pip install opencv-python
pip install fer
pip install pandas
pip install psutil
pip install numpy
```

## Component 1: Adaptive Converter (adaptive_converter.py)

### Purpose
Converts Phantom Camera .cine files to lossless .avi format while maintaining system responsiveness for other tasks.

### Key Features
- Lossless conversion preserving 1000fps frame rate
- Adaptive CPU usage (2-5 cores)
- System-load aware processing
- Auto-pause/resume based on system activity
- Comprehensive logging and error recovery

### Resource Management
- Starting processes: 3 cores
- Minimum processes: 2 cores
- Maximum processes: 5 cores
- Reduces load when system CPU > 75% or memory > 80%
- Increases resources when CPU < 50% and memory < 60%

### Directory Structure Requirements
```
/path/to/source/
├── 2-xxx/
│   └── Video Files/
│       ├── recording.cine
│       └── Follow-Up/
│           └── followup.cine
├── 2-yyy/
└── ...
```

### Usage
```python
from adaptive_converter import AdaptiveConverter

converter = AdaptiveConverter(
    input_dir="/path/to/cine/files",
    output_dir="/path/to/output"
)
converter.process_directory()
```

## Component 2: Adaptive Affect Analyzer (adaptive_analyzer.py)

### Purpose
Performs facial affect analysis on high-speed video recordings with adaptive resource usage.

### Key Features
- Frame-by-frame emotion detection
- Dynamic resource management
- Batch processing with variable sizing
- Automatic progress saving
- Background-friendly operation

### Resource Management
- Starting processes: 3 cores
- Minimum processes: 2 cores
- Maximum processes: 5 cores
- Initial batch size: 20 frames
- Batch size range: 15-25 frames
- Reduces resources when CPU > 75% or memory > 80%
- Increases resources when CPU < 50% and memory < 60%

### Processing Strategy
- Chunks video into sections based on available cores
- Processes chunks in parallel
- Saves results every 1000 frames
- Automatic garbage collection between batches
- System load checks every 100 frames

### Input Requirements
- Lossless .avi files (from adaptive_converter.py)
- Directory structure matching converter output
- Minimum 1920x1080 resolution recommended

### Usage
```python
from adaptive_analyzer import AdaptiveAffectAnalyzer

analyzer = AdaptiveAffectAnalyzer(
    output_dir="/path/to/analysis/output",
    batch_size=20
)
analyzer.process_all_videos("/path/to/avi/files")
```

### Output Structure
```
output_dir/
├── logs/
│   └── analysis_YYYYMMDD_HHMM.log
├── participant_id/
│   └── video_name_analysis.csv
└── master_analysis.csv
```

### CSV Output Columns
- participant_id: Participant identifier
- video: Source video filename
- frame: Frame number
- timestamp: Time in seconds
- face_number: Face identifier if multiple detected
- total_faces: Number of faces in frame
- x, y: Face box coordinates
- faces_found: Boolean detection flag
- Emotion scores: angry, disgust, fear, happy, sad, surprise, neutral

## Error Handling and Recovery
Both components include:
- Automatic retry mechanisms
- Detailed logging of all operations
- Progress saving and recovery points
- Safe interruption capabilities
- Resource monitoring and adaptation

## Performance Considerations
- Conversion time: ~10-12 minutes per 30-second video
- Analysis time: ~20-25 minutes per 30-second video
- System remains responsive for:
  - Web browsing
  - Document editing
  - Code editing
  - Email
  - General productivity tasks

## Limitations
- Processing time varies with system load
- Requires significant storage space
- Memory usage scales with video resolution
- Some tasks may slow processing

## Best Practices
1. Run conversion process first, verify outputs
2. Keep system plugged in and well-ventilated
3. Monitor logs for issues
4. Regular backup of analysis results
5. Consider processing during off-hours
6. Close memory-intensive applications during processing

## Dependencies
```
fer==22.4.0
opencv-python==4.8.0
pandas==2.1.0
psutil==5.9.0
numpy==1.23.5
```

Would you like me to expand on any section or add additional details?
