PhantomConverter: FFmpeg-Based .cine to AVI Converter
Overview
The PhantomConverter script processes .cine files, extracts frames using FFmpeg, and compiles them into AVI files at the original frame rate. It optimizes system resources, supports robust logging, and manages errors effectively for high-speed data processing.

Features
Recursive Processing: Automatically traverses directories to find .cine files for conversion.
System Load Monitoring: Dynamically adjusts process count to optimize performance.
FFmpeg Integration: Uses FFmpeg for frame extraction and AVI conversion.
Robust Logging: Tracks every step and logs errors for troubleshooting.
Temporary File Cleanup: Automatically removes intermediate files after successful conversion.
Customizable Frame Rate: Maintains the original frame rate (e.g., 1000 FPS) during AVI creation.
Requirements
System Installed FFmpeg:

Check FFmpeg availability:
bash
Copy code
ffmpeg -version
Install FFmpeg if necessary:
macOS: brew install ffmpeg
Linux: sudo apt install ffmpeg
Windows: Download FFmpeg
Python Dependencies:

Install required libraries:
bash
Copy code
pip install psutil
How to Use
Set Up Input and Output Directories:

Place .cine files in the input directory.
Define input/output paths in the script or pass them as arguments.
Run the Script:

Execute the Python script:
bash
Copy code
python PhantomConverter.py
Monitor Progress:

Logs will appear in the console and in the logs/ directory under the output path.
Script Workflow
Find .cine Files:

Recursively scans the input directory for .cine files.
Skips files smaller than 10 MB to avoid processing invalid or corrupted files.
Extract Frames:

Uses FFmpeg to extract frames as .tif files:
bash
Copy code
ffmpeg -i <input.cine> -fps_mode passthrough -q:v 2 <output_frames/frame_%08d.tif>
Compile AVI:

Combines extracted frames into a high-quality AVI:
bash
Copy code
ffmpeg -framerate 1000 -i <output_frames/frame_%08d.tif> -c:v ffv1 -pix_fmt bgr24 <output.avi>
Clean Up Temporary Files:

Removes intermediate .tif files after successful AVI creation.
Troubleshooting
Verify FFmpeg Installation:

Confirm FFmpeg is installed and in the system PATH:
bash
Copy code
ffmpeg -version
Check Logs:

Review logs in the logs/ directory for detailed error messages.
Validate Frame Rate:

Confirm the AVI frame rate using FFprobe:
bash
Copy code
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 <output.avi>
Handle Corrupted Files:

Check file size and metadata:
bash
Copy code
ls -lh <file.cine>
ffmpeg -i <file.cine>
Exclude files smaller than 10 MB or re-export from the original source.
Code Snippet Example
Below is the key logic for frame extraction and AVI compilation:

python
Copy code
def extract_frames_with_ffmpeg(self, cine_path, temp_dir):
    """Extract frames from a .cine file using FFmpeg."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = temp_dir / "frame_%08d.tif"

    cmd = [
        'ffmpeg',
        '-i', str(cine_path),
        '-fps_mode', 'passthrough',
        '-q:v', '2',
        str(output_pattern)
    ]

    logging.info(f"Extracting frames from {cine_path.name} to {temp_dir}...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        logging.error(f"FFmpeg frame extraction failed for {cine_path.name}: {result.stderr}")
        return False

    frame_count = len(list(temp_dir.glob("*.tif")))
    if frame_count == 0:
        logging.error(f"No frames were extracted for {cine_path.name}.")
        return False

    logging.info(f"Successfully extracted {frame_count} frames for {cine_path.name}.")
    return True
Validation
Check Extracted Frames:

Ensure .tif frames are saved in the temp_frames/ directory.
Confirm AVI Output:

Verify that the AVI file plays at the correct speed (e.g., 1000 FPS).
Media Playback:

Use VLC or another player for playback.
For slower playback, create a reduced frame rate version:
bash
Copy code
ffmpeg -i <output.avi> -filter:v "setpts=PTS*10" <slower_output.avi>
Future Enhancements
Add support for processing .cine files with different frame rates dynamically.
Integrate file validation for common .cine file errors.
Parallelize FFmpeg tasks for improved performance on multi-core systems.