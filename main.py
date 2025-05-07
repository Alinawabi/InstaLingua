# I want to build a full-featured Python web app called "InstaLingua" that:
# 🎯 Goal:
# A simple Streamlit app to turn any public Instagram video (English audio) into a downloadable Persian-subtitled video.
# This tool should:
# 1. Take a public Instagram video URL as input
# 2. Download the video using yt-dlp
# 3. Extract audio using FFmpeg
# 4. Transcribe English audio to text using OpenAI Whisper
# 5. Translate English transcript to Persian using OpenAI GPT API
# 6. Generate .srt subtitles with timestamps
# 7. Burn Persian subtitles into the video using FFmpeg
# 8. Show preview in browser with a download button
# 9. Save outputs (video, audio, transcript, translation, subtitles) into organized folders
# 10. Load OpenAI API key securely from a secrets.toml file
# ✅ Final app must be deployed using Streamlit and runnable on Raspberry Pi 4 with minimal UI.

# ✅ PHASE 1: Project Setup
# - Create folders: downloads/, audio/, transcripts/, translations/, subtitles/, final/, logs/
# - Create secrets.toml file for storing API key securely
# - Create requirements.txt with:
#   yt-dlp, ffmpeg-python, openai, whisper, srt, toml, streamlit, python-dotenv, pathlib

# ✅ PHASE 2: Instagram Download + Audio Extraction
# - User pastes Instagram video URL
# - Use yt-dlp to download video to downloads/
# - Use ffmpeg to convert video to mono 16kHz .wav audio to audio/

# ✅ PHASE 3: Transcription with Whisper
# - Use tiny or base Whisper model
# - Transcribe .wav to English .txt in transcripts/
# - Keep timestamps for subtitle alignment

# ✅ PHASE 4: Translation to Persian
# - Use OpenAI's Chat API with model "gpt-3.5-turbo"
# - Translate entire transcript to Persian
# - Store in translations/ with matching filename

# ✅ PHASE 5: Generate Timed Persian Subtitles (.srt)
# - Align original timestamps with translated lines
# - Output .srt file using srt library
# - Save to subtitles/

# ✅ PHASE 6: Burn Persian Subtitles into the Video
# - Use FFmpeg to embed Persian subtitles into original video
# - Save to final/ as output.mp4

# ✅ PHASE 7: Streamlit Web UI
# - Streamlit interface with:
#   - Input box for Instagram URL
#   - Progress indicators for each step
#   - Preview final video
#   - Download buttons for: subtitled video, Persian transcript, English transcript
# - Optionally support uploading a local .mp4 file too

# ✅ EXTRAS:
# - Show logs/errors in Streamlit sidebar
# - Style app with emoji, clear steps, minimal layout
# - Ensure Persian subtitles show correctly: load NotoNaskh font or Arial Unicode if needed
# - Make code modular: each step in its own function

# 🚀 GOAL: A clean, deployable Streamlit app (`streamlit run app.py`) that:
# - Runs on Raspberry Pi 4 or small VPS
# - Converts English Instagram videos into professional Persian-subtitled content
# - Requires only OpenAI API key and public IG URL to use
# - Includes `requirements.txt`, `secrets.toml`, and `.streamlit/config.toml` for font/locale settings

# ✅ START NOW with Phase 1 and generate full working code step by step.

import os
from pathlib import Path
import toml
import yt_dlp
import subprocess
import re
from datetime import datetime
import whisper
import json
from openai import OpenAI
import srt
from datetime import timedelta
import shutil

# --- FFmpeg Configuration ---
def get_ffmpeg_path() -> str:
    """
    Get the path to ffmpeg executable.
    Checks in this order:
    1. Custom path from settings
    2. System PATH
    3. Common installation locations
    
    Returns:
        str: Path to ffmpeg executable
        
    Raises:
        FileNotFoundError: If ffmpeg is not found
    """
    # Check if ffmpeg path is configured in settings
    try:
        settings = toml.load('settings.toml')
        custom_path = settings.get('ffmpeg', {}).get('path')
        if custom_path and Path(custom_path).exists():
            return str(custom_path)
    except FileNotFoundError:
        pass  # settings.toml doesn't exist, continue with default checks
    
    # Check if ffmpeg is in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
        
    # Check common installation locations
    common_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',  # Windows default
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        '/usr/bin/ffmpeg',  # Linux
        '/usr/local/bin/ffmpeg',  # macOS
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
            
    raise FileNotFoundError(
        "❌ FFmpeg not found! Please install FFmpeg:\n"
        "1. Download from https://ffmpeg.org/download.html\n"
        "2. Add to PATH or configure path in settings.toml:\n"
        "   [ffmpeg]\n"
        "   path = 'C:/path/to/ffmpeg.exe'"
    )

def validate_ffmpeg():
    """
    Validate that FFmpeg is installed and working.
    Raises FileNotFoundError if FFmpeg is not found or not working.
    """
    try:
        ffmpeg_path = get_ffmpeg_path()
        # Test FFmpeg by getting its version
        result = subprocess.run(
            [ffmpeg_path, '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ FFmpeg found: {ffmpeg_path}")
        return ffmpeg_path
    except subprocess.CalledProcessError:
        raise FileNotFoundError("❌ FFmpeg installation appears to be corrupted. Please reinstall.")
    except FileNotFoundError as e:
        raise e

# --- Phase 1: Project Setup ---
# Define required folders
FOLDERS = [
    'downloads',
    'audio',
    'transcripts',
    'translations',
    'subtitles',
    'final',
    'logs',
]

# Create folders if they don't exist
def setup_folders():
    for folder in FOLDERS:
        Path(folder).mkdir(parents=True, exist_ok=True)

# Load OpenAI API key from secrets.toml
def load_api_key(secrets_path='secrets.toml'):
    if not Path(secrets_path).exists():
        raise FileNotFoundError(f"{secrets_path} not found. Please create it with your OpenAI API key.")
    secrets = toml.load(secrets_path)
    api_key = secrets.get('openai', {}).get('api_key')
    if not api_key:
        raise ValueError("OpenAI API key not found in secrets.toml. Please add it under [openai] section.")
    return api_key

def is_valid_instagram_url(url):
    pattern = r'^https?://(?:www\.)?instagram\.com/(?:p|reel)/[A-Za-z0-9_-]+/?'
    return bool(re.match(pattern, url))

def download_instagram_video(url):
    """
    Download a video from Instagram using yt-dlp.
    
    Args:
        url (str): Public Instagram video URL
        
    Returns:
        str: Path to the downloaded video file
    """
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'best',  # Best quality
        'outtmpl': 'downloads/%(id)s.%(ext)s',  # Output template
        'quiet': True,  # Suppress output
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first to get the ID
            info = ydl.extract_info(url, download=False)
            video_id = info['id']
            
            # Download the video
            ydl.download([url])
            
            # Get the downloaded file path
            video_path = Path('downloads') / f"{video_id}.mp4"
            
            if not video_path.exists():
                raise FileNotFoundError("Video download failed")
                
            return str(video_path)
            
    except Exception as e:
        raise Exception(f"Failed to download Instagram video: {str(e)}")

def extract_audio(video_path: str, audio_output_dir="audio") -> str:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path (str): Path to the video file
        audio_output_dir (str): Directory to save the audio file
        
    Returns:
        str: Path to the extracted audio file
        
    Raises:
        FileNotFoundError: If video file doesn't exist or ffmpeg is not found
        ValueError: If paths are invalid
        subprocess.CalledProcessError: If ffmpeg fails
    """
    # Validate FFmpeg installation first
    ffmpeg_path = validate_ffmpeg()
    
    # Convert to Path objects and resolve to absolute paths
    video_path = Path(video_path).resolve()
    audio_output_dir = Path(audio_output_dir).resolve()
    
    # Validate paths
    if not video_path.exists():
        raise FileNotFoundError(f"❌ Video file not found: {video_path}")
    
    if not audio_output_dir.exists():
        audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create audio filename (same as video but with .wav extension)
    audio_path = audio_output_dir / f"{video_path.stem}.wav"
    
    print(f"📁 Video path: {video_path}")
    print(f"📁 Audio output path: {audio_path}")
    
    # FFmpeg command to extract audio as mono 16kHz WAV
    ffmpeg_cmd = [
        ffmpeg_path,  # Use validated ffmpeg path
        '-y',  # Overwrite output file if exists
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ac', '1',  # Mono
        '-ar', '16000',  # 16kHz
        str(audio_path)
    ]
    
    try:
        # Run FFmpeg command
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"❌ Audio extraction failed - output file not created: {audio_path}")
            
        print(f"✅ Audio extracted successfully to: {audio_path}")
        return str(audio_path)
        
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ FFmpeg failed to extract audio:\nCommand: {' '.join(ffmpeg_cmd)}\nError: {e.stderr}"
        raise Exception(error_msg)
    except Exception as e:
        raise Exception(f"❌ Audio extraction failed: {str(e)}")

def transcribe_audio(audio_path):
    """
    Transcribe audio file to English text using Whisper.
    
    Args:
        audio_path (str): Path to the audio file (.wav)
        
    Returns:
        str: Path to the transcript file (.txt)
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Create transcript filename (same as audio but with .txt extension)
    transcript_path = Path('transcripts') / f"{audio_path.stem}.txt"
    
    try:
        # Load the base model (smaller and faster than large)
        model = whisper.load_model("base")
        
        # Transcribe the audio
        result = model.transcribe(
            str(audio_path),
            language="en",  # Force English
            fp16=False,     # Use CPU
            verbose=False   # Suppress progress
        )
        
        # Save transcript to file
        with open(transcript_path, 'w', encoding='utf-8') as f:
            # Write the full text
            f.write(result["text"])
            
            # Also save the segments with timestamps for later use
            segments_path = Path('transcripts') / f"{audio_path.stem}_segments.json"
            with open(segments_path, 'w', encoding='utf-8') as sf:
                json.dump(result["segments"], sf, indent=2)
        
        if not transcript_path.exists():
            raise FileNotFoundError("Transcript file creation failed")
            
        return str(transcript_path)
        
    except Exception as e:
        raise Exception(f"Whisper transcription failed: {str(e)}")

def translate_transcript(transcript_path):
    """
    Translate English transcript to Persian using OpenAI GPT API.
    
    Args:
        transcript_path (str): Path to the English transcript file
        
    Returns:
        str: Path to the Persian translation file
    """
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    
    # Create translation filename (same as transcript but in translations folder)
    translation_path = Path('translations') / f"{transcript_path.stem}.txt"
    
    try:
        # Load the transcript text
        with open(transcript_path, 'r', encoding='utf-8') as f:
            english_text = f.read()
        
        # Initialize OpenAI client
        client = OpenAI(api_key=load_api_key())
        
        # Prepare the translation prompt
        prompt = f"""Translate the following English text to Persian (Farsi). 
        Maintain the same meaning and tone, but adapt it naturally for Persian speakers.
        Only return the Persian translation, no explanations or notes.

        English text:
        {english_text}"""
        
        # Call GPT API for translation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional English to Persian translator. Translate the given text accurately and naturally."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent translations
            max_tokens=2000   # Adjust based on expected text length
        )
        
        # Extract the translation
        persian_text = response.choices[0].message.content.strip()
        
        # Save translation to file
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(persian_text)
        
        if not translation_path.exists():
            raise FileNotFoundError("Translation file creation failed")
            
        return str(translation_path)
        
    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")

def generate_srt(transcript_path, translation_path):
    """
    Generate timed Persian subtitles (.srt) from transcript segments and translation.
    
    Args:
        transcript_path (str): Path to the English transcript file
        translation_path (str): Path to the Persian translation file
        
    Returns:
        str: Path to the generated .srt file
    """
    transcript_path = Path(transcript_path)
    translation_path = Path(translation_path)
    
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    if not translation_path.exists():
        raise FileNotFoundError(f"Translation file not found: {translation_path}")
    
    try:
        # Load the segments with timestamps
        segments_path = transcript_path.parent / f"{transcript_path.stem}_segments.json"
        if not segments_path.exists():
            raise FileNotFoundError(f"Segments file not found: {segments_path}")
            
        with open(segments_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        # Load the Persian translation
        with open(translation_path, 'r', encoding='utf-8') as f:
            persian_text = f.read()
        
        # Split Persian text into sentences (rough approximation)
        persian_sentences = [s.strip() for s in persian_text.split('.') if s.strip()]
        
        # Create SRT subtitles
        subtitles = []
        for i, segment in enumerate(segments):
            # Convert timestamps to timedelta
            start_time = timedelta(seconds=segment['start'])
            end_time = timedelta(seconds=segment['end'])
            
            # Get corresponding Persian text
            persian_text = persian_sentences[i] if i < len(persian_sentences) else ""
            
            # Create subtitle entry
            subtitle = srt.Subtitle(
                index=i+1,
                start=start_time,
                end=end_time,
                content=persian_text
            )
            subtitles.append(subtitle)
        
        # Generate SRT content
        srt_content = srt.compose(subtitles)
        
        # Save to file
        srt_path = Path('subtitles') / f"{transcript_path.stem}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        if not srt_path.exists():
            raise FileNotFoundError("SRT file creation failed")
            
        return str(srt_path)
        
    except Exception as e:
        raise Exception(f"SRT generation failed: {str(e)}")

def overlay_subtitles(video_path: str, subtitle_path: str, output_path: str) -> str:
    """
    Burn subtitles into the video using FFmpeg with libass.
    
    Args:
        video_path (str): Path to the input video file (.mp4)
        subtitle_path (str): Path to the subtitle file (.srt)
        output_path (str): Path where the output video will be saved
        
    Returns:
        str: Path to the output video file
        
    Raises:
        FileNotFoundError: If input files don't exist or FFmpeg is not found
        ValueError: If paths are invalid
        subprocess.CalledProcessError: If FFmpeg fails
    """
    # Validate FFmpeg installation
    ffmpeg_path = validate_ffmpeg()
    
    # Convert to Path objects and resolve to absolute paths
    video_path = Path(video_path).resolve()
    subtitle_path = Path(subtitle_path).resolve()
    output_path = Path(output_path).resolve()
    
    # Validate input files
    if not video_path.exists():
        raise FileNotFoundError(f"❌ Video file not found: {video_path}")
    if not subtitle_path.exists():
        raise FileNotFoundError(f"❌ Subtitle file not found: {subtitle_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input video: {video_path}")
    print(f"📁 Input subtitles: {subtitle_path}")
    print(f"📁 Output video: {output_path}")
    
    # Convert paths to forward slashes for FFmpeg
    video_path_str = str(video_path).replace('\\', '/')
    subtitle_path_str = str(subtitle_path).replace('\\', '/')
    output_path_str = str(output_path).replace('\\', '/')
    
    # Create ASS style file
    style_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Save ASS style file
    style_path = subtitle_path.parent / f"{subtitle_path.stem}_style.ass"
    with open(style_path, 'w', encoding='utf-8') as f:
        f.write(style_content)
    
    # FFmpeg command using libass
    ffmpeg_cmd = [
        ffmpeg_path,
        '-y',  # Overwrite output file if exists
        '-i', video_path_str,  # Input video
        '-vf', f'ass={style_path}:force_style=\'FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=1,BorderStyle=3\'',  # Use libass filter
        '-c:a', 'copy',  # Copy audio without re-encoding
        output_path_str  # Output path
    ]
    
    try:
        # Run FFmpeg command
        print(f"🔧 Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=True  # Use shell=True for Windows path handling
        )
        
        if not output_path.exists():
            raise FileNotFoundError(f"❌ Video processing failed - output file not created: {output_path}")
            
        print(f"✅ Subtitles burned successfully to: {output_path}")
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ FFmpeg failed to burn subtitles:\nCommand: {' '.join(ffmpeg_cmd)}\nError: {e.stderr}"
        raise Exception(error_msg)
    except Exception as e:
        raise Exception(f"❌ Subtitle burning failed: {str(e)}")
    finally:
        # Clean up temporary style file
        if style_path.exists():
            style_path.unlink()

if __name__ == "__main__":
    try:
        # Setup project structure
        setup_folders()
        
        # Validate FFmpeg installation
        print("\n🔍 Checking FFmpeg installation...")
        validate_ffmpeg()
        
        # Load API key
        api_key = load_api_key()
        print("✅ Project setup complete. API key loaded.")

        # Test with a sample Instagram URL
        test_url = "https://www.instagram.com/p/DEFrnmNP-8k/"  # Replace with a real public Instagram URL
        if not is_valid_instagram_url(test_url):
            raise ValueError("Invalid Instagram URL")

        print("\n1. Downloading video...")
        video_path = download_instagram_video(test_url)
        video_id = Path(video_path).stem
        print(f"✅ Video downloaded to: downloads/{video_id}.mp4")

        print("\n2. Extracting audio...")
        audio_path = extract_audio(video_path)
        print(f"✅ Audio extracted to: audio/{video_id}.wav")

        print("\n3. Transcribing audio...")
        transcript_path = transcribe_audio(audio_path)
        print(f"✅ Transcript saved to: transcripts/{video_id}.txt")

        print("\n4. Translating transcript...")
        translation_path = translate_transcript(transcript_path)
        print(f"✅ Translation saved to: translations/{video_id}.txt")

        print("\n5. Generating subtitles...")
        srt_path = generate_srt(transcript_path, translation_path)
        print(f"✅ Subtitles saved to: subtitles/{video_id}.srt")

        # Verify all required files exist
        required_files = [
            f"downloads/{video_id}.mp4",
            f"audio/{video_id}.wav",
            f"transcripts/{video_id}.txt",
            f"translations/{video_id}.txt",
            f"subtitles/{video_id}.srt"
        ]
        
        print("\n🔍 Verifying output files...")
        all_files_exist = True
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"✅ Found: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
                all_files_exist = False
        
        if all_files_exist:
            print("\n✨ All processing steps completed successfully!")
        else:
            print("\n⚠️ Some files are missing. Please check the errors above.")

        # Subtitle burning step is temporarily disabled
        """
        print("\n6. Burning subtitles into video...")
        output_path = Path('output') / f"{video_id}_with_subs.mp4"
        final_video = overlay_subtitles(video_path, srt_path, str(output_path))
        print(f"✅ Final video with subtitles saved to: {final_video}")
        """

    except FileNotFoundError as e:
        print(f"❌ File not found: {str(e)}")
    except ValueError as e:
        print(f"❌ Invalid input: {str(e)}")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
