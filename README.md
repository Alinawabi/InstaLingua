# InstaLingua

A Python tool that converts public Instagram videos into Persian-subtitled content. The tool downloads Instagram videos, extracts audio, transcribes English speech, translates to Persian, and generates subtitles.

## Features

- Download public Instagram videos
- Extract audio from videos
- Transcribe English audio using OpenAI Whisper
- Translate English to Persian using OpenAI GPT
- Generate timed Persian subtitles (.srt)
- (Coming soon) Burn subtitles into video

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed and configured
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/InstaLingua.git
cd InstaLingua
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings:
   - Copy `settings.toml.example` to `settings.toml`
   - Add your FFmpeg path in `settings.toml`
   - Create `secrets.toml` with your OpenAI API key

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Download the Instagram video
2. Extract audio
3. Transcribe to English
4. Translate to Persian
5. Generate subtitles

Output files will be saved in their respective folders:
- `downloads/`: Original videos
- `audio/`: Extracted audio files
- `transcripts/`: English transcripts
- `translations/`: Persian translations
- `subtitles/`: Generated .srt files

## Configuration

### settings.toml
```toml
[ffmpeg]
path = "C:/path/to/ffmpeg.exe"  # Windows
# path = "/usr/bin/ffmpeg"      # Linux/macOS
```

### secrets.toml
```toml
[openai]
api_key = "your-api-key-here"
```

## Project Structure
```
InstaLingua/
├── main.py              # Main script
├── settings.toml        # FFmpeg configuration
├── requirements.txt     # Python dependencies
├── downloads/          # Downloaded videos
├── audio/             # Extracted audio
├── transcripts/       # English transcripts
├── translations/      # Persian translations
├── subtitles/         # Generated subtitles
└── output/           # Final videos (coming soon)
```

## Future Steps

1. **Subtitle Burning**
   - Implement FFmpeg subtitle burning
   - Add support for custom fonts
   - Improve subtitle positioning

2. **Web Interface**
   - Add Streamlit web UI
   - Support for local video uploads
   - Progress tracking
   - Preview functionality

3. **Performance**
   - Add caching for transcripts
   - Optimize video processing
   - Support for batch processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 