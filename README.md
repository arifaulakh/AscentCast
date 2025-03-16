# Ascent Cast

LLM-powered podcast analysis tool that extracts career insights from podcast transcripts.

## Usage

1. Add your API keys to `.env`:
```
ANTHROPIC_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

2. Run the analyzer:
```bash
python main.py path/to/transcript.pdf --user-context "Your background and learning goals"
```