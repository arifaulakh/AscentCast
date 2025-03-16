import anthropic
import os
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()  # Load environment variables from .env file

@dataclass
class Config:
    """Configuration for the podcast analysis."""
    file_path: Path
    user_context: str
    model: str = "claude-3-7-sonnet-20250219"
    max_tokens: int = 4000
    temperature: float = 1

class PodcastAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def process_transcript(self) -> str:
        """Process the podcast transcript and return insights."""
        try:
            transcript = self._extract_text_from_file()
            insights = self._analyze_transcript(transcript)
            return insights
        except Exception as e:
            return f"Error processing transcript: {str(e)}"

    def _extract_text_from_file(self) -> str:
        """Extract text from the provided file using OCR."""
        try:
            uploaded_file = self.mistral_client.files.upload(
                file={
                    "file_name": self.config.file_path.name,
                    "content": open(self.config.file_path, "rb"),
                },
                purpose="ocr"
            )

            signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id)

            ocr_response = self.mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )

            return "\n".join(page.markdown for page in ocr_response.pages)
        except Exception as e:
            raise Exception(f"OCR processing failed: {str(e)}")

    def _analyze_transcript(self, transcript: str) -> str:
        """Analyze the transcript using Anthropic's Claude."""
        system_prompt = """
        You are an expert in analyzing technology podcasts, extracting key career insights, and providing actionable recommendations tailored to professionals in fast-growing startups.

        Your primary goal is to help users translate insights from industry leaders into concrete actions that accelerate their professional growth. You will analyze podcast transcripts to identify:

        1. Key career takeaways for software engineers, product builders, and startup operators
        2. Skills and strategies that can improve the user's impact in their role
        3. Lessons from top founders and investors that apply to the user's long-term career trajectory
        4. Opportunities for networking, leadership development, and industry positioning

        Ensure your responses are clear, structured, and tailored to the user's career path.
        """

        user_prompt = f"""
        {self.config.user_context}

        Please analyze the following podcast transcript and provide key takeaways that are directly relevant to my career growth, technical development, and long-term trajectory. Structure your response as follows:

        1. Key Career Lessons: What skills, mindsets, and strategies from the podcast are most relevant to my role and future growth?
        2. Actionable Career Moves: What specific actions should I take in the next 6-12 months to develop my skills, expand my network, or increase my impact?
        3. Lessons from Top Operators & Investors: What habits or frameworks from experienced founders, investors, or executives can I apply to my career?
        4. Opportunities to Apply These Insights: How can I integrate these lessons into my current role?

        Here's the transcript:

        {transcript}
        """

        try:
            message = self.anthropic_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}]
                    }
                ]
            )

            return "".join(chunk.text for chunk in message.content)
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Analyze podcast transcripts for career insights")
    parser.add_argument("file_path", type=str, help="Path to the podcast transcript file (PDF)")
    parser.add_argument("--user-context", type=str, 
                      default="I am a professional looking to grow my career in technology and startups.",
                      help="Brief description of your background and what you're looking to learn")
    args = parser.parse_args()

    config = Config(
        file_path=Path(args.file_path),
        user_context=args.user_context
    )

    analyzer = PodcastAnalyzer(config)
    insights = analyzer.process_transcript()
    print(insights)

if __name__ == "__main__":
    main()
