import os
import json
import google.generativeai as genai
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional

# Unified System Prompt forcing strict JSON output
SYSTEM_PROMPT = """
You are a medical extraction assistant for ClariRx, an app that helps elderly patients in India understand their prescriptions.
Your task is to take raw, messy OCR text from a handwritten prescription and extract the medications.

You must output ONLY valid JSON matching this exact structure, with no markdown formatting or extra text:
{
  "type": "prescription",
  "items": [
    {
      "drugName": "Cleaned up drug name with dosage (e.g., Amoxicillin 500mg)",
      "frequency": "Dosage frequency (e.g., 1-0-1 or SOS)",
      "explanationEn": "A simple, jargon-free 1-2 sentence explanation of what the medicine does in English.",
      "explanationHi": "The exact same explanation translated into simple Hindi.",
      "duration": "Duration (e.g., 5 Days, or 'As needed')",
      "instructions": "Any specific instructions (e.g., Take after meals)"
    }
  ]
}

If no medications are found, return {"type": "prescription", "items": []}.
Be highly forgiving of OCR typos (e.g., "Am0x" -> "Amoxicillin").
"""

class LLMOrchestrator:
    def __init__(self):
        # Initialize Gemini
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            # The user specifically requested gemini-2.5-flash
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.gemini_model = None
            
        # Initialize Groq
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            self.groq_client = None

    def extract_with_gemini(self, raw_text: str) -> dict:
        if not self.gemini_model:
            raise ValueError("GEMINI_API_KEY is not set.")
            
        prompt = f"{SYSTEM_PROMPT}\n\nHere is the raw OCR text to process:\n{raw_text}"
        
        # We enforce JSON output using generation config
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print(f"Failed to parse Gemini response: {response.text}")
            return {"type": "prescription", "items": []}

    def extract_with_groq(self, raw_text: str, model_name: str = "llama3-8b-8192") -> dict:
        if not self.groq_client:
            raise ValueError("GROQ_API_KEY is not set.")
            
        response = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Here is the raw OCR text to process:\n{raw_text}"
                }
            ],
            model=model_name,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, AttributeError):
            print(f"Failed to parse Groq response: {response.choices[0].message.content}")
            return {"type": "prescription", "items": []}

    def extract(self, raw_text: str, model_provider: str = "gemini") -> dict:
        """
        Main orchestration method. Routes to the correct model based on provider.
        model_provider can be 'gemini' or 'groq'.
        """
        if model_provider.lower() == "gemini":
            return self.extract_with_gemini(raw_text)
        elif model_provider.lower() == "groq":
            return self.extract_with_groq(raw_text)
        else:
            raise ValueError(f"Unknown model provider: {model_provider}. Choose 'gemini' or 'groq'.")

# Quick standalone test if run directly
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv() # Make sure to pip install python-dotenv if testing locally
    
    orchestrator = LLMOrchestrator()
    sample_ocr_text = "Rx\nPatient: Unknown\nAmox 500 mg 1 - 0 - 1 x 5d\nPara 650mg SOS"
    
    print("Testing Gemini 2.5 Flash...")
    try:
        res = orchestrator.extract(sample_ocr_text, model_provider="gemini")
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Gemini failed: {e}")
        
    print("\nTesting Groq (Llama 3)...")
    try:
        res = orchestrator.extract(sample_ocr_text, model_provider="groq")
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Groq failed: {e}")
