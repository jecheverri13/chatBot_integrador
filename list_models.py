import google.generativeai as genai
import os

print("Listing models...")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
