import os
import google.generativeai as genai

# Set up your Gemini API key (set this via environment variable or direct assignment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBEi-lI4gCL-WunpP3sa0uzSXP6wCRywx8")
genai.configure(api_key=GEMINI_API_KEY)

# Create your GenerativeModel object
model = genai.GenerativeModel('gemini-2.5-flash')

prompt = (
    "Translate this advice for a farmer into very clear spoken Telugu only (no English, be concise, no numbers, only Telugu):\n\n"
    "Detected Tomato early blight, moderate level. Remove infected leaves."
)

response = model.generate_content(prompt)
print(response.text)
