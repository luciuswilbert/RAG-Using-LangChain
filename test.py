import google.generativeai as genai
import pathlib

# ✅ Set your Gemini API key here
genai.configure(api_key="AIzaSyDry3b4ZkOEAwZWIsouZzc1Jrkj0V0qgUM")  # <-- Replace this

# ✅ Load PDF as bytes
pdf_path = pathlib.Path("DisclosureSheet.pdf")
pdf_bytes = pdf_path.read_bytes()

# ✅ Use the model directly (no need for genai.Client())
model = genai.GenerativeModel("models/gemini-1.5-flash")  # or gemini-2.5-flash if available to you

# ✅ Generate content using the PDF file + prompt
response = model.generate_content([
    {
        "inline_data": {
            "mime_type": "application/pdf",
            "data": pdf_bytes
        }
    },
    {
        "text": "Give me the plain text of the document in detailed format and do not miss any information at all. You can give me in the full and understandable form"
    }
])

# ✅ Print the summary result
print(response.text)
