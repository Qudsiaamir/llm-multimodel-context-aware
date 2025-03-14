from fastapi import FastAPI, File, UploadFile
import fitz  # PyMuPDF
from summarizer import Summarizer
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()
logger = logging.getLogger("fastapi")
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def summarize_text(text):
    """Summarize text using BERT Extractive Summarizer."""
    model = Summarizer()
    summary = model(text)  # Adjust min_length as needed
    return summary

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    
    text = extract_text_from_pdf(file_location)
    word_count = len(text.split())

    os.remove(file_location)  # Cleanup after processing
    
    if word_count > 1000:
        logger.info(f"word_count is {word_count} > 1000 , summarising")
        summary = summarize_text(text)
        word_count_summary = len(summary.split())
        logger.info(f"Summarized text word count {word_count_summary} : {summary}")
        return {"message": "Text summarized", "word_count": word_count_summary, "summary": summary}
    else:
        logger.info(f"PDF text  {text}")
        return {"message": "Text extracted", "word_count": word_count, "content": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
