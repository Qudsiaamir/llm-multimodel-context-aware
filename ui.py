import gradio as gr
import requests
import json
import os 

import os
import openai
import datetime
import shutil
import uuid 
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gardio")
# Directories for storing logs and uploads
LOG_DIR = "logs"
UPLOAD_DIR = "uploads"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to generate a unique log filename based on timestamp
def generate_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"conversation_{timestamp}.json")

# Start a new conversation log
current_log_file = generate_log_filename()

def save_uploaded_file(file):
    if isinstance(file, dict):  # Handle Gradio's new file format
        file_path = file["name"]
    else:
        file_path = file
    
    file_name = os.path.basename(file_path)
    save_path = os.path.join(UPLOAD_DIR, f'{uuid.uuid1()}-{file_name}')

    # Save the uploaded file properly
    shutil.copy(file_path, save_path)
    logger.info(f"saving uploaded file {save_path}")

# Function to save chat history
def save_history(history):
    with open(current_log_file, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)
    logger.info(f"saving chat history to file {current_log_file}")


# -------------------------------------------------------------------
# 1) Set up your API key (replace with your real key)
# -------------------------------------------------------------------
openai.api_key = "API KEY"

from mistralai import Mistral
api_key = "API KEY"
model = "mistral-large-latest"
# Set your API key
os.environ["MISTRAL_API_KEY"] = "API KEY"
client = Mistral(api_key=api_key)
# model = "mistral-large-latest"
# chat_response = client.chat.complete(
#     model=model,
#     messages=[{"role":"user", "content":"What is the best French cheese?"}]
# )

# print()
# Initialize memory
conversation_memory = None

# -------------------------------------------------------------------
# 2) Simple “memory” structure
#    This is just a list storing previous interactions (question + answer).
#    In a more advanced setup, you might store this in a vector DB
#    and retrieve the most relevant items for each query.
# -------------------------------------------------------------------
class ConversationMemory:
    def __init__(self):
        self.history = []
    
    def add_interaction(self, user_query, answer):
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})
    
    def get_history(self):
        return self.history

# -------------------------------------------------------------------
# 3) Function to build a prompt from multiple sources
#    We combine:
#      - Short system instruction
#      - Some conversation memory
#      - The current multi-modal context:
#          • Image Caption
#          • Similar Caption
#          • Extracted PDF text
#          • The user’s question
# -------------------------------------------------------------------
def build_prompt(memory, image_caption, similar_caption, pdf_text, user_question):
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant that can combine context from different modalities "
            "including text from PDF documents, image captions, and user-provided text."
        )
    }

    # Turn memory list (history) into a prompt format
    conversation_history = memory.get_history()

    # We create a structured “context” string that merges the different modalities
    context_str = (
        f"Image Caption: {image_caption}\n\n"
        f"Similar Image Caption(s): {similar_caption}\n\n"
        f"PDF Text (relevant excerpt): {pdf_text}\n\n"
        f"User Question: {user_question}\n\n"
        "Use all the above context to provide a concise, factual, and helpful answer."
    )

    user_message = {
        "role": "user",
        "content": context_str
    }

    # The final message list for the chat completion
    messages = [system_message] + conversation_history + [user_message]
    logger.info(f"Question prompt for LLM {messages}")
    return messages


# -------------------------------------------------------------------
# 4) Main function to query the LLM
# -------------------------------------------------------------------
def answer_with_multimodal_context(
    memory: ConversationMemory,
    image_caption: str,
    similar_caption: str,
    pdf_text: str,
    user_question: str,
    model: str = "omni-moderation-latest",
    temperature: float = 0.2
):
    """
    This function:
    1) Builds the prompt with memory + multi-modal input
    2) Sends the prompt to the LLM
    3) Returns the LLM's answer
    4) Stores the interaction in memory
    """
    messages = build_prompt(memory, image_caption, similar_caption, pdf_text, user_question)
    global client
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    #     temperature=temperature
    # )
    logger.info(f"using model {model}")
    chat_response = client.chat.complete(
        model=model,
        messages=messages
        )
    # The assistant answer is in response.choices[0].message["content"]
    # assistant_answer = response.choices[0].message["content"].strip()
    assistant_answer = chat_response.choices[0].message.content
    # rewrite_response(assistant_answer, model, image_caption, similar_caption, pdf_text)
    assistant_answer = rewrite_response_with_mistral(assistant_answer, image_caption, similar_caption, pdf_text)
    # Save the new QA pair to memory
    memory.add_interaction(user_question, assistant_answer)
    logger.info(f"response from model : {assistant_answer}")
    return assistant_answer

from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import SystemMessage, HumanMessage
import os

# Set your API key
# os.environ["MISTRAL_API_KEY"] = "YOUR_MISTRAL_API_KEY"

def rewrite_response_with_mistral(response, image_caption, similar_caption, pdf_text):
    """
    Uses the Mistral large language model to clean the response by removing
    references to sensitive input sources.
    """
    llm = ChatMistralAI(model_name="mistral-large-latest", api_key="API KEY")

    system_prompt = """
    You are an AI post-processing assistant. Your task is to clean a response 
    generated by an AI assistant. The response may contain references to provided 
    inputs such as image captions, similar captions, or extracted PDF text.
    
    Your job:
    1. Remove any direct references to those inputs.
    2. Rephrase sentences that use them without explicitly mentioning the source.
    3. Ensure the response remains coherent and informative.
    4. Dont make any user referances

    also remove refences to this promt 'Use all the above context to provide a concise, factual, and helpful answer.'
    Do NOT mention '[REDACTED]'. Just make it sound natural.
    """

    user_input = f"""
    --- Original Response ---
    {response}

    --- Sensitive Information (Do NOT include) ---
    Image Caption: {image_caption}
    Similar Caption: {similar_caption}
    PDF Text: {pdf_text}

    Please rewrite the response while removing references to these sources.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    logger.info(f"refineing the response from the model")
    # Call the Mistral LLM
    cleaned_response = llm(messages).content
    logger.info(f"Cleaned response from model : {cleaned_response}")
    return cleaned_response



def get_image_caption(file_name):
    logger.info(f"generaaitng image caption")
    url = "http://127.0.0.1:8001/generate-caption/"
    files = {"file": open(file_name, "rb")}
    headers = {"accept": "application/json"}

    response = requests.post(url, headers=headers, files=files)
    logger.info(f"generated caption : {response.json()}")
    return response.json()   

def get_pdf_text(file_name):
    logger.info(f"extracting pdf text.")
    url = "http://127.0.0.1:8000/process-pdf/"
    files = {"file": open(file_name, "rb")}
    headers = {"accept": "application/json"}

    response = requests.post(url, headers=headers, files=files)
    logger.info(f"extracted text : {response.json() }")
    return response.json() 


def get_similarity_caption(query,k=3):
    logger.info(f"searching simlar captions.")
    url = "http://127.0.0.1:8002/search"
    headers = {"Content-Type": "application/json"}
    data = {"query": query , "top_k": k}

    response = requests.post(url, headers=headers, json=data)
    logger.info(f"list of simlar captions : k={k}, {response.json()}")
    return response.json() 
old_files = None
def chatbot_response(user_input, files, history=[]):
    global conversation_memory, old_files
    """Generates a chatbot response based on user input and conversation history, processing files when Send is clicked."""
    history.append({"role": "user", "content": user_input})
    image_caption=''
    pdf_text=''
    similarity_caption=''
    results = []
    if files != old_files:
        old_files = files
        if files:
            for file in files:
                save_uploaded_file(file)
                if file.name.endswith(".pdf"):
                    pdf_text = get_pdf_text(file.name)
                    results.append(file.name)
                elif file.name.endswith(('.png', '.jpg', '.jpeg')):
                    image_caption = get_image_caption(file.name)
                    
                    similarity_caption = get_similarity_caption(image_caption['caption'],k=3)
                    # print('file.name',file.name)
                    results.append(file.name)  
                else:
                    results.append(f"Unsupported file format: {file.name}")

    if not conversation_memory:
        # Initialize memory
        conversation_memory = ConversationMemory()
    # Example inputs
    image_caption_ = image_caption
    similar_caption_ = similarity_caption
    pdf_extracted_text_ = pdf_text
    user_question_ = user_input

    # Make the call
    answer = answer_with_multimodal_context(
        memory=conversation_memory,
        image_caption=image_caption_,
        similar_caption=similar_caption_,
        pdf_text=pdf_extracted_text_,
        user_question=user_question_,
        model="mistral-large-latest",  # or "gpt-4", if you have access
        temperature=0.2
    )

    # print("Assistant Answer:\n", answer)
    
    response = answer
    history.append({"role": "assistant", "content": response})
    # Save updated history
    save_history(history)
    return '', history, results




def clear_chat():
    global conversation_memory
    conversation_memory = None
    global current_log_file
    current_log_file = generate_log_filename()  # New log file
    logger.info(f"clearing convesarions.")
    return '', [], []

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot UI (Gradio)")
    chat = gr.Chatbot(type="messages")  # Updated to use OpenAI-style messages
    user_input = gr.Textbox(placeholder="Type your message...")
    file_input = gr.File( label="Upload Files", interactive=True, file_count='multiple')
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear Chat")
    
    send_btn.click(chatbot_response, inputs=[user_input, file_input, chat], outputs=[user_input, chat, file_input])
    clear_btn.click(clear_chat, inputs=[], outputs=[user_input, chat, file_input])

if __name__ == "__main__":
    demo.launch()
    # os.system('python ./caption_model.py &')
    # os.system('python ./pdf_extract.py &')
    # os.system('python ./similarity_search_app.py &'
'''   
    python ./caption_model.py &
    python ./pdf_extract.py &
    python ./similarity_search_app.py &
'''