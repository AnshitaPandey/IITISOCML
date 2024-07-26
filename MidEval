# Install required libraries
!pip install SpeechRecognition gradio spacy

import spacy
import gradio as gr
import requests
import speech_recognition as sr
import time  # Import time module for retry backoff

# Ensure spaCy model is downloaded and loaded
nlp = spacy.load("en_core_web_sm")

# Set up API URLs and headers for question generation and text-to-speech (TTS)
question_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
question_headers = {"Authorization": "Bearer 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"}

tts_api_url = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
tts_headers = {"Authorization": "Bearer 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"}

# Function to send HTTP request to model and receive response
def query(api_url, headers, payload):
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise ValueError(f"Request to {api_url} failed with status code {response.status_code}")
    if api_url == question_api_url:
        return response.json()  # Return JSON response for question generation
    else:
        return response.content  # Return content (audio bytes) for TTS

# Function to generate a question about a specific domain and position
def generate_question(domain, position, max_new_tokens=500):
    payload = {
        "inputs": f"Generate a question about {domain} related to the position of {position}.",
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "stop_token": "?"  # Stop generation at question mark
        }
    }
    response = query(question_api_url, question_headers, payload)
    generated_text = response[0]["generated_text"]
    question = generated_text.strip()
    return question

# Function to verify the user's answer
def verify_answer(domain, position, user_answer, question):
    payload = {
        "inputs": f"Verify if '{user_answer}' is a correct answer to '{question}' in the context of {domain} and the position of {position}. Praise if correct, else provide hints but never the answer.",
        "parameters": {
            "max_new_tokens": 500,
            "return_full_text": False
        }
    }
    response = query(question_api_url, question_headers, payload)
    generated_text = response[0]["generated_text"]
    verification = generated_text.strip()
    return verification

# Function to generate a follow-up question based on the user's answer
def follow_up_question(domain, position, user_answer):
    doc = nlp(user_answer)  # Use spaCy to extract entities from user's answer
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extracted entities

    payload = {
        "inputs": f"Ask one concise follow-up question based on {domain} related to the position of {position} in context to '{user_answer}'. Consider these entities: {', '.join([f'{ent[0]} ({ent[1]})' for ent in entities])}. Do not repeat previous questions.",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.9,
            "echo": False,
            "return_full_text": False
        }
    }
    response = query(question_api_url, question_headers, payload)
    generated_text = response[0]["generated_text"]
    follow_up = generated_text.strip()
    return follow_up

# Function to generate audio from text using TTS model
def generate_audio(text, max_retries=3, backoff_factor=2):
    retries = 0
    while retries < max_retries:
        try:
            payload = {"inputs": text}
            audio_bytes = query(tts_api_url, tts_headers, payload)
            return audio_bytes
        except requests.exceptions.RequestException as e:
            print(f"Error generating audio: {e}. Retrying in {backoff_factor ** retries} seconds...")
            time.sleep(backoff_factor ** retries)
            retries += 1
    print("Failed to generate audio after maximum retries.")
    return None

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(audio):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"

# Function to conduct the interview
def interview(domain, position):
    question = generate_question(domain, position)  # Generate initial question
    question_audio = generate_audio(question)  # Generate audio for initial question

    # Callback function to handle the interview process
    def interview_callback(audio):
        user_answer = transcribe_audio(audio).strip()  # Transcribe user's audio answer
        if user_answer.lower() == 'finish interview':
            return "Interview finished!", None

        verification = verify_answer(domain, position, user_answer, question)  # Verify user's answer
        follow_up = follow_up_question(domain, position, user_answer)  # Generate follow-up question
        follow_up_audio = generate_audio(follow_up)
        while follow_up_audio is None:
            follow_up_audio = generate_audio(follow_up)  # Retry generating audio for follow-up question

        return f"Verification: {verification}\nFollow-up: {follow_up}", follow_up_audio

    return question, question_audio, interview_callback

# User inputs for domain and position
domain = input("Enter the domain (e.g., technology, healthcare, finance): ")
position = input("Enter the position you're applying for (e.g., software engineer, doctor, financial analyst): ")
question, question_audio, interview_callback = interview(domain, position)  # Start interview

# Create Gradio interface for audio input and text output
interface = gr.Interface(
    fn=interview_callback,
    inputs=gr.Audio(type="filepath", label="Record your answer"),
    outputs=[
        gr.Textbox(label="Interviewer's response"),
        gr.Audio(label="Follow-up question (Audio)")
    ],
    title="Interview",
    description=f"Question: {question}",
    theme="compact"
)

# Launch the initial question and Gradio interface
print(f"Initial question: {question}")
gr.Interface(
    fn=lambda: (question, question_audio),
    inputs=None,
    outputs=[
        gr.Textbox(label="Initial question"),
        gr.Audio(label="Initial question (Audio)")
    ],
    title="Initial Question"
).launch()
interface.launch()
