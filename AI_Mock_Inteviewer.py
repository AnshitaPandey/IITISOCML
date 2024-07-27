import gradio as gr
import cv2
import requests
import os
from collections import Counter
from gtts import gTTS
from huggingface_hub import InferenceClient
import speech_recognition as sr
import time
from IPython.display import Audio, display

# Set up the InferenceClient
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token="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")
API_URL = "https://api-inference.huggingface.co/models/trpakov/vit-face-expression"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def predict(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * 1)
    label_counts = Counter()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            cv2.imwrite("temp.jpg", frame)
            output = query("temp.jpg")
            os.remove("temp.jpg")

            if isinstance(output, list):
                max_label = max(output, key=lambda x: x['score'])['label']
                label_counts[max_label] += 1
        frame_count += 1

    cap.release()

    if len(label_counts) == 0:
        return "No valid frames detected."

    most_common_labels = label_counts.most_common(1)
    result_labels = [label for label, count in most_common_labels]
    return ", ".join(result_labels)

def get_system_prompt(topic):
    return f"You are an AI mock interviewer with expertise in {topic}. Your job is to conduct a realistic and challenging mock interview for the candidate. Your questions should evaluate the candidate's knowledge, problem-solving skills, and ability to apply concepts in practical scenarios. Ensure the interview is thorough, engaging, and covers various aspects of the role without repeating questions."

def generate_question(topic, position, difficulty, previous_questions=[], max_new_tokens=500, question_count=1):
    system_prompt = get_system_prompt(topic)
    messages = [{"role": "system", "content": system_prompt}]
    if question_count == 1:
        messages.append({"role": "user", "content": f"Generate only 1 short question on the future plans of the user in the domain of {topic}."})
    elif question_count == 2:
        messages.append({"role": "user", "content": f"Generate only 1 relevant and concise question about the position of {position}. Don't provide its answer."})
    else:
        messages.append({"role": "user", "content": f"Generate only 1 relevant and concise {difficulty} level question on {topic}. Never provide its answer. Don't generate multiple choice questions. Test the knowledge of the user on {topic}. Make sure the question is not already in the list of previous questions: {', '.join(previous_questions)}."})
    response = client.chat_completion(messages=messages, max_tokens=max_new_tokens, stream=True)
    generated_text = ""
    for chunk in response:
        generated_text += chunk.choices[0].delta.content or ""
    question = generated_text.strip()
    
    # Generate and save speech for the question
    speech_file = "question.mp3"
    tts = gTTS(question)
    tts.save(speech_file)
    
    return question, speech_file

def verify_answer(topic, position, user_answer, question):
    messages = [{"role": "user", "content": f"Tell the user if the answer {user_answer} is right or wrong in relevance to the question {question}. Tell right or wrong only. Add a 1 line explanation. Don't provide the answer to {question}."}]
    response = client.chat_completion(messages=messages, max_tokens=100, stream=True)
    generated_text = ""
    for chunk in response:
        generated_text += chunk.choices[0].delta.content or ""
    verification = generated_text.strip()
    return verification

def generate_hint(topic, user_answer, question):
    messages = [{"role": "user", "content": f"Generate a hint for the question {question} in context to {topic}. Don't provide the answer."}]
    response = client.chat_completion(messages=messages, max_tokens=100, stream=True)
    generated_text = ""
    for chunk in response:
        generated_text += chunk.choices[0].delta.content or ""
    hint = generated_text.strip()
    return hint

def get_correct_answer(topic, position, question):
    messages = [{"role": "user", "content": f"Provide the correct answer to {question} in context to {topic}."}]
    response = client.chat_completion(messages=messages, max_tokens=500, stream=True)
    generated_text = ""
    for chunk in response:
        generated_text += chunk.choices[0].delta.content or ""
    correct_answer = generated_text.strip()
    return correct_answer

def speak(text):
    tts = gTTS(text)
    filename = "temp.mp3"
    tts.save(filename)
    return filename

def play_audio(filename): 
    display(Audio(filename, autoplay=True))
    time.sleep(2)

def welcome_message():
    message = "Welcome to our AI mock interviewer!\n"
    print(message)
    play_audio(speak(message))

def get_user_name():
    user_name = input("Please enter your name to begin: ")
    print("\n")
    greet_message = f"Hi, {user_name}! This sophisticated tool is designed to simulate a real interview environment, helping you practice and refine your interview skills.\n"
    print(greet_message)
    play_audio(speak(greet_message))
    return user_name

def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "Sorry, I did not understand the audio."
    return text

def generate_feedback(correct_answers, total_answers, conversation_history):
    accuracy = correct_answers / total_answers if total_answers > 0 else 0
    history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history])

    feedback_prompt = (
        f"Based on the following conversation history of a mock interview, provide constructive feedback for the user. "
        f"Include the user's strengths and weaknesses, and give advice on how to improve and refine their skills for a real interview.\n\n"
        f"Overall, assess the candidate's performance in terms of clarity, technical knowledge, problem-solving abilities, and communication skills. Highlight areas of improvement and suggest actionable advice to enhance interview readiness."
        f"Provide detailed feedback, mentioning specific instances from the conversation history to support your assessment.\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"Overall accuracy of the user was {accuracy:.2f}. Specify it and comment on it."
    )

    messages = [{"role": "user", "content": feedback_prompt}]
    response = client.chat_completion(messages=messages, max_tokens=600, stream=True)
    feedback = "".join([chunk.choices[0].delta.content or "" for chunk in response])

    print(feedback)
    play_audio(speak(feedback))

def conduct_interview(topic, position, difficulty):
    question_count = 1
    previous_questions = []
    conversation_history = []
    correct_answers = 0
    total_answers = 0
    
    # Initial question
    question, speech_file = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
    previous_questions.append(question)
    
    def handle_answer(audio_file, video_file=None):
        nonlocal question, question_count, correct_answers, total_answers
        
        if video_file:
            sentiment = predict(video_file)
            sentiment_message = f"Detected sentiment: {sentiment}. This might affect the interview flow."
            play_audio(speak(sentiment_message))
            conversation_history.append({"role": "interviewer", "content": sentiment_message})

        user_answer = recognize_speech(audio_file)
        conversation_history.append({"role": "candidate", "content": user_answer})
        if user_answer.lower() == 'finish interview':
            end_message = "Interview finished!"
            play_audio(speak(end_message))
            return end_message, generate_feedback(correct_answers, total_answers, conversation_history), None
        
        if question_count == 1:
            response = "It's great to hear about your innovative ideas. I wish you all the best for your future endeavors!"
            play_audio(speak(response))
            conversation_history.append({"role": "interviewer", "content": response})
        else:
            verification = verify_answer(topic, position, user_answer, question)
            play_audio(speak(verification))
            conversation_history.append({"role": "interviewer", "content": verification})

            if "right" in verification.lower():
                correct_answers += 1
                total_answers += 1
                question_count += 1
                question, speech_file = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
                previous_questions.append(question)
                play_audio(speak(question))
                conversation_history.append({"role": "interviewer", "content": question})
            else:
                total_answers += 1
                hint = generate_hint(topic, user_answer, question)
                play_audio(speak(hint))
                conversation_history.append({"role": "interviewer", "content": hint})

                revised_answer = recognize_speech(audio_file)
                conversation_history.append({"role": "candidate", "content": revised_answer})
                verification = verify_answer(topic, position, revised_answer, question)
                play_audio(speak(verification))
                conversation_history.append({"role": "interviewer", "content": verification})

                if "right" in verification.lower():
                    question_count += 1
                    question, speech_file = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
                    previous_questions.append(question)
                    play_audio(speak(question))
                    conversation_history.append({"role": "interviewer", "content": question})
                else:
                    correct_answer = get_correct_answer(topic, position, question)
                    play_audio(speak(f"Correct answer: {correct_answer}"))
                    conversation_history.append({"role": "interviewer", "content": f"Correct answer: {correct_answer}"})
                    question_count += 1
                    question, speech_file = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
                    previous_questions.append(question)
                    play_audio(speak(question))
                    conversation_history.append({"role": "interviewer", "content": question})
        return question, speech_file

    return question, handle_answer

def gradio_interface():
    with gr.Blocks(theme=gr.themes.Default()) as demo:
        gr.Markdown("### Mock Interview")
        
        # Input fields for initial details
        topic_input = gr.Textbox(label="Topic")
        position_input = gr.Textbox(label="Position")
        difficulty_input = gr.Radio(label="Difficulty", choices=["easy", "medium", "hard"])
        
        # Button to start the interview
        start_button = gr.Button("Start Interview")
        
        # Output components
        question_output = gr.Textbox(label="Question", interactive=False)
        audio_output = gr.Audio(label="Question Audio", type="filepath")
        handle_answer_output = gr.State()
        audio_input = gr.Audio(label="Your Answer", type="filepath")
        video_input = gr.Video(label="Your Video")
        transcribed_output = gr.Textbox(label="Transcribed Answer", interactive=False)
        sentiment_output = gr.Textbox(label="Sentiment Analysis", interactive=False)
        
        # Define the event handler for the start button
        def start_interview(topic, position, difficulty):
            question, handle_answer = conduct_interview(topic, position, difficulty)
            return question, "question.mp3", handle_answer, "", ""
        
        # Define the event handler for the submit button
        def submit_answer(handle_answer, audio, video):
            if video:
                sentiment = predict(video)
                sentiment_output.set(sentiment)
            
            if audio:
                # Handle the transcribed answer
                user_answer = recognize_speech(audio)
                question, speech_file = handle_answer(audio, video)
                
                if "Interview finished!" in question:
                    return question, speech_file, "", sentiment_output.value
                
                transcribed_output.set(user_answer)
                return question, speech_file, transcribed_output.value, sentiment_output.value
            
            # In case no audio file is provided
            return "", "", sentiment_output.value, ""

        start_button.click(fn=start_interview, inputs=[topic_input, position_input, difficulty_input], outputs=[question_output, audio_output, handle_answer_output, transcribed_output, sentiment_output])
        submit_button = gr.Button("Submit Answer")
        submit_button.click(fn=submit_answer, inputs=[handle_answer_output, audio_input, video_input], outputs=[question_output, audio_output, transcribed_output, sentiment_output])
    
    # Launch the Gradio interface
    demo.launch(debug=True)

# Start the Gradio interface
gradio_interface()
