import gradio as gr
import speech_recognition as sr
from gtts import gTTS
from huggingface_hub import InferenceClient
import os
import time
from IPython.display import Audio, display

# Set up the InferenceClient
client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token="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",
)

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
    return question

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
    time.sleep(2)  # Adjust the sleep duration if needed

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
    accuracy = correct_answers / total_answers
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

    question = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
    previous_questions.append(question)
    play_audio(speak(question))
    conversation_history.append({"role": "interviewer", "content": question})

    correct_answers = 0
    total_answers = 0

    def handle_answer(audio_file):
        nonlocal question, question_count, correct_answers, total_answers
        user_answer = recognize_speech(audio_file)
        conversation_history.append({"role": "candidate", "content": user_answer})

        if user_answer.lower() == 'finish interview':
            end_message = "Interview finished!"
            play_audio(speak(end_message))
            return question, end_message  # Return end message as feedback

        if question_count == 1:
            # Provide a fixed response for the first question
            response = "It's great to hear about your innovative ideas. I wish you all the best for your future projects. Let's proceed with the interview."
            play_audio(speak(response))
            conversation_history.append({"role": "interviewer", "content": response})
            # Proceed to the next question
            question_count += 1
            question = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
            previous_questions.append(question)
            play_audio(speak(question))
            conversation_history.append({"role": "interviewer", "content": question})
            return question, response
        else:
            # For subsequent questions, perform verification and hint generation
            verification = verify_answer(topic, position, user_answer, question)
            play_audio(speak(verification))
            conversation_history.append({"role": "interviewer", "content": verification})

            if "right" in verification.lower():
                correct_answers += 1
                total_answers += 1
                question_count += 1
                question = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
                previous_questions.append(question)
                play_audio(speak(question))
                conversation_history.append({"role": "interviewer", "content": question})
                return question, verification
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
                    question = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
                    previous_questions.append(question)
                    play_audio(speak(question))
                    conversation_history.append({"role": "interviewer", "content": question})
                    return question, verification
                else:
                    correct_answer = get_correct_answer(topic, position, question)
                    play_audio(speak(f"Correct answer: {correct_answer}"))
                    conversation_history.append({"role": "interviewer", "content": f"Correct answer: {correct_answer}"})
                    question_count += 1
                    question = generate_question(topic, position, difficulty, previous_questions, question_count=question_count)
                    previous_questions.append(question)
                    play_audio(speak(question))
                    conversation_history.append({"role": "interviewer", "content": question})
                    return question, verification

    return question, handle_answer

def gradio_interface(topic, position, difficulty):
    # Initialize the interview process
    question, handle_answer = conduct_interview(topic, position, difficulty)

    def next_question(audio_file):
        nonlocal question

        if audio_file:
            # Handle the transcribed answer
            question, feedback = handle_answer(audio_file)
            if "Interview finished!" in feedback:
                return question, feedback  # Return response instead of empty string

            return question, feedback  # Return response instead of empty string

        # In case no audio file is provided
        return question, ""

    def transcribe_audio(audio_file):
        # Transcribe the audio file
        return recognize_speech(audio_file)

    with gr.Blocks(theme=gr.themes.Default()) as demo:
        gr.Markdown(f"### Mock Interview on {topic} for {position} position at {difficulty} difficulty level")



        # Textbox for displaying the interview question
        question_output = gr.Textbox(label="Question", interactive=False)
        question_output.value = question

        # Audio input for user's answer
        audio_input = gr.Audio(label="Your Answer", type="filepath")

        # Textbox for displaying the transcribed answer
        transcribed_output = gr.Textbox(label="Transcribed Answer", interactive=False)

        # Button to transcribe audio
        transcribe_button = gr.Button("Transcribe Audio")

        # Button to submit the answer
        submit_button = gr.Button("Submit Answer")

        # Textbox for displaying the interviewer's response
        feedback_output = gr.Textbox(label="Interviewer’s Response", interactive=False)

        # Set up button click events
        transcribe_button.click(fn=transcribe_audio, inputs=audio_input, outputs=transcribed_output)
        submit_button.click(next_question, inputs=audio_input, outputs=[question_output, feedback_output])

    # Launch the Gradio interface
    demo.launch(debug=True)

# Play welcome message
welcome_message()

# Get user name and greet
user_name = get_user_name()

# Get domain and position from user
play_audio(speak("What topic would you like to discuss? "))
topic = input("What topic would you like to discuss? ")

play_audio(speak("What position are you applying for? "))
position = input("What position are you applying for? ")

play_audio(speak("What difficulty level would you like the questions to be? "))
difficulty = input("What difficulty level would you like the questions to be? (easy, medium, hard) ")

# Start the mock interview using Gradio interface
gradio_interface(topic, position, difficulty)
