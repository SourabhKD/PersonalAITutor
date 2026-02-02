import streamlit as st
import ollama
import random
from typing import List, Optional
import logging
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import re  # Used for parsing quiz

# --- CONFIGURATION ---
PROGRESS_FILE = 'progress_data.yaml'

# --- AUTHENTICATION SETUP ---

# Load configuration from the YAML file
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("FATAL: config.yaml not found. Please create it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading config.yaml: {e}")
    st.stop()

# Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- END OF AUTHENTICATION SETUP ---

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PROGRESS TRACKING FUNCTIONS ---

def load_all_progress():
    """Loads the entire progress data file."""
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, 'r') as file:
            return yaml.load(file, Loader=SafeLoader) or {}
    except Exception as e:
        logger.error(f"Error loading progress file: {e}")
        return {}

def save_all_progress(data):
    """Saves the entire progress data file."""
    try:
        with open(PROGRESS_FILE, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error saving progress file: {e}")

def get_user_progress(username):
    """Gets a user's progress data."""
    all_data = load_all_progress()
    return all_data.get(username, {})

def update_user_progress(username, subject, mode_str):
    """Updates and saves a user's progress for one interaction."""
    all_data = load_all_progress()
    
    if username not in all_data:
        all_data[username] = {}
    
    if subject not in all_data[username]:
        all_data[username][subject] = {'explain': 0, 'quiz': 0}
        
    mode_key = 'explain' if mode_str == "Explain a Topic" else 'quiz'
    if mode_key not in all_data[username][subject]:
         all_data[username][subject][mode_key] = 0
         
    all_data[username][subject][mode_key] += 1
    
    save_all_progress(all_data)

# --- END OF PROGRESS TRACKING FUNCTIONS ---


# Function to get available models
@st.cache_data
def get_available_models():
    try:
        models_response = ollama.list()
        model_names = []
        
        # Handle the ListResponse object from ollama
        if hasattr(models_response, 'models'):
            for model in models_response.models:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                elif isinstance(model, dict):
                    name = model.get('name') or model.get('model') or model.get('id')
                    if name:
                        model_names.append(name)
                elif isinstance(model, str):
                    model_names.append(model)
        elif isinstance(models_response, dict) and 'models' in models_response:
            # Fallback for older API format
            for model in models_response['models']:
                if isinstance(model, dict):
                    name = model.get('name') or model.get('model') or model.get('id')
                    if name:
                        model_names.append(name)
                elif isinstance(model, str):
                    model_names.append(model)
        
        preferred_order = ['gemma3:latest', 'gemma3', 'gemma2:2b', 'gemma2', 'llama3', 'mistral', 'deepseek-coder']
        ordered_models = []
        
        for preferred in preferred_order:
            if preferred in model_names:
                ordered_models.append(preferred)
        
        for model in model_names:
            if model not in ordered_models:
                ordered_models.append(model)
                
        return ordered_models
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.info("Make sure Ollama is running: `ollama serve`")
        return []

# --- MAIN APP LOGIC (GATED BY LOGIN) ---

# Check authentication status
if st.session_state["authentication_status"]:
    
    # --- USER IS LOGGED IN ---
    
    # --- MODIFIED: Initialize session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # --- NEW QUIZ STATE ---
    if "quiz_active" not in st.session_state: # <-- CHANGED from quiz_in_progress
        st.session_state.quiz_active = False
    if "quiz_ready" not in st.session_state: # <-- NEW
        st.session_state.quiz_ready = False
    if "current_quiz_questions" not in st.session_state:
        st.session_state.current_quiz_questions = [] # Will be a list of dicts
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = [] # Will be a list of strings
    # --- END OF NEW QUIZ STATE ---

    # --- NEW: Combined state for disabling UI elements ---
    quiz_lock = st.session_state.quiz_active or st.session_state.quiz_ready

    # App title
    st.title("🎓 AI Tutor")
    st.write(f'Welcome, *{st.session_state["name"]}*!')

    # Sidebar for settings
    with st.sidebar:
        st.header("🎯 Learning Preferences")
        
        authenticator.logout()
        st.markdown("---") 

        education_level = st.selectbox(
            "Select your education level",
            ["School", "High School", "Graduate", "PG/PhD"],
            index=1,
            disabled=quiz_lock # <-- CHANGED
        )
        
        subject = st.selectbox(
            "Choose a subject",
            ["Math", "History", "Computer Science", "Physics", "Biology", "Chemistry"],
            index=2,
            disabled=quiz_lock # <-- CHANGED
        )
        
        mode = st.radio(
            "Select mode",
            ["Explain a Topic", "Generate a Quiz"],
            index=0,
            disabled=quiz_lock # <-- CHANGED
        )
        
        available_models = get_available_models()
        if available_models:
            model_name = st.selectbox(
                "AI Model",
                available_models,
                index=0,
                help="Gemma3 is recommended for better performance"
            )
            if 'gemma3' in model_name.lower():
                st.success("✅ Using Gemma3 - Excellent choice!")
            elif 'deepseek-coder' in model_name.lower():
                st.success("✅ Using DeepSeek Coder - Great for coding tasks!")
            elif available_models and not any('gemma3' in m.lower() for m in available_models):
                st.info("💡 Install Gemma3 for better performance: `ollama pull gemma3`")
        else:
            st.error("⚠️ No Ollama models found.")
            st.code("ollama pull gemma3", language="bash")
            st.code("ollama pull llama3\nollama pull deepseek-coder", language="bash")
            model_name = None
        
        st.markdown("---")

        # --- PROGRESS DASHBOARD ---
        st.header("📊 Your Progress")
        username = st.session_state["name"]
        user_progress = get_user_progress(username)
        
        if not user_progress:
            st.info("Start chatting to track your progress!")
        else:
            total_explains = 0
            total_quizzes = 0
            
            for subj, counts in user_progress.items():
                total_explains += counts.get('explain', 0)
                total_quizzes += counts.get('quiz', 0)
            
            total_interactions = total_explains + total_quizzes
            
            st.metric("Total Interactions", total_interactions)
            
            col1, col2 = st.columns(2)
            col1.metric("Topics Explained", total_explains)
            col2.metric("Quizzes Taken", total_quizzes)
            
            with st.expander("View by Subject"):
                for subj, counts in sorted(user_progress.items()):
                    st.markdown(f"**{subj}**")
                    st.text(f"  - Explanations: {counts.get('explain', 0)}")
                    st.text(f"  - Quizzes: {counts.get('quiz', 0)}")
        # --- END OF DASHBOARD ---

        st.markdown("---")
        st.markdown("🔒 **100% Private** – No data leaves your device.")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- ###################################### ---
    # ---      NEW QUIZ START BUTTON           ---
    # --- ###################################### ---
    if st.session_state.quiz_ready:
        st.info("Your quiz is ready!")
        if st.button("🚀 Start Your Quiz!"):
            st.session_state.quiz_active = True
            st.session_state.quiz_ready = False
            st.session_state.current_question_index = 0 # Initialize here
            st.session_state.user_answers = [] # Initialize here
            st.rerun()
    # --- ###################################### ---


    # --- ###################################### ---
    # ---     NEW MULTI-STEP QUIZ UI           ---
    # --- ###################################### ---
    if st.session_state.quiz_active: # <-- CHANGED from quiz_in_progress
        
        # Get the current question index
        index = st.session_state.current_question_index
        total_questions = len(st.session_state.current_quiz_questions)

        # Check if quiz is valid and in progress
        if total_questions > 0 and index < total_questions:
            
            # Get the data for the current question
            quiz_item = st.session_state.current_quiz_questions[index]
            question = quiz_item["question"]
            options = quiz_item["options"]

            with st.form(key=f"quiz_form_{index}"): # Unique key for each question
                st.markdown(f"**Quiz Time! 🧠 - Question {index + 1} of {total_questions}**")
                st.markdown(f"**{question}**") # Make question bold
                
                user_answer = st.radio(
                    "Choose your answer:",
                    options=options,
                    key=f"quiz_choice_{index}", # Use index in key to force re-render
                    index=None # Force a selection
                )
                
                # Dynamic button text
                button_text = "Submit & Next Question"
                if index == total_questions - 1:
                    button_text = "Submit & View Results"
                    
                submitted = st.form_submit_button(button_text)
                
                if submitted:
                    if user_answer:
                        # Store the user's answer
                        st.session_state.user_answers.append(user_answer)
                        
                        # Increment question index
                        st.session_state.current_question_index += 1
                        
                        # Check if quiz is over
                        if st.session_state.current_question_index == total_questions:
                            # --- QUIZ IS OVER - GRADE AND SHOW RESULTS ---
                            
                            score = 0
                            results_markdown = "## 🏁 Quiz Results 🏁\n\n"
                            
                            for i in range(total_questions):
                                q_item = st.session_state.current_quiz_questions[i]
                                user_ans = st.session_state.user_answers[i]
                                correct_ans_str = q_item["answer"].strip()
                                
                                # Robust check
                                correct_letter = correct_ans_str.split('.')[0].strip() + "."
                                is_correct = user_ans.strip().startswith(correct_letter)
                                
                                if is_correct:
                                    score += 1
                                    results_markdown += f"**Question {i+1}: ✅ Correct!**\n"
                                else:
                                    results_markdown += f"**Question {i+1}: ❌ Incorrect.**\n"
                                
                                results_markdown += f"* **Question:** {q_item['question']}\n"
                                results_markdown += f"* **Your answer:** {user_ans}\n"
                                results_markdown += f"* **Correct answer:** {correct_ans_str}\n"
                                results_markdown += f"* **Explanation:** {q_item['explanation']}\n\n"
                                results_markdown += "---\n\n"

                            # Add final score at the top
                            score_summary = f"### You scored **{score} out of {total_questions}**!\n\n---\n\n"
                            final_report = score_summary + results_markdown
                            
                            # Add report to chat
                            st.session_state.messages.append({"role": "assistant", "content": final_report})
                            
                            # --- Clean up quiz state ---
                            st.session_state.quiz_active = False # <-- CHANGED
                            st.session_state.current_quiz_questions = []
                            st.session_state.current_question_index = 0
                            st.session_state.user_answers = []
                            
                        # Rerun to show next question OR the final results
                        st.rerun() 
                    else:
                        st.warning("Please select an answer before submitting.")
        else:
             # Failsafe: If state is broken, reset it.
             st.session_state.quiz_active = False # <-- CHANGED
             st.session_state.quiz_ready = False # <-- NEW
             logger.warning("Quiz state was invalid, resetting.")
             st.rerun()

    # --- ###################################### ---
    # ---   END OF MULTI-STEP QUIZ UI          ---
    # --- ###################################### ---


    # --- MODIFIED: User input (disabled during quiz) ---
    if prompt := st.chat_input(
        f"Ask a {subject} question...", 
        disabled=quiz_lock # <-- CHANGED
    ):
        if not model_name:
            st.error("Please install an Ollama model first!")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # --- MODIFIED: Customize prompt based on mode ---
            if mode == "Explain a Topic":
                custom_prompt = f"""
                You are a {education_level}-level {subject} tutor. 
                Explain the following in a structured, step-by-step manner: 
                "{prompt}"
                
                - Break down complex concepts.
                - Use examples if helpful.
                - Keep explanations clear and concise.
                """
            else:  # Quiz mode
                # --- NEW 10-QUESTION PROMPT ---
                custom_prompt = f"""
                Generate a 10-question {education_level}-level {subject} quiz on the topic: "{prompt}".

                You MUST follow this exact, repeatable format for EACH of the 10 questions:

                [QUESTION X]
                What is the quiz question?
                [OPTIONS X]
                A. Option 1
                B. Option 2
                C. Option 3
                D. Option 4
                [ANSWER X]
                B. Option 2
                [EXPLANATION X]
                This is the explanation for why B is correct.
                """
            # --- END OF MODIFICATION ---
            
            try:
                # Stream response from Ollama
                response = ollama.generate(
                    model=model_name,
                    prompt=custom_prompt,
                    stream=True
                )
                
                for chunk in response:
                    full_response += chunk["response"]
                    message_placeholder.markdown(full_response + "▌")
                
                # --- ###################################### ---
                # ---     NEW QUIZ PARSING LOGIC         ---
                # --- ###################################### ---
                
                if mode == "Explain a Topic":
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                else: # Quiz Mode
                    try:
                        # Use re.findall with DOTALL flag (s) to find all blocks
                        questions = re.findall(r"\[QUESTION \d+\](.*?)(?=\[OPTIONS \d+\])", full_response, re.IGNORECASE | re.DOTALL)
                        options_texts = re.findall(r"\[OPTIONS \d+\](.*?)(?=\[ANSWER \d+\])", full_response, re.IGNORECASE | re.DOTALL)
                        answers = re.findall(r"\[ANSWER \d+\](.*?)(?=\[EXPLANATION \d+\])", full_response, re.IGNORECASE | re.DOTALL)
                        explanations = re.findall(r"\[EXPLANATION \d+\](.*?)(?=\[QUESTION \d+\]|$)", full_response, re.IGNORECASE | re.DOTALL)

                        # Check if we found a consistent number of parts
                        if not (len(questions) == len(options_texts) == len(answers) == len(explanations) and len(questions) > 0):
                            raise ValueError(f"Failed to parse all quiz parts. Found: {len(questions)} Qs, {len(options_texts)} Os, {len(answers)} As, {len(explanations)} Es.")

                        quiz_data = []
                        for q, o_text, a, e in zip(questions, options_texts, answers, explanations):
                            options_list = [opt.strip() for opt in o_text.strip().split('\n') if opt.strip()]
                            if not options_list:
                                raise ValueError("Found a question with no options.")
                            
                            quiz_data.append({
                                "question": q.strip(),
                                "options": options_list,
                                "answer": a.strip(),
                                "explanation": e.strip()
                            })

                        # Store in session state
                        st.session_state.quiz_ready = True # <-- CHANGED
                        st.session_state.quiz_active = False # <-- NEW
                        st.session_state.current_quiz_questions = quiz_data
                        # We no longer initialize index or user_answers here

                        # Display the quiz start message
                        start_message = f"Great! I've prepared a {len(quiz_data)}-question quiz for you. Let's begin!"
                        message_placeholder.markdown(start_message)
                        st.session_state.messages.append({"role": "assistant", "content": start_message})
                        
                        st.rerun() # <-- NEW: Rerun to show the "Start Quiz" button

                    except Exception as e:
                        logger.error(f"Failed to parse quiz response: {e}\nResponse: {full_response}")
                        error_msg = "❌ Sorry, I had trouble formatting the 10-question quiz. The AI's response was not in the correct format. Please try again."
                        message_placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.session_state.quiz_ready = False # <-- CHANGED
                        st.session_state.quiz_active = False # <-- CHANGED
                
                # --- ###################################### ---
                # ---   END OF NEW QUIZ PARSING LOGIC    ---
                # --- ###################################### ---

                # Log the successful interaction (for either mode)
                try:
                    update_user_progress(st.session_state["name"], subject, mode)
                except Exception as e:
                    logger.error(f"Failed to log progress: {e}")
                    pass
                
            except ollama.ResponseError as e:
                error_msg = f"❌ Model '{model_name}' not found. Please install it using: `ollama pull {model_name}`"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg}) # Add error to history
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg}) # Add error to history
        
    # This was moved into the mode-specific logic
    # st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    # --- USER IS NOT LOGGED IN ---
    
    st.title("🎓 AI Tutor")
    st.info("Please log in or sign up to continue.")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        authenticator.login()

    with tab2:
        try:
            if authenticator.register_user(): # Set pre_authorization to True
                st.success('User registered successfully! Please go to the Login tab.')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)

    if st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password, or sign up.')