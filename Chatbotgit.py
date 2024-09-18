import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import nltk
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# Set up the Streamlit page configuration
st.set_page_config(page_title="Software Testing FAQ Chatbot", page_icon="🟠", layout="centered")

# Custom CSS for improved button responsiveness and consistent answer background styling
st.markdown("""
    <style>
    /* Main background color */
    .main {
        background-color: #F5F5DC;
    }
    /* Bot chatbox styling */
    .chatbox.bot {
        background-color: #FFA500;
        color: #000000;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-weight: 600;
    }
    /* User chatbox styling */
    .chatbox.user {
        background-color: #333333;
        color: #FFFFFF;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: right;
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-weight: 600;
    }
    /* Button styling */
    .stButton>button {
        background-color: #FFA500;
        color: #000000;
        border-radius: 10px;
        padding: 5px 15px;
        border: none;
        font-size: 16px;
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s;
    }
    /* Button hover effect for smoother experience */
    .stButton>button:hover {
        background-color: #FFA500;
        transform: scale(1.05);
        cursor: pointer;
    }
    /* Answer chosen background styling */
    .answer-chosen {
        background-color: #FFA500;
        padding: 5px 10px;
        border-radius: 8px;
        color: #000000;
    }
    /* Spinner animation styling */
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #FFA500;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
        display: inline-block;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    /* Typing indicator styling */
    .typing-indicator {
        color: #000000;
        font-style: italic;
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-weight: 600;
    }
    /* Custom dropdown styling */
    .suggestion-dropdown {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 5px;
        max-height: 150px;
        overflow-y: auto;
        position: relative;
        z-index: 9999;
    }
    .suggestion-item {
        padding: 8px;
        cursor: pointer;
    }
    .suggestion-item:hover {
        background-color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

# Download WordNet data
nltk.download('wordnet')

# Load JSON Data
def load_faq_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Updated file path
faq_data = load_faq_data(r'C:\Users\uig83773\Desktop\Software Testing Bot using LLM\FinalDataset.json')

# Extract questions from JSON data
questions = [item['question'] for item in faq_data]
question_embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(questions, convert_to_tensor=True)

# Load Models
generator = AutoModelForCausalLM.from_pretrained('gpt2')
generator_tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Sentiment Analysis Model
sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# Load Intent Recognition Model
intent_model_name = "bhadresh-savani/bert-base-uncased-emotion"
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name)

# Initialize conversation history and state using Streamlit's session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "selected_suggestion" not in st.session_state:
    st.session_state.selected_suggestion = ""

# Function to format the answer nicely
def format_answer(brief, detailed):
    formatted_brief = f"**Brief Answer:**\n\n{brief}"
    
    detailed_parts = detailed.split('\n')
    formatted_detailed = "**Detailed Answer:**\n\n"
    for part in detailed_parts:
        part = part.strip()
        if part:
            if part.startswith('-'):
                formatted_detailed += f"- {part[1:].strip()}\n"
            else:
                formatted_detailed += f"\n{part}\n"
    
    return formatted_brief, formatted_detailed

# Function to generate a response
def generate_response(closest_match):
    formatted_brief, formatted_detailed = format_answer(closest_match['brief_answer'], closest_match['detailed_answer'])
    return {
        "question": closest_match['question'],
        "brief_answer": formatted_brief,
        "detailed_answer": formatted_detailed,
        "image": closest_match['image']
    }

# Function to display the image using the provided images folder path
def display_image(image_name):
    image_path = os.path.join(r'C:\Users\uig83773\Desktop\Software Testing Bot using LLM\Images', image_name)
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, use_column_width=True)
    else:
        st.warning(f"⚠️ Image {image_name} not found.")

# Enhanced function to recognize intent using a BERT-based model
def recognize_intent(user_input):
    greetings = ["hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening"]
    greeting_phrases = ["how are you", "what's up", "how's it going"]
    
    lower_input = user_input.lower()

    # Prioritize detecting FAQ-related queries over greetings
    faq_keywords = [
        "what is", "how do", "explain", "describe", "why", "when", "difference between", "discuss", "technique",
        "types of", "purpose of", "importance of", "how to ensure", "key elements of", "role of", 
        "how to", "challenges of", "how to handle", "best practices for", "strategies for", "approach to",
        "how would you", "how do you ensure", "how to implement", "considerations when", "concept of", 
        "how to validate", "how to manage", "importance of", "key metrics", "how to perform", "techniques used in"
    ]

    if any(keyword in lower_input for keyword in faq_keywords):
        return "faq_query"

    # Check for greetings if no FAQ-related keywords are detected
    if any(greeting in lower_input for greeting in greetings) or any(phrase in lower_input for phrase in greeting_phrases):
        return "greeting"
    
    # Default to FAQ query if no greeting keywords detected
    return "faq_query"

# Function to handle greeting and casual conversation
def handle_greeting(user_input):
    return "Hello! How can I assist you today with your software testing questions?"

# Improved real-time suggestion function for FAQ questions using exact phrase matching and cosine similarity
def suggest_questions(user_input):
    user_intent = recognize_intent(user_input)
    
    if user_intent == "greeting":
        return []

    # Extract the first 3-4 words from the user input
    first_words = " ".join(user_input.lower().split()[:4])

    # Try to find questions that start with these words
    matching_questions = [q for q in questions if q.lower().startswith(first_words)]
    
    # Combine exact matches with additional relevant suggestions
    suggestions = matching_questions[:5]
    
    # If we have fewer than 4-5 suggestions, supplement with cosine similarity results
    if len(suggestions) < 4:
        user_input_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([user_input], convert_to_tensor=True)
        similarities = cosine_similarity(user_input_embedding, question_embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:5]
        
        additional_suggestions = [questions[i] for i in top_indices if questions[i] not in suggestions]
        suggestions.extend(additional_suggestions[:5 - len(suggestions)])
    
    return suggestions

# Main chatbot interaction with multi-turn handling, typing animation, and spinner
def chatbot():
    st.markdown("<div class='chatbox bot'>Hi! I'm your Software Testing Chatbot.</div>", unsafe_allow_html=True)

    choice = st.radio("Would you like to view a list of questions or type your own?", ('View Questions', 'Type Your Own'))

    if choice == 'View Questions':
        st.markdown("### Here are the top 50 questions:")
        selected_question = st.selectbox("Select a question:", questions[:50])

        if selected_question:
            answer_choice = st.radio("Would you like a brief answer, a detailed answer, or both?", 
                                     ('Brief', 'Detailed', 'Both'), key="answer_choice")
            
            if st.button("Submit"):
                closest_match = next((item for item in faq_data if item['question'] == selected_question), None)

                if closest_match is None:
                    st.markdown("❌ Sorry, I couldn't find a close match for your question. Please try rephrasing or ask a different question.")
                    return

                # Show typing indicator before displaying the answer
                with st.spinner("Bot is typing..."):
                    time.sleep(1.5)

                response = generate_response(closest_match)
                st.session_state.conversation_history.append({
                    "user": selected_question,
                    "question": response['question'],
                    "brief_answer": response['brief_answer'],
                    "detailed_answer": response['detailed_answer'],
                    "image": response['image']
                })

                if answer_choice == 'Brief':
                    st.markdown(f"<div class='answer-chosen'>{response['brief_answer']}</div>", unsafe_allow_html=True)
                elif answer_choice == 'Detailed':
                    st.markdown(f"<div class='answer-chosen'>{response['detailed_answer']}</div>", unsafe_allow_html=True)
                elif answer_choice == 'Both':
                    st.markdown(f"<div class='answer-chosen'>{response['brief_answer']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='answer-chosen'>{response['detailed_answer']}</div>", unsafe_allow_html=True)

                display_image(response['image'])

    elif choice == 'Type Your Own':
        # Real-time input capture
        user_input = st.text_input("🔍 Start typing your question:", key="user_input")

        # Real-time suggestions
        if user_input:
            suggestions = suggest_questions(user_input)
            if suggestions:
                st.markdown("<div class='suggestion-dropdown'>", unsafe_allow_html=True)
                for suggestion in suggestions:
                    if st.button(suggestion, key=suggestion):
                        st.session_state.selected_suggestion = suggestion  # Store the selected suggestion
                        break
                st.markdown("</div>", unsafe_allow_html=True)

        # Using a form to submit without an additional submit button
        with st.form("submit_form", clear_on_submit=True):
            if st.form_submit_button("Submit"):
                if st.session_state.selected_suggestion:
                    user_input = st.session_state.selected_suggestion
                else:
                    user_input = st.session_state.get("user_input", user_input)

                closest_match = next((item for item in faq_data if item['question'].lower() == user_input.lower()), None)
                
                if closest_match is None:
                    st.markdown("❌ Sorry, I couldn't find a close match for your question. However, here are some frequently asked questions:")
                    selected_question = st.selectbox("Select a question from the list:", questions[:50])

                    if selected_question:
                        # Show typing indicator before displaying the answer
                        with st.spinner("Bot is typing..."):
                            time.sleep(1.5)

                        closest_match = next((item for item in faq_data if item['question'] == selected_question), None)
                        response = generate_response(closest_match)

                        st.session_state.conversation_history.append({
                            "user": selected_question,
                            "question": response['question'],
                            "brief_answer": response['brief_answer'],
                            "detailed_answer": response['detailed_answer'],
                            "image": response['image']
                        })

                        answer_choice = st.radio("Would you like a brief answer, a detailed answer, or both?", 
                                                ('Brief', 'Detailed', 'Both'), key="answer_choice_after_fallback")

                        if answer_choice == 'Brief':
                            st.markdown(f"<div class='answer-chosen'>{response['brief_answer']}</div>", unsafe_allow_html=True)
                        elif answer_choice == 'Detailed':
                            st.markdown(f"<div class='answer-chosen'>{response['detailed_answer']}</div>", unsafe_allow_html=True)
                        elif answer_choice == 'Both':
                            st.markdown(f"<div class='answer-chosen'>{response['brief_answer']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='answer-chosen'>{response['detailed_answer']}</div>", unsafe_allow_html=True)

                        display_image(response['image'])

                    return

                # Show typing indicator before displaying the answer
                with st.spinner("Bot is typing..."):
                    time.sleep(1.5)

                response = generate_response(closest_match)
                answer_choice = st.radio("Would you like a brief answer, a detailed answer, or both?", 
                                        ('Brief', 'Detailed', 'Both'), key="answer_choice")

                if answer_choice == 'Brief':
                    st.markdown(f"<div class='answer-chosen'>{response['brief_answer']}</div>", unsafe_allow_html=True)
                elif answer_choice == 'Detailed':
                    st.markdown(f"<div class='answer-chosen'>{response['detailed_answer']}</div>", unsafe_allow_html=True)
                elif answer_choice == 'Both':
                    st.markdown(f"<div class='answer-chosen'>{response['brief_answer']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='answer-chosen'>{response['detailed_answer']}</div>", unsafe_allow_html=True)

                display_image(response['image'])

# Streamlit UI
def main():
    chatbot()

    if st.checkbox("📜 Show Conversation History"):
        for entry in st.session_state.conversation_history:
            st.markdown(f"<div class='chatbox user'>{entry['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-chosen'>{entry.get('brief_answer')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-chosen'>{entry.get('detailed_answer')}</div>", unsafe_allow_html=True)
            display_image(entry['image'])

        st.markdown("""
            <hr>
            <div style='text-align: center;'>
                <small>Powered by Hugging Face & Streamlit</small>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
