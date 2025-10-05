import streamlit as st

def apply_custom_css():
    """Applies custom CSS for chat bubbles and overall styling."""
    css = """
    <style>
        /* General App Styling - Dark Theme */
        .stApp {
            background-color: #0f172a; /* Dark Slate Blue */
            color: #e2e8f0; /* Light Slate Grey for text */
        }
        
        /* Main title */
        h1 {
            color: #93c5fd; /* Light Blue */
            text-align: center;
            font-family: 'Arial', sans-serif;
        }

        /* Subheader description text - targeting the centered p tag */
        p[style*='text-align: center;'] {
            color: #94a3b8; /* Lighter Slate Grey */
        }

        /* Chat bubble container styling */
        .stChatMessage {
            border-radius: 20px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2); /* Darker shadow for dark theme */
            border: 1px solid #334155; /* Slate 700 border */
            max-width: 85%;
        }

        /* Assistant message styling (left side) - UPDATED */
        [data-testid="stChatMessage"]:not([style*="flex-direction: row-reverse;"]) {
             background-color: #334155; /* A lighter, distinct slate for the assistant */
             color: #cbd5e1; /* Lighter text for better contrast */
        }

        /* User message styling (right side) - UPDATED */
        [data-testid="stChatMessage"][style*="flex-direction: row-reverse;"] {
             background-color: #4f46e5; /* A rich indigo for the user */
             color: #ffffff; /* White text remains best here */
        }
        
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_header():
    """Renders the main header and description of the app."""
    st.title("🤖 MediLingua: Your Generative Medical Chatbot")
    st.markdown("<p style='text-align: center; font-size: 1.1rem;'>This chatbot uses AI to answer your questions based on a medical dataset.</p>", unsafe_allow_html=True)


def render_sidebar():
    """Renders the sidebar and displays the API key status."""
    with st.sidebar:
        st.header("Configuration")
        # Check if the secret key exists and display a status message
        if st.secrets.get("GOOGLE_API_KEY"):
            st.success("✅ API Key configured successfully.")
        else:
            st.error("❌ Google API Key not found.")
            st.info("Please create a file named .streamlit/secrets.toml and add your GOOGLE_API_KEY to it.")
        
        st.markdown("---")
        st.markdown("Created with Streamlit & Google Gemini.")


def render_chat_history(messages):
    """Renders the chat history from session state."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])