import streamlit as st
import time

def apply_custom_css():
    """Applies custom CSS for proper left-right chat alignment with enhanced colors."""
    css = """
    <style>
        .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }
        .block-container {
            max-width: 900px;
            margin: auto;
            padding-top: 1rem;
        }
        /* Chat message layout override */
        [data-testid="stChatMessage"] {
            display: flex !important;
            align-items: flex-start !important;
            margin-bottom: 0.75rem;
        }
        [data-testid="stChatMessage"] > div[data-testid="stMarkdownContainer"] {
            padding: 0.8rem 1rem;
            border-radius: 16px;
            max-width: 70%;
            line-height: 1.5;
            font-size: 0.95rem;
            word-wrap: break-word;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            transition: all 0.2s ease-in-out;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        /* Assistant (left) */
        [data-testid="stChatMessage"]:has(.stChatMessageContent[data-testid="assistant"]) {
            justify-content: flex-start !important;
        }
        [data-testid="stChatMessage"]:has(.stChatMessageContent[data-testid="assistant"]) 
        > div[data-testid="stMarkdownContainer"] {
            background-color: #1e293b;
            color: #f1f5f9;
            border: 1px solid #334155;
            text-align: left;
        }
        /* User (right) */
        [data-testid="stChatMessage"]:has(.stChatMessageContent[data-testid="user"]) {
            justify-content: flex-end !important;
        }
        [data-testid="stChatMessage"]:has(.stChatMessageContent[data-testid="user"]) 
        > div[data-testid="stMarkdownContainer"] {
            background-color: #2563eb;
            color: white;
            border: 1px solid #1d4ed8;
            text-align: right;
        }
        /* Expander (doctor notes) */
        .streamlit-expanderHeader {
            background: #111827;
            color: #cbd5e1;
            border: 1px solid #374151;
            border-radius: 10px;
        }
        .streamlit-expanderContent {
            background: #0b1220;
            border-left: 2px solid #334155;
        }
        /* Scrollbar style */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb { background-color: #334155; border-radius: 10px; }
        /* Header/title */
        h1 {
            text-align: center;
            color: #60a5fa;
            font-weight: 600;
        }
        p[style*='text-align: center;'] {
            color: #94a3b8;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header():
    st.title("🤖 MediLingua: Your Medical Assistant")
    st.markdown(
        "<p style='text-align: center; font-size: 1.1rem;'>Ask medical questions and get summarized answers from real doctor responses.</p>",
        unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        if st.secrets.get("GOOGLE_API_KEY"):
            st.success("✅ Google API Key configured.")
        else:
            st.error("❌ Missing API Key in `.streamlit/secrets.toml`.")
        st.markdown("---")
        st.markdown("💡 Built with **Streamlit** & **Gemini**.")


def render_chat_history(messages):
    """Render previous messages."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def bot_typing_animation(message_placeholder, final_text, delay=0.02):
    """
    Simulate bot typing animation in chat.
    """
    message_placeholder.markdown("")  # Empty initially
    displayed = ""
    for char in final_text:
        displayed += char
        message_placeholder.markdown(displayed)
        time.sleep(delay)
