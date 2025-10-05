import requests
import json
import streamlit as st
import time
from typing import Union, List

def summarize_with_gemini(
    api_key: str,
    doctor_answer: Union[str, List[str]],
    user_question: str,
    max_retries: int = 2,
) -> str:
    """
    Summarizes one or more doctor's answers into a simple, patient-friendly explanation.
    Uses the Gemini API (2.5-flash-preview-05-20).
    Ensures unrelated content is ignored.
    """
    if not api_key:
        st.warning("⚠️ Google API key missing. Showing full response instead.")
        return (
            "\n\n".join(doctor_answer)
            if isinstance(doctor_answer, list)
            else doctor_answer
        )

    # Combine multiple answers into one coherent input if list provided
    if isinstance(doctor_answer, list):
        combined_answer = "\n\n---\n\n".join(doctor_answer)
    else:
        combined_answer = doctor_answer

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    )

    # Improved summarization prompt to ignore unrelated notes
    prompt = f"""
You are a professional medical assistant AI.
Summarize the following doctor's responses clearly and concisely for a patient.
⚠️ Focus ONLY on information relevant to the user's question. 
Ignore any unrelated medical details.

🩺 User's Question:
"{user_question}"

📋 Doctor's Answer(s):
"{combined_answer}"

✍️ Your Summary (medically correct, easy to understand, one short paragraph, no unrelated content):
"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=60
            )
            response.raise_for_status()
            result = response.json()

            if "candidates" in result and result["candidates"]:
                text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                return text if text else combined_answer

        except requests.exceptions.Timeout:
            st.warning("⏳ Summarization request timed out. Retrying...")
            time.sleep(1)
        except (requests.exceptions.RequestException, KeyError, IndexError):
            st.warning("⚠️ Summarization service error. Retrying...")
            time.sleep(1)

    # Fallback
    st.warning("⚕️ Could not summarize — showing full answer instead.")
    return combined_answer
