def summarize_with_gemini(api_key, doctor_answer, user_question, max_retries=2):
    import requests, json, time, streamlit as st

    if not api_key:
        st.warning("⚠️ Google API Key missing. Showing full answer instead.")
        return doctor_answer if isinstance(doctor_answer, str) else "\n\n".join(doctor_answer)

    combined_answer = "\n\n---\n\n".join(doctor_answer) if isinstance(doctor_answer, list) else doctor_answer

    candidate_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-pro"]

    for model_name in candidate_models:
        prompt = f"""You are a professional AI medical assistant.

Summarize the doctor's responses clearly, accurately, and concisely for the patient.
Focus only on medically relevant information that directly answers the user's question.

User's Question:
"{user_question}"

Doctor's Answer(s):
"{combined_answer}"

Instructions:
- Provide a medically correct, patient-friendly summary in simple, clear language.
- List multiple points as bullets if possible.
- If the user's question lacks personal details (e.g., gender, age, weight), generate a generalized, gender-neutral summary. 
- Avoid gender-specific recommendations (e.g., consulting a gynecologist) unless the query explicitly mentions gender or related details.
- Dont forget to mention potential next steps, treatments, or lifestyle changes if doctor's answer has it mentioned.
- Always include a recommendation to consult a relevant doctor type (e.g., general practitioner, orthopedist) at the end of the summary, unless the doctor's answers already specify a consultation with a specific doctor type.
- For example, for back pain, recommend consulting an orthopedist or general practitioner unless the query or doctor's answers suggest a more specific specialist.
- If the doctor's response does not address the question, respond:
  "There is no information related to your question in the doctor's answer, so I generated the best possible answer based on the information provided."""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
                resp.raise_for_status()
                result = resp.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 404: break
                time.sleep(1)
            except Exception: time.sleep(1)

    st.warning("⚕️ Could not generate summary. Showing original answer.")
    return combined_answer
