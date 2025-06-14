import streamlit as st
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.google import Gemini
from google.generativeai import get_file, upload_file
import google.generativeai as genai
import os
import tempfile
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import time

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("Video Summarizer Agent 📽️")
st.header("This App is Powered by Gemini")

def initialize_agent():
    return Agent(
        name="Video Summarizer",
        tools=[DuckDuckGo()],
        model=Gemini(id="gemini-2.0-flash-exp"),
        markdown=True
    )

multimodal_agent = initialize_agent()

video_file = st.file_uploader(
    "upload a Video File here", type=['mp4', 'mov', 'avi'], help="Upload a video file to summarize"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
       "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
   )
    
    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please provide a query to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    processed_video = upload_file(video_path)

                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    response = multimodal_agent.run(analysis_prompt, videos = [processed_video])
                    
                    st.subheader("Analysis Results")
                    st.markdown(response.content)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                Path(video_path).unlink(missing_ok=True)

    else:
        st.info("Upload a video file to begin analysis.")
                    

st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)    