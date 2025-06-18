import os
import shutil
import streamlit as st
from openai import OpenAI
import yt_dlp
import re
import unicodedata
import random
import string
import logging
import tempfile
import ffmpeg

# --- Utility and Core Functions ---

def setup_logging(log_dir):
    """Sets up logging to a file in the specified directory."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def sanitize_filename(filename):
    """Cleans a string to be a valid filename."""
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    filename = filename.replace(' ', '_')
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    return filename

def extract_audio(video_file, output_file):
    """Extracts audio from a video file and saves it as an MP3 file."""
    st.write(f"Extracting audio from {os.path.basename(video_file)}...")
    try:
        ffmpeg.input(video_file).output(output_file, acodec='libmp3lame', loglevel='quiet').run(overwrite_output=True)
        st.write(f"Audio extracted for {os.path.basename(video_file)}.")
    except FileNotFoundError:
        error_message = "FFmpeg is not installed or not found in system PATH. Please install FFmpeg first."
        st.error(error_message)
        logging.error(error_message)
        raise
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
        st.error(f"An error occurred during audio extraction: {error_message}")
        logging.error(f"FFmpeg error: {error_message}")
        raise

def generate_random_string(length=8):
    """Generates a random string for unique filenames."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def download_facebook_video(url, download_dir):
    """Downloads a video from a URL using yt-dlp."""
    random_id = generate_random_string()
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(download_dir, f'fb_video_{random_id}.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            st.write(f"Downloading video from URL...")
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            st.success(f"Downloaded: {os.path.basename(filename)}")
            return os.path.basename(filename)
        except Exception as e:
            error_message = f"An error occurred during download: {str(e)}"
            st.error(error_message)
            logging.error(error_message)
            return None

def get_gpt4o_response(api_key, user_input, system_prompt):
    """Gets a response from the OpenAI Chat API."""
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        error_message = f"OpenAI API Error: {str(e)}"
        st.error(error_message)
        logging.error(error_message)
        return f"Error: Could not get response from AI. Details: {e}"

def process_single_file(api_key, system_prompt, filename, source_lang_code, paths):
    """Processes a single media file: transcribes, analyzes, and returns the result."""
    client = OpenAI(api_key=api_key)
    file_path = os.path.join(paths['to_transcribe'], filename)
    
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None, None

    st.info(f"Processing {filename}...")
    
    # Extract audio if it's a video file
    audio_file_path = file_path
    if filename.lower().endswith(('.mp4', '.webm', '.mpeg')):
        try:
            sanitized_base = sanitize_filename(os.path.splitext(filename)[0])
            output_audio_file = os.path.join(paths['to_transcribe'], f"{sanitized_base}.mp3")
            extract_audio(file_path, output_audio_file)
            audio_file_path = output_audio_file
        except Exception:
            st.error(f"Could not extract audio from {filename}. Skipping.")
            return None, None

    # Transcribe the audio
    st.write(f"Transcribing {os.path.basename(audio_file_path)}...")
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language=source_lang_code
            )
        transcription_text = transcription.text
        st.text_area("Original Transcription", transcription_text, height=150, key=f"trans_{filename}")
    except Exception as e:
        st.error(f"Transcription Error for {filename}: {str(e)}")
        return None, None

    # Get AI analysis and translation
    st.write(f"Analyzing and translating with GPT-4o...")
    analyzed_text_ai = get_gpt4o_response(api_key, transcription_text, system_prompt)
    st.text_area(f"AI Analysis & Translation", analyzed_text_ai, height=250, key=f"ai_{filename}")
    
    # Combine results into a single text content
    final_text_content = (
        f"--- Original Transcription ---\n{transcription_text}\n\n"
        f"--- AI Analysis & Translation (Hebrew) ---\n{analyzed_text_ai}"
    )
    
    # Create the final .txt filename
    txt_filename = sanitize_filename(os.path.splitext(filename)[0]) + '.txt'
    
    # Cleanup: Move original file and delete temporary audio
    try:
        shutil.move(file_path, os.path.join(paths['done_vids'], filename))
        if audio_file_path != file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
    except Exception as e:
        logging.error(f"Error during file cleanup for {filename}: {e}")

    st.success(f"Finished processing {filename}.")
    return txt_filename, final_text_content

def run_app():
    """The main application logic, shown after password authentication."""
    st.markdown("""
    <div dir="rtl" style="text-align: center;">
        <p style="font-size: 0.9em;">לשאלות ותקלות יש לפנות ל<a href="https://wa.me/972545660439" target="_blank" style="color: #ff6347;">ידידיה הריס</a></p>
    </div>
    """, unsafe_allow_html=True)

    # Retrieve secrets
    api_key = st.secrets["openai_api_key"]
    system_prompt = st.secrets["system_prompt"]

    # Setup temporary directories
    base_temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_transcriber_app")
    paths = {
        'logs': os.path.join(base_temp_dir, 'logs'),
        'to_transcribe': os.path.join(base_temp_dir, 'to_transcribe'),
        'done_vids': os.path.join(base_temp_dir, 'done_vids')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    setup_logging(paths['logs'])

    # --- UI Elements ---
    source_language = st.selectbox(
        "Select source language of the media",
        ["Arabic", "Hebrew", "English"], index=0
    )
    language_codes = {"Arabic": "ar", "Hebrew": "he", "English": "en"}
    source_lang_code = language_codes[source_language]

    fb_video_url = st.text_input("Enter Facebook video URL (optional)")
    uploaded_files = st.file_uploader(
        "Or upload local audio/video files",
        type=['mp4', 'm4a', 'mp3', 'webm', 'mpga', 'wav', 'mpeg'],
        accept_multiple_files=True
    )

    # Save uploaded files to the temporary directory
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(paths['to_transcribe'], uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

    if st.button("Start Transcription and Translation", type="primary"):
        st.session_state.processed_files = {} # Clear previous results
        selected_files = []
        
        if fb_video_url:
            with st.spinner("Downloading video..."):
                downloaded_file = download_facebook_video(fb_video_url, paths['to_transcribe'])
                if downloaded_file:
                    selected_files.append(downloaded_file)
        
        files_in_dir = os.listdir(paths['to_transcribe'])
        selected_files.extend([f for f in files_in_dir if f not in selected_files])
        
        if selected_files:
            with st.spinner("Processing files... This may take a few minutes."):
                for filename in selected_files:
                    txt_filename, content = process_single_file(api_key, system_prompt, filename, source_lang_code, paths)
                    if txt_filename and content:
                        st.session_state.processed_files[txt_filename] = content
            st.success("All files processed!")
        else:
            st.warning("Please upload at least one file or provide a valid Facebook video URL.")

    # Display download buttons using session state
    if 'processed_files' in st.session_state and st.session_state.processed_files:
        st.markdown("---")
        st.header("Download Transcripts")
        for filename, content in st.session_state.processed_files.items():
            st.download_button(
                label=f"⬇️ Download {filename}",
                data=content,
                file_name=filename,
                mime="text/plain"
            )

# --- Main Execution with Password Gate ---
def main():
    st.set_page_config(layout="centered")
    st.title("Audio/Video Transcription App")

    try:
        correct_password = st.secrets["login_pass"]
    except (KeyError, FileNotFoundError):
        st.error("FATAL: `login_pass` not found in Streamlit secrets. The app cannot start.")
        st.info("Please create a .streamlit/secrets.toml file and add a `login_pass` key.")
        return

    # Password input
    password = st.text_input("Enter Password to Continue", type="password")

    if password == correct_password:
        run_app()
    elif password: # If user entered something, but it's wrong
        st.error("Password incorrect. Please try again.")
    else: # If the field is empty
        st.info("Please enter the password to use the application.")

if __name__ == "__main__":
    main()
