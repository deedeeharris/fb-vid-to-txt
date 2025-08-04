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
import sys
import tempfile
import ffmpeg

# --- Utility and Core Functions ---

def setup_logging(log_dir):
    """
    Sets up logging to output to both a file and the console (terminal).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the lowest level to capture all logs

    # Prevent adding handlers multiple times in Streamlit's rerun-heavy environment
    if not logger.handlers:
        # Formatter for the logs
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler - saves logs to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler - prints logs to the console/terminal
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    logging.info(f"Logging initialized. Log file at: {log_file}")


def sanitize_filename(filename):
    """Cleans a string to be a valid filename."""
    logging.info(f"Sanitizing filename: '{filename}'")
    # Remove problematic characters like parentheses, etc.
    sanitized = re.sub(r'[\\/*?:"<>|()]', "", filename)
    sanitized = sanitized.replace(' ', '_')
    # Normalize unicode characters
    sanitized = unicodedata.normalize('NFKD', sanitized).encode('ASCII', 'ignore').decode('ASCII')
    logging.info(f"Sanitized name: '{sanitized}'")
    return sanitized

def extract_audio(video_file, output_file):
    """Extracts audio from a video file and saves it as an MP3 file."""
    logging.info(f"Starting audio extraction from '{os.path.basename(video_file)}' to '{os.path.basename(output_file)}'.")
    st.write(f"Extracting audio from {os.path.basename(video_file)}...")
    try:
        # Remove existing output file if it exists to avoid conflicts
        if os.path.exists(output_file):
            os.remove(output_file)
            logging.info(f"Removed existing output file: {output_file}")
        
        ffmpeg.input(video_file).output(output_file, acodec='libmp3lame', loglevel='quiet').run(overwrite_output=True)
        
        # Validate that the output file was created successfully
        if not os.path.exists(output_file):
            error_message = f"Audio extraction failed: Output file '{os.path.basename(output_file)}' was not created."
            st.error(error_message)
            logging.error(error_message)
            raise RuntimeError(error_message)
        
        # Check if the output file has content
        if os.path.getsize(output_file) == 0:
            error_message = f"Audio extraction failed: Output file '{os.path.basename(output_file)}' is empty."
            st.error(error_message)
            logging.error(error_message)
            raise RuntimeError(error_message)
        
        st.write(f"Audio extracted for {os.path.basename(video_file)}.")
        logging.info(f"Audio extraction successful. Output file size: {os.path.getsize(output_file)} bytes.")
    except FileNotFoundError:
        error_message = "FFmpeg is not installed or not found in system PATH. Please install FFmpeg first."
        st.error(error_message)
        logging.critical(error_message) # Use critical for fatal setup errors
        raise
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr is not None else str(e)
        st.error(f"An error occurred during audio extraction: {error_message}")
        logging.error(f"FFmpeg error: {error_message}")
        raise

def generate_random_string(length=8):
    """Generates a random string for unique filenames."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def download_facebook_video(url, download_dir):
    """Downloads a video from a URL using yt-dlp."""
    logging.info(f"Attempting to download video from URL: {url}")
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
            base_filename = os.path.basename(filename)
            st.success(f"Downloaded: {base_filename}")
            logging.info(f"Successfully downloaded video to '{base_filename}'")
            return base_filename
        except Exception as e:
            error_message = f"An error occurred during download: {str(e)}"
            st.error(error_message)
            logging.error(error_message)
            return None

def get_gpt4o_response(api_key, user_input, system_prompt):
    """Gets a response from the OpenAI Chat API."""
    logging.info("Sending request to OpenAI GPT-4o API.")
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Corrected from gpt-4.1 to a valid model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0 # As specified for deterministic output
        )
        logging.info("Successfully received response from GPT-4o API.")
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
    
    logging.info(f"--- Starting processing for file: {filename} ---")
    
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        logging.error(f"File not found during processing: {file_path}")
        return None, None

    st.info(f"Processing {filename}...")
    
    audio_file_path = file_path
    if filename.lower().endswith(('.mp4', '.webm', '.mpeg')):
        try:
            sanitized_base = os.path.splitext(filename)[0]
            output_audio_file = os.path.join(paths['to_transcribe'], f"{sanitized_base}.mp3")
            logging.info(f"Attempting to extract audio from video file: {filename}")
            extract_audio(file_path, output_audio_file)
            audio_file_path = output_audio_file
            logging.info(f"Successfully extracted audio to: {os.path.basename(output_audio_file)}")
        except Exception as e:
            error_msg = f"Could not extract audio from {filename}. Error: {str(e)}"
            st.error(error_msg)
            logging.error(f"Audio extraction failed for {filename}: {type(e).__name__}: {str(e)}")
            return None, None

    logging.info(f"Starting transcription for '{os.path.basename(audio_file_path)}'...")
    st.write(f"Transcribing {os.path.basename(audio_file_path)}...")
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language=source_lang_code
            )
        transcription_text = transcription.text
        st.text_area("Original Transcription", transcription_text, height=150, key=f"trans_{filename}")
        logging.info("Transcription successful.")
    except Exception as e:
        st.error(f"Transcription Error for {filename}: {str(e)}")
        logging.error(f"Transcription failed for {filename}: {e}")
        return None, None

    st.write(f"Analyzing and translating with GPT-4.1...")
    analyzed_text_ai = get_gpt4o_response(api_key, transcription_text, system_prompt)
    st.text_area(f"AI Analysis & Translation", analyzed_text_ai, height=250, key=f"ai_{filename}")
    
    final_text_content = (
        f"--- Original Transcription ---\n{transcription_text}\n\n"
        f"--- AI Analysis & Translation (Hebrew) ---\n{analyzed_text_ai}"
    )
    
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    
    try:
        logging.info(f"Moving processed file '{filename}' to done folder.")
        shutil.move(file_path, os.path.join(paths['done_vids'], filename))
        if audio_file_path != file_path and os.path.exists(audio_file_path):
            logging.info(f"Removing temporary audio file '{os.path.basename(audio_file_path)}'.")
            os.remove(audio_file_path)
    except Exception as e:
        logging.error(f"Error during file cleanup for {filename}: {e}")

    st.success(f"Finished processing {filename}.")
    logging.info(f"--- Successfully finished processing file: {filename} ---")
    return txt_filename, final_text_content

def run_app():
    """The main application logic, shown after password authentication."""
    st.markdown("""
    <div dir="rtl" style="text-align: center;">
        <p style="font-size: 0.9em;">לשאלות ותקלות יש לפנות ל<a href="https://wa.me/972545660439" target="_blank" style="color: #ff6347;">ידידיה הריס</a></p>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.secrets["openai_api_key"]
    system_prompt = st.secrets["system_prompt"]

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

    # Use a relative directory path that's writable in Streamlit Community Cloud
    base_temp_dir = os.path.join(".", "temp_files")
    paths = {
        'logs': os.path.join(base_temp_dir, 'logs'),
        'to_transcribe': os.path.join(base_temp_dir, 'to_transcribe'),
        'done_vids': os.path.join(base_temp_dir, 'done_vids')
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    if uploaded_files:
        logging.info(f"Detected {len(uploaded_files)} uploaded files.")
        for uploaded_file in uploaded_files:
            sanitized_name = sanitize_filename(uploaded_file.name)
            save_path = os.path.join(paths['to_transcribe'], sanitized_name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logging.info(f"Saved uploaded file to '{save_path}'.")

    if st.button("Start Transcription and Translation", type="primary"):
        logging.info("'Start' button clicked.")
        st.session_state.processed_files = {}
        selected_files = []
        
        if fb_video_url:
            with st.spinner("Downloading video..."):
                downloaded_file = download_facebook_video(fb_video_url, paths['to_transcribe'])
                if downloaded_file:
                    selected_files.append(downloaded_file)
        
        files_in_dir = os.listdir(paths['to_transcribe'])
        logging.info(f"Files found in 'to_transcribe' directory: {files_in_dir}")
        selected_files.extend([f for f in files_in_dir if f not in selected_files])
        
        if selected_files:
            logging.info(f"Starting processing for {len(selected_files)} files: {selected_files}")
            successful_files = 0
            failed_files = 0
            with st.spinner("Processing files... This may take a few minutes."):
                for filename in selected_files:
                    txt_filename, content = process_single_file(api_key, system_prompt, filename, source_lang_code, paths)
                    if txt_filename and content:
                        st.session_state.processed_files[txt_filename] = content
                        successful_files += 1
                    else:
                        failed_files += 1
            
            if failed_files == 0:
                st.success(f"All {successful_files} files processed successfully!")
                logging.info(f"All {successful_files} files processed successfully.")
            elif successful_files == 0:
                st.error(f"Failed to process all {failed_files} files.")
                logging.error(f"Failed to process all {failed_files} files.")
            else:
                st.warning(f"Processed {successful_files} files successfully, {failed_files} files failed.")
                logging.warning(f"Processing completed with {successful_files} successes and {failed_files} failures.")
        else:
            st.warning("Please upload at least one file or provide a valid Facebook video URL.")
            logging.warning("No files selected or found for processing.")

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

def main():
    st.set_page_config(layout="centered")
    st.title("Audio/Video Transcription App")

    # Setup logging at the very beginning
    # Use a relative directory path that's writable in Streamlit Community Cloud
    base_temp_dir = os.path.join(".", "temp_files")
    log_path = os.path.join(base_temp_dir, 'logs')
    os.makedirs(log_path, exist_ok=True)
    setup_logging(log_path)

    try:
        correct_password = st.secrets["login_pass"]
    except (KeyError, FileNotFoundError):
        st.error("FATAL: `login_pass` not found in Streamlit secrets. The app cannot start.")
        logging.critical("`login_pass` not found in Streamlit secrets.")
        return

    password = st.text_input("Enter Password to Continue", type="password")

    if password == correct_password:
        logging.info("Password correct. Loading main application.")
        run_app()
    elif password:
        st.error("Password incorrect. Please try again.")
        logging.warning("Incorrect password entered.")
    else:
        st.info("Please enter the password to use the application.")

if __name__ == "__main__":
    main()
