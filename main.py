import os
import shutil
import streamlit as st
from deep_translator import GoogleTranslator
from openai import OpenAI
import yt_dlp
import re
import unicodedata
import random
import string
import logging
import tempfile

import ffmpeg # for extracting audio from the vid file, make sure ffmpeg-python is installed

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
    # Remove invalid characters
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    return filename

def translate_text_google(text, lang='hebrew'):
    """
    Translates text using Google Translate.
    Note: This function is defined but not used in the primary workflow,
    as GPT-4o is used for a more advanced translation/analysis.
    """
    if '\n' in text:
        text = text.replace('\n','. ')
    text = text + '.'
    try:
        translated_text = GoogleTranslator(source='auto', target=lang).translate(text)
    except Exception as e:
        error_message = f"Google Translate Error: {str(e)}"
        st.error(error_message)
        logging.error(error_message)
        translated_text = text
    return translated_text

def extract_audio(video_file, output_file):
    """
    Extracts audio from a video file and saves it as an MP3 file using ffmpeg-python.
    """
    st.write(f"Extracting audio from {os.path.basename(video_file)}...")
    try:
        ffmpeg.input(video_file).output(output_file, acodec='libmp3lame', loglevel='quiet').run(overwrite_output=True)
        st.write(f"Audio extracted and saved to {os.path.basename(output_file)}")
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
            # yt-dlp already saves the file, we just need the final path
            filename = ydl.prepare_filename(info)
            st.success(f"Downloaded: {os.path.basename(filename)}")
            return os.path.basename(filename)
        except Exception as e:
            error_message = f"An error occurred during download: {str(e)}"
            st.error(error_message)
            logging.error(error_message)
            return None

def get_gpt4o_response(api_key, user_input, system_prompt, temperature=0.7, model='gpt-4o'):
    """Gets a response from the OpenAI Chat API."""
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        error_message = f"OpenAI API Error: {str(e)}"
        st.error(error_message)
        logging.error(error_message)
        return f"Error: Could not get response from AI. Details: {e}"


def transcribe_and_process_files(api_key, system_prompt, selected_files, source_lang_code, paths):
    """Main function to transcribe, translate, and manage files."""
    client = OpenAI(api_key=api_key)
    supported_formats = ('.mp4', '.m4a', '.mp3', '.webm', '.mpga', '.wav', '.mpeg')
    final_txt_path = None

    for filename in selected_files:
        if not filename.lower().endswith(supported_formats):
            continue

        file_path = os.path.join(paths['to_transcribe'], filename)
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            continue

        st.info(f"Processing {filename}...")
        
        # If the file is a video, extract the audio
        audio_file_path = file_path
        if filename.lower().endswith(('.mp4', '.webm', '.mpeg')):
            try:
                # Use a sanitized name for the audio file
                sanitized_base = sanitize_filename(os.path.splitext(filename)[0])
                output_audio_file = os.path.join(paths['to_transcribe'], f"{sanitized_base}.mp3")
                extract_audio(file_path, output_audio_file)
                audio_file_path = output_audio_file
            except Exception as e:
                st.error(f"Could not extract audio from {filename}. Skipping.")
                logging.error(f"Audio extraction failed for {filename}: {e}")
                continue

        st.write(f"Transcribing {os.path.basename(audio_file_path)}...")
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language=source_lang_code
                )
            transcription_text = transcription.text
            st.write("Transcription successful.")
            st.text_area("Original Transcription", transcription_text, height=150)
        except Exception as e:
            error_message = f"Transcription Error for {filename}: {str(e)}"
            st.error(error_message)
            logging.error(error_message)
            continue

        # Save the transcription and AI analysis to a text file
        txt_filename = sanitize_filename(os.path.splitext(filename)[0]) + '.txt'
        final_txt_path = os.path.join(paths['transcribed_txts'], txt_filename)

        with open(final_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write("--- Original Transcription ---\n")
            txt_file.write(transcription_text)
            txt_file.write("\n\n")

            # Get AI analysis and translation
            st.write(f"Analyzing and translating with GPT-4o...")
            analyzed_text_ai = get_gpt4o_response(api_key, transcription_text, system_prompt)
            st.write("AI analysis complete.")
            st.text_area(f"AI Analysis & Translation for {filename}", analyzed_text_ai, height=250)
            
            txt_file.write("--- AI Analysis & Translation (Hebrew) ---\n")
            txt_file.write(analyzed_text_ai)

        # Move the processed original file to the 'done' folder
        try:
            shutil.move(file_path, os.path.join(paths['done_vids'], filename))
        except Exception as e:
            logging.error(f"Could not move original file {filename}: {e}")

        # If an audio file was extracted, remove it
        if audio_file_path != file_path and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except Exception as e:
                logging.error(f"Could not delete temporary audio file {audio_file_path}: {e}")
        
        st.success(f"Finished processing {filename}. Results saved to {final_txt_path}")

    return final_txt_path

# --- Streamlit App Main Function ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="centered")
    st.title("Audio/Video Transcription and Translation App")

    # Setup temporary directories for the session
    base_temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_transcriber_app")
    paths = {
        'logs': os.path.join(base_temp_dir, 'logs'),
        'to_transcribe': os.path.join(base_temp_dir, 'to_transcribe'),
        'transcribed_txts': os.path.join(base_temp_dir, 'transcribed_txts'),
        'done_vids': os.path.join(base_temp_dir, 'done_vids')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    setup_logging(paths['logs'])

    # Add contact info
    st.markdown("""
    <div dir="rtl" style="text-align: center;">
        <p style="font-size: 0.9em;">לשאלות ותקלות יש לפנות ל<a href="https://wa.me/972545660439" target="_blank" style="color: #ff6347;">ידידיה הריס</a></p>
    </div>
    """, unsafe_allow_html=True)

    # Check for secrets
    try:
        api_key = st.secrets["openai_api_key"]
        system_prompt = st.secrets["system_prompt"]
    except (KeyError, FileNotFoundError):
        st.error("ERROR: `openai_api_key` or `system_prompt` not found in Streamlit secrets.")
        st.info("Please create a .streamlit/secrets.toml file with your credentials. See documentation for details.")
        return

    # --- UI Elements ---
    
    # Language selection
    source_language = st.selectbox(
        "Select source language of the media",
        ["Arabic", "Hebrew", "English"],
        index=0  # Default to Arabic
    )
    language_codes = {"Arabic": "ar", "Hebrew": "he", "English": "en"}
    source_lang_code = language_codes[source_language]

    # Input for Facebook video URL
    fb_video_url = st.text_input("Enter Facebook video URL (optional)")

    # File uploader for local files
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

    # List files available for processing
    try:
        files_in_dir = os.listdir(paths['to_transcribe'])
        files_to_process = [f for f in files_in_dir if f.lower().endswith(('.mp4', '.m4a', '.mp3', '.webm', '.mpga', '.wav', '.mpeg'))]
    except FileNotFoundError:
        files_to_process = []

    if not files_to_process and not fb_video_url:
        st.info("Upload a file or provide a Facebook URL to begin.")

    if st.button("Start Transcription and Translation", type="primary"):
        selected_files = []
        
        if fb_video_url:
            with st.spinner("Downloading video..."):
                downloaded_file = download_facebook_video(fb_video_url, paths['to_transcribe'])
                if downloaded_file:
                    selected_files.append(downloaded_file)
        
        # Add any uploaded files to the list to be processed
        if files_to_process:
            selected_files.extend(files_to_process)
        
        # Remove duplicates
        selected_files = list(set(selected_files))

        if selected_files:
            with st.spinner("Processing files... This may take a few minutes."):
                final_txt_path = transcribe_and_process_files(api_key, system_prompt, selected_files, source_lang_code, paths)
            
            st.success("All selected files have been processed.")
            if final_txt_path:
                st.info(f"You can find the final text file here: {final_txt_path}")
                # os.startfile is not cross-platform and won't work in cloud deployments.
                # Providing the path is a more robust solution.
        else:
            st.warning("Please upload at least one file or provide a valid Facebook video URL.")

if __name__ == "__main__":
    main()
