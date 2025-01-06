from flask import Flask, render_template, request
import yt_dlp
import os
import whisper
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Initialize Hugging Face T5 model for summarization
summarizer = pipeline("summarization", model="t5-small", framework="pt")

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle transcription
@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    # Get the YouTube URL from the form
    youtube_url = request.form.get('youtube_url')
    if not youtube_url:
        return "Error: Please provide a valid YouTube URL."

    # Download audio from the YouTube video
    try:
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'noplaylist': True,
            'outtmpl': './uploads/audio.%(ext)s',  # Force saving audio as 'audio.wav'
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract audio and ignore the video title
            info_dict = ydl.extract_info(youtube_url, download=True)
            # Directly set the filename to 'audio.wav'
            audio_file_path = './uploads/audio.wav'
    except Exception as e:
        return f"Error downloading audio: {str(e)}"

    # Transcribe the audio
    try:
        model = whisper.load_model("base")  # Use 'small', 'medium', or 'large' for better accuracy
        result = model.transcribe(audio_file_path)
        transcription = result['text']
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

    # Summarize the transcription using T5
    try:
        summary = summarize_text(transcription)
    except Exception as e:
        return f"Error summarizing transcription: {str(e)}"

    # Clean up by deleting the downloaded audio file after processing
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    # Return the summary
    return render_template('index.html', summary=summary)

# Function to summarize transcription using T5
def summarize_text(text):
    # Use Hugging Face's T5 model for summarization
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

if __name__ == '__main__':
    # Ensure the uploads folder exists
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')

    app.run(debug=True)
