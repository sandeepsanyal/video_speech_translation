import os
import threading
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
from pydub import AudioSegment
from faster_whisper import WhisperModel
from googletrans import Translator
import asyncio
import time
from gtts import gTTS

def extract_audio_from_video(video_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile("output/audio.wav")
        print("Audio extracted from video.")
    except Exception as e:
        print(f"Error extracting audio: {e}")

def transcribe_audio_worker():
    try:
        model_size = "base"  # You can choose other models like "tiny", "small", "medium", "large"
        model = WhisperModel(model_size, device="cpu")
        segments, info = model.transcribe("output/audio.wav", language="en")

        transcription = ""
        for segment in segments:
            transcription += f"{segment.text.strip()}\n"

        with open('output/transcription.txt', 'w') as file:
            file.write(transcription)
        print("Audio transcribed.")
    except Exception as e:
        print(f"Error transcribing audio: {e}")

async def translate_text():
    try:
        translator = Translator()
        with open('output/transcription.txt', 'r') as file:
            transcription = file.read()

        # Use asyncio to handle the coroutine
        translation = await translator.translate(transcription, src='en', dest='hi')
        with open('output/translation.txt', 'w') as file:
            file.write(translation.text)
        print("Text translated.")
    except Exception as e:
        print(f"Error translating text: {e}")

def translate_text_to_audio(text_file_path):
    try:
        with open(text_file_path, 'r') as file:
            translation = file.read()
        tts = gTTS(translation, lang='hi')
        tts.save("output/translated_audio.mp3")
        return AudioSegment.from_mp3("output/translated_audio.mp3")
    except Exception as e:
        print(f"Error translating text to audio: {e}")

def generate_combined_audio(video_clip):
    try:
        audio_segment = AudioSegment.from_wav("output/audio.wav")
        translated_audio_segment = translate_text_to_audio("output/translation.txt")
        translated_audio_segment.export("output/translated_audio.wav", format="wav")
        return AudioFileClip("output/translated_audio.wav")
    except Exception as e:
        print(f"Error generating combined audio: {e}")

def overlay_text_on_video(video_path, final_audio_clip):
    try:
        video_clip = VideoFileClip(video_path)
        with open('output/translation.txt', 'r') as file:
            translation = file.read()

        # Split translation into 5-second chunks
        translations = split_text_into_chunks(translation, 5)  # Assuming 120 words per minute

        # Extract start and end times of each segment in the video
        video_start_time = video_clip.start
        video_end_time = video_clip.end

        # Calculate the number of seconds in the video
        total_video_duration = video_end_time - video_start_time

        # Create a list to store text clips with their corresponding start and end times
        text_clips = []

        for i, text_chunk in enumerate(translations):
            # Calculate the start time for this chunk
            start_time_seconds = (i * 5) / total_video_duration
            # Ensure the start time is within the video bounds
            start_time_seconds = max(0, min(start_time_seconds, 1))

            # Create a TextClip with the text and position it at the bottom center
            txt_clip = TextClip(text_chunk, fontsize=24, color='white', font='Arial-Bold')
            txt_clip = txt_clip.set_position(('center', 'bottom'))

            # Calculate the end time for this chunk
            end_time_seconds = ((i + 1) * 5) / total_video_duration
            # Ensure the end time is within the video bounds
            end_time_seconds = max(0, min(end_time_seconds, 1))

            # Set the duration and start/end times for the text clip
            txt_clip = txt_clip.set_duration(end_time_seconds - start_time_seconds)
            txt_clip = txt_clip.set_start(start_time_seconds)

            # Append the text clip to the list
            text_clips.append(txt_clip)

        # Create a CompositeVideoClip with the video and the text clips
        final_video = CompositeVideoClip([video_clip] + text_clips)
        return final_video.set_audio(final_audio_clip)
    except Exception as e:
        print(f"Error overlaying text on video: {e}")

def split_text_into_chunks(text, duration_per_chunk):
    try:
        import re
        words = re.findall(r'\w+', text)
        num_words_per_chunk = len(words) // (duration_per_chunk * 24)  # 24fps as a rough estimate
        chunks = [' '.join(words[i:i+num_words_per_chunk]) for i in range(0, len(words), num_words_per_chunk)]
        return chunks
    except Exception as e:
        print(f"Error splitting text into chunks: {e}")

async def main():
    video_path = input("Enter the path to the video file: ")

    # Start threads for audio extraction and transcription
    threads = []
    for target in [extract_audio_from_video, transcribe_audio_worker]:
        thread = threading.Thread(target=target, args=(video_path,) if target == extract_audio_from_video else (), daemon=True)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Run the translation asynchronously
    await translate_text()

    video_clip = VideoFileClip(video_path)
    final_audio_clip = generate_combined_audio(video_clip)

    # Overlay text on video with the new audio
    final_video_clip = overlay_text_on_video(video_path, final_audio_clip)

    # Save the video with the new audio and subtitles
    output_video_path = "output/translated_video.mp4"
    try:
        final_video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        print(f"Translated video saved to: {output_video_path}")
    except Exception as e:
        print(f"Error saving video: {e}")

if __name__ == "__main__":
    asyncio.run(main())
