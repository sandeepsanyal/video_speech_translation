import cv2
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
import torch
from faster_whisper import WhisperModel
from queue import Queue
import threading
import time
import subprocess
import sys
from gtts import gTTS

# Initialize queues for communication between threads
audio_queue = Queue()
text_queue = Queue()
translated_text_queue = Queue()

def extract_audio_from_video(video_path):
    video_clip = VideoFileClip(video_path)
    chunk_size = 6  # in seconds
    num_chunks = int(video_clip.duration / chunk_size) + 1
    
    audio_folder = r"output"
    os.makedirs(audio_folder, exist_ok=True)
    
    for i in range(num_chunks):
        start_time = i * chunk_size
        end_time = min((i + 1) * chunk_size, video_clip.duration)
        
        audio_chunk = video_clip.subclip(start_time, end_time).audio
        audio_file_path = os.path.join(audio_folder, f"extracted_audio_{i}.wav")
        audio_chunk.write_audiofile(audio_file_path)
        print(f"Extracted audio chunk: {audio_file_path}")
        audio_queue.put((start_time, end_time, audio_file_path))

def transcribe_audio(model, audio_file):
    result = model.transcribe(audio_file)
    text = " ".join([segment['text'] for segment in result["segments"]])
    print(f"Transcribed Text from {audio_file}: {text}")
    return text

def transcribe_audio_worker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "small"  # You can choose other sizes like 'tiny', 'small', 'medium', 'large'
    model = WhisperModel(model_size, device=device)
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        while True:
            if not audio_queue.empty():
                start_time, end_time, audio_file = audio_queue.get()
                future = executor.submit(transcribe_audio, model, audio_file)
                futures.append((start_time, end_time, future))
            
            # Collect results from completed futures
            for start_time, end_time, future in list(futures):
                if future.done():
                    text = future.result()
                    text_queue.put((start_time, end_time, text))
                    futures.remove((start_time, end_time, future))

def translate_text():
    from googletrans import Translator
    translator = Translator()
    
    while True:
        if not text_queue.empty():
            start_time, end_time, text = text_queue.get()
            try:
                translated_text = translator.translate(text, src='en', dest='es').text
                print(f"Translated Text: {translated_text}")
                translated_text_queue.put((start_time, end_time, translated_text))
            except Exception as e:
                print(f"Translation error: {e}")

def generate_tts_audio(translated_text):
    tts = gTTS(text=translated_text, lang='es')
    audio_file_path = os.path.join("output", f"tts_audio_{time.time()}.mp3")
    tts.save(audio_file_path)
    print(f"Generated TTS Audio: {audio_file_path}")
    return audio_file_path

def generate_combined_audio(video_clip):
    combined_audio_clips = []
    while True:
        if not translated_text_queue.empty():
            start_time, end_time, translated_text = translated_text_queue.get()
            tts_audio_path = generate_tts_audio(translated_text)
            tts_audio_clip = AudioFileClip(tts_audio_path)
            tts_audio_clip = tts_audio_clip.set_start(start_time).set_end(end_time)
            combined_audio_clips.append(tts_audio_clip)
        
        if len(combined_audio_clips) > 0:
            break
    
    return concatenate_videoclips(combined_audio_clips)

def overlay_text_on_video(video_path, final_audio_clip):
    video_clip = VideoFileClip(video_path)
    
    frame_count = 0
    frames_per_second = int(video_clip.fps)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / frames_per_second

        # Check for new translated texts that should be added to the buffer
        while not translated_text_queue.empty():
            start_time, end_time, translated_text = translated_text_queue.get()
            text_buffer.append((start_time, end_time, translated_text))
        
        # Remove old texts from the buffer
        text_buffer = [entry for entry in text_buffer if current_time <= entry[1]]
        
        # Overlay texts on the frame
        y_offset = 50  # Starting y-coordinate
        for start_time, end_time, text in text_buffer:
            cv2.putText(frame, text, (50, y_offset), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 40  # Move to the next line for the next subtitle
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = input("Enter the path to the video file: ")
    
    # Start threads for audio extraction, transcription, and translation
    threading.Thread(target=extract_audio_from_video, args=(video_path,), daemon=True).start()
    threading.Thread(target=transcribe_audio_worker, daemon=True).start()
    threading.Thread(target=translate_text, daemon=True).start()

    # Generate combined audio from translated text
    video_clip = VideoFileClip(video_path)
    final_audio_clip = generate_combined_audio(video_clip)

    # Overlay text on video with the new audio
    overlay_text_on_video(video_path, final_audio_clip)

    # Save the video with the new audio
    output_video_path = "output/translated_video.mp4"
    final_video_clip = video_clip.set_audio(final_audio_clip)
    final_video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
    print(f"Translated video saved to: {output_video_path}")

if __name__ == "__main__":
    main()