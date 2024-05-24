import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
from youtube_transcript_api import YouTubeTranscriptApi
import faiss
import numpy as np
import openai
import pandas as pd
import pickle

# Function to extract YouTube links using Selenium
def get_youtube_links(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")

    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    # List to store video URLs
    all_urls = []

    # Fetch video links
    driver.get(url + "/videos")
    time.sleep(5)  # Wait for page to load
    links = driver.find_elements(By.TAG_NAME, 'a')
    video_urls = [link.get_attribute('href') for link in links if link.get_attribute('href') and 'watch?v=' in link.get_attribute('href')]
    all_urls.extend(video_urls)

    # Fetch shorts links
    driver.get(url + "/shorts")
    time.sleep(5)  # Wait for page to load
    links = driver.find_elements(By.TAG_NAME, 'a')
    shorts_urls = [link.get_attribute('href') for link in links if link.get_attribute('href') and 'shorts/' in link.get_attribute('href')]
    all_urls.extend(shorts_urls)

    driver.quit()
    
    # Remove duplicates and ensure valid YouTube links
    all_urls = list(set([url for url in all_urls if 'youtube.com/watch?v=' in url or 'youtube.com/shorts/' in url]))
    
    return all_urls

# Function to extract video IDs from URLs
def extract_video_ids(urls):
    video_ids = [url.split('watch?v=')[-1] if 'watch?v=' in url else url.split('shorts/')[-1] for url in urls]
    return list(set(video_ids))  # Remove duplicates

# Function to get transcriptions
def get_transcriptions(video_ids):
    transcriptions = []
    for video_id in video_ids:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = ' '.join([t['text'] for t in transcript])
            transcriptions.append(text)
        except Exception as e:
            st.write(f"Could not fetch transcript for video {video_id}: {e}")
            transcriptions.append(None)
    return transcriptions

# Function to chunk text into 50-word chunks
def chunk_text(text, chunk_size=50):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Embedding function provided by user
def get_embedding(text, api_key, model="text-embedding-3-small"):
    client = OpenAI(api_key=api_key)
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Streamlit UI
st.title("YouTube Transcription and Embedding Generator")
st.write("A valid URL should look something like this: https://www.youtube.com/@pratik_AI")

openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
youtube_channel_url = st.text_input("Enter the YouTube channel URL:")

if openai_api_key and youtube_channel_url:
    openai.api_key = openai_api_key

    if st.button("Start Processing"):
        st.write("Fetching YouTube links...")
        urls = get_youtube_links(youtube_channel_url)
        video_ids = extract_video_ids(urls)
        st.write(f"Found {len(video_ids)} video(s) and short(s).")

        st.write("Fetching transcriptions...")
        transcriptions = get_transcriptions(video_ids)
        st.write(f"Retrieved {len(transcriptions)} transcriptions.")

        st.write("Generating embeddings and creating FAISS index...")
        all_embeddings = []
        all_chunks = []
        for url, transcription in zip(urls, transcriptions):
            if transcription:
                chunks = chunk_text(transcription)
                for chunk in chunks:
                    embedding = get_embedding(chunk, openai_api_key)
                    all_chunks.append((url, chunk))
                    all_embeddings.append(embedding)

        dimension = len(all_embeddings[0])
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(all_embeddings))

        st.write("Saving FAISS index and transcriptions to files...")
        faiss.write_index(faiss_index, "faiss_index.idx")
        with open("transcriptions.pkl", "wb") as f:
            pickle.dump(all_chunks, f)
        st.write("Files saved: faiss_index.idx and transcriptions.pkl")

        st.session_state['faiss_index'] = faiss_index
        st.session_state['all_chunks'] = all_chunks
        st.session_state['api_key'] = openai_api_key
        st.success("Processing complete. You can now ask questions.")

# Function to answer questions using GPT-4 and FAISS
def answer_question(question, faiss_index, all_chunks, gptkey):
    embedding = get_embedding(question, gptkey)
    D, I = faiss_index.search(np.array([embedding]), k=5)
    relevant_chunks = [all_chunks[i] for i in I[0]]
    context = " ".join([chunk for url, chunk in relevant_chunks])
    system_prompt = "Use the following context to answer the question."
    user_prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    expected_format = "JSON"
    client = OpenAI(api_key=gptkey)
    chat_completion, *_ = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"system_prompt : {system_prompt}"},
            {"role": "user", "content": f"user_prompt : {user_prompt}"},
            {"role": "user", "content": f"expected_JSON_format : {expected_format}"}
        ],
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
    ).choices
    content = chat_completion.message.content
    return content

if 'faiss_index' in st.session_state and 'all_chunks' in st.session_state:
    question = st.text_input("Ask a question about the channel content:")
    if question:
        answer = answer_question(question, st.session_state['faiss_index'], st.session_state['all_chunks'], st.session_state['api_key'])
        st.write(f"Answer: {answer}")

st.write("Now you can use the FAISS index and transcriptions to create a question-answer bot with GPT-4.")
