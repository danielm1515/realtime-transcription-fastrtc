import os
import logging
import json
import asyncio
import uuid
from typing import AsyncGenerator
import aiohttp

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from langchain_ollama import ChatOllama
import chromadb
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    AlgoOptions,
    SileroVadOptions,
    audio_to_bytes,
)

from utils.logger_config import setup_logging
from utils.device import get_device
from utils.turn_server import get_rtc_credentials
from utils.model import initialize_whisper_model

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


UI_MODE = os.getenv("UI_MODE", "fastapi").lower() # gradio | fastapi
UI_TYPE = os.getenv("UI_TYPE", "base").lower() # base | screen
APP_MODE = os.getenv("APP_MODE", "local").lower() # local | deployed
TURN_PROVIDER = os.getenv("TURN_PROVIDER", "hf-cloudflare") # hf-cloudflare | cloudflare | twilio

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
LANGUAGE = os.getenv("LANGUAGE", "english")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "aya:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://localhost:8000/tts")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "english")

logger.info(f"""
    --------------------------------------
    Configuration (environment variables):
    - UI_MODE: {UI_MODE}
    - UI_TYPE: {UI_TYPE}
    - APP_MODE: {APP_MODE}
    - TURN_PROVIDER: {TURN_PROVIDER}
    - MODEL_ID: {MODEL_ID}
    - LANGUAGE: {LANGUAGE}
    - OLLAMA_MODEL: {OLLAMA_MODEL}
    - OLLAMA_BASE_URL: {OLLAMA_BASE_URL}
    - TTS_SERVER_URL: {TTS_SERVER_URL}
    - TTS_LANGUAGE: {TTS_LANGUAGE}
    --------------------------------------
""")

# ──────────────────────────────────────────────────────────────────────────────
# Local long-term memory (ChromaDB)
# ──────────────────────────────────────────────────────────────────────────────
class LocalMemory:
    def __init__(self, persist_dir="memory_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("long_term_memory")

    def add(self, text: str, user_id: str = "default"):
        doc_id = f"{user_id}_{uuid.uuid4()}"
        self.collection.add(
            documents=[text],
            metadatas=[{"user": user_id}],
            ids=[doc_id]
        )

    def search(self, query: str, user_id: str = "default", limit: int = 5):
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"user": user_id}
        )
        docs = results.get("documents", [])
        return [d for sublist in docs for d in sublist]  # flatten

# Initialize models and memory
transcribe_pipeline = initialize_whisper_model(
    model_id=MODEL_ID,
    try_compile=True, # Set to False to disable trying to compile the model
    try_use_flash_attention=True, # Set to False to disable trying to use flash attention
    device=get_device(force_cpu=False) # Set to False to use GPU if available
)

# Initialize Ollama LLM
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7
)

# Initialize memory
USER_ID = "daniel"
memory = LocalMemory()

# Persistent user profile (facts)
USER_FACTS = [
    "The user's name is Daniel Mamre.",
    "Daniel is a senior full-stack engineer and ML enthusiast.",
    "He works on the TIBA SPARK ecosystem.",
    "He is building Hebrew TTS and AI Agents.",
]

# Store facts once (only if not already in DB)
try:
    for fact in USER_FACTS:
        memory.add(fact, user_id=USER_ID)
    logger.info("User facts stored in memory")
except Exception as e:
    logger.warning(f"Could not store user facts: {e}")

# Global conversation history
conversation_history = []

async def generate_tts_audio(text: str) -> bytes:
    """Generate TTS audio from text using the TTS server"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": text,
                "lang": TTS_LANGUAGE
            }
            async with session.post(TTS_SERVER_URL, json=payload) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    logger.info(f"Generated TTS audio for text: {text[:50]}...")
                    return audio_data
                else:
                    logger.error(f"TTS server error: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return None

async def process_with_llm(transcript: str, webrtc_id: str = None) -> AsyncGenerator[str, None]:
    """Process transcript with LLM and yield streaming response"""
    try:
        # Save user input to memory
        memory.add(transcript, user_id=USER_ID)
        conversation_history.append(f"User: {transcript}")
        
        # Retrieve relevant memory
        relevant = memory.search(transcript, user_id=USER_ID, limit=5)
        memories_str = "\n".join(f"- {m}" for m in relevant)
        
        # Facts prompt
        facts_prompt = "\n".join(f"- {fact}" for fact in USER_FACTS)
        
        # Include short-term conversation (last 5 turns)
        short_term = "\n".join(conversation_history[-5:])
        
        system_context = f"""
        You are Daniel's AI assistant with access to his memory.
        Always give short, clear answers (1–2 sentences max).
        
        Facts about Daniel:
        {facts_prompt}

        Relevant Long-Term Memories:
        {memories_str}

        Recent Conversation:
        {short_term}
        """
        
        # Stream LLM response
        full_response = ""
        async for chunk in llm.astream([
            {"role": "system", "content": system_context},
            {"role": "user", "content": transcript}
        ]):
            if hasattr(chunk, 'content') and chunk.content and chunk.content.strip():
                full_response += chunk.content
                yield chunk.content
        
        # Save response to memory and conversation history
        if full_response.strip():
            memory.add(full_response, user_id=USER_ID)
            conversation_history.append(f"Assistant: {full_response}")
            logger.info(f"LLM response saved: {full_response[:50]}...")
            
            # Generate TTS for the complete response
            if webrtc_id:
                tts_audio = await generate_tts_audio(full_response)
                if tts_audio:
                    tts_audio_store[webrtc_id] = tts_audio
                    logger.info(f"TTS audio stored for webrtc_id: {webrtc_id}")
            
    except Exception as e:
        logger.error(f"Error in LLM processing: {e}")
        error_msg = "Sorry, I encountered an error processing your request."
        yield error_msg

async def transcribe(audio: tuple[int, np.ndarray]):
    sample_rate, audio_array = audio
    logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    
    outputs = transcribe_pipeline(
        audio_to_bytes(audio),
        chunk_length_s=5,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': LANGUAGE,
        },
        #return_timestamps="word"
    )
    transcript = outputs["text"].strip()
    yield AdditionalOutputs(transcript)


logger.info("Initializing FastRTC stream")
stream = Stream(
    handler=ReplyOnPause(
        transcribe,
        algo_options=AlgoOptions(
            # Duration in seconds of audio chunks passed to the VAD model (default 0.6) 
            audio_chunk_duration=0.5,
            # If the chunk has more than started_talking_threshold seconds of speech, the user started talking (default 0.2)
            started_talking_threshold=0.1,
            # If, after the user started speaking, there is a chunk with less than speech_threshold seconds of speech, the user stopped speaking. (default 0.1)
            speech_threshold=0.1,
            # Max duration of speech chunks before the handler is triggered, even if a pause is not detected by the VAD model. (default -inf)
            max_continuous_speech_s=15
        ),
        model_options=SileroVadOptions(
            # Threshold for what is considered speech (default 0.5)
            threshold=0.5,
            # Final speech chunks shorter min_speech_duration_ms are thrown out (default 250)
            min_speech_duration_ms=250,
            # Max duration of speech chunks, longer will be split at the timestamp of the last silence that lasts more than 100ms (if any) or just before max_speech_duration_s (default float('inf')) (used internally in the VAD algorithm to split the audio that's passed to the algorithm)
            max_speech_duration_s=10,
            # Wait for ms at the end of each speech chunk before separating it (default 2000)
            min_silence_duration_ms=400,
            # Chunk size for VAD model. Can be 512, 1024, 1536 for 16k s.r. (default 1024)
            window_size_samples=1024,
            # Final speech chunks are padded by speech_pad_ms each side (default 400)
            speech_pad_ms=200,
        ),
    ),
    # send-receive: bidirectional streaming (default)
    # send: client to server only
    # receive: server to client only
    modality="audio",
    mode="send",
    additional_outputs=[
        gr.Textbox(label="Transcript"),
    ],
    additional_outputs_handler=lambda current, new: current + " " + new,
    rtc_configuration=get_rtc_credentials(provider=TURN_PROVIDER) if APP_MODE == "deployed" else None,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
stream.mount(app)

@app.get("/")
async def index():
    if UI_TYPE == "base":
        html_content = open("static/index.html").read()
    elif UI_TYPE == "screen":
        html_content = open("static/index-screen.html").read()

    rtc_configuration = await get_rtc_credentials(provider=TURN_PROVIDER) if APP_MODE == "deployed" else None
    logger.info(f"RTC configuration: {rtc_configuration}")
    html_content = html_content.replace("__INJECTED_RTC_CONFIG__", json.dumps(rtc_configuration))
    return HTMLResponse(content=html_content)

# Store LLM streams and TTS audio per webrtc_id
llm_streams = {}
tts_audio_store = {}

@app.get("/transcript")
def _(webrtc_id: str):
    logger.debug(f"New transcript stream request for webrtc_id: {webrtc_id}")
    async def output_stream():
        try:
            async for output in stream.output_stream(webrtc_id):
                transcript = output.args[0]
                logger.debug(f"Sending transcript for {webrtc_id}: {transcript[:50]}...")
                
                # Process with LLM and store the async generator
                if transcript.strip():
                    llm_streams[webrtc_id] = process_with_llm(transcript, webrtc_id)
                
                yield f"event: output\ndata: {transcript}\n\n"
        except Exception as e:
            logger.error(f"Error in transcript stream for {webrtc_id}: {str(e)}")
            raise

    return StreamingResponse(output_stream(), media_type="text/event-stream")

@app.get("/llm-response")
def _(webrtc_id: str):
    logger.debug(f"New LLM response stream request for webrtc_id: {webrtc_id}")
    async def llm_output_stream():
        try:
            if webrtc_id in llm_streams:
                llm_stream = llm_streams[webrtc_id]
                async for chunk in llm_stream:
                    # Only send non-empty chunks
                    if chunk and chunk.strip():
                        logger.debug(f"Sending LLM chunk for {webrtc_id}: {chunk[:50]}...")
                        yield f"event: llm-output\ndata: {chunk}\n\n"
                
                # Send completion event before cleaning up
                yield f"event: llm-complete\ndata: Stream complete\n\n"
                
                # Clean up after streaming is complete
                del llm_streams[webrtc_id]
                logger.info(f"LLM stream completed and cleaned up for {webrtc_id}")
            else:
                # Send a proper "no stream" event instead of just logging a warning
                logger.debug(f"No LLM stream found for webrtc_id: {webrtc_id}")
                yield f"event: no-stream\ndata: No stream available\n\n"
        except Exception as e:
            logger.error(f"Error in LLM response stream for {webrtc_id}: {str(e)}")
            yield f"event: error\ndata: Error processing LLM response\n\n"

    return StreamingResponse(llm_output_stream(), media_type="text/event-stream")

@app.get("/tts-audio")
def _(webrtc_id: str):
    """Serve TTS audio for a specific webrtc_id"""
    logger.debug(f"TTS audio request for webrtc_id: {webrtc_id}")
    
    if webrtc_id in tts_audio_store:
        audio_data = tts_audio_store[webrtc_id]
        # Clean up after serving
        del tts_audio_store[webrtc_id]
        logger.info(f"TTS audio served and cleaned up for webrtc_id: {webrtc_id}")
        
        return StreamingResponse(
            iter([audio_data]), 
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename=tts_{webrtc_id}.wav",
                "Cache-Control": "no-cache"
            }
        )
    else:
        logger.debug(f"No TTS audio found for webrtc_id: {webrtc_id}")
        return HTMLResponse(content="No TTS audio available", status_code=404)


if __name__ == "__main__":

    server_name = os.getenv("SERVER_NAME", "localhost")
    port = os.getenv("PORT", 7860)
    
    if UI_MODE == "gradio":
        logger.info("Launching Gradio UI")
        stream.ui.launch(
            server_port=port, 
            server_name=server_name,
            ssl_verify=False,
            debug=True
        )
    else:
        import uvicorn
        logger.info("Launching FastAPI server")
        uvicorn.run(app, host=server_name, port=port)