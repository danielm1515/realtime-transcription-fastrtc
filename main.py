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
        self.conversation_collection = self.client.get_or_create_collection("conversation_history")
        self.user_facts_collection = self.client.get_or_create_collection("user_facts")

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
    
    def save_conversation_turn(self, message: str, user_id: str = "default", turn_number: int = 0):
        """Save a conversation turn to persistent storage"""
        doc_id = f"{user_id}_turn_{turn_number}_{uuid.uuid4()}"
        self.conversation_collection.add(
            documents=[message],
            metadatas={"user": user_id, "turn": turn_number, "timestamp": str(uuid.uuid4())},
            ids=[doc_id]
        )
    
    def load_conversation_history(self, user_id: str = "default", limit: int = 50):
        """Load recent conversation history from persistent storage"""
        try:
            results = self.conversation_collection.query(
                query_texts=["conversation"],
                n_results=limit,
                where={"user": user_id}
            )
            
            if results and results.get("documents"):
                # Get documents with metadata
                docs = results["documents"][0] if results["documents"] else []
                metadatas = results["metadatas"][0] if results.get("metadatas") else []
                
                # Sort by turn number if available
                history_items = []
                for i, doc in enumerate(docs):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    turn = metadata.get("turn", 0)
                    history_items.append((turn, doc))
                
                # Sort by turn number and return messages
                history_items.sort(key=lambda x: x[0])
                return [item[1] for item in history_items[-limit:]]
            
            return []
        except Exception as e:
            logger.warning(f"Could not load conversation history: {e}")
            return []
    
    def save_user_fact(self, fact: str, user_id: str = "default", fact_type: str = "general"):
        """Save a new fact about the user"""
        doc_id = f"{user_id}_fact_{uuid.uuid4()}"
        self.user_facts_collection.add(
            documents=[fact],
            metadatas={"user": user_id, "fact_type": fact_type, "timestamp": str(uuid.uuid4())},
            ids=[doc_id]
        )
        logger.info(f"Saved new user fact: {fact}")
    
    def get_user_facts(self, user_id: str = "default", limit: int = 20):
        """Retrieve all known facts about the user"""
        try:
            results = self.user_facts_collection.query(
                query_texts=["facts about user"],
                n_results=limit,
                where={"user": user_id}
            )
            
            if results and results.get("documents"):
                docs = results["documents"][0] if results["documents"] else []
                return docs
            
            return []
        except Exception as e:
            logger.warning(f"Could not load user facts: {e}")
            return []
    
    def update_user_fact(self, old_fact: str, new_fact: str, user_id: str = "default"):
        """Update an existing fact about the user"""
        try:
            # First, try to find the old fact
            results = self.user_facts_collection.query(
                query_texts=[old_fact],
                n_results=1,
                where={"user": user_id}
            )
            
            if results and results.get("ids") and results["ids"][0]:
                # Delete the old fact
                fact_id = results["ids"][0][0]
                self.user_facts_collection.delete(ids=[fact_id])
                logger.info(f"Deleted old fact: {old_fact}")
            
            # Add the new fact
            self.save_user_fact(new_fact, user_id)
            
        except Exception as e:
            logger.warning(f"Could not update user fact: {e}")
            # Fallback: just add the new fact
            self.save_user_fact(new_fact, user_id)

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

# Initial user profile (facts) - will be stored in dynamic facts system
INITIAL_USER_FACTS = [
    "The user's name is Daniel Mamre.",
    "Daniel is a senior full-stack engineer and ML enthusiast.",
    "He works on the TIBA SPARK ecosystem.",
    "He is building Hebrew TTS and AI Agents.",
]

# Store initial facts once (only if not already in DB)
try:
    existing_facts = memory.get_user_facts(user_id=USER_ID)
    if not existing_facts:  # Only add if no facts exist yet
        for fact in INITIAL_USER_FACTS:
            memory.save_user_fact(fact, user_id=USER_ID, fact_type="initial")
        logger.info("Initial user facts stored in dynamic facts system")
    else:
        logger.info(f"Found {len(existing_facts)} existing user facts")
except Exception as e:
    logger.warning(f"Could not store initial user facts: {e}")

# Per-user conversation history storage
user_conversation_histories = {}

# Load existing conversation history from persistent storage
try:
    existing_history = memory.load_conversation_history(user_id=USER_ID, limit=20)
    if existing_history:
        user_conversation_histories[USER_ID] = existing_history
        logger.info(f"Loaded {len(existing_history)} conversation turns from persistent storage")
    else:
        user_conversation_histories[USER_ID] = []
        logger.info("No existing conversation history found, starting fresh")
except Exception as e:
    logger.warning(f"Could not load conversation history: {e}")
    user_conversation_histories[USER_ID] = []

# Global turn counter for conversation persistence
conversation_turn_counter = len(user_conversation_histories.get(USER_ID, []))

async def extract_new_facts(transcript: str, response: str, user_id: str = USER_ID) -> None:
    """Extract and save new facts about the user from the conversation"""
    try:
        # Create a simple prompt to identify new facts
        fact_extraction_prompt = f"""
        Analyze this conversation and identify any NEW facts about the user that should be remembered.
        Only extract clear, factual information that is NEW and not already known.
        
        User said: "{transcript}"
        Assistant responded: "{response}"
        
        Current known facts about the user:
        {chr(10).join(f"- {fact}" for fact in memory.get_user_facts(user_id=user_id, limit=10))}
        
        If there are any NEW facts to remember, list them one per line starting with "FACT:".
        If no new facts, respond with "NO_NEW_FACTS".
        
        Examples of facts to remember:
        - Personal preferences (favorite food, color, etc.)
        - Current projects or work
        - Technical skills or tools used
        - Personal experiences mentioned
        - Future plans or goals
        - Family information
        - Location or background info
        
        Only extract clear, factual statements. Don't extract opinions or temporary states.
        """
        
        # Use LLM to extract facts
        fact_response = ""
        async for chunk in llm.astream([
            {"role": "user", "content": fact_extraction_prompt}
        ]):
            if hasattr(chunk, 'content') and chunk.content:
                fact_response += chunk.content
        
        # Parse the response and save new facts
        if fact_response and "NO_NEW_FACTS" not in fact_response.upper():
            lines = fact_response.strip().split('\n')
            for line in lines:
                if line.strip().startswith("FACT:"):
                    new_fact = line.replace("FACT:", "").strip()
                    if new_fact and len(new_fact) > 5:  # Basic validation
                        memory.save_user_fact(new_fact, user_id=user_id, fact_type="learned")
                        logger.info(f"Learned new fact about user: {new_fact}")
        
    except Exception as e:
        logger.warning(f"Error extracting facts: {e}")

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
    global conversation_turn_counter
    
    try:
        # Get or create conversation history for this user
        if USER_ID not in user_conversation_histories:
            user_conversation_histories[USER_ID] = []
        
        conversation_history = user_conversation_histories[USER_ID]
        
        # Save user input to memory and conversation history
        memory.add(transcript, user_id=USER_ID)
        user_message = f"User: {transcript}"
        conversation_history.append(user_message)
        
        # Save user message to persistent storage
        try:
            memory.save_conversation_turn(user_message, user_id=USER_ID, turn_number=conversation_turn_counter)
            conversation_turn_counter += 1
        except Exception as e:
            logger.warning(f"Could not save user message to persistent storage: {e}")
        
        # Retrieve relevant memory
        relevant = memory.search(transcript, user_id=USER_ID, limit=5)
        memories_str = "\n".join(f"- {m}" for m in relevant)
        
        # Get current user facts (dynamic)
        current_facts = memory.get_user_facts(user_id=USER_ID, limit=20)
        facts_prompt = "\n".join(f"- {fact}" for fact in current_facts)
        
        # Include more conversation history (last 10 turns instead of 5)
        short_term = "\n".join(conversation_history[-10:])
        
        system_context = f"""
        You are Daniel's AI assistant with access to his memory.
        Always give short, clear answers (1–2 sentences max).
        
        IMPORTANT: If Daniel mentions new information about himself (preferences, projects, experiences, etc.), 
        you should remember it for future conversations. Pay attention to facts like:
        - Personal preferences or interests
        - Current projects he's working on
        - Technical skills or tools he uses
        - Personal experiences or stories
        - Future plans or goals
        
        Current Facts about Daniel:
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
            assistant_message = f"Assistant: {full_response}"
            conversation_history.append(assistant_message)
            
            # Save assistant message to persistent storage
            try:
                memory.save_conversation_turn(assistant_message, user_id=USER_ID, turn_number=conversation_turn_counter)
                conversation_turn_counter += 1
            except Exception as e:
                logger.warning(f"Could not save assistant message to persistent storage: {e}")
            
            # Extract new facts from this conversation turn
            try:
                await extract_new_facts(transcript, full_response, user_id=USER_ID)
            except Exception as e:
                logger.warning(f"Could not extract facts from conversation: {e}")
            
            # Keep conversation history manageable (last 50 messages)
            if len(conversation_history) > 50:
                user_conversation_histories[USER_ID] = conversation_history[-50:]
            
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
    
    # Enhanced speech detection to avoid hallucinations
    audio_float = audio_array.astype(np.float32)
    
    # Calculate multiple audio metrics for better speech detection
    audio_rms = np.sqrt(np.mean(audio_float ** 2))
    audio_max = np.max(np.abs(audio_float))
    audio_std = np.std(audio_float)
    
    # Check for speech-like characteristics
    # Real speech has higher energy variation and peaks
    has_speech_energy = audio_rms > 0.005  # Lower threshold for quiet speech
    has_speech_peaks = audio_max > 0.02    # Real speech has amplitude peaks
    has_speech_variation = audio_std > 0.003  # Speech has variation, silence doesn't
    
    # Only skip if ALL indicators suggest no speech (very conservative)
    if not (has_speech_energy or has_speech_peaks or has_speech_variation):
        logger.debug(f"No speech detected - RMS: {audio_rms:.4f}, Max: {audio_max:.4f}, Std: {audio_std:.4f}")
        return
    
    logger.debug(f"Speech detected - RMS: {audio_rms:.4f}, Max: {audio_max:.4f}, Std: {audio_std:.4f}")
    
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
    
    # Only filter if we're confident it's a hallucination AND audio was very quiet
    # This way we don't filter real speech
    if transcript and audio_rms < 0.003 and audio_max < 0.01:
        # Only filter very common single-word hallucinations on very quiet audio
        very_common_hallucinations = ["you", "thank you", "thanks"]
        if transcript.lower().strip('.') in very_common_hallucinations:
            logger.debug(f"Filtered likely hallucination on quiet audio: '{transcript}' (RMS: {audio_rms:.4f})")
            return
    
    # Yield all other transcripts (including real "תודה רבה" when you actually speak)
    if transcript and len(transcript.strip()) > 0:
        yield AdditionalOutputs(transcript)


logger.info("Initializing FastRTC stream")
stream = Stream(
    handler=ReplyOnPause(
        transcribe,
        algo_options=AlgoOptions(
            # Duration in seconds of audio chunks passed to the VAD model (default 0.6) 
            audio_chunk_duration=0.8,
            # If the chunk has more than started_talking_threshold seconds of speech, the user started talking (default 0.2)
            started_talking_threshold=0.25,
            # If, after the user started speaking, there is a chunk with less than speech_threshold seconds of speech, the user stopped speaking. (default 0.1)
            speech_threshold=0.03,
            # Max duration of speech chunks before the handler is triggered, even if a pause is not detected by the VAD model. (default -inf)
            max_continuous_speech_s=30
        ),
        model_options=SileroVadOptions(
            # Threshold for what is considered speech (default 0.5)
            threshold=0.3,
            # Final speech chunks shorter min_speech_duration_ms are thrown out (default 250)
            min_speech_duration_ms=150,
            # Max duration of speech chunks, longer will be split at the timestamp of the last silence that lasts more than 100ms (if any) or just before max_speech_duration_s (default float('inf')) (used internally in the VAD algorithm to split the audio that's passed to the algorithm)
            max_speech_duration_s=25,
            # Wait for ms at the end of each speech chunk before separating it (default 2000)
            min_silence_duration_ms=1500,
            # Chunk size for VAD model. Can be 512, 1024, 1536 for 16k s.r. (default 1024)
            window_size_samples=1024,
            # Final speech chunks are padded by speech_pad_ms each side (default 400)
            speech_pad_ms=600,
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

@app.get("/user-facts")
def get_user_facts():
    """Get current facts about the user (for debugging)"""
    try:
        facts = memory.get_user_facts(user_id=USER_ID, limit=50)
        return {
            "user_id": USER_ID,
            "facts_count": len(facts),
            "facts": facts
        }
    except Exception as e:
        logger.error(f"Error retrieving user facts: {e}")
        return {"error": str(e)}, 500

@app.post("/add-user-fact")
def add_user_fact(fact_data: dict):
    """Manually add a fact about the user"""
    try:
        fact = fact_data.get("fact", "").strip()
        fact_type = fact_data.get("type", "manual")
        
        if not fact:
            return {"error": "Fact text is required"}, 400
        
        memory.save_user_fact(fact, user_id=USER_ID, fact_type=fact_type)
        return {"message": "Fact added successfully", "fact": fact}
    except Exception as e:
        logger.error(f"Error adding user fact: {e}")
        return {"error": str(e)}, 500


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