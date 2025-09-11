import os
import logging
import json

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
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
TURN_PROVIDER = os.getenv("TURN_PROVIDER", "cloudflare") # hf-cloudflare | cloudflare | twilio

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
LANGUAGE = os.getenv("LANGUAGE", "english")
ENABLE_CONSOLE_PRINT = os.getenv("ENABLE_CONSOLE_PRINT", "true").lower() == "true"

logger.info(f"""
    --------------------------------------
    Configuration (environment variables):
    - UI_MODE: {UI_MODE}
    - UI_TYPE: {UI_TYPE}
    - APP_MODE: {APP_MODE}
    - TURN_PROVIDER: {TURN_PROVIDER}
    - MODEL_ID: {MODEL_ID}
    - LANGUAGE: {LANGUAGE}
    - ENABLE_CONSOLE_PRINT: {ENABLE_CONSOLE_PRINT}
    --------------------------------------
""")

transcribe_pipeline = initialize_whisper_model(
    model_id=MODEL_ID,
    try_compile=True, # Set to False to disable trying to compile the model
    try_use_flash_attention=True, # Set to False to disable trying to use flash attention
    device=get_device(force_cpu=False), # Set to False to use GPU if available
    enable_console_print=ENABLE_CONSOLE_PRINT
)

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
    yield AdditionalOutputs(outputs["text"].strip())


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

@app.get("/transcript")
def _(webrtc_id: str):
    logger.debug(f"New transcript stream request for webrtc_id: {webrtc_id}")
    async def output_stream():
        try:
            async for output in stream.output_stream(webrtc_id):
                transcript = output.args[0]
                logger.debug(f"Sending transcript for {webrtc_id}: {transcript[:50]}...")
                yield f"event: output\ndata: {transcript}\n\n"
        except Exception as e:
            logger.error(f"Error in transcript stream for {webrtc_id}: {str(e)}")
            raise

    return StreamingResponse(output_stream(), media_type="text/event-stream")


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