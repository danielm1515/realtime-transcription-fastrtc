import os
import logging

import gradio as gr
import numpy as np

from utils.logger_config import setup_logging
from utils.device import get_device
from utils.model import initialize_whisper_model

from dotenv import load_dotenv
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)


MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
ENABLE_CONSOLE_PRINT = os.getenv("ENABLE_CONSOLE_PRINT", "true").lower() == "true"

transcribe_pipeline = initialize_whisper_model(
    model_id=MODEL_ID,
    try_compile=True,
    try_use_flash_attention=True,
    device=get_device(force_cpu=False),
    enable_console_print=ENABLE_CONSOLE_PRINT
)

async def transcribe(stream, audio: tuple[int, np.ndarray]):
    sample_rate, audio_array = audio
    logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    
    # Convert to mono if stereo
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    
    audio_array = audio_array.astype(np.float32)
    audio_array /= np.max(np.abs(audio_array))
    
    if stream is not None:
        stream = np.concatenate((stream, audio_array))
    else:
        stream = audio_array

    outputs = transcribe_pipeline(
        {"sampling_rate": sample_rate, "raw": audio_array},
        chunk_length_s=10,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': 'english',
        },
        #return_timestamps="word"
    )
    return stream, outputs["text"].strip()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Audio Stream", streaming=True)
        with gr.Column():
            transcript = gr.Textbox(label="Transcript", value="")
        
        state = gr.State()
        audio_input.stream(
            transcribe, 
            inputs=[state, audio_input], 
            outputs=[state, transcript],
            stream_every=2
        )

        clear_button = gr.Button("Clear")
        clear_button.click(
            lambda: None, # clear the state
            outputs=[state]
        ).then(
            lambda: "", # clear the transcript
            outputs=[transcript]
        )

if __name__ == "__main__":
    
    server_name = os.getenv("SERVER_NAME", "localhost")
    port = os.getenv("PORT", 7860)
    
    demo.launch(
        server_name=server_name,
        server_port=port,
        debug=True
    )
