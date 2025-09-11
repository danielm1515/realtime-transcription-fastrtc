import logging
import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available

from utils.device import get_torch_and_np_dtypes

logger = logging.getLogger(__name__)

def initialize_whisper_model(
        model_id: str, 
        try_compile: bool, 
        try_use_flash_attention: bool, 
        device: str,
        enable_console_print: bool = True
    ) -> pipeline:
    """Initialize Whisper model with optimal configuration.
    
    Args:
        model_id: The ID of the model to load.
        try_compile: Whether to try to compile the model.
        try_use_flash_attention: Whether to try to use flash attention.
        device: The device to load the model on.
        enable_console_print: Whether to enable live transcription printing to console.

    Returns:
        pipeline: The transcribe pipeline.
    """
    use_device_map = device == "cuda"
    torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
    try_compile_model = (
        try_compile and device in ["cuda", "mps"] and (device != "mps" or torch_version >= (2, 7))
    )
    torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16=False)
    
    logger.info(f"Initializing model {model_id} on {device} with dtype {torch_dtype}")

    # Try flash attention first, fallback to SDPA if warmup fails
    model, use_flash_attention = load_model_with_flash_attention_test(
        model_id=model_id,
        torch_dtype=torch_dtype,
        use_device_map=use_device_map,
        device=device,
        try_use_flash_attention=try_use_flash_attention
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    transcribe_pipeline = create_pipeline(model, processor, torch_dtype)
    compiled = compile_model_if_possible(transcribe_pipeline, try_compile_model)
    warmup_model(transcribe_pipeline, np_dtype, audio_length=16000)
    
    # Wrap with console printing if enabled
    if enable_console_print:
        transcribe_pipeline = create_live_transcription_wrapper(transcribe_pipeline, enable_console_print)
        logger.info("Live console transcription printing enabled")
    
    logger.info(f"Model initialized successfully")
    logger.info(f"""
        --------------------------------------
        Model Configuration:
        - Device: {device}
        - Torch dtype: {torch_dtype}
        - Numpy dtype: {np_dtype}
        - Flash attention: {use_flash_attention}
        - Compiled: {compiled}
        - Console printing: {enable_console_print}
        --------------------------------------
    """)
    return transcribe_pipeline

def load_model_with_flash_attention_test(
        model_id: str, 
        torch_dtype: torch.dtype, 
        use_device_map: bool, 
        device: str, 
        try_use_flash_attention: bool
    ) -> tuple[AutoModelForSpeechSeq2Seq, bool]:
    """Load model with flash attention if possible, fallback to SDPA if not."""
    try_flash = try_use_flash_attention and device == "cuda" and is_flash_attn_2_available()
    
    if try_flash:
        try:
            model = _load_model(
                model_id=model_id, 
                torch_dtype=torch_dtype, 
                use_device_map=use_device_map, 
                device=device, 
                use_flash_attention=True
            )
            
            # Test with warmup - if this fails, don't use flash attention
            if _test_model_warmup(model, model_id, torch_dtype):
                logger.info("Flash attention enabled and tested successfully")
                return model, True
            else:
                logger.warning("Flash attention warmup failed, falling back to SDPA")
                model = _load_model(
                    model_id=model_id, 
                    torch_dtype=torch_dtype, 
                    use_device_map=use_device_map, 
                    device=device, 
                    use_flash_attention=False
                )
                return model, False
                
        except Exception as e:
            logger.warning(f"Flash attention initialization failed: {e}, falling back to SDPA")
    
    # Load without flash attention
    model = _load_model(model_id, torch_dtype, use_device_map, device, use_flash_attention=False)
    return model, False

def warmup_model(transcribe_pipeline, np_dtype, audio_length=16000):
    """Run dummy input through pipeline to warm it up."""
    warmup_audio = np.random.rand(audio_length).astype(np_dtype)
    transcribe_pipeline(warmup_audio)

def _load_model(
    model_id: str,
    torch_dtype: torch.dtype,
    use_device_map: bool,
    device: str,
    use_flash_attention: bool = False
) -> AutoModelForSpeechSeq2Seq:
    """Load model with specified attention implementation."""
    attn_implementation = "flash_attention_2" if use_flash_attention else "sdpa"
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=attn_implementation,
            device_map="auto" if use_device_map else None,
        )
        if not use_device_map:
            model.to(device)
        return model
    except RuntimeError as e:
        logger.warning(f"Model loading with device_map failed: {e}", exc_info=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=attn_implementation,
            device_map=None,
        )
        model.to(device)
        return model

def _test_model_warmup(model, model_id, torch_dtype):
    """Test if model works with a small warmup input."""
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        test_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype
        )
        warmup_model(test_pipeline, np.float32, audio_length=1600)
        return True
    except Exception as e:
        logger.debug(f"Model warmup test failed: {e}")
        return False

def create_pipeline(model, processor, torch_dtype):
    """Create ASR pipeline with specified model and processor."""
    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype
    )

def compile_model_if_possible(transcribe_pipeline, try_compile: bool):
    """Attempt to compile model for better performance if supported."""
    if not try_compile:
        logger.info("Model compilation skipped (requirements not met)")
        return False

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available in this PyTorch version")
        return False

    try:
        transcribe_pipeline.model = torch.compile(transcribe_pipeline.model, mode="max-autotune")
        logger.info("Model compiled successfully")
        return True
    except Exception as e:
        logger.error(f"Model compilation failed: {e}", exc_info=True)
        return False

def create_live_transcription_wrapper(transcribe_pipeline, enable_console_print: bool = True):
    """Create a wrapper around the transcription pipeline that prints results to console.
    
    Args:
        transcribe_pipeline: The original transcription pipeline
        enable_console_print: Whether to print transcriptions to console
        
    Returns:
        A wrapper function that behaves like the original pipeline but prints results
    """
    def transcription_wrapper(*args, **kwargs):
        # Call the original pipeline
        result = transcribe_pipeline(*args, **kwargs)
        
        # Print to console if enabled
        if enable_console_print and isinstance(result, dict) and "text" in result:
            transcript_text = result["text"].strip()
            if transcript_text:  # Only print if there's actual text
                print(f"🎤 LIVE TRANSCRIPTION: {transcript_text}")
        
        return result
    
    # Copy attributes from original pipeline to wrapper
    for attr_name in dir(transcribe_pipeline):
        if not attr_name.startswith('__') and attr_name != '__call__':
            try:
                setattr(transcription_wrapper, attr_name, getattr(transcribe_pipeline, attr_name))
            except (AttributeError, TypeError):
                pass  # Skip attributes that can't be set
    
    return transcription_wrapper