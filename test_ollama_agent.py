#!/usr/bin/env python3
"""
Test the conversation agent with Ollama and memory
"""

import asyncio
import logging
from utils.conversation import ConversationAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_conversation_agent():
    """Test the conversation agent with Ollama and memory"""
    print("🧪 Testing Conversation Agent with Ollama and Memory...")
    
    try:
        # Initialize the agent
        agent = ConversationAgent(
            model_name="gpt-oss:20b",
            piper_server_url="http://localhost:8000",
            user_id="daniel"
        )
        
        print("✅ Conversation Agent initialized successfully")
        
        # Test a simple conversation
        print("\n🗣️  Testing conversation...")
        user_input = "Hello, how are you?"
        print(f"User: {user_input}")
        
        print("Assistant: ", end="", flush=True)
        response = ""
        async for chunk in agent.process_user_input(user_input):
            print(chunk, end="", flush=True)
            response += chunk
        
        print(f"\n\n✅ Full response: {response}")
        
        # Test memory
        print("\n🧠 Testing memory...")
        user_input2 = "What's my name?"
        print(f"User: {user_input2}")
        
        print("Assistant: ", end="", flush=True)
        response2 = ""
        async for chunk in agent.process_user_input(user_input2):
            print(chunk, end="", flush=True)
            response2 += chunk
        
        print(f"\n\n✅ Memory response: {response2}")
        
        # Test TTS
        print("\n🔊 Testing TTS...")
        agent.speak_response("שלום דניאל, אני עובד טוב!")
        print("✅ TTS test completed (check if you heard Hebrew audio)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_conversation_agent())
