# tests/test_llm_smoke.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from clients.llm_client import LLMClient
from core.types import Message

def test_complete_smoke():
    client = LLMClient()
    msgs = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say hello in one short sentence.")
    ]
    reply = client.complete(msgs, max_tokens=32)
    assert isinstance(reply, str) and len(reply) > 0
    print("✅ reply:", reply)



if __name__ == "__main__":
    test_complete_smoke()
