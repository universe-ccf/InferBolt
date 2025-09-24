# tests/test_llm_smoke.py
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
