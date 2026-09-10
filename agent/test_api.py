# test_api.py
from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from your environment
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[{"role": "user", "content": "Say hello in exactly 5 words."}]
)
print(response.content[0].text)