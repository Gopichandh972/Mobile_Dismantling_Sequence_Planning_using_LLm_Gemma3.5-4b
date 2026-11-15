
import json
import requests

URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"

payload = {
    "model": MODEL,
    "stream": False,
    "format": {
        "type": "array",
        "items": {"type": "string"}
    },
    "messages": [
        {
            "role": "system",
            "content": (
                "You organize phone disassembly operations. "
                "Return only a JSON array of operation strings, "
                "preserving the supplied operations and their order."
            )
        },
        {
            "role": "user",
            "content": (
                "Organize these operations into their original sequence: "
                '["Battery removal", "Screw removal", "Motherboard removal"]'
            )
        }
    ],
    "options": {
        "temperature": 0,
        "num_predict": 128
    }
}

response = requests.post(URL, json=payload, timeout=180)
response.raise_for_status()

result = response.json()
content = result["message"]["content"]
operations = json.loads(content)

assert isinstance(operations, list), "Response is not a JSON list"
assert all(isinstance(op, str) for op in operations), "Invalid list elements"

print("Model:", MODEL)
print("Valid JSON:", True)
print("Returned operations:", operations)
print("Python API test passed.")
