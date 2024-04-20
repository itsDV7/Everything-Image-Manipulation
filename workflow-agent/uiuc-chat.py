import requests

url = "https://uiuc.chat/api/chat-api/chat"
headers = {
    'Content-Type': 'application/json',
}
data = {
    "model": "gpt-4-hackathon",
    "messages": [
        {
            "role": "system",
            "content": "Your system prompt here"
        },
        {
            "role": "user",
            "content": "Propose a computational framework for this hackathon. Step by step guideline."
        }
    ],
    "openai_key": "dc528eaf83724782914e171f3bbdaeda",
    "temperature": 0.1,
    "course_name": "Team-4",
    "stream": True,
    "api_key": "uc_a94414b8bd5141e1bc301a16812ffa03"
}

response = requests.post(url, headers=headers, json=data)
print(response.text)