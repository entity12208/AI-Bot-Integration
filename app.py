from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import requests
import json

# Initialize Flask app
app = Flask(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///ai_data.db')
Session = sessionmaker(bind=engine)
session = Session()

class Memory(Base):
    __tablename__ = 'memory'
    id = Column(Integer, primary_key=True)
    prompt = Column(Text)
    response = Column(Text)

Base.metadata.create_all(engine)

# Gemini API details
API_KEY = 'AIzaSyDJykebLC7sEZYp_uLArefy5M6kCreklsw'
TEXT_API_URL = 'https://api.google.com/v1/generateContent'
IMAGE_API_URL = 'https://api.google.com/v1/generateImage'

def generate_text(prompt):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gemini-1.5-pro",
        "prompt": prompt,
        "max_tokens": 100
    }
    response = requests.post(TEXT_API_URL, headers=headers, data=json.dumps(data))
    return response.json()

def generate_image(prompt):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gemini-1.5-pro",
        "prompt": prompt,
        "max_images": 1
    }
    response = requests.post(IMAGE_API_URL, headers=headers, data=json.dumps(data))
    return response.json()

def store_memory(prompt, response):
    memory = Memory(prompt=prompt, response=response)
    session.add(memory)
    session.commit()

def learn_and_generate(prompt):
    past_memory = session.query(Memory).all()
    past_responses = [memory.response for memory in past_memory]
    refined_prompt = f"{prompt}\n\nPrevious responses:\n" + "\n".join(past_responses)
    new_response = generate_text(refined_prompt)
    store_memory(prompt, new_response['choices'][0]['text'])
    return new_response['choices'][0]['text']

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    response = learn_and_generate(prompt)
    return jsonify({"response": response})

@app.route('/generate-image', methods=['POST'])
def generate_image_route():
    data = request.json
    prompt = data.get('prompt')
    response = generate_image(prompt)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
