from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from sqlalchemy import create_engine, Column, String, Integer, Text, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import requests
import json
import os
from dotenv import load_dotenv
from functools import lru_cache
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key')

# API endpoints
TEXT_API_URL = 'https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent'
IMAGE_API_URL = 'https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent'

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///ai_data.db')
Session = sessionmaker(bind=engine)
session = Session()

# Model for embeddings
encoder = SentenceTransformer('all-MiniLM-L6-v2')

class User(UserMixin, Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True)
    password = Column(String(120))
    email = Column(String(120), unique=True)

class Memory(Base):
    __tablename__ = 'memory'
    id = Column(Integer, primary_key=True)
    prompt = Column(Text)
    response = Column(Text)
    embedding = Column(Text)
    sentiment_score = Column(Float)
    timestamp = Column(Float)
    user_id = Column(Integer)
    
    Index('idx_timestamp', timestamp)

Base.metadata.create_all(engine)

@login_manager.user_loader
def load_user(user_id):
    return session.query(User).get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if session.query(User).filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if session.query(User).filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password, email=email)
        session.add(new_user)
        session.commit()
        
        login_user(new_user)
        return redirect(url_for('home'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = session.query(User).filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@lru_cache(maxsize=1000)
def generate_text(prompt):
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "contents": [{"parts":[{"text": prompt}]}]
        }
        response = requests.post(
            f"{TEXT_API_URL}?key={os.getenv('GEMINI_API_KEY')}", 
            headers=headers, 
            json=data
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def get_similar_responses(prompt, user_id, limit=5):
    prompt_embedding = encoder.encode(prompt)
    memories = session.query(Memory).filter_by(user_id=user_id).all()
    similarities = []
    
    for memory in memories:
        stored_embedding = np.array(json.loads(memory.embedding))
        similarity = np.dot(prompt_embedding, stored_embedding)
        similarities.append((similarity, memory))
    
    similarities.sort(reverse=True)
    return similarities[:limit]

def learn_and_generate(prompt, user_id):
    similar_responses = get_similar_responses(prompt, user_id)
    
    context = "\n".join([
        f"[Similarity: {sim:.2f}] {mem.response}"
        for sim, mem in similar_responses
    ])
    
    refined_prompt = f"{prompt}\n\nRelevant context:\n{context}"
    response = generate_text(refined_prompt)
    
    embedding = encoder.encode(prompt)
    
    memory = Memory(
        prompt=prompt,
        response=response['text'],
        embedding=json.dumps(embedding.tolist()),
        timestamp=datetime.now().timestamp(),
        user_id=user_id
    )
    session.add(memory)
    session.commit()
    
    return response['text']

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
@login_required
def chat():
    data = request.json
    response = learn_and_generate(data['prompt'], current_user.id)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=True,
        ssl_context=None,
        threaded=True
    )

