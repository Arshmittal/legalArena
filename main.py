from urllib.parse import unquote
from http.client import HTTPException
import os
import math
import logging
import pandas as pd
from flask import Flask, Response, json, jsonify, redirect, url_for, session, request, render_template, url_for,  flash
from flask_session import Session
from flask import Flask, request, jsonify, render_template
from authlib.integrations.flask_client import OAuth
from pydantic import BaseModel, Field
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, time, timedelta
import requests
from typing import Optional
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from bson.json_util import dumps, loads
import json
import time 
from together import Together  # Import the Together API client


from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime


# Load environment variables
load_dotenv()
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)


# Define the feedback model
class FeedbackModel(BaseModel):
    name: str
    feedback: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)   

app = Flask(__name__, static_folder="static", template_folder="templates")
bcrypt = Bcrypt(app)

app.secret_key = os.getenv("SECRET_KEY")
dummy_user = {
    "name": "Tester",
    "email": "tester@example.com",
    "picture": "/static/default-profile.png",
    "last_login": datetime.utcnow()
}

CORS(app, supports_credentials=True)


app.config['SESSION_COOKIE_NAME'] = 'google-login-session'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'   # Same domain => 'Lax' is okay
app.config['SESSION_COOKIE_SECURE'] = True      # Ensure HTTPS
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=60)
app.config['SESSION_PERMANENT'] = True



app.logger.setLevel(logging.INFO)

# Add this line to handle proxy headers correctly
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)




# Configure Logging - Enhanced with file and console handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Optional file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configure MongoDB - With connection verification
try:
    client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    # Verify connection is working
    client.server_info()
    logger.info("Successfully connected to MongoDB")
    
    db = client['legal_chatbot']
    users_collection = db['users']
    # Create separate collections for each model
    llama_responses_collection = db['llama_responses']
    deepseek_responses_collection = db['deepseek_responses']
    votes_collection = db['tool_votes']
    votes_collection = db['model_votes']
    elo_ratings_collection = db['model_elo_ratings']
    tools_collection = db['tools']
    feedback_collection = db['feedback']
except Exception as e:
    logger.critical(f"Failed to connect to MongoDB: {e}")
    raise

# API URLs & Secrets
# Dify API Configuration
DIFY_WORKFLOW_URL = os.getenv("DIFY_WORKFLOW_URL")
DIFY_WORKFLOW_SECRET = os.getenv("DIFY_WORKFLOW_SECRET")

# Together API Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Model names
LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

# OAuth Setup
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
SECRET_KEY = os.getenv('SECRET_KEY')

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'prompt': 'select_account'
    }
)


def call_together_api(question, model_name):
    """Call the Together API to generate a response using the specified model"""
    try:
        # Initialize Together client
        together_client = Together(api_key=TOGETHER_API_KEY)
        
        # Make the API call
        response = together_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"Legal Question: {question}"}],
            temperature=0.7,
            max_tokens=1024
        )
        
        # Extract and return the response content
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Together API with model {model_name}: {e}")
        return f"Error: Unable to get response from {model_name}."


def call_dify_workflow(question, model_answer, prev_model_answer=None, model_name=None):
    """Calls the Dify API to evaluate the answer with 5 different metrics."""
    headers = {
        "Authorization": f"Bearer {DIFY_WORKFLOW_SECRET}",
        "Content-Type": "application/json"
    }
   
    # If no previous answer exists, use the current answer as comparison
    if not prev_model_answer:
        prev_model_answer = model_answer
    
    # Log original sizes before truncation
    logger.debug(f"Original sizes: question={len(question) if question else 0}, answer={len(model_answer) if model_answer else 0}, prev_answer={len(prev_model_answer) if prev_model_answer else 0}")
    
    # Different truncation sizes based on model (DeepSeek might need smaller sizes)
    max_answer_len = 4000 if model_name == "deepseek" else 5000
    
    # Ensure inputs are strings and truncate if necessary to avoid payload issues
    question_trimmed = str(question)[:1000] if question else ""
    model_answer_trimmed = str(model_answer)[:max_answer_len] if model_answer else ""
    prev_model_answer_trimmed = str(prev_model_answer)[:max_answer_len] if prev_model_answer else ""
    
    # Check for potentially problematic characters in the responses
    model_answer_trimmed = clean_response(model_answer_trimmed)
    prev_model_answer_trimmed = clean_response(prev_model_answer_trimmed)
   
    payload = {
        "workflow_id": os.getenv("DIFY_WORKFLOW_ID"),
        "inputs": {
            "question": question_trimmed,
            "user_answer": model_answer_trimmed,
            "prev_answer": prev_model_answer_trimmed
        },
        "response_mode": "blocking",
        "user": "abc-123"
    }
    
    logger.debug(f"Calling Dify API for {model_name} evaluation with payload length: question={len(question_trimmed)}, answer={len(model_answer_trimmed)}, prev_answer={len(prev_model_answer_trimmed)}")
    
    # Add retry logic for robustness
    max_retries = 3
    retry_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            response = requests.post(DIFY_WORKFLOW_URL, json=payload, headers=headers)
            
            # Log actual response for debugging if there's an error
            if response.status_code != 200:
                logger.warning(f"Dify API error response: {response.text}")
            else:
                # Log the complete response structure for debugging
                logger.debug(f"Dify API complete response: {response.json()}")
            response.raise_for_status()
            result = response.json().get("data", {}).get("outputs", {})
            logger.debug(f"Dify API parsed outputs: {result}") 
            logger.debug(f"Dify API response received successfully on attempt {attempt+1}")
            return result
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error calling Dify API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed to call Dify API after {max_retries} attempts")
                # Return a default response structure instead of None
                return {"evaluation_results": "{\"accuracy\": 0, \"completeness\": 0, \"helpfulness\": 0, \"clarity\": 0, \"comparison\": 0}"}

def clean_response(text):
    """Clean a response to remove potentially problematic characters"""
    if not text:
        return ""
    
    # Remove or replace characters that might cause issues with APIs
    # This is a simple implementation - you might need to adjust based on specific problems
    cleaned = text
    
    # Handle common problematic characters
    cleaned = cleaned.replace('\u0000', '')  # Null bytes
    
    # Handle unbalanced quotes/brackets that might break JSON
    quote_count = cleaned.count('"')
    if quote_count % 2 != 0:
        # If unbalanced quotes, remove the last character which might be causing issues
        cleaned = cleaned[:-1]
    
    return cleaned

def parse_evaluation_metrics(evaluation):
    """Parse evaluation results with better error handling and different possible formats"""
    if not evaluation:
        logger.warning("Empty evaluation response")
        return {"accuracy": 0, "completeness": 0, "helpfulness": 0, "clarity": 0, "comparison": 0}
    
    # Log the actual structure received
    logger.debug(f"Evaluation structure received: {evaluation}")
    
    # Try different possible locations for the metrics
    if "evaluation_results" in evaluation:
        try:
            return json.loads(evaluation["evaluation_results"])
        except Exception as e:
            logger.error(f"Error parsing evaluation_results: {e}")
    
    # Try if metrics are directly in the response
    metrics_keys = ["accuracy", "completeness", "helpfulness", "clarity", "comparison"]
    direct_metrics = {k: evaluation.get(k, 0) for k in metrics_keys if k in evaluation}
    if direct_metrics and len(direct_metrics) == len(metrics_keys):
        return direct_metrics
    
    # Check other possible keys
    for key in evaluation:
        if isinstance(evaluation[key], str):
            try:
                parsed = json.loads(evaluation[key])
                if isinstance(parsed, dict) and any(k in parsed for k in metrics_keys):
                    return {k: parsed.get(k, 0) for k in metrics_keys}
            except:
                pass
    
    logger.warning("Could not find evaluation metrics in response")
    return {"accuracy": 0, "completeness": 0, "helpfulness": 0, "clarity": 0, "comparison": 0}
def get_previous_model_response(question, model_type):
    """Get previous response for a model if it exists"""
    collection = llama_responses_collection if model_type == "llama" else deepseek_responses_collection
    result = collection.find_one({"question": question})
    return result.get("response") if result else None

@app.route('/')
def home():
    # Log session info for debugging
    app.logger.info(f"Session at home route: {dict(session)}")
    user = session.get('user')
    
    # If user cookie exists but session doesn't have user, try to restore
    if not user and request.cookies.get('user_email'):
        email = request.cookies.get('user_email')
        app.logger.info(f"Attempting to restore session for: {email}")
        user_from_db = users_collection.find_one({"email": email})
        if user_from_db:
            user = {
                "_id": str(user_from_db["_id"]),
                "name": user_from_db.get("name", "User"),
                "email": user_from_db["email"],
                "picture": user_from_db.get("picture", "/static/default-profile.png"),
                "auth_type": "email"
            }
            session['user'] = user
    
    return render_template('index.html', user=user)


@app.route('/login')
def login():
    session.clear()
    session['oauth_state'] = os.urandom(16).hex()
    session.modified = True
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(
        redirect_uri=redirect_uri,
        state=session['oauth_state']
    )


@app.route('/google/callback')
def google_callback():
    try:
        state = request.args.get('state')
        stored_state = session.get('oauth_state')

        if not state or not stored_state or state != stored_state:
            raise ValueError("State verification failed")
        
        session.pop('oauth_state', None)

        token = google.authorize_access_token()
        if not token:
            raise ValueError("Failed to get access token")

        # Get user info from Google
        resp = google.get('https://www.googleapis.com/oauth2/v3/userinfo', token=token)
        user_info = resp.json()
        
        if not user_info or 'email' not in user_info:
            raise ValueError("Failed to get user info")

        # Store user data and OAuth token in MongoDB
        user_data = {
            "name": user_info.get("name", "User"),
            "email": user_info["email"],
            "picture": user_info.get("picture", "/static/user.png"),
            "last_login": datetime.utcnow(),
            "oauth_token": token  # Store OAuth token for future API requests
        }

        try:
            result = users_collection.update_one(
                {"email": user_data["email"]},
                {"$set": user_data},
                upsert=True
            )
            logger.info(f"User data updated: matched={result.matched_count}, modified={result.modified_count}, upserted_id={result.upserted_id}")
        except Exception as e:
            logger.error(f"Error saving user data to MongoDB: {e}")

        session.permanent = True
        session['user'] = user_data

        return redirect(url_for('home'))

    except Exception as e:
        logger.error(f"Error in Google callback: {str(e)}")
        session.clear()
        return render_template('error.html', error="Authentication failed. Please try again.")


def sanitize_for_json(obj):
    """Convert MongoDB document to JSON-serializable dict"""
    if isinstance(obj, dict):
        # Convert ObjectId to string in dictionary
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Process each item in list
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, ObjectId):
        # Convert ObjectId to string
        return str(obj)
    else:
        # Return other types as is
        return obj

@app.route('/register', methods=['POST'])
def register():
    try:
        # Check if the request is a form submission or JSON
        if request.is_json:
            data = request.get_json()
            name = data.get('name')
            email = data.get('email')
            password = data.get('password')
            confirm_password = data.get('confirmPassword')
        else:
            # Handle traditional form submission
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirmPassword')
        
        # Validate inputs
        if not all([name, email, password, confirm_password]):
            if request.is_json:
                return jsonify({"success": False, "message": "All fields are required"}), 400
            flash("All fields are required", "error")
            return redirect(url_for('home'))
        
        if password != confirm_password:
            if request.is_json:
                return jsonify({"success": False, "message": "Passwords do not match"}), 400
            flash("Passwords do not match", "error")
            return redirect(url_for('home'))
        
        # Check if user already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            if request.is_json:
                return jsonify({"success": False, "message": "Email already registered"}), 400
            flash("Email already registered", "error")
            return redirect(url_for('home'))
        
        # Hash password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Create user
        user_data = {
            "name": name,
            "email": email,
            "password": hashed_password,
            "picture": "/static/user.png",
            "last_login": datetime.utcnow(),
            "created_at": datetime.utcnow(),
            "auth_type": "email"
        }
        
        # Insert the user and get the inserted ID
        result = users_collection.insert_one(user_data)
        
        # Create a clean version for the session (without password)
        session_user = {
            "_id": str(result.inserted_id),  # Convert ObjectId to string
            "name": name,
            "email": email,
            "picture": "/static/user.png",
            "last_login": datetime.utcnow().isoformat(),  # Convert datetime to string
            "created_at": datetime.utcnow().isoformat(),
            "auth_type": "email"
        }
        
        # Store in session
        session.permanent = True
        session['user'] = session_user
        
        if request.is_json:
            return jsonify({"success": True, "message": "Registration successful!", "redirect": url_for('home')}), 200
        
        flash("Registration successful! Welcome to Legal AI Arena.", "success")
        return redirect(url_for('home'))
    
    except Exception as e:
        logger.error(f"Error in registration: {str(e)}")
        if request.is_json:
            return jsonify({"success": False, "message": f"Registration failed: {str(e)}"}), 500
        flash(f"Registration failed: {str(e)}", "error")
        return redirect(url_for('home'))

from flask import jsonify, request, redirect, url_for, flash, session, make_response

@app.route('/login-with-email', methods=['POST'])
def login_with_email():
    try:
        # Check if the request is a form submission or JSON
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
        else:
            # Handle traditional form submission
            email = request.form.get('email')
            password = request.form.get('password')
       
        # Validate inputs
        if not all([email, password]):
            if request.is_json:
                return jsonify({"success": False, "message": "Email and password are required"}), 400
            flash("Email and password are required", "error")
            return redirect(url_for('home'))
       
        # Find user
        user = users_collection.find_one({"email": email})
        if not user:
            if request.is_json:
                return jsonify({"success": False, "message": "Invalid email or password"}), 401
            flash("Invalid email or password", "error")
            return redirect(url_for('home'))
       
        # Verify password
        if 'password' not in user:
            if request.is_json:
                return jsonify({"success": False, "message": "Please login with Google"}), 400
            flash("Please login with Google", "error")
            return redirect(url_for('home'))
       
        if not bcrypt.check_password_hash(user['password'], password):
            if request.is_json:
                return jsonify({"success": False, "message": "Invalid email or password"}), 401
            flash("Invalid email or password", "error")
            return redirect(url_for('home'))
       
        # Create a clean version for the session
        session_user = sanitize_for_json({
            "_id": str(user["_id"]),  # Ensure _id is string for JSON serialization
            "name": user.get("name", "User"),
            "email": user["email"],
            "picture": user.get("picture", "/static/user.png"),
            "last_login": datetime.utcnow().isoformat(),  # Serialize datetime
            "auth_type": "email"
        })
       
        # Update last login time
        users_collection.update_one(
            {"email": email},
            {"$set": {"last_login": datetime.utcnow()}}
        )
       
        # Set session data
        session.permanent = True
        session['user'] = session_user
        
        # Create a response object
        if request.is_json:
            resp = make_response(jsonify({
                "success": True,
                "message": "Login successful",
                "redirect": url_for('home')
            }))
        else:
            resp = make_response(redirect(url_for('home')))
        
        # Set cookies to ensure persistence
        resp.set_cookie('logged_in', 'true', max_age=604800)  # 7 days
        
        return resp
       
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        if request.is_json:
            return jsonify({"success": False, "message": f"Login failed: {str(e)}"}), 500
        flash(f"Login failed: {str(e)}", "error")
        return redirect(url_for('home'))
    
@app.route('/login_page', methods=['GET'])
def login_page():
   return redirect(url_for('login'))


@app.route('/check-login-status', methods=['GET'])
def check_login_status():
    user = session.get('user')
    if user:
        return jsonify({'loggedIn': True})
    return jsonify({'loggedIn': False})


@app.route("/tools")
def tool():
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("tools.html", user=session["user"])


@app.route("/compare")
def compare():
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("compare.html", user=session["user"])


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

def check_question_legality(question):
    """
    Use Together AI API with Llama-3.3-70B-Instruct-Turbo-Free to determine if a question is legal
    
    Parameters:
    - question: The question text to check
    
    Returns:
    - Boolean: True if legal, False if illegal
    """
    try:
        import requests
        import os
        import json
        
        # Together AI API configuration
        TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")  # Make sure this is in your environment variables
        TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
        LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        
        # Headers for API request
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Create a prompt for the legality check
        prompt = f"""<|begin_of_text|><|system|>
You are a helpful AI assistant. Your task is to determine if the following question is legal and appropriate to answer.

A question is ILLEGAL if it:
1. Asks for instructions on illegal activities (hacking, fraud, etc.)
2. Requests information on creating weapons, drugs, or dangerous items
3. Seeks advice on how to harm others or oneself 
4. Contains hate speech or promotes discrimination
5. Asks for private/personal information about specific individuals
6. Explicitly requests content that would be illegal in most jurisdictions

Analyze the question carefully and respond with EXACTLY one word, either "LEGAL" or "ILLEGAL".
<|user|>
Question: "{question}"

Is this question legal and appropriate to answer? Answer with only LEGAL or ILLEGAL.
<|assistant|>
"""
        
        # Payload for API request
        payload = {
            "model": LLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["<|user|>", "<|system|>", "<|end_of_text|>"]
        }
        
        # Make API request
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        result = response.json()
        
        # Extract the generated text from response
        generated_text = result.get("choices", [{}])[0].get("text", "").strip().upper()
        
        # Determine if response indicates the question is legal
        is_legal = "LEGAL" in generated_text and "ILLEGAL" not in generated_text
        
        logger.info(f"Question legality check (Together AI): '{question}' -> {is_legal}")
        logger.debug(f"Raw model response: {generated_text}")
        
        return is_legal
        
    except Exception as e:
        logger.error(f"Error checking question legality with Together AI: {e}")
        # Log the detailed error for debugging
        import traceback
        logger.error(traceback.format_exc())
        # Default to allowing it if the legal check fails
        return True


@app.route("/query-models", methods=["POST", "GET"])
def query_models():
    """Endpoint to query both LLM models and evaluate their responses sequentially"""
    print("==== QUERY MODELS ENDPOINT HIT ====")
    if "user" not in session:
        print("ERROR: User not logged in")
        return jsonify({"error": "User not logged in"}), 401
   
    # Handle GET request for EventSource
    if request.method == "GET":
        question = request.args.get("question")
    else:  # POST request
        data = request.json
        question = data.get("question")
       
    print(f"Question received: {question}")
   
    if not question:
        print("ERROR: No question provided")
        return jsonify({"error": "Question is required"}), 400
   
    logger.info(f"Processing question: {question}")
    
    # Check question legality
    is_legal = check_question_legality(question)
    if not is_legal:
        logger.warning(f"Illegal question detected and flagged: {question}")
        # We'll still process the question but mark it as flagged
        # This allows admins to review it but prevents rating impacts
   
    # Create response object that will be updated as models complete
    response_data = {
        "question": question,
        "is_legal": is_legal,  # Add flag for legal status
        "model_a": {
            "name": "eBrevia",
            "response": "",
            "metrics": {},
            "status": "pending"
        },
        "model_b": {
            "name": "Equally.ai",
            "response": "",
            "metrics": {},
            "status": "pending"
        },
        "votes": {
            "counts": {
                "a": 0,
                "b": 0,
                "tie": 0,
                "bad": 0
            },
            "user_vote": None
        }
    }
   
    # Get existing vote counts for this question
    vote_counts = {
        "a": votes_collection.count_documents({"question": question, "vote_type": "a"}),
        "b": votes_collection.count_documents({"question": question, "vote_type": "b"}),
        "tie": votes_collection.count_documents({"question": question, "vote_type": "tie"}),
        "bad": votes_collection.count_documents({"question": question, "vote_type": "bad"})
    }
   
    # Check if user has already voted
    user_id = session["user"].get("id") or session["user"]["email"]
    user_vote = votes_collection.find_one({
        "question": question,
        "user_id": user_id
    })
   
    response_data["votes"]["counts"] = vote_counts
    response_data["votes"]["user_vote"] = user_vote["vote_type"] if user_vote else None
   
    # Extract user info from session before passing to generator
    user_email = session["user"]["email"]
   
    # This will enable us to stream the response
    return Response(
        stream_model_responses(question, response_data, user_email),
        content_type="text/event-stream"
    )
def stream_model_responses(question, response_data, user_email):
    """Generator function to stream model responses as they become available"""
    # user_email is now passed as a parameter instead of accessing from session
    timestamp = datetime.utcnow()
    
    # Start with the initial state
    yield f"data: {json.dumps(response_data)}\n\n"
    
    # Process Llama model first
    try:
        # Get previous response
        prev_llama_response = get_previous_model_response(question, "llama")
        
        # Query Llama model
        logger.info("Querying Llama model...")
        print("Calling Together API for Llama model...")
        llama_response = call_together_api(question, LLAMA_MODEL)
        print(f"Llama response received (length: {len(llama_response) if llama_response else 0})")
        
        # Update response object with Llama results
        response_data["model_a"]["response"] = llama_response
        response_data["model_a"]["status"] = "evaluating"
        
        # Send partial update
        yield f"data: {json.dumps(response_data)}\n\n"
        
        # Evaluate Llama response
        logger.info(f"Evaluating Llama response...")
        llama_evaluation = call_dify_workflow(
            question, 
            llama_response,
            prev_llama_response if prev_llama_response else None,
            model_name="llama"
        )
        llama_metrics = parse_evaluation_metrics(llama_evaluation)
        print(f"Llama metrics: {llama_metrics}")
        
        # Update response with metrics
        response_data["model_a"]["metrics"] = llama_metrics
        response_data["model_a"]["status"] = "complete"
        
        # Send updated data
        yield f"data: {json.dumps(response_data)}\n\n"
        
        # Save Llama response to MongoDB
        llama_doc = {
            "question": question,
            "response": llama_response,
            "evaluation": llama_evaluation,
            "metrics": llama_metrics,
            "updated_at": timestamp,
            "updated_by": user_email
        }
        
        # Save to database in background
        try:
            result = llama_responses_collection.update_one(
                {"question": question},
                {"$set": llama_doc},
                upsert=True
            )
            logger.info(f"Llama response saved: matched={result.matched_count}, modified={result.modified_count}, upserted_id={result.upserted_id}")
        except Exception as e:
            logger.error(f"Error saving Llama response to MongoDB: {e}")
            print(f"ERROR saving Llama response: {e}")
            
    except Exception as e:
        logger.error(f"Error processing Llama model: {e}")
        response_data["model_a"]["response"] = f"Error: {str(e)}"
        response_data["model_a"]["status"] = "error"
        yield f"data: {json.dumps(response_data)}\n\n"
    
    # Now process DeepSeek model
    try:
        # Get previous response
        prev_deepseek_response = get_previous_model_response(question, "deepseek")
        
        # Query DeepSeek model
        logger.info("Querying DeepSeek model...")
        print("Calling Together API for DeepSeek model...")
        deepseek_response = call_together_api(question, DEEPSEEK_MODEL)
        print(f"DeepSeek response received (length: {len(deepseek_response) if deepseek_response else 0})")
        
        # Update response object with DeepSeek results
        response_data["model_b"]["response"] = deepseek_response
        response_data["model_b"]["status"] = "evaluating"
        
        # Send partial update
        yield f"data: {json.dumps(response_data)}\n\n"
        
        # Evaluate DeepSeek response
        logger.info(f"Evaluating DeepSeek response...")
        deepseek_evaluation = call_dify_workflow(
            question, 
            deepseek_response,
            prev_deepseek_response if prev_deepseek_response else None,
            model_name="deepseek"
        )
        deepseek_metrics = parse_evaluation_metrics(deepseek_evaluation)
        print(f"DeepSeek metrics: {deepseek_metrics}")
        
        # Update response with metrics
        response_data["model_b"]["metrics"] = deepseek_metrics
        response_data["model_b"]["status"] = "complete"
        
        # Send final data
        yield f"data: {json.dumps(response_data)}\n\n"
        
        # Save DeepSeek response to MongoDB
        deepseek_doc = {
            "question": question,
            "response": deepseek_response,
            "evaluation": deepseek_evaluation,
            "metrics": deepseek_metrics,
            "updated_at": timestamp,
            "updated_by": user_email
        }
        
        # Save to database in background
        try:
            result = deepseek_responses_collection.update_one(
                {"question": question},
                {"$set": deepseek_doc},
                upsert=True
            )
            logger.info(f"DeepSeek response saved: matched={result.matched_count}, modified={result.modified_count}, upserted_id={result.upserted_id}")
        except Exception as e:
            logger.error(f"Error saving DeepSeek response to MongoDB: {e}")
            print(f"ERROR saving DeepSeek response: {e}")
            
    except Exception as e:
        logger.error(f"Error processing DeepSeek model: {e}")
        response_data["model_b"]["response"] = f"Error: {str(e)}"
        response_data["model_b"]["status"] = "error"
        yield f"data: {json.dumps(response_data)}\n\n"
    
    # Send complete message to signify end of stream
    response_data["status"] = "complete"
    yield f"data: {json.dumps(response_data)}\n\n"
def get_recent_questions():
    """Get a list of recently asked questions from both model collections"""
    if "user" not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    try:
        # Get questions from both collections
        logger.debug("Fetching recent questions from MongoDB (both collections)")
        
        # Query from Llama collection
        llama_pipeline = [
            {"$sort": {"updated_at": -1}},
            {"$project": {"_id": 0, "question": 1, "updated_at": 1}}
        ]
        llama_questions = list(llama_responses_collection.aggregate(llama_pipeline))
        
        # Query from DeepSeek collection
        deepseek_pipeline = [
            {"$sort": {"updated_at": -1}},
            {"$project": {"_id": 0, "question": 1, "updated_at": 1}}
        ]
        deepseek_questions = list(deepseek_responses_collection.aggregate(deepseek_pipeline))
        
        # Combine and deduplicate questions
        all_questions = llama_questions + deepseek_questions
        all_questions.sort(key=lambda x: x.get("updated_at", datetime.min), reverse=True)
        
        # Remove duplicates keeping the most recent
        unique_questions = []
        seen_questions = set()
        
        for q in all_questions:
            if q["question"] not in seen_questions:
                unique_questions.append(q)
                seen_questions.add(q["question"])
        
        # Limit to 10 most recent
        recent_questions = unique_questions[:10]
        
        logger.info(f"Found {len(recent_questions)} recent unique questions")
        
        return jsonify({
            "questions": [q["question"] for q in recent_questions]
        })
    except Exception as e:
        logger.error(f"Error fetching recent questions: {e}")
        return jsonify({"error": "Failed to fetch recent questions", "questions": []}), 500


@app.route("/get-tool-votes", methods=["POST"])
def get_tool_votes():
    if "user" not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    data = request.json
    tool_ids = data.get("toolIds", [])
    
    votes = {}
    try:
        for tool_id in tool_ids:
            vote_doc = votes_collection.find_one({"tool_id": tool_id})
            if vote_doc:
                votes[tool_id] = vote_doc["votes"]
            else:
                votes[tool_id] = 0
        
        logger.debug(f"Retrieved votes for {len(tool_ids)} tools: {votes}")
        return jsonify({"votes": votes})
    except Exception as e:
        logger.error(f"Error retrieving tool votes: {e}")
        return jsonify({"error": "Failed to retrieve votes", "votes": {}}), 500


@app.route("/vote-for-tool", methods=["POST"])
def vote_for_tool():
    if "user" not in session:
        return jsonify({"error": "User not logged in", "success": False}), 401
    
    data = request.json
    tool_id = data.get("toolId")
    user_email = session["user"]["email"]
    
    if not tool_id:
        return jsonify({"error": "Tool ID is required", "success": False}), 400
    
    try:
        # Check if user has already voted for this tool
        user_vote = votes_collection.find_one({
            "tool_id": tool_id,
            "voters": user_email
        })
        
        if user_vote:
            logger.info(f"User {user_email} has already voted for tool {tool_id}")
            return jsonify({
                "success": False,
                "message": "You have already voted for this tool"
            })
        
        # Update vote count and add user to voters list
        logger.info(f"Recording vote from {user_email} for tool {tool_id}")
        result = votes_collection.update_one(
            {"tool_id": tool_id},
            {
                "$inc": {"votes": 1},
                "$push": {"voters": user_email}
            },
            upsert=True
        )
        
        logger.debug(f"Vote recorded: matched={result.matched_count}, modified={result.modified_count}, upserted_id={result.upserted_id}")
        
        # Get updated vote count
        vote_doc = votes_collection.find_one({"tool_id": tool_id})
        vote_count = vote_doc["votes"] if vote_doc else 1
        
        return jsonify({
            "success": True,
            "votes": vote_count
        })
    except Exception as e:
        logger.error(f"Error recording vote: {e}")
        return jsonify({"error": "Failed to record vote", "success": False}), 500

TOOL_CATEGORIES = [
    "AI/ML", "Data Processing", "Communication", 
    "Analytics", "Payment Processing", "E-commerce",
    "Authentication", "Mapping/Location", "Storage"
]

@app.route('/vendor-onboarding')
def vendor_onboarding():
    # Check if user is logged in, redirect to login if not
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Pass categories to the template
    return render_template('vendor_onboarding.html', categories=TOOL_CATEGORIES)

@app.route('/documentation')
def documentation():
    # Check if user is logged in, redirect to login if not
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Pass categories to the template
    return render_template('doc.html')

@app.route('/api/submit-tool', methods=['POST'])
def submit_tool():
    try:
        print("Request received at /api/submit-tool")
        tool_data = request.json
        print(f"Received data: {tool_data}")
        
        # Basic validation
        required_fields = ['name', 'price', 'endpoint', 'category', 'description']
        for field in required_fields:
            if field not in tool_data or not tool_data[field]:
                return jsonify({"success": False, "message": f"Missing required field: {field}"}), 400
        
        # Optional validation for price as number
        try:
            tool_data['price'] = float(tool_data['price'])
        except ValueError:
            return jsonify({"success": False, "message": "Price must be a number"}), 400
        
        # Check if category is valid
        if tool_data['category'] not in TOOL_CATEGORIES:
            return jsonify({"success": False, "message": "Invalid category"}), 400
            
        # Insert into MongoDB
        print("Attempting to insert into MongoDB...")
        result = tools_collection.insert_one(tool_data)
        print(f"Insert successful, ID: {result.inserted_id}")
        
        return jsonify({
            "success": True, 
            "message": "Tool submitted successfully!",
            "tool_id": str(result.inserted_id)
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/categories')
def get_categories():
    return jsonify(TOOL_CATEGORIES)

@app.route("/api/feedback", methods=["POST"])
def create_feedback():
    """
    Store feedback in MongoDB
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data or "name" not in data or "feedback" not in data:
            return jsonify({
                "success": False,
                "message": "Name and feedback are required"
            }), 400
            
        # Create feedback document
        feedback_doc = {
            "name": data["name"],
            "feedback": data["feedback"],
            "timestamp": datetime.now()
        }
        
        # Insert the feedback into MongoDB
        result = feedback_collection.insert_one(feedback_doc)
        
        # Return success response with the inserted ID
        return jsonify({
            "success": True,
            "message": "Feedback submitted successfully",
            "feedback_id": str(result.inserted_id)
        }), 201
    except Exception as e:
        # Log the error (in a production environment, use a proper logger)
        print(f"Error storing feedback: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to store feedback. Please try again later."
        }), 500


def calculate_custom_rating(votes, current_rating=500, model_name=None):
    """
    Calculate custom rating based on unique voters with diminishing returns
   
    Parameters:
    - votes: List of all vote documents
    - current_rating: Current rating of the model
    - model_name: Name of the model ("eBrevia" or "Equally.ai")
   
    Returns:
    - New custom rating value
    """
    # Group votes by user to apply diminishing returns
    user_votes = {}
    for vote in votes:
        user_id = vote["user_id"]
        if user_id not in user_votes:
            user_votes[user_id] = []
        user_votes[user_id].append(vote)
   
    # Calculate vote weight for each user based on their number of votes
    user_weights = {}
    for user_id, user_vote_list in user_votes.items():
        # Diminishing returns formula: weight = 1 / sqrt(vote_count)
        vote_count = len(user_vote_list)
        user_weights[user_id] = 1 / math.sqrt(vote_count)
   
    # Calculate total score for this model
    total_score = 0
   
    for vote in votes:
        user_id = vote["user_id"]
        weight = user_weights[user_id]
        vote_type = vote["vote_type"]
        
        # Model-specific scoring
        if model_name == "eBrevia":  # Model A
            if vote_type == "a":  # Win for model A
                total_score += weight
            elif vote_type == "b":  # Loss for model A
                total_score -= weight
        elif model_name == "Equally.ai":  # Model B
            if vote_type == "b":  # Win for model B
                total_score += weight
            elif vote_type == "a":  # Loss for model B
                total_score -= weight
        # Ties don't affect the score
   
    # Calculate rating change (multiplier can be adjusted)
    rating_change = total_score * 10
   
    # Apply the change to current rating
    new_rating = current_rating + rating_change
   
    # Ensure rating stays within reasonable bounds
    return max(0, min(1000, new_rating))


# Update the MongoDB schema to include custom rating
def initialize_ratings():
    """Initialize both ELO and custom ratings for models if they don't exist"""
    models = ["eBrevia", "Equally.ai"]
    default_elo = 1500  # Standard starting ELO
    default_custom = 500  # Starting custom rating
    
    for model in models:
        existing = elo_ratings_collection.find_one({"model_name": model})
        if not existing:
            elo_ratings_collection.insert_one({
                "model_name": model,
                "elo_rating": default_elo,
                "custom_rating": default_custom,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_matches": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            logger.info(f"Initialized ratings for {model}")
        elif "custom_rating" not in existing:
            # Update existing models to include custom rating
            elo_ratings_collection.update_one(
                {"model_name": model},
                {"$set": {"custom_rating": default_custom}}
            )
            logger.info(f"Added custom rating to existing model {model}")
def initialize_elo_ratings():
    """Initialize ELO ratings for models if they don't exist"""
    models = ["eBrevia", "Equally.ai"]
    default_elo = 1500  # Standard starting ELO
    
    for model in models:
        existing = elo_ratings_collection.find_one({"model_name": model})
        if not existing:
            elo_ratings_collection.insert_one({
                "model_name": model,
                "elo_rating": default_elo,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_matches": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            logger.info(f"Initialized ELO rating for {model}")

# Call this during application startup
initialize_elo_ratings()

# Helper function to calculate new ELO ratings
def calculate_elo(rating_a, rating_b, outcome_a):
    """
    Calculate new ELO ratings based on match outcome
    
    Parameters:
    - rating_a: Current ELO rating of model A
    - rating_b: Current ELO rating of model B
    - outcome_a: Result for model A (1 for win, 0.5 for tie, 0 for loss)
    
    Returns:
    - Tuple of (new_rating_a, new_rating_b)
    """
    # K-factor determines how much ratings change after each match
    k_factor = 32
    
    # Calculate expected outcome using ELO formula
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    
    # Calculate new ratings
    new_rating_a = rating_a + k_factor * (outcome_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - outcome_a) - expected_b)
    
    return new_rating_a, new_rating_b
@app.route('/leaderboard')
def leaderboard():
    """Model leaderboard page route"""
    # Check if user is logged in - this is used to determine if nav buttons should be disabled
    user = session.get('user', None)
    return render_template('leaderboard.html', user=user)
@app.route("/cast-vote", methods=["POST"])
def cast_vote():
    """Endpoint to cast a vote between models and update both ELO and custom ratings"""
    if "user" not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    logger.info("Cast vote endpoint hit")
    
    data = request.json
    if not data:
        logger.error("No JSON data received in request")
        return jsonify({"error": "No data received"}), 400
        
    question = data.get("question")
    vote_type = data.get("vote_type")
    
    logger.info(f"Vote request received - question: {question}, vote_type: {vote_type}")
    
    if not question or not vote_type:
        return jsonify({"error": "Question and vote type are required"}), 400
    
    if vote_type not in ["a", "b", "tie", "bad"]:
        return jsonify({"error": "Invalid vote type"}), 400
    
    user_email = session["user"]["email"]
    user_id = session["user"].get("id") or user_email
    
    # Check if user has already voted on this question
    existing_vote = votes_collection.find_one({
        "question": question,
        "user_id": user_id
    })
    
    if existing_vote:
        return jsonify({
            "error": "You have already voted on this question",
            "previous_vote": existing_vote["vote_type"]
        }), 409
    
    # Check question legality first
    is_legal = check_question_legality(question)
    if not is_legal and vote_type != "bad":
        logger.warning(f"Illegal question detected: {question}")
        return jsonify({
            "error": "This question has been flagged as potentially illegal or inappropriate"
        }), 400
    
    # Get current ratings
    model_a_data = elo_ratings_collection.find_one({"model_name": "eBrevia"})
    model_b_data = elo_ratings_collection.find_one({"model_name": "Equally.ai"})
    
    if not model_a_data or not model_b_data:
        initialize_ratings()
        model_a_data = elo_ratings_collection.find_one({"model_name": "eBrevia"})
        model_b_data = elo_ratings_collection.find_one({"model_name": "Equally.ai"})
    
    rating_a_elo = model_a_data["elo_rating"]
    rating_b_elo = model_b_data["elo_rating"]
    rating_a_custom = model_a_data.get("custom_rating", 500)
    rating_b_custom = model_b_data.get("custom_rating", 500)
    
    # Process vote and update ratings
    timestamp = datetime.utcnow()
    
    # Create vote record
    vote_doc = {
        "question": question,
        "user_id": user_id,
        "user_email": user_email,
        "vote_type": vote_type,
        "created_at": timestamp
    }
    
    # Update model stats based on vote type
    if vote_type != "bad":  # Only update ratings for non-"bad" votes
        wins_a = model_a_data["wins"]
        losses_a = model_a_data["losses"]
        ties_a = model_a_data["ties"]
        total_a = model_a_data["total_matches"]
        
        wins_b = model_b_data["wins"]
        losses_b = model_b_data["losses"]
        ties_b = model_b_data["ties"]
        total_b = model_b_data["total_matches"]
        
        # Update ELO ratings
        if vote_type == "a":
            # Model A wins, B loses
            outcome_a = 1.0
            new_rating_a_elo, new_rating_b_elo = calculate_elo(rating_a_elo, rating_b_elo, outcome_a)
            wins_a += 1
            losses_b += 1
        elif vote_type == "b":
            # Model B wins, A loses
            outcome_a = 0.0
            new_rating_a_elo, new_rating_b_elo = calculate_elo(rating_a_elo, rating_b_elo, outcome_a)
            losses_a += 1
            wins_b += 1
        elif vote_type == "tie":
            # Tie
            outcome_a = 0.5
            new_rating_a_elo, new_rating_b_elo = calculate_elo(rating_a_elo, rating_b_elo, outcome_a)
            ties_a += 1
            ties_b += 1
        
        # Get previous votes for both models
        previous_votes = list(votes_collection.find({
            "vote_type": {"$in": ["a", "b", "tie"]},
            "question": {"$ne": question}  # Exclude current vote as it's not saved yet
        }))
        
        # Create model-specific vote lists
        model_a_votes = []
        model_b_votes = []
        
        # Process previous votes for each model's perspective
        for vote in previous_votes:
            # For model A: "a"=win, "b"=loss, "tie"=tie
            model_a_votes.append(vote.copy())
            
            # For model B: "b"=win, "a"=loss, "tie"=tie
            model_b_vote = vote.copy()
            if vote["vote_type"] == "a":
                model_b_vote["vote_type"] = "b"  # Loss for model B
            elif vote["vote_type"] == "b":
                model_b_vote["vote_type"] = "a"  # Win for model B
            model_b_votes.append(model_b_vote)
        
        # Add current vote to both models' vote lists with correct perspective
        current_vote_a = vote_doc.copy()
        current_vote_b = vote_doc.copy()
        
        # For model B, flip "a" and "b" vote types to represent wins/losses correctly
        if current_vote_b["vote_type"] == "a":
            current_vote_b["vote_type"] = "b"  # Loss for model B
        elif current_vote_b["vote_type"] == "b":
            current_vote_b["vote_type"] = "a"  # Win for model B
        
        model_a_votes.append(current_vote_a)
        model_b_votes.append(current_vote_b)
        all_votes = previous_votes + [vote_doc]
        # Calculate custom ratings
        # Don't flip votes; instead pass the model name to the calculation function
        new_rating_a_custom = calculate_custom_rating(all_votes, rating_a_custom, "eBrevia")
        new_rating_b_custom = calculate_custom_rating(all_votes, rating_b_custom, "Equally.ai")
        
        # Update ratings in database
        elo_ratings_collection.update_one(
            {"model_name": "eBrevia"},
            {"$set": {
                "elo_rating": new_rating_a_elo,
                "custom_rating": new_rating_a_custom,
                "wins": wins_a,
                "losses": losses_a,
                "ties": ties_a,
                "total_matches": total_a + 1,
                "updated_at": timestamp
            }}
        )
        
        elo_ratings_collection.update_one(
            {"model_name": "Equally.ai"},
            {"$set": {
                "elo_rating": new_rating_b_elo,
                "custom_rating": new_rating_b_custom,
                "wins": wins_b,
                "losses": losses_b,
                "ties": ties_b,
                "total_matches": total_b + 1,
                "updated_at": timestamp
            }}
        )
    
    # Save the vote
    result = votes_collection.insert_one(vote_doc)
    logger.info(f"Vote saved with ID: {result.inserted_id}")
    
    # Get current vote counts for this question
    vote_counts = {
        "a": votes_collection.count_documents({"question": question, "vote_type": "a"}),
        "b": votes_collection.count_documents({"question": question, "vote_type": "b"}),
        "tie": votes_collection.count_documents({"question": question, "vote_type": "tie"}),
        "bad": votes_collection.count_documents({"question": question, "vote_type": "bad"})
    }
    
    # Get updated ratings
    model_a_updated = elo_ratings_collection.find_one({"model_name": "eBrevia"})
    model_b_updated = elo_ratings_collection.find_one({"model_name": "Equally.ai"})
    
    logger.info(f"Vote successful - returning vote counts: {vote_counts}")
    
    return jsonify({
        "success": True,
        "vote_recorded": vote_type,
        "vote_counts": vote_counts,
        "model_ratings": {
            "eBrevia": {
                "elo_rating": model_a_updated["elo_rating"],
                "custom_rating": model_a_updated.get("custom_rating", 500),
                "wins": model_a_updated["wins"],
                "losses": model_a_updated["losses"],
                "ties": model_a_updated["ties"]
            },
            "Equally.ai": {
                "elo_rating": model_b_updated["elo_rating"],
                "custom_rating": model_b_updated.get("custom_rating", 500),
                "wins": model_b_updated["wins"],
                "losses": model_b_updated["losses"],
                "ties": model_b_updated["ties"]
            }
        }
    })

# Update endpoint to get model ratings to include custom ratings
@app.route("/get-model-ratings", methods=["GET"])
def get_model_ratings():
    """Endpoint to get current ELO and custom ratings for all models"""
    if "user" not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    # Get ratings from database
    ratings = list(elo_ratings_collection.find({}, {
        "_id": 0,
        "model_name": 1,
        "elo_rating": 1,
        "custom_rating": 1,
        "wins": 1,
        "losses": 1,
        "ties": 1,
        "total_matches": 1
    }))
    
    # Ensure custom_rating exists for all models
    for model in ratings:
        if "custom_rating" not in model:
            model["custom_rating"] = 500
    
    return jsonify({
        "ratings": ratings
    })

@app.route("/question-votes/<path:question>", methods=["GET"])
def get_question_votes(question):
    """Get votes for a specific question"""
    if "user" not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    # URL decode the question parameter
    question = unquote(question)
    
    # Get vote counts
    vote_counts = {
        "a": votes_collection.count_documents({"question": question, "vote_type": "a"}),
        "b": votes_collection.count_documents({"question": question, "vote_type": "b"}),
        "tie": votes_collection.count_documents({"question": question, "vote_type": "tie"}),
        "bad": votes_collection.count_documents({"question": question, "vote_type": "bad"})
    }
    
    # Check if current user has voted
    user_id = session["user"].get("id") or session["user"]["email"]
    user_vote = votes_collection.find_one({
        "question": question,
        "user_id": user_id
    })
    
    return jsonify({
        "question": question,
        "vote_counts": vote_counts,
        "user_vote": user_vote["vote_type"] if user_vote else None,
        "total_votes": sum(vote_counts.values())
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
