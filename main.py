import os
import logging
import pandas as pd
from flask import Flask, json, jsonify, redirect, url_for, session, request, render_template, url_for,  flash
from flask_session import Session
from flask import Flask, request, jsonify, render_template
from authlib.integrations.flask_client import OAuth
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, time, timedelta
import requests
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
    

app = Flask(__name__, static_folder="static", template_folder="templates")
bcrypt = Bcrypt(app)

app.secret_key = os.getenv("SECRET_KEY")
dummy_user = {
    "name": "Tester",
    "email": "tester@example.com",
    "picture": "/static/default-profile.png",
    "last_login": datetime.utcnow()
}

CORS(app,
     supports_credentials=True,
     origins=["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:3000"])


app.config['SESSION_COOKIE_NAME'] = 'google-login-session'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = True  # ✅ only for local dev
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
    tools_collection = db['tools']
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
            
            response.raise_for_status()
            result = response.json().get("data", {}).get("outputs", {})
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
    """Parse evaluation results with better error handling and defaults"""
    if not evaluation or "evaluation_results" not in evaluation:
        logger.warning("Missing evaluation results")
        return {"accuracy": 0, "completeness": 0, "helpfulness": 0, "clarity": 0, "comparison": 0}
    
    try:
        return json.loads(evaluation["evaluation_results"])
    except Exception as e:
        logger.error(f"Error parsing evaluation results: {e}")
        # Return default metrics
        return {"accuracy": 0, "completeness": 0, "helpfulness": 0, "clarity": 0, "comparison": 0}


def get_previous_model_response(question, model_type):
    """Get the previous response for this question and model if it exists"""
    try:
        # Select the appropriate collection based on model type
        collection = llama_responses_collection if model_type == "llama" else deepseek_responses_collection
        
        prev_response = collection.find_one({"question": question})
        
        if prev_response:
            logger.debug(f"Found previous response for {model_type} model and question: {question[:30]}...")
            return prev_response.get("response")
        logger.debug(f"No previous response found for {model_type} model and question: {question[:30]}...")
        return None
    except Exception as e:
        logger.error(f"Error retrieving previous model response: {e}")
        return None


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


@app.route("/query-models", methods=["POST"])
def query_models():
    """Endpoint to query both LLM models and evaluate their responses"""
    if "user" not in session:
        return jsonify({"error": "User not logged in"}), 401
   
    data = request.json
    question = data.get("question")
   
    if not question:
        return jsonify({"error": "Question is required"}), 400
   
    logger.info(f"Processing question: {question}")
   
    # Get previous responses for each model (if they exist)
    prev_llama_response = get_previous_model_response(question, "llama")
    prev_deepseek_response = get_previous_model_response(question, "deepseek")
   
    # Query the models for new responses using Together API
    logger.info("Querying Llama model...")
    llama_response = call_together_api(question, LLAMA_MODEL)
    logger.info("Querying DeepSeek model...")
    deepseek_response = call_together_api(question, DEEPSEEK_MODEL)
    
    # Log response lengths for debugging
    logger.debug(f"Response lengths - Llama: {len(llama_response)}, DeepSeek: {len(deepseek_response)}")
   
    # Process responses to ensure they're valid
    try:
        # Evaluate Llama response
        logger.info(f"Evaluating Llama response...")
        llama_evaluation = call_dify_workflow(
            question, 
            llama_response,
            prev_llama_response if prev_llama_response else None,
            model_name="llama"
        )
        llama_metrics = parse_evaluation_metrics(llama_evaluation)
    except Exception as e:
        logger.error(f"Error during Llama evaluation: {e}")
        llama_metrics = {"accuracy": 0, "completeness": 0, "helpfulness": 0, "clarity": 0, "comparison": 0}
    
    # Process DeepSeek with a small delay to avoid API rate limits
    time.sleep(1)
    
    try:
        # Evaluate DeepSeek response
        logger.info(f"Evaluating DeepSeek response...")
        deepseek_evaluation = call_dify_workflow(
            question, 
            deepseek_response,
            prev_deepseek_response if prev_deepseek_response else None,
            model_name="deepseek"
        )
        deepseek_metrics = parse_evaluation_metrics(deepseek_evaluation)
    except Exception as e:
        logger.error(f"Error during DeepSeek evaluation: {e}")
        deepseek_metrics = {"accuracy": 0, "completeness": 0, "helpfulness": 0, "clarity": 0, "comparison": 0}
    
    # Save the model responses to MongoDB
    timestamp = datetime.utcnow()
    user_email = session["user"]["email"]
   
    # Save Llama response to llama collection
    llama_doc = {
        "question": question,
        "response": llama_response,
        "evaluation": llama_evaluation,
        "metrics": llama_metrics,
        "updated_at": timestamp,
        "updated_by": user_email
    }
   
    try:
        result = llama_responses_collection.update_one(
            {"question": question},
            {"$set": llama_doc},
            upsert=True
        )
        logger.info(f"Llama response saved: matched={result.matched_count}, modified={result.modified_count}, upserted_id={result.upserted_id}")
    except Exception as e:
        logger.error(f"Error saving Llama response to MongoDB: {e}")
   
    # Save DeepSeek response to deepseek collection
    deepseek_doc = {
        "question": question,
        "response": deepseek_response,
        "evaluation": deepseek_evaluation,
        "metrics": deepseek_metrics,
        "updated_at": timestamp,
        "updated_by": user_email
    }
   
    try:
        result = deepseek_responses_collection.update_one(
            {"question": question},
            {"$set": deepseek_doc},
            upsert=True
        )
        logger.info(f"DeepSeek response saved: matched={result.matched_count}, modified={result.modified_count}, upserted_id={result.upserted_id}")
    except Exception as e:
        logger.error(f"Error saving DeepSeek response to MongoDB: {e}")
   
    return jsonify({
        "question": question,
        "model_a": {
            "name": "Meta Llama 3.3 70B",
            "response": llama_response,
            "metrics": llama_metrics
        },
        "model_b": {
            "name": "DeepSeek R1 Distill Llama 70B",
            "response": deepseek_response,
            "metrics": deepseek_metrics
        }
    })


@app.route("/get-recent-questions", methods=["GET"])
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

# Add these two routes to your Flask app

@app.route("/get-user-vote", methods=["POST"])
def get_user_vote():
    if "user" not in session:
        return jsonify({"error": "User not logged in"}), 401
   
    data = request.json
    comparison_id = data.get("comparisonId")
    user_email = session["user"]["email"]
   
    if not comparison_id:
        return jsonify({"error": "Comparison ID is required", "voted": False}), 400
   
    try:
        # Check all possible tool IDs for this comparison
        tool_ids = [
            f"{comparison_id}_model_a",
            f"{comparison_id}_model_b",
            f"{comparison_id}_tie",
            f"{comparison_id}_both_bad"
        ]
        
        for tool_id in tool_ids:
            # Check if user has voted for this tool
            vote_doc = votes_collection.find_one({
                "tool_id": tool_id,
                "voters": user_email
            })
            
            if vote_doc:
                logger.info(f"User {user_email} has voted for {tool_id} in comparison {comparison_id}")
                return jsonify({
                    "voted": True,
                    "toolId": tool_id
                })
        
        # User hasn't voted for this comparison
        return jsonify({
            "voted": False,
            "toolId": None
        })
    except Exception as e:
        logger.error(f"Error checking user vote: {e}")
        return jsonify({"error": "Failed to check vote", "voted": False}), 500

@app.route("/change-vote", methods=["POST"])
def change_vote():
    if "user" not in session:
        return jsonify({"error": "User not logged in", "success": False}), 401
   
    data = request.json
    comparison_id = data.get("comparisonId")
    previous_tool_id = data.get("previousToolId")
    new_tool_id = data.get("newToolId")
    user_email = session["user"]["email"]
   
    if not comparison_id or not new_tool_id:
        return jsonify({"error": "Comparison ID and new tool ID are required", "success": False}), 400
   
    try:
        # If user had a previous vote, remove it
        if previous_tool_id:
            votes_collection.update_one(
                {"tool_id": previous_tool_id},
                {
                    "$inc": {"votes": -1},
                    "$pull": {"voters": user_email}
                }
            )
            logger.info(f"Removed previous vote from {user_email} for {previous_tool_id}")
        
        # Add new vote
        votes_collection.update_one(
            {"tool_id": new_tool_id},
            {
                "$inc": {"votes": 1},
                "$push": {"voters": user_email}
            },
            upsert=True
        )
        logger.info(f"Added new vote from {user_email} for {new_tool_id}")
        
        # Get updated vote counts for all tools in this comparison
        tool_ids = [
            f"{comparison_id}_model_a",
            f"{comparison_id}_model_b",
            f"{comparison_id}_tie",
            f"{comparison_id}_both_bad"
        ]
        
        votes = {}
        for tool_id in tool_ids:
            vote_doc = votes_collection.find_one({"tool_id": tool_id})
            if vote_doc:
                votes[tool_id] = vote_doc["votes"]
            else:
                votes[tool_id] = 0
        
        return jsonify({
            "success": True,
            "votes": votes
        })
    except Exception as e:
        logger.error(f"Error changing vote: {e}")
        return jsonify({"error": "Failed to change vote", "success": False}), 500

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


if __name__ == "__main__":
    app.run(debug=True)