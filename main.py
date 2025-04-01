import os
import logging
import pandas as pd
from flask import Flask, json, jsonify, redirect, url_for, session, request, render_template
from flask_session import Session
from authlib.integrations.flask_client import OAuth
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import requests
from werkzeug.utils import secure_filename
from together import Together  # Import the Together API client


# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

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


def call_dify_workflow(question, model_answer, prev_model_answer=None):
    """Calls the Dify API to evaluate the answer with 5 different metrics."""
    headers = {
        "Authorization": f"Bearer {DIFY_WORKFLOW_SECRET}",
        "Content-Type": "application/json"
    }
    
    # If no previous answer exists, use the current answer as comparison
    if not prev_model_answer:
        prev_model_answer = model_answer
    
    payload = {
        "workflow_id": os.getenv("DIFY_WORKFLOW_ID"),
        "inputs": {
            "question": question[:48],  # Trim if needed
            "user_answer": model_answer,
            "prev_answer": prev_model_answer  # Using previous answer as comparison
        },
        "response_mode": "blocking",
        "user": "abc-123"
    }
    logger.debug(f"Calling Dify API for model evaluation with payload: {json.dumps(payload, indent=2)}")
    try:
        logger.debug(f"Calling Dify API with payload: {payload}")
        response = requests.post(DIFY_WORKFLOW_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json().get("data", {}).get("outputs", {})
        logger.debug(f"Dify API response: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Dify API: {e}")
        # Return a default response structure instead of None
        return {"evaluation_results": "{\"accuracy\": 0, \"completeness\": 0, \"helpfulness\": 0, \"clarity\": 0, \"comparison\": 0}"}


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


@app.route("/")
def home():
    return render_template("index.html", user=session.get("user"))


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
            "picture": user_info.get("picture", "/static/default-profile.png"),
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
    
    # Evaluate each model's response with better error handling
    logger.info(f"Evaluating Llama response...")
    if prev_llama_response:
        # Only do comparison if there's a previous response
        llama_evaluation = call_dify_workflow(question, llama_response, prev_llama_response)
    else:
        # Otherwise, just evaluate the current response without comparison
        llama_evaluation = call_dify_workflow(question, llama_response)
    llama_metrics = parse_evaluation_metrics(llama_evaluation)
    
    logger.info(f"Evaluating DeepSeek response...")
    if prev_deepseek_response:
       logger.debug(f"Using previous DeepSeek response for comparison: {prev_deepseek_response}")
       deepseek_evaluation = call_dify_workflow(question, deepseek_response, prev_deepseek_response)
    else:
       logger.debug("No previous DeepSeek response found, evaluating standalone response")
       deepseek_evaluation = call_dify_workflow(question, deepseek_response)
    deepseek_metrics = parse_evaluation_metrics(deepseek_evaluation)
    logger.debug(f"DeepSeek evaluation result: {deepseek_evaluation}")
    
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
if __name__ == "__main__":
    app.run(debug=True)