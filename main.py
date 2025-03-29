import os
import logging
import pandas as pd
from flask import Flask, json, jsonify, redirect, url_for, session, request, render_template, url_for
from flask_session import Session
from authlib.integrations.flask_client import OAuth
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime

# Load environment variables
load_dotenv()


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
dummy_user = {
    "name": "Tester",
    "email": "tester@example.com",
    "picture": "/static/default-profile.png",
    "last_login": datetime.utcnow()
}



app.config['SESSION_COOKIE_NAME'] = 'google-login-session'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=60)  # Increased session lifetime
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = True  # Set to True in production with HTTPS

app.logger.setLevel(logging.INFO)

# Add this line to handle proxy headers correctly
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)




# Configure Logging
logging.basicConfig(level=logging.DEBUG)

# Configure MongoDB
client = MongoClient(os.getenv("MONGO_URI"))

db = client['legal_chatbot']
users_collection = db['users']
file_data_collection = db['file_data']

# API URLs & Secrets
GROUND_TRUTH_FILE = "ground_truth.json"

try:
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)
except Exception as e:
    logging.error(f"Error loading ground truth file: {e}")
    ground_truth_data = {}

# Dify API Configuration
DIFY_WORKFLOW_URL = os.getenv("DIFY_WORKFLOW_URL")
DIFY_WORKFLOW_SECRET = os.getenv("DIFY_WORKFLOW_SECRET")

# Allowed File Extensions
ALLOWED_EXTENSIONS = {"csv", "xlsx"}

# OAuth Setup
# def configure_oauth(app):
#     # oauth = OAuth(app)
#     # google = oauth.register(
#     #     name='google',
#     #     client_id=os.getenv('GOOGLE_CLIENT_ID'),
#     #     client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
#     #     authorize_url='https://accounts.google.com/o/oauth2/auth',
#     #     access_token_url='https://oauth2.googleapis.com/token',
#     #     api_base_url='https://www.googleapis.com/oauth2/v3/',
#     #     userinfo_endpoint='https://www.googleapis.com/oauth2/v3/userinfo',
#     #     client_kwargs={'scope': 'openid email profile'},
#     #     server_metadata_url="https://accounts.google.com/.well-known/openid-configuration"
#     # )
#     # return oauth, google

API_URL = os.getenv('API_URL')
API_KEY = os.getenv('API_KEY')
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

# oauth, google = configure_oauth(app)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def call_dify_workflow(question, user_answer, ground_truth):
    """ Calls the Dify API to evaluate the answer with 5 different metrics. """
    headers = {
        "Authorization": f"Bearer {DIFY_WORKFLOW_SECRET}",
        "Content-Type": "application/json"
    }
    payload = {
        "workflow_id": os.getenv("DIFY_WORKFLOW_ID"),
        "inputs": {
            "question": question[:48],  # Trim if needed
            "user_answer": user_answer,
            "ground_truth": ground_truth
        },
        "response_mode": "blocking",
        "user": "abc-123"
    }

    try:
        response = requests.post(DIFY_WORKFLOW_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("data", {}).get("outputs", {})
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Dify API: {e}")
        return None



    
@app.route("/")
def home():
    return render_template("index.html", user=session.get("user"))

# @app.route("/login")
# def login():
#     session.clear()
#     state = os.urandom(16).hex()  # Generate a unique state
#     session['oauth_state'] = state  # Store state in session
#     session.modified = True  # Ensure session updates
#     redirect_uri = url_for('google_callback', _external=True)
#     return google.authorize_redirect(redirect_uri, state=state, prompt="select_account")  # ✅ Force account selection

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

        users_collection.update_one(
            {"email": user_data["email"]},
            {"$set": user_data},
            upsert=True
        )

        session.permanent = True
        session['user'] = user_data

        return redirect(url_for('home'))

    except Exception as e:
        app.logger.error(f"Error in Google callback: {str(e)}")
        session.clear()
        return render_template('error.html', error="Authentication failed. Please try again.")

    
    




@app.route('/login_page', methods=['GET'])
def login_page():
   return redirect(url_for('login'))




@app.route('/check-login-status', methods=['GET'])
def check_login_status():
    user = session.get('user')  # Retrieve the user from the session
    if user:  # Check if the user exists in the session
        return jsonify({'loggedIn': True})  # User is logged in
    return jsonify({'loggedIn': False})  # User is not logged in





def load_ground_truth_data(ground_truth_file):
    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth_list = json.load(f)
        
        # Convert list to dictionary using 'user_question' as key
        ground_truth_dict = {
            item['user_question']: item['ground_truth'] 
            for item in ground_truth_list
        }
        
        return ground_truth_dict
    
    except Exception as e:
        logging.error(f"Error loading ground truth file: {e}")
        return {}

# Replace the existing ground truth loading code with this
ground_truth_data = load_ground_truth_data(GROUND_TRUTH_FILE)
# @app.route('/google/callback')
# def google_callback():
#     try:
#         token = google.authorize_access_token()
#         user_info = google.get('https://www.googleapis.com/oauth2/v3/userinfo').json()
#         if 'email' not in user_info:
#             raise ValueError("Invalid user info received")
#         session["user"] = user_info
#         if not users_collection.find_one({"email": user_info["email"]}):
#             users_collection.insert_one({
#                 "email": user_info["email"],
#                 "name": user_info.get("name", ""),
#                 "picture": user_info.get("picture", ""),
#                 "created_at": datetime.now()
#             })
#         return redirect(url_for("home"))
#     except Exception as e:
#         logging.error(f"Google OAuth Error: {str(e)}")
#         return jsonify({"error": "Authentication failed"}), 500
    
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

# API Calls


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Only CSV/XLSX allowed."}), 400
       
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit(".", 1)[1].lower()
        
        if file_ext == "csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        if "Legal Questions" not in df.columns or "Answer" not in df.columns:
            return jsonify({"error": "CSV/XLSX must contain 'Legal Questions' and 'Answer' columns"}), 400
        
        results = []
        aggregated_scores = {
            "Legal Reasoning Accuracy": 0,
            "Citation Reliability": 0,
            "Hallucination Rate": 0,
            "Task Performance": 0,
            "User Preference Ranking": 0
        }
        total_questions = 0
        
        file_metadata = {
            "filename": filename,
            "file_type": file_ext,
            "upload_date": datetime.utcnow(),
            "data": []
        }
        
        for _, row in df.iterrows():
            question = row["Legal Questions"]
            print(question)

            user_answer = row["Answer"]
            ground_truth = ground_truth_data.get(question, None)
            
            if not ground_truth:
                results.append({"question": question, "error": "Ground truth not found"})
                continue
            
            evaluation_scores = call_dify_workflow(question, user_answer, ground_truth)
            
            if not evaluation_scores:
                results.append({"question": question, "error": "Evaluation failed"})
                continue
            
            results.append({
                "question": question,
                "user_answer": user_answer,
                "ground_truth": ground_truth,
                "evaluation_scores": evaluation_scores
            })
            
            file_metadata["data"].append({
                "question": question,
                "user_answer": user_answer,
                "ground_truth": ground_truth,
                "evaluation_scores": evaluation_scores
            })
            
            
            if "evaluation_results" in evaluation_scores:
                metrics = json.loads(evaluation_scores["evaluation_results"])
                for key in aggregated_scores:
                    if key in metrics:
                        aggregated_scores[key] += metrics[key]
                total_questions += 1
            
        
        if total_questions > 0:
            for key in aggregated_scores:
                aggregated_scores[key] /= total_questions
        
        file_data_collection.insert_one(file_metadata)
        
        return jsonify({
            "results": results,
            "aggregated_scores": aggregated_scores
        })
    
    else:
        data = request.json
        question = data.get("question")
        user_answer = data.get("user_answer")
        
        if not question or not user_answer:
            return jsonify({"error": "Missing required inputs"}), 400
        
        ground_truth = ground_truth_data.get(question, None)
        
        if not ground_truth:
            return jsonify({"error": "Ground truth not found for question"}), 404
        
        evaluation_scores = call_dify_workflow(question, user_answer, ground_truth)
        
        if not evaluation_scores:
            return jsonify({"error": "Evaluation failed"}), 500
        
        return jsonify({
            "question": question,
            "user_answer": user_answer,
            "ground_truth": ground_truth,
            "evaluation_scores": evaluation_scores
        })


if __name__ == "__main__":
    app.run(debug=True)
