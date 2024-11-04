from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
import requests
from flask_session import Session
from functools import wraps
import os
import pandas as pd
from datetime import timedelta

app = Flask(__name__)

# Session configuration
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session_client'  # Separate from server sessions
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
Session(app)

# Server configuration
SERVER_URL = 'http://127.0.0.1:5000'  # Update if server is hosted elsewhere
REQUEST_TIMEOUT = 10  # seconds

def handle_server_error(error):
    """Common error handling for server communication"""
    app.logger.error(f"Server communication error: {str(error)}")
    if isinstance(error, requests.exceptions.Timeout):
        flash("Server request timed out. Please try again.", "error")
    elif isinstance(error, requests.exceptions.ConnectionError):
        flash("Cannot connect to server. Please check if the server is running.", "error")
    else:
        flash("An error occurred. Please try again later.", "error")
    return None

def check_session_valid():
    """Check if current session is valid"""
    return 'user_id' in session and session.get('server_session_valid', False)

def session_required(f):
    """Decorator to ensure valid session exists"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not check_session_valid():
            # Initialize new session
            try:
                response = requests.post(
                    f'{SERVER_URL}/send_filter_values',
                    timeout=REQUEST_TIMEOUT
                )
                if response.status_code == 200:
                    session['server_session_valid'] = True
                    session.permanent = True
                else:
                    flash("Failed to initialize session. Please try again.", "error")
                    return redirect(url_for('index'))
            except requests.exceptions.RequestException as e:
                handle_server_error(e)
                return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Initialize session and render main page"""
    try:
        response = requests.post(
            f'{SERVER_URL}/send_filter_values',
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            specs, dates, PUR_NAMEs, POLICY_NAMEs = response.json()
            session["filters_all"] = response.json()
            session['server_session_valid'] = True
            session.permanent = True
            return render_template(
                'index3.html',
                specs=specs,
                dates=dates,
                PUR_NAMEs=PUR_NAMEs,
                POLICY_NAMEs=POLICY_NAMEs
            )
        else:
            flash("Failed to load initial data. Please refresh the page.", "error")
            return render_template('error.html')
    except requests.exceptions.RequestException as e:
        handle_server_error(e)
        return render_template('error.html')

@app.route('/get_filter_values', methods=['POST'])
@session_required
def get_filter_values():
    """Get filter values based on user selection"""
    try:
        data = request.get_json()
        session["selected_filters"] = data
        print(data, "&*&*^^%&")
        response = requests.post(
            f'{SERVER_URL}/get_filter_values',
            json=data,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Failed to get filter values"}), response.status_code
            
    except requests.exceptions.RequestException as e:
        handle_server_error(e)
        return jsonify({"error": "Server communication error"}), 500

@app.route('/apply_filters', methods=['POST'])
@session_required
def apply_filters():
    """Apply selected filters and get filtered data"""
    try:
        data = request.get_json()
        print(data)
        response = requests.post(
            f'{SERVER_URL}/apply_filters',
            json=data,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result_data = response.json()
            session["filtered_df"] = pd.DataFrame(result_data).to_json()
            df_html = pd.DataFrame(result_data).to_html(
                classes='table table-striped',
                index=False
            )
            return jsonify({'df_html': df_html})
        else:
            return jsonify({"error": "Failed to apply filters"}), response.status_code
            
    except requests.exceptions.RequestException as e:
        handle_server_error(e)
        return jsonify({"error": "Server communication error"}), 500

@app.route('/generate_image', methods=['POST'])
@session_required
def generate_image():
    """Generate visualization for selected row"""
    try:
        row_index = request.json.get('index')
        if not session.get("filtered_df"):
            return jsonify({"error": "No filtered data available"}), 400
            
        data = [session["filtered_df"], row_index]
        response = requests.post(
            f'{SERVER_URL}/generate_image',
            json=data,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Failed to generate image"}), response.status_code
            
    except requests.exceptions.RequestException as e:
        handle_server_error(e)
        return jsonify({"error": "Server communication error"}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    flash("The requested page was not found.", "error")
    return render_template('error.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    flash("An internal server error occurred.", "error")
    return render_template('error.html'), 500

if __name__ == '__main__':
    # Create session directory if it doesn't exist
    os.makedirs('flask_session_client', exist_ok=True)
    app.run(port=5005, debug=True)
