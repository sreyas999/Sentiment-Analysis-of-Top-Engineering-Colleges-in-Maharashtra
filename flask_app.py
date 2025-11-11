from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pickle
from flask_mail import Mail, Message
from flask_wtf.csrf import CSRFProtect
from flask_wtf.csrf import generate_csrf
from forms import LoginForm




from sklearn.feature_extraction.text import CountVectorizer
import secrets

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'An@2621#'
csrf = CSRFProtect(app)

# Function to get a database connection
def get_db_connection():
    conn = sqlite3.connect('flask_users.db') 
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL,
            reset_token TEXT
        )
    ''')

    # Check if reset_token column exists, and add it if necessary
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'reset_token' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")

    # Create feedback table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            college TEXT,
            keyword TEXT,
            name TEXT,
            email TEXT,
            rating INTEGER,
            comments TEXT
        )
    ''')

    conn.commit()
    conn.close()

init_db()

df = pd.read_csv('sentiment_result.csv')

# Load models and vectorizer
with open('svm_model.pkl', 'rb') as svm_model_file:
    svm_model = pickle.load(svm_model_file)

with open('lr_model.pkl', 'rb') as lr_model_file:
    lr_model = pickle.load(lr_model_file)

with open('vectorizer.pkl', 'rb') as vector_file:
    vectorizer = pickle.load(vector_file)

plt.switch_backend('agg')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    college = data.get('college', 'N/A')
    keyword = data.get('keyword', 'N/A')
    name = data['name']
    email = data['email']
    rating = data['rating']
    comments = data['comments']

    save_feedback_to_database(college, keyword, name, email, rating, comments)

    return jsonify({'title': 'Feedback Submitted', 'text': 'Thank you for your feedback!', 'icon': 'success'})

def save_feedback_to_database(college, keyword, name, email, rating, comments):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO feedback (college, keyword, name, email, rating, comments)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (college, keyword, name, email, rating, comments))

    conn.commit()
    conn.close()

def plot_pie_chart(avg_positive, avg_negative, avg_neutral):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [avg_positive, avg_negative, avg_neutral]
    colors = ['#77DD77', '#FF6961', '#AEC6CF']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=140, explode=explode, wedgeprops=dict(width=0.4)
    )

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title('Average Sentiment Analysis')

    img_stream = BytesIO()
    fig.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_data = base64.b64encode(img_stream.read()).decode('utf-8')

    plt.close(fig)
    return img_data

def perform_sentiment_analysis(college_df, keyword):
    keyword_df = college_df[college_df['Review'].str.contains(keyword, case=False, regex=True)]
    reviews_vectorized = vectorizer.transform(keyword_df['Review'])

    svm_sentiments = svm_model.predict(reviews_vectorized)
    lr_sentiments = lr_model.predict(reviews_vectorized)

    combined_sentiments = [f"{svm}_{lr}" for svm, lr in zip(svm_sentiments, lr_sentiments)]

    positive_count = sum(1 for score in combined_sentiments if score == 'Positive_Positive')
    negative_count = sum(1 for score in combined_sentiments if score == 'Negative_Negative')
    neutral_count = sum(1 for score in combined_sentiments if score == 'Neutral_Neutral')

    total_reviews = len(combined_sentiments)
    avg_positive = positive_count / total_reviews
    avg_negative = negative_count / total_reviews
    avg_neutral = neutral_count / total_reviews

    return positive_count, negative_count, neutral_count, avg_positive, avg_negative, avg_neutral

@app.route('/')
def index():
    colleges = sorted(df['college_name'].unique())
    just_logged_in = session.pop('just_logged_in', False)  
    just_logged_out = session.pop('just_logged_out', False)
    return render_template('index.html', colleges=colleges, just_logged_in=just_logged_in,just_logged_out=just_logged_out)

@app.route('/overview')
def overview():
    return render_template('overview.html')

@app.route('/feedback')
def feedback():
    college = "Your College"
    keyword = "Your Keyword"
    return render_template('feedback.html', college=college, keyword=keyword)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'username' in session:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = request.form['email']

        if password != confirm_password:
            flash('Passwords do not match. Please enter matching passwords.', 'error')
            return render_template('signup.html', csrf_token=generate_csrf())

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username or email already exists. Choose different ones.', 'error')
            conn.close()
          
            return render_template('signup.html', csrf_token=generate_csrf())
        else:
            cursor.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                           (username, hashed_password, email))
            conn.commit()
            conn.close()
           
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
            
    return render_template('signup.html', csrf_token=generate_csrf())

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()  # Assuming you have a LoginForm defined using Flask-WTF
    if 'username' in session:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['just_logged_in'] = True 
            flash('Login successful!', 'success')
            return redirect(url_for('index'))  
        else:
            flash('Invalid username or password. Please try again.', 'error')
               
    return render_template('login.html', form=form, csrf_token=generate_csrf())

@app.route('/logout')
def logout():
    session.pop('username', None)
    session['just_logged_out'] = True 
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))


# Initialize Flask-Mail
mail = Mail(app)


# Update the forgot_password route
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user:
            # Generate reset token
            token = secrets.token_urlsafe(16)
            cursor.execute("UPDATE users SET reset_token = ? WHERE email = ?", (token, email))
            conn.commit()
            conn.close()

            # Send reset password email
            reset_link = url_for('reset_password', token=token, _external=True)
            send_reset_password_email(email, reset_link)

            flash(f'Password reset link has been sent to your email: {email}', 'info')
        else:
            flash('No account found with that email. Please try again.', 'error')
            conn.close()

    return render_template('forgot_password.html')


def send_reset_password_email(email, reset_link):
    msg = Message('Reset Your Password', sender='shreyaspanchal276@gmail.com', recipients=[email])
    msg.body = f'Click the following link to reset your password: {reset_link}'
    mail.send(msg)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE reset_token = ?", (token,))
    user = cursor.fetchone()

    if not user:
        flash('Invalid or expired token. Please request a new password reset.', 'error')
        conn.close()
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match. Please enter matching passwords.', 'error')
            return redirect(url_for('reset_password', token=token))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        cursor.execute("UPDATE users SET password = ?, reset_token = NULL WHERE id = ?",
                       (hashed_password, user['id']))
        conn.commit()
        conn.close()

        flash('Password reset successful. You can now log in with your new password.', 'success')
        return redirect(url_for('login'))

    conn.close()
    return render_template('reset_password.html', token=token)

@app.route('/result', methods=['GET', 'POST'])
def result():
    if 'username' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))

    try:
        if request.method == 'POST':
            college_name = request.form['college']
            keyword = request.form['keyword']

            college_df = df[df['college_name'] == college_name]
            positive_count, negative_count, neutral_count, avg_positive, avg_negative, avg_neutral = perform_sentiment_analysis(
                college_df, keyword)

            img_data = plot_pie_chart(avg_positive, avg_negative, avg_neutral)

            review_count = len(college_df)

            return render_template('result.html', college=college_name, keyword=keyword, img_data=img_data,
                                   review_count=review_count, positive_count=positive_count,
                                   negative_count=negative_count, neutral_count=neutral_count,
                                   avg_positive=avg_positive, avg_negative=avg_negative, avg_neutral=avg_neutral)

        colleges = sorted(df['college_name'].unique())
        return render_template('result.html', colleges=colleges)

    except Exception as e:
        error_message = "Oops! An error occurred while processing your request. Please try again later."
        return render_template('error.html', error_message=error_message)

@app.errorhandler(404)
def page_not_found(error):
    error_message = "Oops! The page you are looking for does not exist."
    return render_template('error.html', error_message=error_message), 404

@app.errorhandler(500)
def internal_server_error(error):
    error_message = "Oops! Something went wrong on the server. Please try again later."
    return render_template('error.html', error_message=error_message), 500

engineering_colleges = [
    {"name": "Indian Institute of Technology Bombay", "website": "https://www.iitb.ac.in/"},
    {"name": "Visvesvaraya National Institute of Technology, Nagpur", "website": "https://vnit.ac.in/"},
    {"name": "College Of Engineering Pune", "website": "https://www.coep.org.in/"},
    {"name": "Army Institute of Technology, Pune", "website": "https://www.aitpune.com/"},
    {"name": "Veermata Jijabai Technological Institute, Mumbai", "website": "https://vjti.ac.in/"},
]

@app.route('/index2')
def index2():
    return render_template('index2.html', colleges=engineering_colleges)

@app.route('/college/<int:index>')
def college(index):
    if 0 <= index < len(engineering_colleges):
        college = engineering_colleges[index]
        return redirect(college['website'])
    else:
        return "Invalid college index"

@app.route('/sentiment_reviews')
def sentiment_reviews():
    try:
        df_sentiment = pd.read_csv('sentiment_result.csv')
        reviews_data = df_sentiment[['college_name', 'Keyword', 'Review', 'sentiment']].values.tolist()
        return render_template('reviews.html', reviews_data=reviews_data)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(threaded=True, debug=True)
