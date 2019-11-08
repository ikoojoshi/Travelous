from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from rs import generate_recommendations

app = Flask(__name__)
app.secret_key = """ Internet and Web Programming Project
"""

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'rabbit99'
app.config['MYSQL_DB'] = 'travelous'

mysql = MySQL(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('username', 'none')
   session.pop('loggedin', None)
   session.pop('id', None)
   # Redirect to login page
   return redirect(url_for('login'))

@app.route('/destinations', methods=['GET', 'POST'])
def destinations():

    data = request.form.get('dest', 'Destination')
    try:
        history = pd.read_csv("data/history.csv")
        questions = pd.read_csv("data/questions.csv")
        queries = generate_recommendations(session['userno'], 5, history, questions)
        print(queries[:5])
        session['places'] = queries[:5]
    except:
        queries = ["Dresden", "Zermatt", "Mahe", "Munich"];
    session['city'] = [None]*5
    session['country'] = [None]*5
    session['description'] = [None]*5
    session['views'] = [None]*5
    session['website'] = [None]*5
    session['images'] = [None]*5
    for i in range(4):
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        city = queries[i]
        cursor.execute('SELECT * FROM destinations WHERE city = "'+city+'"')
        # Fetch one record and return result
        account = cursor.fetchone()
        if account:
            # Create session data, we can access this data in other routes
            session['city'][i] = queries[i]
            session['country'][i] = account['country']
            session['description'][i] = account['description']
            session['views'][i] = account['views']
            session['website'][i] = account['website']
            session['images'][i] = 'images/destinations/' + queries[i] +'.jpg'

    return render_template("destinations.html", dest = data, preview=queries[:5], length=4)


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if 'username' in request.form and 'password' in request.form:
            username = request.form['username']
            password = request.form['password']
            session['username'] = username
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM users WHERE userid = %s AND password = %s', (username, password))
            # Fetch one record and return result
            account = cursor.fetchone()
            if account:
                # Create session data, we can access this data in other routes
                session['loggedin'] = True
                session['username'] = account['userid']
                session['fname'] = account['fname']
                session['lname'] = account['lname']
                session['userno'] = account['userno']
                # Redirect to home page
                return redirect(url_for('index'))
            else:
                # Account doesnt exist or username/password incorrect
                return render_template("login.html", msg="Incorrect username or password")

        else:
            redirect(url_for('login'))
    return render_template("login.html", msg="")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'username' in request.form and 'pwd' in request.form and 'email' in request.form:
            # Create variables for easy access
            username = request.form['username']
            fname = request.form['fname']
            lname = request.form['lname']
            password = request.form['pwd']
            cpassword = request.form['cpwd']
            email = request.form['email']
            age = request.form['age']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM users WHERE userid = "'+ username+'"')
            account = cursor.fetchone()
            # If account exists show error and validation checks
            if account:
                return render_template("register.html", msg="Account already exists")
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                return render_template("register.html", msg='Invalid email address!')
            elif not re.match(r'[A-Za-z0-9]+', username):
                return render_template("register.html", msg='Username must contain only characters and numbers!')
            elif not password==cpassword:
                return render_template("register.html", msg='Password and Confirm Password do not match')
            elif not username or not password or not email:
                return render_template("register.html", msg='Please fill out the form!')
            else:
                session['username'] = request.form['username']
                session['fname'] = request.form['fname']
                session['lname'] = request.form['lname']
                session['userno'] = 12347
                # Account doesnt exists and the form data is valid, now insert new account into accounts table
                cursor.execute("INSERT INTO users VALUES (%s, %s, %s, %s, %s, 1, 20, 12347)", (username, fname, lname, email, password))
                mysql.connection.commit()
            return redirect(url_for('index'))
        else:
            return render_template("register.html", msg="Error Registering..")

    return render_template("register.html", msg="")


@app.route('/search')
def search():
    input = request.args.get('input')
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    print(input)
    cursor.execute('SELECT * FROM destinations WHERE city="' + input + '"')
    # Fetch one record and return result
    account = cursor.fetchone()
    if account:
        # Create session data, we can access this data in other routes
        session['city'][0] = account['city']
        session['country'][0] = account['country']
        session['description'][0] = account['description']
        session['views'][0] = account['views']
        session['website'][0] = account['website']
        session['images'][0] = 'images/destinations/' + account['city'] + '.jpg'

    return render_template("search.html", length=1)

@app.route('/housing')
def housing():
    return render_template("housing.html")

@app.route('/transport')
def transport():
    return render_template("transport.html")

@app.route('/visa')
def visa():
    return render_template("visa.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == "__main__":
    app.run(debug=True)