import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, session, redirect, url_for
from flask_mysqldb import MySQL

def n_neighbours(df, n):
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:n].index,
                                      index=['top{}'.format(i) for i in range(1, n + 1)]), axis=1)
    return df


def similar_questions(user1, user2):
    common_questions = history_s[history_s['userID'] == user1].merge(history_s[history_s['userID'] == user2],
                                                                     on="questionID", how="inner")
    return common_questions.merge(questions, on='questionID')


def generate_recommendations(user, n, history, questions):
    # Normalise visits
    history['visits'] = (history['visits'] - history['visits'].min()) / (
                history['visits'].max() - history['visits'].min())
    userhistory = history.groupby(by='userID', as_index=False)['visits'].mean()

    history_s = pd.merge(history, userhistory, on='userID')
    history_s['norm_rating'] = history_s['visits_x'] - history_s['visits_y']
    temp = pd.pivot_table(history_s, values='visits_x', index='userID', columns='questionID')
    final = pd.pivot_table(history_s, values='norm_rating', index='userID', columns='questionID')

    # replacing by question
    final_question = final.fillna(final.mean(axis=0))

    # replacing by user average
    final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)

    # user similarity on final_user
    b = cosine_similarity(final_user)
    np.fill_diagonal(b, 0)
    similarity_with_user = pd.DataFrame(b, index=final_user.index)
    similarity_with_user.columns = final_user.index

    # user similarity on final_shop
    cosine = cosine_similarity(final_question)
    np.fill_diagonal(cosine, 0)
    similarity_with_question = pd.DataFrame(cosine, index=final_question.index)
    similarity_with_question.columns = final_user.index

    # top n neighbours for each user and question
    sim_user_u = n_neighbours(similarity_with_user, n)
    sim_user_m = n_neighbours(similarity_with_question, n)

    history_s.userID = history_s.userID.astype(str)
    history_s.questionID = history_s.questionID.astype(str)
    question_user = history_s.groupby('userID')['questionID'].apply(lambda x: ','.join(x))

    question_user.index = question_user.index.astype(int)
    question_by_user = temp.columns[temp[temp.index == user].notna().any()].tolist()
    a = sim_user_m[sim_user_m.index == user].values
    b = a.squeeze().tolist()
    d = question_user[question_user.index.isin(b)]
    l = ','.join(d.values)
    question_similar_users = l.split(',')
    questionslist = list(set(question_similar_users) - set(list(map(str, question_by_user))))
    questionslist = list(map(int, questionslist))
    score = []
    for item in questionslist:
        item = int(item)
        c = final_question.loc[:, item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = userhistory.loc[userhistory['userID'] == user, 'visits'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = similarity_with_question.loc[user, index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score', 'correlation']
        fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume / deno)
        score.append(final_score)
    data = pd.DataFrame({'questionID': questionslist, 'score': score})
    recommendations = data.sort_values(by='score', ascending=False)
    questionname = recommendations.merge(questions, how='inner', on='questionID')

    return questionname.Name.values.tolist()


#DELETE

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