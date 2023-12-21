from flask import Flask, jsonify, request, render_template, url_for, flash, redirect, session
# from flaskext.mysql import MySQL
import pymysql


from functools import wraps
from image_recomend import Images

# Recomendation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import Recommender
import csv
import re
import gensim.downloader as api
from gensim.models import KeyedVectors
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sent2vec.vectorizer import Vectorizer


app = Flask(__name__, static_url_path='/static')
# word2vec_model = api.load("word2vec-google-news-300")
word2vec_model = api.load('glove-twitter-25')
sent2vec_vectorizer = Vectorizer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the GloVe model
# word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
# app.config['SECRET_KEY'] = 'ec830e5ae057c5b08f5a435a7b13e891'
app.secret_key = 'ec830e5ae057c5b08f5a435a7b13e891'

# Config MySQL
# app.config['MYSQL_DATABASE_HOST'] = "localhost"
# app.config['MYSQL_DATABASE_PORT'] = 3306
# app.config['MYSQL_DATABASE_USER'] = 'root'
# app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
# app.config['MYSQL_DATABASE_DB'] = 'myflaskapp'
# app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# init MYSQL
# mysql = MySQL()
# mysql.init_app(app)
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='mohammed',
    db='myflaskapp',
    cursorclass=pymysql.cursors.DictCursor  # This returns results as dictionaries
)


@app.route('/')
def main():

    try:

        if session['user_id']:

            user_ids = session['user_id']

            data = pd.read_csv('place_4.csv')
            data['all_place'] = data['place'].map(str) + " - " + data['type']

            song_grouped = data.groupby(['all_place']).agg(
                {'rating': 'count'}).reset_index()
            grouped_sum = song_grouped['rating'].sum()
            song_grouped['percentage'] = song_grouped['rating'].div(
                grouped_sum)*100
            song_grouped.sort_values(['rating', 'all_place'], ascending=[0, 1])

            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=0)
            pm = Recommender.popularity_recommender_py()
            pm.create(train_data, 'user_id', 'all_place')

            # user_id = 10
            content_based = pm.recommend(user_ids)
            place = []
            for index, row in content_based.iterrows():
                print(row['all_place'])
                place.append(row['all_place'])

            is_model = Recommender.item_similarity_recommender_py()
            is_model.create(train_data, 'user_id', 'all_place')

            # user_id = 10
            print(user_ids)
            user_ids = int(user_ids)
            user_items = is_model.get_user_items(user_ids)
            place_personal = []
            for user_item in user_items:
                print(user_item)
                place_personal.append(user_item)
                
            return render_template('index.html', content_based=place, personal_recomendation=place_personal)
    except:
        return render_template('login.html')  






@app.route('/reg', methods=['GET', 'POST'])
def reg():
    return render_template('register.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    # conn=mysql.connect()
    cursor = conn.cursor()
    if (cursor.execute("INSERT INTO users(user_name,user_id, password) VALUES(%s,%s,%s)", ('2', '1', '1'))):
        print("data gone")
        conn.commit()
        cursor.close()

    return 'data inserted'


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return render_template('index.html')
    if request.method == 'POST':
        user_name = request.form['user_name']
        password = request.form['password']

        # conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE user_name = %s ", [user_name])

        # cursor.execute("SELECT * FROM users")

        user_data = cursor.fetchone()
        print(user_data)

        cursor.close()

        if user_data and password == user_data['password']:
            session['logged_in'] = True
            session['username'] = user_name
            session['user_id'] = user_data['user_id']
            flash('You are now logged in', 'success')
            return redirect(url_for('main'))
        else:
            flash('Login failed. Please check your credentials.', 'danger')

    return render_template('login.html')

# Check if user logged in


def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap


# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))


@app.route('/search_city', methods=['GET', 'POST'])
def search_city():

    try:
        if session['user_id']:
            data = pd.read_csv('place_4.csv')
            city = data['city'].unique()

            return render_template('search_city.html', city=city)
    except:
        return render_template('login.html')



@app.route('/view_places/<int:city_id>', methods=['GET', 'POST'])
def view_places(city_id):
    try:
        # Connect to the database
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='mohammed',
            db='myflaskapp',
            cursorclass=pymysql.cursors.DictCursor
        )

        with conn.cursor() as cursor:
            # Assuming you have a 'city_name' column in your 'cities' table
            query = f"SELECT * FROM places WHERE city_id = '{city_id}'"
            cursor.execute(query)
            city_places = cursor.fetchall()
             # Assuming you have a 'city_name' column in your 'cities' table
            city_query = f"SELECT city_name FROM cities WHERE city_id = {city_id}"
            cursor.execute(city_query)
            city_name = cursor.fetchone()['city_name']

            # Get the list of places that the user has visited
            # Fetch visited places for the current user
            user_id = session.get('user_id')
            if user_id:
                visited_query = "SELECT place_id FROM visited_places WHERE user_id = %s"
                cursor.execute(visited_query, (user_id,))
                visited_places = {row['place_id'] for row in cursor.fetchall()}
            else:
                visited_places = set()

        dt = []
        for place in city_places:
            new_data = {
                'place': place['place_name'],
                'place_id': place['place_id'],
                'rating': place['rating'],
                'type': place['type'],
                'image_url': place['image_url'],
                'is_visited': place['place_id'] in visited_places
            }
            dt.append(new_data)

        return render_template('view_city_place.html', data=dt, city_name=city_name)
    except Exception as e:
        print('An error occurred:', str(e))
        flash('An error occurred while fetching the city places.', 'error')
        return render_template('login.html')

# @app.route('/rating/<city_name>/<place>/<types>', methods=['GET', 'POST'])
# def rating(city_name, place, types):
#     try:
#         if session['user_id']:
#             if request.method == 'POST':
#                 user_id = session['user_id']
#                 rating = request.form['rating']
#                 print(city_name, place, types, rating)
#                 with open('place_4.csv', 'a') as newFile:
#                     newFileWriter = csv.writer(newFile)
#                     newFileWriter.writerow(
#                         [1, city_name, place, rating, types, user_id])

#                 # return render_template('rating.html')
#                 flash('Rating Done!', 'success')
#                 return redirect(url_for('view_city', city_name=city_name))

#     except:
#         print('except in rating')
#         return render_template('login.html')
@app.route('/rating/<int:place_id>', methods=['GET', 'POST'])
def rating(place_id):
        if 'user_id' in session:
            if request.method == 'POST':
                user_id = session['user_id']
                rating = request.form['rating']
                comment = request.form['comment']

                # Connect to the database
                conn = pymysql.connect(
                    host='localhost',
                    user='root',
                    password='mohammed',
                    db='myflaskapp',
                    cursorclass=pymysql.cursors.DictCursor
                )

                with conn.cursor() as cursor:
                    # Assuming you have a 'reviews' table
                    query = "INSERT INTO reviews (user_id, place_id, rating, comment_text) VALUES (%s, %s, %s, %s)"
                    cursor.execute(query, (user_id, place_id, rating, comment))
                    conn.commit()
                    place_query = f"SELECT city_id FROM places WHERE place_id = {place_id}"
                    cursor.execute(place_query)
                    city_id = cursor.fetchone()['city_id']
                flash('Rating and comment added successfully!', 'success')
                return redirect(url_for('view_places', city_id=city_id))

        else:
            flash('You need to be logged in to submit a rating.', 'error')
            return render_template('login.html')

@app.route('/mark_visited/<int:place_id>', methods=['POST'])
def mark_visited(place_id):
        if 'user_id' in session:
            user_id = session['user_id']

            # Connect to the database
            conn = pymysql.connect(
                host='localhost',
                user='root',
                password='mohammed',
                db='myflaskapp',
                cursorclass=pymysql.cursors.DictCursor
            )

            with conn.cursor() as cursor:
                # Check if the place is already marked as visited
                query = "SELECT * FROM visited_places WHERE user_id = %s AND place_id = %s"
                cursor.execute(query, (user_id, place_id))
                result = cursor.fetchone()
                place_query = f"SELECT city_id FROM places WHERE place_id = {place_id}"
                cursor.execute(place_query)
                city_id = cursor.fetchone()['city_id']

                if result:
                    flash('Place is already marked as visited!', 'info')
                else:
                    # Insert a new record into the visited_places table
                    insert_query = "INSERT INTO visited_places (user_id, place_id) VALUES (%s, %s)"
                    cursor.execute(insert_query, (user_id, place_id))
                    conn.commit()
                    flash('Place marked as visited successfully!', 'success')

            return redirect(url_for('view_places', city_id=city_id))

        else:
            flash('You need to be logged in to mark a place as visited.', 'error')
            return render_template('login.html')

 

@app.route('/search', methods=['POST', 'GET'])
def search():
    try:
        if session['user_id']:
            if request.method == 'POST':
                inputs = request.form['data']
                print(inputs)

                return render_template('search.html')
    except:
        return render_template('login.html')


@app.route('/index')
def index():
    places = pd.read_csv('place_4.csv')
    places = places[['city', 'rating']]
    all_places = places.groupby(['city']).mean()
    dt = []
    for index, rows in all_places.iterrows():
        new_data = {}
        new_data['city'] = index
        new_data['rating'] = rows['rating']
        dt.append(new_data)
    return render_template('indexpage.html', places=dt)


@app.route('/city/<city_name>')
def display_city_places(city_name):
    data = pd.read_csv('place_4.csv')
    places = data[data['city'] == city_name.lower().capitalize()]
    places = places[['place', 'type', 'rating']]
    all_places = places.groupby(['place', 'type']).mean()
    dt = []
    for index, rows in all_places.iterrows():
        new_data = {}
        new_data['place'] = index[0]
        new_data['rating'] = rows['rating']
        new_data['type'] = index[1]
        dt.append(new_data)

    return render_template('city_places.html', city_name=city_name, places=dt)
# Define the route for Marrakech




# Define the route for Rabat
@app.route('/rabat')
def rabat():
    data = pd.read_csv('place_4.csv')
    places = data[data['city'] == 'Rabat']
    places = places[['place', 'type', 'rating']]
    all_places = places.groupby(['place', 'type']).mean()
    dt = []
    for index, rows in all_places.iterrows():
        new_data = {}
        new_data['place'] = index[0]
        new_data['rating'] = rows['rating']
        new_data['type'] = index[1]
        dt.append(new_data)
    return render_template('rabat.html', places=dt)


# Create a Sent2Vec Vectorizer
sent2vec_vectorizer = Vectorizer()


@app.route('/commentSimilarity', methods=['GET', 'POST'])
def comment_similarity():
    if request.method == 'POST':
        comment1 = request.form['comment1']
        comment2 = request.form['comment2']

        # Preprocess comments
        preprocessed_comment1 = preprocess_comment(comment1)
        preprocessed_comment2 = preprocess_comment(comment2)

        # Calculate similarity using Sent2Vec
        similarity_sent2vec = sent2vec_similarity(
            preprocessed_comment1, preprocessed_comment2)

        # Calculate max distance between words
        max_dist = max_distance_between_words(
            preprocessed_comment1, preprocessed_comment2)

        # Calculate min distance between words
        min_dist = min_distance_between_words(
            preprocessed_comment1, preprocessed_comment2)

        return render_template('commentSimilarity.html',
                               similarity_sent2vec=similarity_sent2vec,
                               max_distance=max_dist,
                               min_distance=min_dist)

    return render_template('commentSimilarity.html')


def preprocess_comment(comment):
    # Convert to lowercase
    comment = comment.lower()

    # Remove punctuation
    comment = comment.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(comment)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a string
    preprocessed_comment = ' '.join(tokens)

    return preprocessed_comment


def sent2vec_similarity(comment1, comment2):
    # Fit the model and transform comments to embeddings
    sent2vec_vectorizer.run([comment1, comment2])
    # Get the sentence embeddings
    embedding1 = sent2vec_vectorizer.vectors[0].reshape(1, -1)
    embedding2 = sent2vec_vectorizer.vectors[1].reshape(1, -1)

    # Calculate cosine similarity between embeddings
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity


def max_distance_between_words(comment1, comment2):
    # Split the comments into words
    words1 = comment1.split()
    words2 = comment2.split()

    # Load the Word2Vec model

    # Calculate the max distance between words
    max_distance = 0
    for word1 in words1:
        for word2 in words2:
            distance = word2vec_model.similarity(word1, word2)
            max_distance = max(max_distance, distance)

    return max_distance


def min_distance_between_words(comment1, comment2):
   # Split the comments into words
    words1 = comment1.split()
    words2 = comment2.split()

    # Load the Word2Vec model

    # Calculate the max distance between words
    min_distance = float('inf')
    for word1 in words1:
        for word2 in words2:
            distance = word2vec_model.similarity(word1, word2)
            min_distance = min(min_distance, distance)

    return min_distance

user_comments_file = 'commentsRatings.csv'
user_comments = pd.read_csv(user_comments_file)
def get_user_comment(user_id):
    user_comment = user_comments[user_comments['user_id'] == user_id]['comment'].values
    if len(user_comment) > 0:
        return user_comment[0]
    else:
        return None
    
def calculate_recommendations(user_id):
    user_comment = get_user_comment(user_id)  # get the user's comment
    similarities = []

    for index, row in places.iterrows():
        place_description = row['description']
        similarity_score = sent2vec_similarity(user_comment, place_description)
        # similarity_score = max_distance_between_words(user_comment, place_description)
        similarities.append((index, similarity_score))

    # Sort in descending order of similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Select the top 5 places
    top_5_recommendations = similarities[:5]
    
    return top_5_recommendations

places = pd.read_csv('place_4_description.csv')

@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    recommendations = calculate_recommendations(user_id)

    #'places'  DataFrame all place details
    recommended_places = places.loc[[index for index, _ in recommendations]]

    # Convert to JSON and return
    # result = recommended_places.to_json(orient='records')
    # return jsonify(result)
    return render_template('recommendations.html', places=recommended_places.to_dict(orient='records'))

@app.route('/offers')
def offers():
    return render_template('offers.html')


@app.route('/index_msg')
def index_msg():
    return render_template('index.html')


@app.route('/image')
def image():
    return render_template('imagesearch.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        recomend = Images()
        data = recomend.Upload()

        location = data
        location = re.findall('([^\/]+$)', location)
        new_loc = "queries/"+location[0]
        print(new_loc)
        print("Working")

        res_loc = recomend.predict(new_loc)

        print(res_loc)
        res_name = res_loc[:-7]
        if res_name == 'Taj_Mahal':
            return render_template('main copy.html')
        elif res_name == 'qutub_minar':
            return render_template('main copy 2.html')
        elif res_name == 'Mysore_Palace':
            return render_template('mysore.html')

        elif res_name == 'Jantar_mantar':
            return render_template('jantar mantar.html')
        elif res_name == 'hawa_mahal':
            return render_template('hawa mahal.html')

        elif res_name == 'red_fort':
            return render_template('red fort.html')

        elif res_name == 'gateway':
            return render_template('gateway of india.html')

        elif res_name == 'lotus_temple':
            return render_template('lotus temple.html')

        elif res_name == 'Virupaksha':
            return render_template('virupaksha temple.html')
        elif res_name == 'gol_gumbaz':
            return render_template('gol gumbaz.html')
        elif res_name == 'golden_temple':
            return render_template('golden temple.html')

        elif res_name == 'Jama_Masjid':
            return render_template('jama masjid.html')
        else:
            return render_template('image.html')
    return render_template('indexpage.html')


if __name__ == "__main__":
    app.run(debug=True)
