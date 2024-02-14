from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD, NMF
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sqlalchemy import create_engine
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load e-commerce dataset (replace with your own dataset)
# Assume you have a SQLite database with columns: 'user_id', 'product_id', 'rating', 'category', 'brand', 'timestamp'
# Replace with MySQL or MongoDB configurations as needed
database_type = 'mongodb'  # Change to 'mysql' or 'mongodb' as needed

if database_type == 'sqlite':
    database_url = 'sqlite:///ecommerce_data.db'
    engine = create_engine(database_url)
    ecommerce_data = pd.read_sql_query('SELECT * FROM purchase_history;', engine)
    product_data = pd.read_sql_query('SELECT * FROM product_info;', engine)
elif database_type == 'mysql':
    # Connect to MySQL database and fetch data
    # Replace 'your_username', 'your_password', 'your_database', and 'your_host' with your MySQL configurations
    import mysql.connector
    connection = mysql.connector.connect(
        user='your_username',
        password='your_password',
        database='your_database',
        host='your_host'
    )
    ecommerce_data = pd.read_sql_query('SELECT * FROM purchase_history;', connection)
    product_data = pd.read_sql_query('SELECT * FROM product_info;', connection)
    connection.close()
elif database_type == 'mongodb':
    # Connect to MongoDB and fetch data
    # Replace 'your_username', 'your_password', 'your_database', and 'your_host' with your MongoDB configurations
    from pymongo import MongoClient
    client = MongoClient('mongodb+srv://your_username:your_password@your_host/your_database')
    db = client.your_database
    ecommerce_data = pd.DataFrame(list(db.purchase_history.find()))
    product_data = pd.DataFrame(list(db.product_info.find()))
    client.close()

# Surprise expects a specific format for the dataset
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(ecommerce_data[['user_id', 'product_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Build and train collaborative filtering model (KNN)
sim_options = {
    'name': 'cosine',
    'user_based': False
}
knn_model = KNNBasic(sim_options=sim_options)
knn_model.fit(trainset)

# Build and train matrix factorization models (SVD, NMF)
svd_model = SVD()
svd_model.fit(trainset)

nmf_model = NMF()
nmf_model.fit(trainset)

# Content-Based Filtering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
product_data['description'] = product_data['description'].fillna('')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_data['description'])

# Function to get content-based recommendations
def get_content_based_recommendations(product_id, num_recommendations=5):
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix[product_id])
    product_scores = list(enumerate(cosine_similarities))
    product_scores = sorted(product_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    recommended_product_ids = [x[0] for x in product_scores]
    return product_data['product_id'].iloc[recommended_product_ids]

# Hybrid Recommendation combining collaborative filtering and content-based filtering
def hybrid_recommendation(user_id, product_id, num_recommendations=5):
    # Get collaborative filtering recommendations
    knn_recommendations = knn_model.get_neighbors(product_id, k=num_recommendations)

    # Get SVD predictions
    svd_prediction = svd_model.predict(user_id, product_id).est

    # Get NMF predictions
    nmf_prediction = nmf_model.predict(user_id, product_id).est

    # Get content-based recommendations
    content_based_recommendations = get_content_based_recommendations(product_id, num_recommendations)

    # Combine recommendations from different models
    hybrid_recommendations = list(set(knn_recommendations) | set(content_based_recommendations))
    hybrid_recommendations = [product for product in hybrid_recommendations if product != product_id]  # Exclude the input product
    hybrid_recommendations = sorted(hybrid_recommendations, key=lambda x: svd_prediction, reverse=True)

    return hybrid_recommendations[:num_recommendations]

# API route for recommendation for known user

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = data['user_id']
        product_id = data['product_id']
        num_recommendations = data.get('num_recommendations', 5)

        recommendations = hybrid_recommendation(user_id, product_id, num_recommendations=num_recommendations)

        return jsonify({'status': 'success', 'recommendations': recommendations})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# API route for general recommendations for unknown user
@app.route('/general_recommend', methods=['GET'])
def general_recommend():
    try:
        num_recommendations = int(request.args.get('num_recommendations', 5))

        # General recommendations based on popularity (or other criteria)
        top_rated_products = ecommerce_data.groupby('product_id')['rating'].mean().sort_values(ascending=False).index[:num_recommendations].tolist()

        return jsonify({'status': 'success', 'recommendations': top_rated_products})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# API route for user history-based recommendations
@app.route('/user_history_recommend', methods=['POST'])
def user_history_recommend():
    try:
        data = request.json
        user_id = data['user_id']
        num_recommendations = data.get('num_recommendations', 5)

        # User history recommendations based on past purchases
        user_history_recommendations = ecommerce_data[ecommerce_data['user_id'] == user_id]['product_id'].unique()[:num_recommendations]

        return jsonify({'status': 'success', 'recommendations': user_history_recommendations})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# API route for user preferences-based recommendations
@app.route('/user_preferences_recommend', methods=['POST'])
def user_preferences_recommend():
    try:
        data = request.json
        user_id = data['user_id']
        num_recommendations = data.get('num_recommendations', 5)

        # User preferences recommendations based on categories and brands
        user_preferences_recommendations = ecommerce_data[(ecommerce_data['user_id'] == user_id) & (ecommerce_data['rating'] >= 4)]
        user_preferences_recommendations = user_preferences_recommendations.groupby('product_id')['rating'].mean().sort_values(ascending=False).index[:num_recommendations].tolist()

        return jsonify({'status': 'success', 'recommendations': user_preferences_recommendations})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# API route for popular products
@app.route('/popular_products', methods=['GET'])
def popular_products():
    try:
        num_recommendations = int(request.args.get('num_recommendations', 5))

        # Popular products based on overall ratings
        popular_products = ecommerce_data.groupby('product_id')['rating'].mean().sort_values(ascending=False).index[:num_recommendations].tolist()

        return jsonify({'status': 'success', 'recommendations': popular_products})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# API route for new products
@app.route('/new_products', methods=['GET'])
def new_products():
    try:
        num_recommendations = int(request.args.get('num_recommendations', 5))

        # New products based on timestamp (recently added)
        new_products = ecommerce_data.sort_values(by='timestamp', ascending=False)['product_id'].unique()[:num_recommendations]

        return jsonify({'status': 'success', 'recommendations': new_products})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
