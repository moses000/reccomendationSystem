
# E-Commerce Recommendation System API

This Flask application provides a RESTful API for a hybrid recommendation system in an e-commerce context. The recommendation system combines collaborative filtering (KNN), matrix factorization (SVD, NMF), and content-based filtering to suggest products to users based on their preferences and behavior.

## Setup

1. Install the required packages using `pip install -r requirements.txt`.
2. Modify the database configuration in the code (`database_type`, `database_url`, MongoDB/MYSQL configurations) to connect to your database.
3. Run the Flask application using `python app.py`.

## API Endpoints

### 1. `/recommend` (POST)

- **Description:** Get recommendations for a known user based on their purchase history and product preferences.
- **Request Body:** `{ "user_id": "user_id_here", "product_id": "product_id_here", "num_recommendations": 5 }`
- **Response:** `{ "status": "success", "recommendations": ["recommended_product_id_1", "recommended_product_id_2", ...] }`

### 2. `/general_recommend` (GET)

- **Description:** Get general recommendations for unknown users based on popularity.
- **Request:** `GET /general_recommend?num_recommendations=5`
- **Response:** `{ "status": "success", "recommendations": ["recommended_product_id_1", "recommended_product_id_2", ...] }`

### 3. `/user_history_recommend` (POST)

- **Description:** Get recommendations based on a user's purchase history.
- **Request Body:** `{ "user_id": "user_id_here", "num_recommendations": 5 }`
- **Response:** `{ "status": "success", "recommendations": ["recommended_product_id_1", "recommended_product_id_2", ...] }`

### 4. `/user_preferences_recommend` (POST)

- **Description:** Get recommendations based on a user's preferences (e.g., high-rated products).
- **Request Body:** `{ "user_id": "user_id_here", "num_recommendations": 5 }`
- **Response:** `{ "status": "success", "recommendations": ["recommended_product_id_1", "recommended_product_id_2", ...] }`

### 5. `/popular_products` (GET)

- **Description:** Get popular products based on overall ratings.
- **Request:** `GET /popular_products?num_recommendations=5`
- **Response:** `{ "status": "success", "recommendations": ["recommended_product_id_1", "recommended_product_id_2", ...] }`

### 6. `/new_products` (GET)

- **Description:** Get new products based on the timestamp of when they were added.
- **Request:** `GET /new_products?num_recommendations=5`
- **Response:** `{ "status": "success", "recommendations": ["recommended_product_id_1", "recommended_product_id_2", ...] }`

## Notes

- Replace the database configurations and dataset with your own.
- Adjust the number of recommendations (`num_recommendations`) as needed for each endpoint.
- This README assumes you have basic knowledge of Flask, RESTful APIs, and recommendation systems.
