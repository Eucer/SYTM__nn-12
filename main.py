from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bson import ObjectId
import math
import uvicorn
from bson.objectid import ObjectId

app = FastAPI()

# CORS middleware settings
origins = [
    "http://localhost:5173",
    "https://www.douvery.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(
    "mongodb+srv://germys:LWBVI45dp8jAIywv@douvery.0oma0vw.mongodb.net/Production"
)
db = client["Production"]
products_collection = db["products"]
user_events_collection = db["user_events"]

# Leer los datos y almacenarlos en un DataFrame
product_data = pd.DataFrame(list(products_collection.find()))
product_data["_id"] = product_data["_id"].apply(lambda x: str(x))


# Agregar una columna "full_text" que contenga información adicional sobre el producto
product_data["full_text"] = (
    product_data["category"]
    + " "
    + product_data["subCategory"]
    + " "
    + product_data["name"]
)

# Crear un vectorizador para convertir las cadenas de texto a vectores numéricos
vectorizer = CountVectorizer(stop_words="english")
product_vectors = vectorizer.fit_transform(product_data["full_text"])

# Calcular la similitud entre los productos basada en los vectores numéricos de sus descripciones
similarities = cosine_similarity(product_vectors)


def convert_to_serializable(value):
    if isinstance(value, ObjectId):
        return str(value)
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_serializable(v) for v in value]
    elif isinstance(value, float) and (
        value == float("inf") or value == float("-inf") or math.isnan(value)
    ):
        return None
    return value


def products_to_json(products):
    serializable_products = [
        {key: convert_to_serializable(value) for key, value in product.items()}
        for product in products
    ]
    return serializable_products


def recommend_products(product_id, limit=25):
    product_row = product_data[product_data["_id"] == product_id].iloc[0]
    product_similarities = similarities[product_row.name]
    closest_product_indices = product_similarities.argsort()[::-1][1 : (limit + 1)]
    closest_products = product_data.iloc[closest_product_indices]
    return products_to_json(closest_products.to_dict("records"))


def get_last_viewed_products(user_id):
    user_product_views = list(
        user_events_collection.find(
            {"userId": ObjectId(user_id), "type": "view-product-page"},
            {"_id": 0, "productId": 1},
        ).sort("timestamp", -1)
    )
    product_ids = []
    for event in user_product_views:
        if str(event["productId"]) not in product_ids:
            product_ids.append(str(event["productId"]))
        if len(product_ids) == 5:
            break
    # print(
    #     f"Last viewed products for user {user_id}: {product_ids}"
    # )  # Debug print statement
    return product_ids


def recommend_products_based_on_last_viewed(user_id, limit=25):
    last_viewed_products = get_last_viewed_products(user_id)
    combined_recommendations = []
    for product in last_viewed_products:
        product_recommendations = recommend_products(product, limit)
        combined_recommendations += product_recommendations
    sorted_recommendations = sorted(
        combined_recommendations,
        key=lambda x: combined_recommendations.count(x),
        reverse=True,
    )
    final_recommendations = []
    for recommendation in sorted_recommendations:
        if recommendation not in final_recommendations:
            final_recommendations.append(recommendation)
        if len(final_recommendations) == limit:
            break
    return final_recommendations


@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Welcome to my notes application, use the /docs route to proceed"
    }


@app.get("/recommend_products/{product_id}")
def get_recommendations(product_id: str, limit: int = 25):
    recommendations = recommend_products(product_id, limit)
    return recommendations


@app.get("/recommend_products_based_on_last_viewed/{user_id}")
def get_recommendations_based_on_last_viewed(user_id: str, limit: int = 25):
    recommendations = recommend_products_based_on_last_viewed(user_id, limit)
    return recommendations
