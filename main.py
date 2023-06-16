from fastapi import FastAPI
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from bson.objectid import ObjectId


from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson.objectid import ObjectId  # Import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import asyncio

# Definición de las variables globales y de la aplicación FastAPI
app = FastAPI()

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

client = None
product_data = None
product_likes_data = None
vectorizer = None
similarities = None


def convert_objectid_to_str(x):
    if isinstance(x, ObjectId):  # ObjectId should be available now
        return str(x)
    return x


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


# Definición de las funciones para cargar los datos, convertirlos a un formato serializable, y hacer las recomendaciones
async def load_data():
    global product_data
    global product_likes_data
    global client
    global vectorizer
    global similarities

    print("Loading data...")

    client = MongoClient(
        "mongodb+srv://germys:LWBVI45dp8jAIywv@douvery.0oma0vw.mongodb.net/Production"
    )
    db = client["Production"]

    product_collection = db["products"]
    raw_product_data = list(product_collection.find())
    for document in raw_product_data:
        for key, value in document.items():
            if isinstance(value, ObjectId):
                document[key] = str(value)

    product_data = pd.DataFrame(raw_product_data)
    print(product_data.dtypes)

    likes_collection = db["productuserlikesdislikes"]
    raw_product_likes_data = list(likes_collection.find())
    for document in raw_product_likes_data:
        for key, value in document.items():
            if isinstance(value, ObjectId):
                document[key] = str(value)

    product_likes_data = pd.DataFrame(raw_product_likes_data)
    product_likes_data.drop("_id", axis=1, inplace=True, errors="raise")
    print(product_likes_data.dtypes)
    vectorizer = CountVectorizer(stop_words="english")
    product_vectors = vectorizer.fit_transform(product_data["description"])

    similarities = cosine_similarity(product_vectors)

    print("Data loaded successfully")


@app.on_event("startup")
async def startup_event():
    await load_data()


@app.get("/recommend_products/{product_id}")
def get_recommendations(product_id: str, limit: int = 25):
    # Obtiene los índices de los productos similares en el DataFrame
    product_matches = product_data[product_data["_id"] == product_id]

    if len(product_matches) > 0:
        product_index = product_matches.index[0]
    else:
        return {"error": "No se encontró el producto con el ID especificado"}

    similar_product_indices = similarities[product_index].argsort()[-limit:][::-1]

    # Obtiene los productos similares
    similar_products = product_data.iloc[similar_product_indices]

    # Convierte los productos a un formato JSON serializable
    try:
        json_compatible_item_data = jsonable_encoder(similar_products)
        return JSONResponse(
            content=json_compatible_item_data, json_encoder=CustomJSONEncoder
        )
    except Exception as e:
        print(f"Error while serializing the data: {e}")


@app.get("/recommend_products_based_on_likes/{user_id}")
def get_recommendations_based_on_likes(user_id: str, limit: int = 25):
    liked_products = product_likes_data[
        (product_likes_data["userId"] == user_id) & (product_likes_data["like"] == True)
    ]

    if liked_products.empty:
        random_products = product_data.sample(n=limit)
        # Convierte los objetos numpy.int64 a int antes de la serialización.
        return random_products.astype(int).to_dict("records")

    all_recommendations = []
    for product in liked_products["productId"]:
        recommendations = get_recommendations(product, limit)
        print(liked_products["productId"])
        if isinstance(recommendations, dict) and "error" in recommendations:
            # Si se produce un error, devuelve el diccionario de error
            return recommendations
        elif recommendations is not None:
            all_recommendations += recommendations

    all_recommendations_df = pd.DataFrame(all_recommendations)
    if "id" not in all_recommendations_df.columns:
        return {"error": "La columna '_id' no se encuentra en el DataFrame"}

    recommendations_counts = all_recommendations_df["dui"].value_counts()
    most_frequently_recommended = recommendations_counts[:limit].index
    most_frequently_recommended_products = product_data[
        product_data["_id"].isin(most_frequently_recommended)
    ]
    # Convierte los objetos numpy.int64 a int antes de la serialización.
    return (
        most_frequently_recommended_products.astype(int)
        .applymap(convert_objectid_to_str)
        .to_dict("records")
    )


# Ejecución del servidor
if __name__ == "__main__":
    nest_asyncio.apply()
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
    asyncio.run(startup_event())
    uvicorn.run(app, host="0.0.0.0", port=8000)
