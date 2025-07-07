import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import load_model, Model
from PIL import Image
import io
from tensorflow.keras.preprocessing import image as keras_image


from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import create_engine, Column, Integer, String, DateTime


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/docs")


app.mount("/images", StaticFiles(directory="dataset/images_compressed"), name="images")

full_model = load_model("resnet50.h5")
feature_extractor = Model(
    inputs=full_model.input, outputs=full_model.get_layer("Feature_extractor").output
)

with open("clothing_embeddings.pkl", "rb") as f:
    image_paths, embeddings = pickle.load(f)

nn = NearestNeighbors(n_neighbors=15, metric="cosine")
nn.fit(embeddings)


def preprocess_img(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    x = keras_image.img_to_array(img)
    x = x / 255.0
    return np.expand_dims(x, axis=0)


Base = declarative_base()


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    price = Column(Integer)
    description = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    category_id = Column(Integer)
    image = Column(String)
    brand_id = Column(Integer)


DRIVER = "ODBC Driver 17 for SQL Server"
DATABASE_URL = (
    "mssql+pyodbc://@./QLBanQuanAo"
    f"?driver={DRIVER.replace(' ', '+')}"
    "&trusted_connection=yes"
)

engine = create_engine(DATABASE_URL, fast_executemany=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/search-by-image/")
async def search_by_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    img_bytes = await file.read()
    x = preprocess_img(img_bytes)
    emb = feature_extractor.predict(x)

    dists, idxs = nn.kneighbors(emb)
    dists, idxs = dists[0].tolist(), idxs[0].tolist()

    found_files = [os.path.basename(image_paths[i]) for i in idxs]

    products = db.query(Product).filter(Product.image.in_(found_files)).all()
    prod_map = {p.image: p for p in products}

    results = []
    for fname, dist in zip(found_files, dists):
        prod = prod_map.get(fname)
        results.append(
            {
                "id": prod.id if prod else None,
                "name": prod.name if prod else None,
                "price": prod.price if prod else None,
                "description": prod.description if prod else None,
                "image": f"{fname}",
                "distance": float(dist),
            }
        )

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=5000, reload=True)
