import base64
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model_def import QuickDrawCNN

MODEL_PATH = Path(__file__).parent / "model" / "quickdraw_cnn.pth"

CLASSES = [
    "apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope",
    "eyeglasses", "guitar", "hammer", "hat", "ice cream", "leaf", "scissors",
    "star", "t-shirt", "pants", "lightning", "tree",
]

app = FastAPI(title="Doodle Recognizer")

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def catch_all_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


model = QuickDrawCNN(input_size=28, num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


def _decode_image(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    return Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale


def _preprocess(image: Image.Image) -> torch.Tensor:
    """
    Canvas drawings are dark strokes on a light/white background.
    The model was trained on Quick Draw bitmaps: white strokes on a black
    background, normalized to [0, 1]. Resize to 28x28 and invert to match.
    """
    image = image.resize((28, 28), Image.LANCZOS)
    tensor = torch.from_numpy(
        __import__("numpy").array(image)
    ).float()
    tensor = 255.0 - tensor  # invert: dark-on-light -> light-on-dark
    tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor


@app.get("/")
def home() -> dict[str, str]:
    return {"status": "ok", "message": "Doodle Recognizer backend is running", "docs": "/docs"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/classes")
def get_classes() -> dict[str, list[str]]:
    return {"classes": CLASSES}


@app.post("/predict")
def predict(payload: dict):
    image_data = payload.get("image")
    if not image_data:
        raise HTTPException(status_code=422, detail="Request body must include an 'image' field (base64 data URL).")

    try:
        image = _decode_image(image_data)
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {error}") from error

    tensor = _preprocess(image)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    top5 = torch.topk(probabilities, k=5)
    predictions = [
        {"label": CLASSES[idx], "confidence": round(float(prob), 4)}
        for prob, idx in zip(top5.values.tolist(), top5.indices.tolist())
    ]

    return {
        "predicted_label": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "top5": predictions,
    }
