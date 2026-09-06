# Doodle Recognizer

Draw one of 20 objects (apple, star, tree, guitar, etc.) on a canvas in your
browser. The drawing is sent to a FastAPI backend, which runs it through a
small CNN and returns the top-5 most likely predictions with confidence
scores.

## Credit

The trained model weights and CNN architecture come from
[**vietnh1009/QuickDraw**](https://github.com/vietnh1009/QuickDraw)
(MIT licensed), which trained a CNN on a 20-category subset of Google's
[Quick, Draw!](https://quickdraw.withgoogle.com/) dataset. That repo ships
desktop-only demos (OpenCV windows); this project wraps the same trained
model in a FastAPI backend and a browser canvas frontend so it can run as a
normal deployable web app instead.

**Categories**: apple, book, bowtie, candle, cloud, cup, door, envelope,
eyeglasses, guitar, hammer, hat, ice cream, leaf, scissors, star, t-shirt,
pants, lightning, tree.

## Project structure

```
quickdraw-web/
├── main.py              FastAPI backend
├── model_def.py          The CNN architecture (matches the original repo)
├── model/
│   └── quickdraw_cnn.pth   Extracted trained weights (~2.5MB, included)
├── index.html             Canvas frontend
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `index.html` in your browser. It's pointed at
`http://localhost:8000` by default — see the `BACKEND_URL` constant near the
top of the `<script>` tag in `index.html` if you run the backend on a
different port.

## API endpoints

| Method | Path        | Description                                          |
|--------|-------------|-------------------------------------------------------|
| GET    | `/`         | Health/status check                                    |
| GET    | `/healthz`  | Health check (used by hosting platforms)                |
| GET    | `/classes`  | Returns the list of 20 recognizable categories           |
| POST   | `/predict`  | Body: `{"image": "data:image/png;base64,..."}` — returns top-5 predictions |

## Deploying

Same pattern as a typical FastAPI + static-frontend project:

- **Backend → Render** (or Railway/Fly.io): push this repo, set the start
  command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Frontend → Netlify** (or GitHub Pages): deploy `index.html` as a static
  site, and update `BACKEND_URL` in `index.html` to your deployed backend's
  URL before deploying.

## How the drawing is processed

1. The canvas drawing (dark strokes on a white background) is sent to the
   backend as a base64 PNG.
2. It's converted to grayscale, resized to 28×28, and inverted (the model
   was trained on light strokes on a dark background, matching Quick Draw's
   bitmap format).
3. The CNN outputs a probability for each of the 20 classes; the top 5 are
   returned.

## Extending to more categories

The original repo's `train.py` and `src/config.py` support retraining on
any subset of Quick Draw's 345 total categories — update `CLASSES` there,
download the corresponding `.npy`/`.npz` files from the
[Quick Draw dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset),
retrain, then swap in the new weights here (re-run the same state_dict
extraction step used to produce `model/quickdraw_cnn.pth`).
