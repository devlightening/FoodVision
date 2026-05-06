# FoodVision

FoodVision is a computer vision course project: a web-supported food image classification prototype built on the Hugging Face `ethz/food101` dataset.

## Initial scope (10 classes)

Selected Food-101 classes (popular/recognizable in Turkey):

- pizza (Pizza)
- hamburger (Hamburger)
- french_fries (Patates Kızartması)
- ice_cream (Dondurma)
- baklava (Baklava)
- donuts (Donut)
- omelette (Omlet)
- chicken_wings (Tavuk Kanat)
- steak (Biftek / Et)
- sushi (Sushi)

## Project structure

This repo contains the prototype under `food-vision-project/`:

- `backend/`: Flask API + prediction utilities
- `notebooks/`: Python scripts for dataset exploration and training
- `web/`: Minimal HTML/CSS/JS UI for testing predictions
- `mobile/`: Placeholder (later: React Native + Expo)
- `dataset/`: Generated exploration artifacts (ignored by git)

## Setup (Windows / PowerShell)

From the `food-vision-project` folder:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset exploration

```powershell
py notebooks/food101_explore.py
```

Expected outputs:

- Prints all 101 class names
- Verifies the selected 10 classes exist
- Prints filtered counts and per-class distribution
- Saves a sample grid image to `dataset/selected_food101_samples.png`

## Train the model (prototype)

```powershell
py notebooks/train_food_model.py
```

This trains a MobileNetV2 transfer learning model for a small number of epochs (default: 5) and writes:

- `backend/model/food_model.keras`
- `backend/model/class_mapping.json`

## Run the backend

```powershell
py backend/app.py
```

Health check:

- http://127.0.0.1:5000/health

Prediction:

- `POST /predict` with form-data key `image`

## Run the web UI

Open `web/index.html` in your browser (double-click is fine).

If you hit CORS or file restrictions in the browser, use a simple static server:

```powershell
py -m http.server 5173
```

Then open:

- http://127.0.0.1:5173/web/

## Notes

- This is a working prototype foundation, not a final accuracy-optimized model.
- Next phases will add richer evaluation, image preprocessing visualizations, and a mobile app.

