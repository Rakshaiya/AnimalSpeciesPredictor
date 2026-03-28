# Animal Species Classifier — MobileNetV2 + Flask

A deep learning web application that classifies images into 10 animal species using a fine-tuned MobileNetV2 model, served through a Flask web interface.

---

## Demo

Upload any animal image → get instant species prediction.

---

## Supported Animals

| Label | Species |
|-------|---------|
| cane | Dog |
| cavallo | Horse |
| elefante | Elephant |
| farfalla | Butterfly |
| gallina | Chicken |
| gatto | Cat |
| mucca | Cow |
| pecora | Sheep |
| ragno | Spider |
| scoiattolo | Squirrel |

---

## How It Works

1. User uploads an animal image via the web interface
2. Image is resized to 128×128 and normalized
3. Fine-tuned MobileNetV2 model predicts the species
4. Result is displayed on screen with the uploaded image

---

## Tech Stack

- **Python**
- **TensorFlow / Keras** — MobileNetV2 fine-tuned model
- **Flask** — web framework
- **NumPy** — array operations
- **Werkzeug** — secure file handling
- **HTML/CSS** — frontend interface

---

## Project Structure

```
animal-classifier/
│
├── app.py                  # Flask application
├── models/
│   └── mobilenetv2_animals_finetuned.h5   # Trained model
├── uploads/                # Uploaded images (auto-created)
├── templates/
│   └── index.html          # Frontend template
└── requirements.txt        # Dependencies
```

---

## Setup & Run

**1. Clone the repository:**
```bash
git clone https://github.com/Rakshaiya/animal-species-classifier.git
cd animal-species-classifier
```

**2. Install dependencies:**
```bash
pip install flask tensorflow numpy werkzeug
```

**3. Make sure the model file is in place:**
```
models/mobilenetv2_animals_finetuned.h5
```

**4. Run the app:**
```bash
python app.py
```

**5. Open in browser:**
```
http://127.0.0.1:5000
```

---

## Model Details

- **Architecture** — MobileNetV2 (pre-trained on ImageNet, fine-tuned)
- **Input size** — 128 × 128 × 3
- **Output** — 10 animal classes
- **Dataset** — Animals-10 dataset (Italian class labels mapped to English)
- **Training** — Transfer learning with fine-tuning on top layers

---

## Requirements

```
flask
tensorflow
numpy
werkzeug
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Author

**Rakshaiya Yadav G**
- GitHub: [@Rakshaiya](https://github.com/Rakshaiya)
- Email: rakshaiya115@gmail.com
