from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import base64
import io

app = Flask(__name__)
model = load_model("emnist_model.h5")

# EMNIST Balanced: 
EMNIST_LABELS = (
    [str(i) for i in range(10)] +       # 0-9
    [chr(i) for i in range(65, 91)] +   # A-Z
    [chr(i) for i in range(97, 123)]    # a-z
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_data = data["image"].split(",")[1]

    # Decode base64 image
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGBA")

    
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("L")

   
    img_array = np.array(img)
    inverted = 255 - img_array
    rows = np.any(inverted > 30, axis=1)
    cols = np.any(inverted > 30, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    
        pad = 20
        rmin = max(0, rmin - pad)
        rmax = min(img_array.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(img_array.shape[1] - 1, cmax + pad)
        img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img)

    
    img_array = 255 - img_array

    
    img_array = np.transpose(img_array)

    # Normalize
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    pred = model.predict(img_array)
    idx = int(np.argmax(pred))
    result = EMNIST_LABELS[idx]
    confidence = float(np.max(pred)) * 100

    return jsonify({"result": result, "confidence": round(confidence, 1)})

if __name__ == "__main__":
    app.run(debug=True)