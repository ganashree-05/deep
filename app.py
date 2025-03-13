from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from main import predict_image  # Import the prediction function from main.py

app = Flask(__name__)

# ✅ Set Upload Folder for User Images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ✅ Home Page Route
@app.route("/")
def home():
    return render_template("index.html")

# ✅ About Page Route
@app.route("/about")
def about():
    return render_template("about.html")

# ✅ Logout Page Route
@app.route("/logout")
def logout():
    return render_template("logout.html")

# ✅ Route: Handle Image Upload & Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)  # No file uploaded

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)  # No file selected

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)  # Save uploaded file

        # ✅ Perform Deepfake Detection
        result, confidence = predict_image(file_path)

        # ✅ Render index.html with Prediction Result
        return render_template("index.html", prediction=result, confidence=confidence, filename=filename)

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
