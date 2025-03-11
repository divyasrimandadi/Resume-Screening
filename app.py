from flask import Flask, request, render_template
import os
from ml_model import analyze_resume  # ✅ Updated import

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "resume" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["resume"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Analyze the resume
    result = analyze_resume(file_path)
    return f"Screening Result: {result}"

if __name__ == "__main__":
    app.run(debug=True)
