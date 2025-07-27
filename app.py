
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import os
from uuid_tracker import process_uploaded_video, process_webcam_video

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    video = request.files["video"]
    if video.filename != "":
        filepath = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(filepath)
        out_vid, out_log = process_uploaded_video(filepath)
        return render_template("index.html", output_video=out_vid, log_file=out_log)
    return redirect(url_for("home"))

@app.route("/webcam", methods=["POST"])
def webcam():
    out_vid, out_log = process_webcam_video()
    return render_template("index.html", output_video=out_vid, log_file=out_log)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(directory=os.path.dirname(filename), path=os.path.basename(filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
