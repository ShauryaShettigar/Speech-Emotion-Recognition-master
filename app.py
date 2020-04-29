from flask import Flask, render_template, request, send_from_directory
import os
from helper_funcs import *

[fname, scaler] = create_and_save()

model = load(fname)


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'audio/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files['file']
    filename = file.filename
    destination = "/".join([target, filename])
    file.save(destination)
    outname = predict(destination, model, scaler)

    return render_template("complete.html", output=outname.upper())


@app.route('/upload/<filename>')
def prev_img(filename):
    return send_from_directory('audio', filename)


@app.route('/res/<filename>')
def res_img(filename):
    return send_from_directory('results', filename)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
