from flask import Flask, render_template, request, redirect, url_for
import os
from detector import run

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        eps = float(request.form.get("eps"))
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            output_filename1, output_filename2 = run(filepath,eps)

            return render_template('index.html', input_image=filename, output_image=[output_filename1,output_filename2])

    return render_template('index.html')

@app.route('/Report')
def Report():
    return render_template('Report.html')


@app.route('/Code')
def Code():
    return render_template('Code.html')

if __name__ == '__main__':
    app.run(debug=True)
