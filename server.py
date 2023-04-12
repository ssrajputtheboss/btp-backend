from flask import Flask, flash, request, redirect, url_for, render_template, make_response, jsonify, json
import os
import urllib.request
from flask_sqlalchemy import SQLAlchemy
import uuid
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps
from process import generate_summary, setup
from threading import Thread
from flask_cors import cross_origin

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'secret key'
# database name
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# creates SQLALCHEMY object
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(70), unique=True)
    password = db.Column(db.String(80))


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        if not token:
            return jsonify({'message': 'Token is missing !!'}), 401

        try:
            # decoding the payload to fetch the stored details
            data = jwt.decode(
                token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query\
                .filter_by(public_id=data['public_id'])\
                .first()
        except Exception as e:
            return jsonify({
                'message': 'Token is invalid !!'
            }), 401
        # returns the current logged in users context to the routes
        return f(current_user, *args, **kwargs)

    return decorated


"""@app.route('/user', methods =['GET'])
@token_required
def get_all_users(current_user):
    # querying the database
    # for all the entries in it
    users = User.query.all()
    # converting the query objects
    # to list of jsons
    output = []
    for user in users:
        # appending the user data json
        # to the response list
        output.append({
            'public_id': user.public_id,
            'name' : user.name,
            'email' : user.email
        })
  
    return jsonify({'users': output})"""


@app.route('/getsummaries', methods=["GET"])
@cross_origin()
@token_required
def get_summaries(user):
    token = request.headers['x-access-token']
    data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
    files = []
    if os.path.isdir(os.path.join('static/summaries', data["public_id"])):
        files = os.listdir(os.path.join('static/summaries', data["public_id"]))
        files = list(filter(lambda x: x.endswith(".mp4"), files))
    return make_response(jsonify({'summaries': files}))


@app.route('/getuploads', methods=["GET"])
@cross_origin()
@token_required
def get_uploads(user):
    token = request.headers['x-access-token']
    data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
    files = []
    if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], data["public_id"])):
        files = os.listdir(os.path.join(
            app.config['UPLOAD_FOLDER'], data["public_id"]))
        files = list(filter(lambda x: x.endswith(".mp4"), files))
    return make_response(jsonify({'uploads': files}))


@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    # creates dictionary of form data
    auth = json.loads(request.data)

    if not auth or not auth.get('email') or not auth.get('password'):
        # returns 401 if any email or / and password is missing
        return make_response(
            'Could not verify',
            401,
            {'WWW-Authenticate': 'Basic realm ="Login required !!"'}
        )

    user = User.query\
        .filter_by(email=auth.get('email'))\
        .first()

    if not user:
        # returns 401 if user does not exist
        return make_response(
            'Could not verify',
            401,
            {'WWW-Authenticate': 'Basic realm ="User does not exist !!"'}
        )

    if check_password_hash(user.password, auth.get('password')):
        # generates the JWT Token
        token = jwt.encode({
            'public_id': user.public_id,
            'exp': datetime.utcnow() + timedelta(hours=24 * 30)
        }, app.config['SECRET_KEY'], algorithm="HS256")

        return make_response(jsonify({'token': token, 'user': {'name': user.name, 'email': user.email}}), 201)
    # returns 403 if password is wrong
    return make_response(
        'Could not verify',
        403,
        {'WWW-Authenticate': 'Basic realm ="Wrong Password !!"'}
    )


@app.route('/signup', methods=['POST'])
@cross_origin()
def signup():
    # creates a dictionary of the form data
    data = json.loads(request.data)

    # gets name, email and password
    name, email = data.get('name'), data.get('email')
    password = data.get('password')
    # checking for existing user
    user = User.query\
        .filter_by(email=email)\
        .first()
    if not user:
        # database ORM object
        user = User(
            public_id=str(uuid.uuid4()),
            name=name,
            email=email,
            password=generate_password_hash(password)
        )
        # insert user
        db.session.add(user)
        db.session.commit()
        token = token = jwt.encode({
            'public_id': user.public_id,
            'exp': datetime.utcnow() + timedelta(hours=24 * 30)
        }, app.config['SECRET_KEY'], algorithm="HS256")

        return make_response(jsonify(
            {'token': token, 'user': {'name': user.name, 'email': user.email}}
        ))
        # return make_response('Successfully registered.', 201)
    else:
        # returns 202 if user already exists
        return make_response('User already exists. Please Log in.', 202)


# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# return redirect(url_for('static', filename='uploads/' + filename), code=301)

'''@app.route('/upload')
def upload_form():
	return render_template('upload.html')'''


@cross_origin
@app.route('/video/<filename>', methods=["GET"])
def get_video(filename):
    args = request.args
    token = args.get('token')
    if not token:
        make_response(jsonify({}), 401)
    data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
    return redirect(url_for('static', filename=('summaries/' + data['public_id'] + '/' + filename)), code=301)


@app.route('/upload', methods=['POST'])
@cross_origin()
@token_required
def upload_video(user):
    if 'file' not in request.files:
        return make_response(jsonify({'message': 'No video selected for uploading'}), 401)
    file = request.files['file']
    token = request.headers['x-access-token']
    data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
    if file.filename == '':
        return make_response(jsonify({'message': 'No video selected for uploading'}), 401)
    elif os.path.isfile(f'static/summaries/{data["public_id"]}/' + file.filename):
        return make_response(jsonify({'message': 'summary already generated for this file'}), 401)
    elif os.path.isfile(UPLOAD_FOLDER + file.filename):
        return make_response(jsonify({'message': 'Video already queued for summary generation'}), 401)
    else:
        filename = secure_filename(file.filename)
        if not os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], data["public_id"])):
            os.mkdir(os.path.join(
                app.config['UPLOAD_FOLDER'], data["public_id"]))
        file.save(os.path.join(
            app.config['UPLOAD_FOLDER'], data["public_id"], filename))
        flash('generating summary')
        sigma = int(request.form.get('sigma'))
        if sigma < 5:
            sigma = 5
        if sigma > 100:
            sigma = 100
        task = Thread(target=generate_summary, args=(
            data["public_id"], filename, sigma,), daemon=True)
        task.start()
        # generate_summary(filename)
        # print('upload_video filename: ' + filename)
        return make_response(jsonify({'message': 'success'}, 200))


@app.route('/display/<filename>')
@token_required
def display_video(filename):
    # print('display_video filename: ' + filename)
    # print(url_for('static', filename='summaries/' + filename))
    return redirect(url_for('static', filename='summaries/' + filename), code=301)


if __name__ == "__main__":
    setup()
    app.run(debug=True, host="172.16.202.38")
