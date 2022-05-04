import hashlib
import json
import pathlib
import pickle
from io import BytesIO
import pandas as pd
import requests
from PIL import Image
from flask import Flask, request, Response, send_file
from bs4 import BeautifulSoup
from datasets import PetFinderDataset
from utils import WordCounter, CharacterCounter

NOT_SPECIFIED = "Not Specified"

app = Flask(__name__)


def hasher(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def load_options(type: str):
    frame = dataset.dataframe[dataset.dataframe['Type'] == type]
    categorical_columns = list(frame.select_dtypes(exclude=['number']).columns)
    categorical_columns.remove('Name')
    categorical_columns.remove('Description')
    mapping = {}
    for col in categorical_columns:
        mapping[col] = list(frame[col].unique())

    mapping['Breed'] = list(set(mapping.pop('Breed1') + mapping.pop('Breed2')))
    mapping['Color'] = list(set(mapping.pop('Color1') + mapping.pop('Color2') + mapping.pop('Color3')))
    return mapping


def load_model():
    return pickle.load(open('models/model.pkl', 'rb'))


dataset = PetFinderDataset()
model = load_model()


def center_crop_image(image):
    width, height = image.size

    if width > height:
        crop_width = height
        crop_height = height
    else:
        crop_width = width
        crop_height = width

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = (width + crop_width) // 2
    bottom = (height + crop_height) // 2

    return image.crop((left, top, right, bottom)).resize((512, 512))


@app.route("/v1/imagepreview")
def route_image_preview():
    form_validation = ['Type', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3']
    params = request.args.to_dict()

    for key in form_validation:
        if key not in params:
            params[key] = NOT_SPECIFIED

    if not request.args.get('Type'):
        return Response(json.dumps({'message': "Missing Type Parameter", 'status': 400}), status=400, mimetype='application/json')

    url = f'https://www.bing.com/images/search?q=' \
          f'{params["Breed1"] + "%20" if params["Breed1"] != NOT_SPECIFIED else ""}' \
          f'{"cross%20" + params["Breed2"] + "%20" if params["Breed2"] != NOT_SPECIFIED else ""}' \
          f'{"color%20" + params["Color1"] + "%20" if params["Color1"] != NOT_SPECIFIED else ""}' \
          f'{params["Color2"] + "%20" if params["Color2"] != NOT_SPECIFIED else ""}' \
          f'{params["Color3"] if params["Color3"] != NOT_SPECIFIED else ""}' \
          f'{params["Type"]}%20' \
          f'&FORM=HDRSC2'

    if pathlib.Path(f'backend/cache/{hasher(url)}.jpg').exists():
        image = Image.open(f'backend/cache/{hasher(url)}.jpg')
        return serve_pil_image(image)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    link = json.loads(soup.find('ul', {'class': 'dgControl_list'}).find('a').get('m'))['murl']

    image = Image.open(requests.get(link, stream=True, timeout=5).raw)
    image = center_crop_image(image)
    pathlib.Path('cache').mkdir(parents=True, exist_ok=True)
    image.save(f'cache/{hasher(url)}.jpg')
    return serve_pil_image(image)


@app.route("/v1/inference")
def route_inference():
    form_validation = [
        'Type', 'Name', 'Gender', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3',
        'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'Description'
    ]
    inputs = request.args.to_dict()

    error_messages = []

    for key in form_validation:
        if key not in inputs:
            error_messages.append(f'Missing Parameter: {key}')
    if len(error_messages) > 0:
        return Response(json.dumps({'message': error_messages, 'status': 400}), status=400, mimetype='application/json')

    try:
        _y = model.predict_proba(pd.DataFrame([inputs]))
        return Response(json.dumps({'probability': {
            'Today': _y[0, 0],
            'Less Than A Week': _y[0, 1],
            'Less Than A Month': _y[0, 2],
            'Less Than 3 Month': _y[0, 3],
            'More Than 3 Month': _y[0, 4],
        }, 'status': 200}), status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'message': str(e), 'status': 500}), status=500, mimetype='application/json')


@app.route("/v1/options", methods=['GET'])
def route_options():
    type = request.args.get('Type')
    return Response(json.dumps(load_options(type)), status=200, mimetype='application/json')