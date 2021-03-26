#https://github.com/mohshawky5193/dog-breed-classifier/blob/master/web-app/web-app-classifier.py
#deployed at https://dog-breed-classifier-udacity.herokuapp.com/

import os
import io
from flask import Flask,request,jsonify,render_template
from fastai.basic_train import load_learner
from fastai.vision import open_image
import torch
from PIL import Image 

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def render_page():
    return render_template('cat-breed-detector.html')

@app.route('/uploadajax',methods=['POST'])
def upload_file():
    """
    retrieve the image uploaded and make sure it is an image file
    """
    file = request.files['file']
    image_extensions=['jpg', 'jpeg', 'png']
    
    if file.filename.split('.')[1] not in image_extensions:
        return jsonify('Please upload an appropriate image file')
    
    """
    Load the trained model in export.pkl 
    """
    learn = load_learner(path = ".")
    
    """
    Perform prediction
    """
    #image_bytes = file.read()
    #img = Image.open(io.BytesIO(image_bytes))
    
    img = open_image(file)
    
    pred_class,pred_idx,outputs = learn.predict(img)
    # i = pred_idx.item()
    # classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    # prediction = classes[i]
    
    # def predict(url):
    # img = fetch_image(url)
    # pred_class,pred_idx,outputs = learn.predict(img)
    res =  zip (learn.data.classes, outputs.tolist())
    predictions = sorted(res, key=lambda x:x[1], reverse=True)
    top_predictions = predictions[0:1]
    # plant = pred_class 
    # namme = top_predictions.replace('___', '\n DISEASE: ')
    # name1 = namme.replace('_', ' ')
    # name2 = name1.replace('(', '')
    # name3 = name2.replace(')', '')
    # name4 = name3.replace('"', '')
    # name5 = name4.replace('[', '')
    # name6 = name5.replace(']', '')

    for a, b in top_predictions:
        bnb = f"'{a}'"
        outp = b
    return jsonify({'disease': bnb, 'score': outp})    
    # pprint.pprint( top_predictions)
    # return img.resize(500)
   
    
    
if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))


