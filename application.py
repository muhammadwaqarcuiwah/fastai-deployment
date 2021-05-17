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
    classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    treatment = ['Apple Scab can be treated with: Propizol® Fungicide (Crabapples only) or PHOSPHO-jet. *Propizol is for ornamental use only.', 'Captan and sulfur products are labeled for control of both scab and black rot. A scab spray program including these chemicals may help prevent the frog-eye leaf spot of black rot, as well as the infection of fruit.', 'If you see the lesions on the apple leaves or fruit, it is too late to control the fungus. In that case, you should focus on purging infected leaves and fruit from around your tree. Spraying apple trees with copper can be done to treat cedar apple rust and prevent other fungal infections.', 'Plant is Healthy', 'Plant is Healthy', 'apply just prior to leaf fall and again next spring (several weeks prior to bud break) allow for thorough wetting of foliage, tree trunks, and scaffold limbs. apply 10-20 gallons per acre just prior to leaf fall. ... always read the pesticide label.', 'Plant is Healthy', 'Airborne spores are spread locally and regionally from corn debris. Management strategies for gray leaf spot include tillage, crop rotation and planting resistant hybrids. Fungicides may be needed to prevent significant loss when plants are infected early and environmental conditions favor disease', 'Numerous fungicides are available for rust control. Products containing mancozeb, pyraclostrobin, pyraclostrobin + metconazole, pyraclostrobin + fluxapyroxad, azoxystrobin + propiconazole, trifloxystrobin + prothioconazole can be used to control the disease.', 'Treating northern corn leaf blight involves using fungicides. For most home gardeners this step is not needed, but if you have a bad infection, you may want to try this chemical treatment. The infection usually begins around the time of silking, and this is when the fungicide should be applied.', 'Plant is Healthy', 'Presently, there are no effective management strategies for measles. Wine grape growers with small vineyards will often have field crews remove infected fruit prior to harvest.', 'The most widely used fungicide to control diseases of grapevine is Bordeaux mixture, a copper fungicide', 'Plant is Healthy', 'Citrus Greening Disease, or HLB, is considered one of the most devastating citrus diseases, and there is no known cure. ACPs do not always carry the disease, but the disease can only be spread by the insect or through infected propagating material.', 'Compounds available for use on peach and nectarine for bacterial spot include copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan; however, repeated applications are typically necessary for even minimal disease control.', 'Plant is Healthy', 'Seed treatment with hot water, soaking seeds for 30 minutes in water pre-heated to 125 F/51 C, is effective in reducing bacterial populations on the surface and inside the seeds. However, seed germination may be affected by heat treatment if not done accurately, while the risk is relatively low with bleach treatment.', 'Plant is Healthy', 'Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease; late maturing are more resistant than early maturing varieties. Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible.', 'The severe late blight can be effectively managed with prophylactic spray of mancozeb at 0.25% followed by cymoxanil+mancozeb or dimethomorph+mancozeb at 0.3% at the onset of disease and one more spray of mancozeb at 0.25% seven days after application of systemic fungicides in West Bengal [50]', 'Plant is Healthy', 'Plant is Healthy', 'Plant is Healthy', 'Combine one tablespoon baking soda and one-half teaspoon of liquid, non-detergent soap with one gallon of water, and spray the mixture liberally on the plants. Mouthwash. The mouthwash you may use on a daily basis for killing the germs in your mouth can also be effective at killing powdery mildew spores.', 'Treating Strawberry Leaf Scorch Since this fungal pathogen overwinters on the fallen leaves of infected plants, proper garden sanitation is key. This includes the removal of infected garden debris from the strawberry patch, as well as the frequent establishment of new strawberry transplants.', 'Plant is Healthy', 'A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit.', 'Tomatoes that have early blight require immediate attention before the disease takes over the plants. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable. Both of these treatments are organic.', 'For the home gardener, fungicides that contain maneb, mancozeb, chlorothanolil, or fixed copper can help protect plants from late tomato blight. Repeated applications are necessary throughout the growing season as the disease can strike at any time.', 'When treating tomato plants with fungicide, be sure to cover all areas of the plant that are above the soil, especially the underside of leaves, where the disease often forms. Calcium chloride-based sprays are recommended for treating leaf mold issues. Organic fungicide options are also available.', 'Organic fungicides can go a long way towards treating and preventing fungal infections like septoria leaf spot. Fungicides containing copper and potassium bicarbonate will help contain the fungal disease and keep it from spreading. Start spraying as soon as you notice symptoms of septoria leaf spot.', 'For sensitive plants, try 1 part alcohol to 3 parts water, and for hardier plants, try a 1 to 1 mixture. Dish soap solution: Using a mixture of 1 liter of warm water and 1 teaspoon of liquid dish soap, either mix the solution well in a spray bottle, or mix it into a bucket and wash the plant with a cloth or sponge.', 'Remove old plant debris at the end of the growing season; otherwise, the spores will travel from debris to newly planted tomatoes in the following growing season, thus beginning the disease anew.', 'Use only virus-and whitefly-free tomato and pepper transplants. Transplants should be treated with Capture (bifenthrin) or Venom (dinotefuran) for whitefly adults and Oberon for eggs and nymphs. Imidacloprid or thiamethoxam should be used in transplant houses at least seven days before shipping.', 'There are no cures for viral diseases such as mosaic once a plant is infected. Fungicides will NOT treat this viral disease. Plant resistant varieties when available or purchase transplants from a reputable source. Do NOT save seed from infected crops.', 'Plant is Healthy']
    plants = ['Apple', 'Apple', 'Apple', 'Apple', 'Blueberry', 'Cherry', 'Cherry', 'Corn', 'Corn', 'Corn', 'Corn', 'Grape', 'Grape', 'Grape', 'Orange', 'Peach', 'Peach', 'Pepper', 'Pepper', 'Potato', 'Potato', 'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Strawberry', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato']
    diseases = ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn (maize) Common rust ', 'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy', 'Grape Esca (Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape healthy', 'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot', 'Peach healthy', 'Pepper, bell Bacterial spot', 'Pepper, bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot', 'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus', 'Tomato healthy']

    # prediction = classes[i]
    for a, b in top_predictions:
        bnb = a
        outp = b
    
    index = classes.index(bnb) 
    treat = treatment[index]
    plant = plants[index]
    disease = diseases[index]
    return jsonify({'plant':plant, 'disease': disease, 'score': outp, 'treatment': treat, 'index': index})    
    # pprint.pprint( top_predictions)
    # return img.resize(500)
   
    
    
if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))


