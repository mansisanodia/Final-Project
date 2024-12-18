from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load models
crop_model = joblib.load('D:/project/crop_rf_model.pkl')
fertilizer_model = joblib.load('D:/project/fertilizer_rf_model.pkl')
disease_model = load_model("D:/project/disease_model.h5")
class_indices = np.load("D:/project/class_indices.npy", allow_pickle=True)
class_labels = {i: label for i, label in enumerate(class_indices)}

# Fertilizer dictionary and data
fertilizer_data = pd.read_csv('D:/project/Fertilizer1.csv')
fertilizer_dic = {
    # Add your fertilizer recommendation logic here
}
def get_fertilizer_advice(n, p, k):
    if n > 50:
        return fertilizer_dic['NHigh']
    elif n < 20:
        return fertilizer_dic['Nlow']
    elif p > 50:
        return fertilizer_dic['PHigh']
    elif p < 20:
        return fertilizer_dic['Plow']
    elif k > 50:
        return fertilizer_dic['KHigh']
    elif k < 20:
        return fertilizer_dic['Klow']
    else:
        return "Soil nutrient levels are balanced. No specific fertilizer recommendations."
    
# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

disease_info = {
    "Cashew anthracnose": {
        "cause": "Caused by the fungus *Colletotrichum gloeosporioides*, often due to high humidity and wet conditions.",
        "prevention": "Ensure proper pruning for air circulation, avoid overcrowding, and apply fungicides when needed.",
        "cure": "Use copper-based fungicides or systemic fungicides during the early stages of infection."
    },
    "Cashew gumosis": {
        "cause": "Caused by fungal pathogens or injuries leading to excessive gum exudation.",
        "prevention": "Avoid injuries during cultivation and ensure proper drainage.",
        "cure": "Apply fungicides like Bordeaux mixture to infected areas."
    },
    "Cashew healthy": {
        "note": "No action needed; the plant is healthy."
    },
    "Cashew leaf miner": {
        "cause": "Caused by larvae of moths that burrow into leaves.",
        "prevention": "Monitor and remove infested leaves, and use pheromone traps to reduce moth populations.",
        "cure": "Apply neem oil or insecticides specifically targeting leaf miners."
    },
    "Cashew red rust": {
        "cause": "Caused by *Cephaleuros virescens*, an algal pathogen.",
        "prevention": "Maintain proper field hygiene and avoid water stagnation.",
        "cure": "Use copper-based fungicides for effective control."
    },
    "Cassava bacterial blight": {
        "cause": "Caused by *Xanthomonas axonopodis* bacteria.",
        "prevention": "Use disease-free planting material and ensure crop rotation.",
        "cure": "Remove and destroy infected plants and apply bactericides."
    },
    "Cassava brown spot": {
        "cause": "Caused by the fungus *Cercospora spp.*.",
        "prevention": "Plant resistant varieties and ensure good field sanitation.",
        "cure": "Apply fungicides like Mancozeb at early stages."
    },
    "Cassava green mite": {
        "cause": "Damage caused by *Mononychellus tanajoa*, a type of mite.",
        "prevention": "Encourage natural predators like lady beetles and plant resistant varieties.",
        "cure": "Apply acaricides or neem-based products."
    },
    "Cassava healthy": {
        "note": "No action needed; the plant is healthy."
    },
    "Cassava mosaic": {
        "cause": "Caused by the Cassava Mosaic Virus, transmitted by whiteflies.",
        "prevention": "Use virus-free planting material and control whitefly populations.",
        "cure": "Remove and destroy infected plants. Plant resistant varieties."
    },
    "Corn Common_Rust": {
        "cause": "Caused by the fungus *Puccinia sorghi*.",
        "prevention": "Plant resistant hybrids and rotate crops.",
        "cure": "Apply fungicides at early stages of infection."
    },
    "Corn Gray_Leaf_Spot": {
        "cause": "Caused by *Cercospora zeae-maydis* fungus.",
        "prevention": "Ensure crop rotation and plant resistant hybrids.",
        "cure": "Use fungicides containing strobilurins or triazoles."
    },
    "Corn Healthy": {
        "note": "No action needed; the plant is healthy."
    },
    "Corn Northern_Leaf_Blight": {
        "cause": "Caused by the fungus *Setosphaeria turcica*.",
        "prevention": "Use resistant hybrids and ensure crop rotation.",
        "cure": "Apply fungicides as needed."
    },
    "Maize fall armyworm": {
        "cause": "Caused by larvae of the moth *Spodoptera frugiperda*.",
        "prevention": "Use pheromone traps and practice crop rotation.",
        "cure": "Apply insecticides or biological controls like *Bacillus thuringiensis*."
    },
    "Maize grasshopper": {
        "cause": "Damage caused by grasshoppers feeding on foliage.",
        "prevention": "Monitor fields regularly and encourage natural predators.",
        "cure": "Apply insecticides if infestation is severe."
    },
    "Maize healthy": {
        "note": "No action needed; the plant is healthy."
    },
    "Maize leaf beetle": {
        "cause": "Caused by beetles feeding on leaves.",
        "prevention": "Encourage natural predators and monitor crops regularly.",
        "cure": "Use appropriate insecticides."
    },
    "Maize leaf blight": {
        "cause": "Caused by fungal pathogens.",
        "prevention": "Ensure crop rotation and plant resistant varieties.",
        "cure": "Apply fungicides when symptoms appear."
    },
    "Maize leaf spot": {
        "cause": "Caused by fungal or bacterial infections.",
        "prevention": "Practice good field sanitation and crop rotation.",
        "cure": "Apply specific fungicides or bactericides."
    },
    "Maize streak virus": {
        "cause": "Caused by the Maize Streak Virus, transmitted by leafhoppers.",
        "prevention": "Control leafhopper populations and plant resistant varieties.",
        "cure": "Remove and destroy infected plants."
    },
    "Potato Early_Blight": {
        "cause": "Caused by the fungus *Alternaria solani*.",
        "prevention": "Plant resistant varieties and avoid overhead irrigation.",
        "cure": "Apply fungicides such as Mancozeb or Chlorothalonil."
    },
    
}

fertilizer_dic = {
        'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:

        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

        <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli

        <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods""",

        'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

        <br/>5. Add composted manure to the soil.

        <br/>6. Plant Nitrogen fixing plants like peas or beans.

        <br/>7. <i>Use NPK fertilizers with high N value.

        <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

        'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        <br/>5. Use crop rotations to decrease high phosphorous levels""",

        'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

        <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.

        <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

        <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

        <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

        <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

        'KHigh': """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

        <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

        <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """,

        'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:

        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium
        """
    }



@app.route('/')
def home():
    return render_template('index002.html')

@app.route('/crop')
def crop_page():
    return render_template('crop001.html')

@app.route('/fertilizer')
def fertilizer_page():
    return render_template('fertilizer001.html')

@app.route('/disease')
def disease_page():
    return render_template('disease001.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = [float(request.form[field]) for field in ['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
    input_features = pd.DataFrame([data], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    crop_prediction = crop_model.predict(input_features)
    return jsonify({"recommended_crop": crop_prediction[0]})

@app.route('/fert_recommend', methods=['POST'])
def fert_recommend():
    crop = request.form['cropname'].capitalize()
    ph = float(request.form['pH'])
    if crop not in fertilizer_data['Crop'].unique():
        return "Invalid crop input"
    encoded_crop = list(fertilizer_data['Crop'].unique()).index(crop)
    input_features = pd.DataFrame([[encoded_crop, ph]], columns=["Crop", "pH"])
    fert_predict = fertilizer_model.predict(input_features)
    fertilizer_advice = get_fertilizer_advice(fert_predict[0][0],fert_predict[0][1] ,fert_predict[0][2])
    return render_template('fertilizer_result.html', fertilizer=fert_predict, advice=fertilizer_advice)

@app.route('/predict_disease', methods=['POST']) 
def predict_disease():
    file = request.files.get('image')
    if not file or not allowed_file(file.filename):
        return "Invalid file"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    img = load_img(filepath, target_size=(128, 128))
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    predictions = disease_model.predict(img_array)
    predicted_class = class_labels.get(np.argmax(predictions), "Unknown Disease")
    os.remove(filepath)
    disease_details = disease_info.get(predicted_class, {
        "cause": "Unknown cause",
        "prevention": "No prevention info available",
        "cure": "No cure info available"
    })
    return render_template('disease_result.html', prediction=predicted_class, cause=disease_details.get("cause", "N/A"), prevention=disease_details.get("prevention", "N/A"), cure=disease_details.get("cure", "N/A"))

if __name__ == '__main__':
    app.run(debug=True)
