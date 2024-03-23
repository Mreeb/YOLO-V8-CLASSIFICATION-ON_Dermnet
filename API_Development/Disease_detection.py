from ultralytics import YOLO
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from google_link import link

app = Flask(__name__)

CORS(app)

model = YOLO('5disease.pt')

@app.route("/classification", methods=["POST"])
def classification():
    try:

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"})

        image_file = request.files['image']

        # Save the uploaded image to a temporary file
        temp_image_path = "temp_image.jpg"
        image_file.save(temp_image_path)

        results = model(temp_image_path, save=True, conf=0.7)

        # Extracting Confidence and Class

        if isinstance(results, list):
            results = results[0]

        tensor = results.probs.top5conf
        list1 = tensor.tolist()
        list2 = results.probs.top5
        result_dict = {key: value for key, value in zip(list2, list1)}
        dict2 = result_dict
        dict1 = results.names
        result = {dict1[key]: value for key, value in dict2.items()}

        try:
            result['no_Disease'] = result.pop('Unlabeled')
        except:
            pass

        top_confidence = 100 * float(results.probs.top1conf.item())
        print(top_confidence)

        flag = True
        if top_confidence <= 60:
            flag = False

        conf = list()
        diseases = list()
        links = list()


        for key, value in result.items():
            result[key] = round(value * 100, 2)

            conf.append(result[key])
            diseases.append(key)
            links.append(link(key))

        combined_list = [[disease, confidence, link] for disease, confidence, link in zip(diseases, conf, links)]

        print(result)


        response = {
            "top": [round(top_confidence, 2), flag],
            "Result": combined_list
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=9090)