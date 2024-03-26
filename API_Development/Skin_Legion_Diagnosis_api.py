from ultralytics import YOLO
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from google_link import link
import os


app = Flask(__name__)

CORS(app)

@app.route("/skin_disease", methods=["POST"])
def classification():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"})

        image_file = request.files['image']
        user_id = request.form.get("user_id")

        model = YOLO('5disease.pt')

        burn_dir = "TEST_BURN_IMAGES"
        os.makedirs(burn_dir, exist_ok=True)

        temp_image_path = f"{burn_dir}/burn_image_{user_id}.jpg"
        image_file.save(temp_image_path)

        results = model(temp_image_path, save=True, conf=0.7)

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
            result['No_disease'] = result.pop('Unlabeled')
        except:
            pass

        try:
            result['Nail-Onychopathies'] = result.pop('Nail Onychopathies')
        except:
            pass


        top_confidence = 100 * float(results.probs.top1conf.item())
        print(top_confidence)

        flag = True
        if top_confidence <= 65:
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

        result = sorted(combined_list, key=lambda x: x[1], reverse=True)

        response = {
            "top": [round(top_confidence, 2), flag],
            "Result": result
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/skin_burn", methods=["POST"])
def skin_burn():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"})

        image_file = request.files['image']
        user_id = request.form.get("user_id")

        model = YOLO("Skin_burn.pt")

        disease_dir = "TEST_DISEASE_IMAGES"
        os.makedirs(disease_dir, exist_ok=True)

        temp_image_path = f"{disease_dir}/burn_image_{user_id}.jpg"
        image_file.save(temp_image_path)

        results = model(temp_image_path, save=True, conf=0.7)

        if isinstance(results, list):
            results = results[0]

        tensor = results.probs.data
        conf = tensor.tolist()

        per_conf = list()

        for c in conf:
            per_conf.append(round(c*100, 2))

        labels = results.names
        degree = [value for value in labels.values()]

        print(degree)

        index_to_replace = degree.index("3nd degree burn")
        degree[index_to_replace] = "3rd degree burn"

        combined_list = [[burn_degree, confidence] for burn_degree, confidence in zip(degree, per_conf)]

        result = sorted(combined_list, key=lambda x: x[1], reverse=True)

        response = {
            "Result": result
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port = 5000)
