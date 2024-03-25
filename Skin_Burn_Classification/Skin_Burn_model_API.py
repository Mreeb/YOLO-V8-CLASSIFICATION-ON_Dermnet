from ultralytics import YOLO
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

model = YOLO("Skin_burn.pt")


@app.route("/skin_burn", methods=["POST"])

def skin_burn():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"})

        image_file = request.files['image']
        user_id = request.form.get("user_id")

        # Save the uploaded image to a temporary file
        temp_image_path = f"burn_image_{user_id}.jpg"
        image_file.save(temp_image_path)

        results = model(temp_image_path, save=True, conf=0.7)

        if isinstance(results, list):
            results = results[0]

        tensor = results.probs.data
        conf = tensor.tolist()

        labels = results.names
        degree = [value for value in labels.values()]

        combined_list = [[burn_degree, confidence] for burn_degree, confidence in zip(degree, conf)]

        response = {
            "Result": combined_list
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port = 5000)


# from ultralytics import YOLO
#
# model = YOLO("Skin_burn.pt")
#
# image = "burn_test3.jpg"
#
# results = model(image, save=True, conf=0.7)
#
# if isinstance(results, list):
#     results = results[0]
#
# tensor = results.probs.data
# conf = tensor.tolist()
#
# labels = results.names
# degree = [value for value in labels.values()]
#
# combined_list = [[burn_degree, confidence] for burn_degree, confidence in zip(degree, conf)]
#
#
# print(combined_list)
