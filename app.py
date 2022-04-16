from flask import Flask, jsonify, request
from classifier import  get_prediction

app = Flask(__name__)

#creating additional route
@app.route('/')
def index():
  return "Welcome to the home page"

# Now we'll start writing our route. We need a post request to send the
# image to the prediction model and our route name would be 'predict-digit'.
@app.route("/predict-digit", methods=["POST"])

# Inside this route we'll write a function called predict_data().We'll use the request function
# to get the files from the digit key.We'll also use the get_prediction function here and return
# the prediction in the json format with the status code of 200.

def predict_data():
  # image = cv2.imdecode(np.fromstring(request.files.get("digit").read(), np.uint8), cv2.IMREAD_UNCHANGED)
  image = request.files.get("digit")
  prediction = get_prediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

if __name__ == "__main__":
  app.run(debug=True)
