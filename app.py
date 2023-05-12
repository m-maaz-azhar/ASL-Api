from function import *
from keras.models import model_from_json
from flask import Flask, request
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
api = Api(app)

class PredictNumber(Resource):
    def post(self):
        json_file = open("./models/digitsModel.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("./models/digitsModel.h5")

        image = request.files["image"]
        filename = secure_filename(image.filename)
        image.save(os.path.join("uploads", filename))

        # 1. New detection variables
        sequence = []
        sentence = []
        accuracy=[]
        predictions = []
        threshold = 0.8 

        i = 0
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while True:
                i += 1
                print(i," =====")
                frame = cv2.imread("uploads/" + filename)
                frame = cv2.resize(frame,(300,400),interpolation=cv2.INTER_LINEAR)

                cropframe=frame[40:400,0:300]

                frame=cv2.rectangle(frame,(0,40),(300,400),255,2)

                image, results = mediapipe_detection(cropframe, hands)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                try: 
                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(digitsActions[np.argmax(res)])
                        predictions.append(np.argmax(res))
                        
                        
                        if np.unique(predictions[-10:])[0]==np.argmax(res): 
                            if res[np.argmax(res)] > threshold: 
                                if len(sentence) > 0: 
                                    if digitsActions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(digitsActions[np.argmax(res)])
                                        accuracy.append(str(res[np.argmax(res)]*100))
                                else:
                                    sentence.append(digitsActions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)]*100)) 

                        if len(sentence) > 1: 
                            sentence = sentence[-1:]
                            accuracy=accuracy[-1:]

                except Exception as e:
                    pass
                
                print("Output: -"+' '.join(sentence)+''.join(accuracy))

                if i > 32:
                    break
                if ''.join(sentence) != '':
                    break
        if ''.join(sentence) != '':
            
            return {"result": {"success": True, "answer": ' '.join(sentence)}}, 200
        else:
            return {"result": {"success": False, "answer": ' '.join(sentence)}}, 200

class PredictAlphabet(Resource):
    def post(self):
        json_file = open("./models/alphaModel.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("./models/alphaModel.h5")

        image = request.files["image"]
        filename = secure_filename(image.filename)
        image.save(os.path.join("uploads", filename))

        # 1. New detection variables
        sequence = []
        sentence = []
        accuracy=[]
        predictions = []
        threshold = 0.8 
        i=0
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while True:
                i+=1
                frame = cv2.imread("uploads/" + filename)
                frame = cv2.resize(frame,(300,450),interpolation=cv2.INTER_LINEAR)

                cropframe=frame[40:400,0:300]

                frame=cv2.rectangle(frame,(0,40),(300,400),255,2)
  
                image, results = mediapipe_detection(cropframe, hands)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                try: 
                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(alphaActions[np.argmax(res)])
                        predictions.append(np.argmax(res))
                        
                        
                        if np.unique(predictions[-10:])[0]==np.argmax(res): 
                            if res[np.argmax(res)] > threshold: 
                                if len(sentence) > 0: 
                                    if alphaActions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(alphaActions[np.argmax(res)])
                                        accuracy.append(str(res[np.argmax(res)]*100))
                                else:
                                    sentence.append(alphaActions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)]*100)) 

                        if len(sentence) > 1: 
                            sentence = sentence[-1:]
                            accuracy=accuracy[-1:]

                except Exception as e:
                    pass
                
                print("Output: -"+' '.join(sentence)+''.join(accuracy))

                if i > 32:
                    break

                if ''.join(sentence) != '':
                    break
        if ''.join(sentence) != '':
            return {"result": {"success": True, "answer": ' '.join(sentence)}}, 200
        else:
            return {"result": {"success": False, "answer": ' '.join(sentence)}}, 200

api.add_resource(PredictNumber, "/PredictNumber")
api.add_resource(PredictAlphabet, "/PredictAlphabet")

if __name__ == "__main__":
    app.run()


