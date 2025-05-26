from flask import Flask, render_template, Response
import cv2
import numpy as np
import pickle
import mediapipe as mp
import os
import pygame

app = Flask(__name__)

class SignLanguageClassifier:
    def __init__(self, model_path='./model.p', capture_device=0):
        self.model_dict = pickle.load(open(model_path, 'rb'))
        self.model = self.model_dict['model']
        self.cap = cv2.VideoCapture(capture_device)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.labels_dict = {
            0: {'label': 'Bad', 'audio': 'audio/bad.mp3'},
            1: {'label': 'Rock/Yes', 'audio': 'audio/rock2.mp3'},
            2: {'label': 'Paper/Hello', 'audio': 'audio/paper2.mp3'},
            3: {'label': 'Scissor', 'audio': 'audio/scissor.mp3'},
            4: {'label': 'Good/Okey', 'audio': 'audio/good.mp3'},
            5: {'label': 'ILoveYou', 'audio': 'audio/ILY.mp3'},
            6: {'label': 'No', 'audio': 'audio/no.mp3'},
            7: {'label': 'Help', 'audio': 'audio/help.mp3'},
            8: {'label': 'Im angry', 'audio': 'audio/angry.mp3'},
            9: {'label': 'Im happy', 'audio': 'audio/happy.mp3'},
            10: {'label': 'So Delicious', 'audio': 'audio/delicious.mp3'}
        }

        self.cap.set(3, 640)  # Width
        self.cap.set(4, 480)  # Height


        # Initialize Pygame for audio playback
        pygame.mixer.init()
        self.sounds = {index: pygame.mixer.Sound(audio_path) for index, data in self.labels_dict.items() for audio_path in [os.path.join(os.path.dirname(__file__), data['audio'])]}

        # Error handling: basic
        if not self.cap.isOpened():
            print("Error: Unable to open video capture.")

    def generate_frames(self):
        
        while True:    
            data_aux = [] 
            x_ = []  
            y_ = []

            ret, frame = self.cap.read()
            print("Capture success:", ret)

            if not ret:
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    print("Hand landmarks:", results.multi_hand_landmarks)

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Ensure that data_aux has a length of 84
                data_aux = data_aux[:84] + [0] * max(0, 84 - len(data_aux))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = self.model.predict([np.asarray(data_aux)])

                predicted_character = int(prediction[0])
                self.play_sound(predicted_character)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, self.labels_dict[predicted_character]['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def play_sound(self, character):
        # Stop any currently playing sound
        pygame.mixer.stop()

        if character in self.sounds:
            self.sounds[character].play()

    def end_detection(self):
        self.cap.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect/<int:show_camera>')
def detect(show_camera):
    if show_camera:
        return render_template('detect.html')
    else:
        return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(classifier.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    classifier = SignLanguageClassifier(model_path='./model.p')
    app.run(debug=True)
