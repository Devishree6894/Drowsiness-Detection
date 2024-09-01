import cv2
import dlib
import os
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import winsound
from datetime import datetime
import pygame
from flask import Flask, render_template, Response, url_for, redirect, g
import sqlite3

app = Flask(__name__)

# Function to get the database connection
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('driver_status2.db', check_same_thread=False)
    return db

# Function to get the database cursor
def get_cursor():
    db = get_db()
    return db.cursor()

# Function to insert status information
def insert_status(state):
    cursor = get_cursor()
    current_time=datetime.now().replace(microsecond=0)
    cursor.execute('''INSERT INTO driver_status (time,state) VALUES (?,?)''', (current_time,state,))
    db = get_db()
    db.commit()

# Function to fetch status information
def fetch_status():
    cursor = get_cursor()
    cursor.execute('''SELECT * FROM driver_status''')
    return cursor.fetchall()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()



cap=None

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load('sounds/beeep.wav')

# Function to play alarm sound
def play_alarm():
    duration = 500  # milliseconds
    freq = 600  # Hz
    winsound.Beep(freq, duration)


def mouth_aspect_ratio(landmarks):  
    # grab the indexes of the facial landmarks for the mouth
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    mouth = landmarks[mStart:mEnd]
    #vertical mouth dist
    A = compute(mouth[2],mouth[10])
    B = compute(mouth[4],mouth[8])

    #horizontal dist
    C = compute(mouth[0],mouth[6])

    mar = ( A + B ) / (2.0 * C)

    return mar

def compute(ptA,ptB):
    dist=np.linalg.norm(ptA-ptB)
    return dist

def eye_aspect_ratio(eye):
    #complete the euclidean distance between the horizontal
    #vertical eye landmarks (x-y)-coordinates
    up=compute(eye[1],eye[5])+compute(eye[2],eye[4])
    down=compute(eye[0],eye[3])

    ea_Ratio = up / ( 2.0 * down )

    return ea_Ratio

def end_ea_ratio(landmarks):
    (lStart,lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart,rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye=landmarks[lStart:lEnd]
    rightEye=landmarks[rStart:rEnd]

    leftEAR=eye_aspect_ratio(leftEye)
    rightEAR=eye_aspect_ratio(rightEye)

    #Average of left and right EAR
    ear = (leftEAR + rightEAR) / 2.0

    return ear

def generate_frame():

    global cap
    if cap is None:
        cap=cv2.VideoCapture(0)

    cap.set(3,640)
    cap.set(4,360)

    width=1000
    height=600
    
    imageBackground=cv2.imread('static/image/bg.png')
    resized_image = cv2.resize(imageBackground, (width, height))

    #importing the mode images into a list
    folderModePath = 'Images'
    modePath = os.listdir(folderModePath)
    imgModeList = []
    for path in modePath:
        imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
    #print(len(imgModeList))

    imgModeResized = [cv2.resize(img, (240, 240)) for img in imgModeList]


    s=cv2.imread('Images/Layer 4.png')
    status=cv2.resize(s, (240, 240))
    #Constants
    YAWN_THRESH = 0.80
    SLEEPY_THRESH = 0.25
    
    FCOUNTER = 0
    SCOUNTER = 0
    DCOUNTER = 0
    ACTIVE = 0

    
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    while True:
        #getting img by webcam
        success,frame=cap.read()

        if not success:
            cap.release()
            break

        else:
            #converting the image to gray scale
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #get faces into webcam's image
            faces= detector(gray)

            #detected face in faces array
            for face in faces:

                #make the prediction and transform it to numpy arrray
                landmarks = predictor(gray,face)
                landmarks = face_utils.shape_to_np(landmarks)

                #Draw all the coordinate points on image
                for n in range (0,68):
                    (x,y)=landmarks[n]
                    cv2.circle(frame,(x,y),1,(255,255,255),-1)
        
                ear=end_ea_ratio(landmarks)
                mar = mouth_aspect_ratio(landmarks) 

        
                if ear < SLEEPY_THRESH:
                    SCOUNTER += 1
                    DCOUNTER=0
                    ACTIVE=0
                    FCOUNTER=0

                    if SCOUNTER>30:
                        status=imgModeResized[0]
                        if pygame.mixer.music.get_busy() == 0:  # Check if music is not already playing
                            pygame.mixer.music.play()
                        with app.app_context():  # Add this line
                            insert_status("sleepy")

                
                elif mar > YAWN_THRESH :
                    SCOUNTER=0
                    ACTIVE=0
                    DCOUNTER+=1
                    FCOUNTER=0
                    if(DCOUNTER>6):
                        status=imgModeResized[2]
                        pygame.mixer.music.stop()
                        play_alarm()
                        with app.app_context(): 
                            insert_status("drowsy")

                else:
                    SCOUNTER=0
                    DCOUNTER=0
                    ACTIVE+=1
                    FCOUNTER=0
                    if ACTIVE>4:
                        status=imgModeResized[1]
                        pygame.mixer.music.stop()
                        with app.app_context():
                            insert_status("active")
                

                fps_str=f"EAR:{ear}"
                cv2.putText(frame,fps_str, (7, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2) 
                cv2.putText(frame,f"MAR:{mar}", (7, 45), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2) 


            if not faces:
                SCOUNTER=0
                DCOUNTER=0
                ACTIVE=0
                FCOUNTER+=1
                if FCOUNTER>15:
                    status = imgModeResized[3]
                    pygame.mixer.music.stop()
                    with app.app_context():
                        insert_status("noDriver")
         
            resized_frame=cv2.resize(frame,(700,446))

            resized_image[111:111+446,34:34+700] = resized_frame
            resized_image[1:1+240, width-240:width] = status
        

        ret,buffer = cv2.imencode(".jpeg",resized_image)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    
#########################################################################################


@app.route("/")
def index():
    return render_template("home.html")

@app.route("/aboutus")
def about_us():
    return render_template("aboutus.html")

# Route to start sleep detection and redirect to the detection page (start.html)
@app.route("/start_detection", methods=['GET'])
def start_detection():
    return render_template("start.html")

@app.route("/video")
def video():
    return Response(
    generate_frame(),mimetype="multipart/x-mixed-replace; boundary=frame"
    )

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
@app.route("/status")
@app.route("/status")
def status():
    # Fetch status information from the database
    status_data = fetch_status()
    
    # Extract the timestamp and state of the most recent data point
    latest_data = status_data[-1]
    latest_timestamp = latest_data[1]
    latest_state = latest_data[2]
    
    # Define a dictionary to map status strings to numeric values
    status_mapping = {"noDriver": -1, "sleepy": 0, "drowsy": 1, "active": 2}
    
    # Map the status string of the most recent data point to a numeric value
    numeric_state = status_mapping[latest_state]
    
    # Create a color map for the numeric value
    cmap = {-1: 'blue', 0: 'orange', 1: 'red', 2: 'green'}
    
    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the most recent data point as a vertical line
    ax.axvline(x=latest_timestamp, color=cmap[numeric_state], linewidth=2)
    
    # Set y-axis ticks and labels
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_yticklabels(["noDriver", "sleepy", "drowsy", "active"])
    
    plt.xlabel('Timestamp')
    plt.ylabel('Status')
    plt.title('Driver Status Over Time')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return "Graph plotted successfully!"



@app.route("/stop", methods=['GET'])
def stop_detection():
    global cap
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    cap = None
    return "Video capture stopped successfully!"

@app.route("/quit",methods=['GET'])
def quit_detection():
    global cap
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    cap = None
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)