import face_recognition
import cv2
import numpy as np
import os
import time
import tkinter as tk
#from twilio.rest import Client
import paho.mqtt.client as mqtt
import random

broker_url = "broker.mqttdashboard.com"
broker_port = 1883

flag = 0

now_name = ''

client = mqtt.Client()
client.connect(broker_url, broker_port)

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

video_capture = cv2.VideoCapture(0)

width, height = 800, 600

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

path = 'faces_detected'


def rename(re_in):
    os.rename(path+"\\"+".jpg",path+"\\"+"{}.jpg".format(re_in))

def make_screenshot(img, counter):
    """Takes a screenshot and saves the frame/image

    :param img: colored image of the video capture process
    :param input_name: name for the image
    """
    img_path = path
    if not os.path.exists(path):
        os.makedirs(img_path)
    cv2.imwrite(os.path.join(
        img_path, 'NewMember_{0}.jpg'.format(counter)), img)
    known_face_names.append('NewMember_{0}.jpg'.format(counter))
    known_face_encodings.append(face_recognition.face_encodings(img)[0])



# Load images and names
faces = os.listdir(path)
known_face_names = []
known_face_encodings = []

for face_name in faces:
    known_name = os.path.splitext(face_name)[0]
    known_name = known_name.capitalize()
    known_face_names.append(known_name)

print(known_face_names)
print(faces)

for face in faces:
    img = face_recognition.load_image_file(path+"\\"+face)
    #x = faces.index(face)
    known_face_encodings.append(face_recognition.face_encodings(img)[0])
    
print(len(known_face_encodings))


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

name = ''
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.475)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            #if face_distances.all() == face_encoding.:
             #   name = "Unknown"
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    cv2.rectangle(frame, (width, 400), (0, 500), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, "Name: "+now_name, (50, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(frame, "Date and Time: " + time.ctime(), (50, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    if name != "Unknown":
        if flag == 1:
            flag = 0
            client.publish(topic="alkaf", payload=flag, qos=0, retain=False)
    else:
        if flag == 0:
            flag = 1
            client.publish(topic="alkaf", payload=flag, qos=0, retain=False)
    #elif name == 
    if name == "Unknown":
        cv2.putText(frame, "Press 's' to save your picture and name", (50, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        now_name = name

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 215, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (255, 215, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (0, 0, 0), 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[top:top + bottom-top, left:left + right-left]
        roi_color = frame[top:top + bottom-top, left:left + right-left]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.6, 22)
        '''
        now = datetime.datetime.now()
        old_time = 0
        new_time = now.minute
        if len(smile) == 0:
            while True:
                if new_time - old_time > 5:
                    # Your Account SID from twilio.com/console
                    account_sid = "AC62b310bb2d6a32baf846aed556b585b5"
                    # Your Auth Token from twilio.com/console
                    auth_token  = "8d0fd71f037c4f86496946be3538bafa"
                    client = Client(account_sid, auth_token)
                    message = client.messages.create(
                       to="+966534791690",
                       from_="+13342471056",
                       body="Hosnia is not happy today ")
                    print(message.sid)
                    old_time = new_time
                break
'''
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (62, 0, 0), 2)        


    file = open("information.txt" ,"a")
    if name != '':
        file.write(name +"--"+ time.ctime() + "\n")
    file.close()

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Hit 'q' on the keyboard to quit or 's' to save!
    k = cv2.waitKey(1)
    cnt = random.randrange(0, 1000, 3)
    if k == ord('r'):
        master = tk.Tk()
        master.title("Save Information")
        master.geometry("300x300")
        tk.Label(master, text="Name: ").grid(row=3)

        e1 = tk.Entry(master)
        e1.grid(row=1, column=2)

        tk.Button(master, text='Quit', command=master.quit).grid(row=3, column=0, sticky=tk.W, pady=4)
        tk.Button(master, text='Screenshot', command=make_screenshot(frame, cnt)).grid(row=3, column=1, sticky=tk.W, pady=4)
        cnt += 1



        #def get_in():
         #   return e1.get()
        master.mainloop()



    elif k == ord('q'):
        break



# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



