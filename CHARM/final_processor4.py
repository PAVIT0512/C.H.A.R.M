#Model with face_recognition and gemini
import google.generativeai as genai
from email import message
from re import T
import re
from gtts import gTTS
import speech_recognition as sr
import os
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import wolframalpha 
import wikipedia
import datetime
import warnings
# import tkinter as tk
# from tkinter import ttk
# from tkinter import END
# from tkinter import INSERT
import smtplib
import face_recognition
import cv2
import csv
import time
from time import sleep
from csv import writer
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
# from pydub import AudioSegment
# from pydub.playback import play
# import simpleaudio as sa
import vlc
#import RPi.GPIO as GPIO
# import frame_viewer as fv
from keras.models import load_model
import json
import random
from nltk.tokenize import sent_tokenize

# Configure Gemini API
genai.configure(api_key="AIzaSyDWUz6hsjHlSLI2HJAyG_HuvzQmy5ZFlas")
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure WolframAlpha API
WOLFRAMALPHA_APP_ID = "44YXHU-TV6AJRQ6HT"


model_chatbot= load_model('chatbot_model2.h5')
intents = json.loads(open('job_intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model_chatbot.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.8
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
    
def speak(text):
    tts = gTTS(text=text, lang="en")
    file = "sound.mp3"
    tts.save(file)
    
    vlc_instance = vlc.Instance()
    player = vlc_instance.media_player_new()
    media = vlc_instance.media_new_path(file)
    player.set_media(media)
    player.play()
    while player.get_state() != vlc.State.Ended:
        time.sleep(1)
    player.stop()
    vlc_instance.release()
    os.remove(file)

def gemini_search(query):
    try:
        response = model.generate_content(query)
        clean_response = re.sub(r'[^\w\s]', '', response.text)
        return clean_response
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def wolfram_search(query):
    try:
        client = wolframalpha.Client(WOLFRAMALPHA_APP_ID)
        res = client.query(query)
        return next(res.results).text if res['@success'] == 'true' else None
    except Exception as e:
        print(f"WolframAlpha API error: {e}")
        return None

def wikipedia_search(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=2)
    except wikipedia.exceptions.PageError:
        return None
    
def internet_audio():
    text = Message_audio
    try:
        # Try Gemini search first
        result = gemini_search(text)
        print(result)
        if result:
            speak(result)
            return

        # Fall back to WolframAlpha
        result = wolfram_search(text)
        if result:
            speak(result)
            return

        # Fall back to Wikipedia
        result = wikipedia_search(text)
        if result:
            speak(result)
            return

        # If no results from any source
        speak('Sorry, I cannot understand or find relevant information.')
    except Exception as e:
        print(f"Exception in internet_audio(): {e}")
        speak('Sorry, I cannot understand or find relevant information.')
        reply()
              
def get_audio():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,duration=1)
        # r.energy_threshold()
        print("say anything : ")
        audio= r.listen(source)
        try:
            global text
            text = r.recognize_google(audio)
            print(text)
        except:
            print("sorry could not recognize ")
            speak("Could not understand ,please come again")
            reply()
    return text
        
def timeaudio():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    speak(current_time)    
        
def date_today():
    today = datetime.date.today()
    d1 = today.strftime("%d/%m/%Y")
    speak(d1)
   
def getResponse_audio(ints, intents_json):
    global tag
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    global result
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            speak(result)
            break    
    if(tag=='goodbye'):
        process_with_recognition()
    elif(tag=='internet_audio'):
        internet_audio()
    return result

def speak_recognition(name):
    sentences = [
        f"Hey, we have met before. Your name is {name} right!",
        f"I found you in my database. You are {name}, right!",
        f"Nice to see you again {name} !",
        f"Hello {name} ",
        f"It's good to have you back! You are {name}, right!",
        f"Ah, {name}! It's you.",
        f"Hey {name} Long time no see! ",
    ]
    speak(random.choice(sentences))

def speak_recognition_not_recognise():
    sentences_not_recognise = [
        f"Hey, have we met before ?",
        f"I didn't found you in my database. By the way",
        f"Nice to see you !",
        f"Hello there ",
    ]
    speak(random.choice(sentences_not_recognise))

def recognise():
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open("encodings.pickle", "rb").read())
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    fps = FPS().start()
    t0 = time.time()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            best_match_index = np.argmin(face_distances)
            probability = 1 - face_distances[best_match_index]  # Calculate probability

            if matches[best_match_index]:
                name = data["names"][best_match_index]
                print(f"Recognized {name} with {probability*100:.2f}% confidence")
                speak_recognition(name)
                vs.stop()
                return name
            else:
                print("Unknown Detected with confidence below threshold")
                speak_recognition_not_recognise()
                vs.stop()
                return None
            
            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        t1 = time.time()
        num_seconds = t1 - t0
        if num_seconds > 3:
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()
      

def reply():
       global Message_audio
       speak("What do you want to know")
       Message_audio = get_audio()
       Message_audio=Message_audio.lower()
       if 'bye-bye' in Message_audio:
         process_with_recognition()
       if 'time' in Message_audio:
         timeaudio()
       elif 'date' in Message_audio:
         date_today()
       elif 'suggestion' in Message_audio:
            speak('What suggestion would you like to give ?')
            suggestion_audio=get_audio()
            smtplibObj=smtplib.SMTP('smtp.gmail.com', 587)
            smtplibObj.ehlo()
            smtplibObj.starttls()
            smtplibObj.login("pavitnarang14@gmail.com" ,"glzininpfrxdjdao")
            smtplibObj.sendmail("pavitnarang14@gmail.com","pavitnarang0512@gmail.com",suggestion_audio)
            smtplibObj.quit()
            speak('Your suggestion has been sent')
       ints=predict_class(Message_audio, model)
       if ints==[]:
           ints=[{'intent': 'internet_audio', 'probability': '0.9999997615814209'}]             
       getResponse_audio(ints, intents) 
      
            

def person():
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()  # Assuming camera source is 2, adjust as needed
    time.sleep(2.0)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    person_detected = False

    try:
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=500)  # Resize frame for faster processing (optional)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                center = x + w // 2, y + h // 2
                radius = w // 2
                frame = cv2.circle(frame, center, radius, (0, 255, 0), 3)

            
            if len(faces) > 0:
                print("Person Detected")
                person_detected = True
                break
            else:
                print("Person Not Detected")
                person_detected = False
                break

    except Exception as e:
        print(f"Exception in person(): {e}")
        # Optionally handle or log the exception here

    finally:
        
        vs.stop()

    return person_detected

def person():
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()  # Assuming camera source is 2, adjust as needed
    time.sleep(2.0)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    person_detected = False

    try:
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=500)  # Resize frame for faster processing (optional)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                center = x + w // 2, y + h // 2
                radius = w // 2
                frame = cv2.circle(frame, center, radius, (0, 255, 0), 3)

            
            if len(faces) > 0:
                print("Person Detected")
                person_detected = True
                break
            else:
                print("Person Not Detected")
                person_detected = False
                break

    except Exception as e:
        print(f"Exception in person(): {e}")
        # Optionally handle or log the exception here

    finally:
        
        vs.stop()

    return person_detected

def process_with_recognition():
    if person()==True:
        name = recognise()
        if name:
            speak(f" My name is Charm")
            while True: 
                reply()
        else:
            speak(f" My name is Charm")
            while True:
                reply()
    elif person()==False:
        process_with_recognition()

print("CHARM is running")
while True:
    process_with_recognition()