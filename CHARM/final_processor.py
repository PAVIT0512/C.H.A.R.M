from email import message
from re import T
from gtts import gTTS
import speech_recognition as sr
import playsound
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
import tkinter as tk
from tkinter import ttk
from tkinter import END
from tkinter import INSERT
import smtplib
import face_recognition
import cv2
import csv
# from PIL import Image, ImageTk
from time import sleep
from csv import writer


from keras.models import load_model
model = load_model('chatbot_model2.h5')
import json
import random
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
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.8
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def speak(text):
    tts=gTTS(text=text, lang="en")
    file="sound.mp3"
    tts.save(file)
    playsound.playsound(file)
    os.remove("sound.mp3")

def internet():
    app_id = "44YXHU-TV6AJRQ6HT" 
    text=Message
    try:
        client = wolframalpha.Client(app_id) 

        res = client.query(text) 

        answer = next(res.results).text 
        msgs.insert(tk.END, "       "+ answer)
        speak(answer)

    except:
        result = wikipedia.summary(text, sentences = 1)
        msgs.insert(tk.END,"       "+result)
        speak(result)

def internet_audio():
    app_id = "44YXHU-TV6AJRQ6HT" 
    text=Message_audio
    try:
        client = wolframalpha.Client(app_id) 

        res = client.query(text) 

        answer = next(res.results).text 
        msgs_audio.insert(tk.END,"       "+answer)
        speak(answer)
    except:
        result = wikipedia.summary(text, sentences = 1)
        msgs_audio.insert(tk.END,  "       "+result)
        speak(result)
        

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
            
        except:
            print("sorry could not recognize ")
            exit()
    return text


def time():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    msgs.insert(tk.END, "       "+ current_time)
    speak(current_time)    
def time_audio():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    msgs_audio.insert(tk.END,  "       "+current_time)
    speak(current_time)     

def date_today():
    today = datetime.date.today()
    d1 = today.strftime("%d/%m/%Y")
    msgs.insert(tk.END,  d1)
    speak(d1)
def date_today_audio():
    today = datetime.date.today()
    d1 = today.strftime("%d/%m/%Y")
    msgs_audio.insert(tk.END,  d1)
    speak(d1)    

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    global result
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            msgs.insert(tk.END, "Charm: " + result)
            speak(result)
            break    
    if(tag=='goodbye'):
        result= random.choice(["See you!", "Have a nice day", "Sure Bye", "Sayoonara", "Sure dude", "Ciao"])
        msgs.insert(tk.END, "Charm: " + result)
        speak(result)
        exit()
    elif(tag=='internet'):
        internet()  
    return result

def getResponse_audio(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    global result
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            msgs_audio.insert(tk.END, "Charm: " + result)
            speak(result)
            break    
    if(tag=='goodbye'):
        result= random.choice(["See you!", "Have a nice day", "Sure Bye", "Sayoonara", "Sure dude", "Ciao"])
        msgs_audio.insert(tk.END, "Charm: " + result)
        speak(result)
        exit()
    elif(tag=='internet_audio'):
        internet_audio()
    return result


    
print("CHARM is running")

window= tk.Tk()
window.title('Welcome to CHARM')
window.geometry("925x500")
window.configure(bg="#fff")
window.resizable(True,True)

def Options():
    def Chat():
        btn_chatbot.destroy()
        btn_tour.destroy()
        btn_attendance.destroy()
        btn_suggestion.destroy()
        btn_information.destroy()
        btn_clubs.destroy()
        photoL.destroy()

        def back_to_options():
            btn_textmain.destroy()
            btn_speechmain.destroy()
            btn_back_to_options.destroy()
            phototext.destroy()
            Options()
        def text():
            def chatbox():
                global Message
                Message=textF.get()
                Message=Message.lower()
                ints=predict_class(Message, model)
                msgs.insert(tk.END, "You: " + Message)
                if ints==[]:
                    ints=[{'intent': 'internet', 'probability': '0.9999997615814209'}]
                res=getResponse(ints, intents) 
                
                if 'time' in Message:
                    time()
                elif 'date' in Message:
                    date_today()
                textF.delete(0,tk.END)
                msgs.yview(tk.END)
            def enter_function(event):
                btn_text.invoke()

            btn_textmain.destroy()
            btn_speechmain.destroy()
            btn_back_to_options.destroy()

            global expression
            expression = " "          # global variable 
            def press(num):
                global expression
                expression=expression + str(num)
                name_equation.set(expression)  

            def clear():
                global expression
                expression = " "
                name_equation.set(expression)

            btn_textmain.destroy()
            btn_speechmain.destroy()
            btn_back_to_options.destroy()
            phototext.destroy()
            name_equation = tk.StringVar()
            img = tk.PhotoImage(file="charm.png")
            photochatL=tk.Label(window,image=img,bg='white')
            photochatL.image=img
            photochatL.pack(pady=10)
            frame_chatbot_scrollbar=tk.Scrollbar(window)
            global msgs
            msgs=tk.Listbox(window,width=90,height=10,yscrollcommand=frame_chatbot_scrollbar.set)
            frame_chatbot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            frame_chatbot_scrollbar.config(command=msgs.yview)
            msgs.pack(side=tk.LEFT, fill=tk.BOTH,  pady=10)
            msgs.place(x=60,y=80)
            textF=tk.Entry(window,width=82,font=("Verdana",12),textvariable = name_equation)
            textF.pack(side=tk.BOTTOM,fill=tk.BOTH, pady=10)
            textF.place(x=60,y=260)
            msgs.insert(tk.END, "Welcome to the CHARM")
            msgs.insert(tk.END, "Say anything:")
            q = ttk.Button(window,text = 'Q' , width = 4, command = lambda : press('q'))
            q.place(x=50,y=300)

            w = ttk.Button(window,text = 'W' , width = 4, command = lambda : press('w'))
            w.place(x=100,y=300)

            E = ttk.Button(window,text = 'E' , width = 4, command = lambda : press('e'))
            E.place(x=150,y=300)

            R = ttk.Button(window,text = 'R' , width = 4, command = lambda : press('r'))
            R.place(x=200,y=300)

            T = ttk.Button(window,text = 'T' , width = 4, command = lambda : press('t'))
            T.place(x=250,y=300)

            Y = ttk.Button(window,text = 'Y' , width = 4, command = lambda : press('y'))
            Y.place(x=300,y=300)

            U = ttk.Button(window,text = 'U' , width = 4, command = lambda : press('u'))
            U.place(x=350,y=300)

            I = ttk.Button(window,text = 'I' , width = 4, command = lambda : press('i'))
            I.place(x=400,y=300)

            O = ttk.Button(window,text = 'O' , width = 4, command = lambda : press('o'))
            O.place(x=450,y=300)

            P = ttk.Button(window,text = 'P' , width = 4 ,command = lambda : press('p'))
            P.place(x=500,y=300)

            cur = ttk.Button(window,text = '{' , width = 4, command = lambda : press('{'))
            cur.place(x=550,y=300)

            cur_c = ttk.Button(window,text = '}' , width = 4, command = lambda : press('}'))
            cur_c.place(x=600,y=300)

            back_slash = ttk.Button(window,text = '\\' , width = 4, command = lambda : press('\\'))
            back_slash.place(x=650,y=300)

            delete = ttk.Button(window,text = 'Del' , width = 6, command = clear)
            delete.place(x=700,y=300)

# Second Line Button

            A = ttk.Button(window,text = 'A' , width = 4, command = lambda : press('a'))
            A.place(x=50,y=330)

            S = ttk.Button(window,text = 'S' , width = 4, command = lambda : press('s'))
            S.place(x=100,y=330)

            D = ttk.Button(window,text = 'D' , width = 4, command = lambda : press('d'))
            D.place(x=150,y=330)

            F = ttk.Button(window,text = 'F' , width = 4, command = lambda : press('f'))
            F.place(x=200,y=330)


            G = ttk.Button(window,text = 'G' , width = 4, command = lambda : press('g'))
            G.place(x=250,y=330)


            H = ttk.Button(window,text = 'H' , width = 4, command = lambda : press('h'))
            H.place(x=300,y=330)


            J = ttk.Button(window,text = 'J' , width = 4, command = lambda : press('j'))
            J.place(x=350,y=330)


            K = ttk.Button(window,text = 'K' , width = 4, command = lambda : press('k'))
            K.place(x=400,y=330)

            L = ttk.Button(window,text = 'L' , width = 4, command = lambda : press('l'))
            L.place(x=450,y=330)


            semi_co = ttk.Button(window,text = ';' , width = 4, command = lambda : press(';'))
            semi_co.place(x=500,y=330)


            d_colon = ttk.Button(window,text = '"' , width = 4, command = lambda : press('"'))
            d_colon.place(x=550,y=330)

            open_b = ttk.Button(window,text = '(' , width = 4, command = lambda : press('('))
            open_b.place(x= 600 , y=330 )

            close_b = ttk.Button(window,text = ')' , width = 4, command = lambda : press(')'))
            close_b.place(x = 650 , y = 330 )   

# third line Button

            Z = ttk.Button(window,text = 'Z' , width = 4, command = lambda : press('z'))
            Z.place(x=50,y=360)


            X = ttk.Button(window,text = 'X' , width = 4, command = lambda : press('x'))
            X.place(x=100,y=360)


            C = ttk.Button(window,text = 'C' , width = 4, command = lambda : press('c'))
            C.place(x=150,y=360)


            V = ttk.Button(window,text = 'V' , width = 4, command = lambda : press('v'))
            V.place(x=200,y=360)

            B = ttk.Button(window, text= 'B' , width = 4 , command = lambda : press('b'))
            B.place(x=250,y=360)


            N = ttk.Button(window,text = 'N' , width = 4, command = lambda : press('n'))
            N.place(x=300,y=360)


            M = ttk.Button(window,text = 'M' , width = 4, command = lambda : press('m'))
            M.place(x=350,y=360)


            left = ttk.Button(window,text = '<' , width = 4, command = lambda : press('<'))
            left.place(x=400,y=360)


            right = ttk.Button(window,text = '>' , width = 4, command = lambda : press('>'))
            right.place(x=450,y=360)


            slas = ttk.Button(window,text = '/' , width = 4, command = lambda : press('/'))
            slas.place(x=500,y=360)


            q_mark = ttk.Button(window,text = '?' , width = 4, command = lambda : press('?'))
            q_mark.place(x=550,y=360)


            coma = ttk.Button(window,text = ',' , width = 4, command = lambda : press(','))
            coma.place(x=600,y=360)

            dot = ttk.Button(window,text = '.' , width = 4, command = lambda : press('.'))
            dot.place(x=650,y=360)

#Fourth Line Button

            space = ttk.Button(window,text = '   Space   ' , width = 18,command = lambda : press(' '))
            space.place(x=300,y=390)
            
            window.bind('<Return>', enter_function)
            btn_text=tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0,text="Text",font=("Verdana",12),command=chatbox)
            btn_text.place(x=300, y=450)
            def back_frame1():
                frame_chatbot_scrollbar.destroy()
                msgs.destroy()
                btn_text.destroy()
                textF.destroy()
                photochatL.destroy()
                btn_back.destroy()
                q.destroy()
                w.destroy()
                E.destroy()
                R.destroy()
                T.destroy()
                Y.destroy()
                U.destroy()
                I.destroy()
                O.destroy()
                P.destroy()
                A.destroy()
                S.destroy()
                D.destroy()
                F.destroy()
                G.destroy()
                H.destroy()
                J.destroy()
                K.destroy()
                L.destroy()
                Z.destroy()
                X.destroy()
                C.destroy()
                V.destroy()
                B.destroy()
                N.destroy()
                M.destroy()
                cur.destroy()
                cur_c.destroy()
                back_slash.destroy()
                delete.destroy()
                semi_co.destroy()
                open_b.destroy()
                close_b.destroy()
                d_colon.destroy()
                left.destroy()
                right.destroy()
                q_mark.destroy()
                coma.destroy()
                slas.destroy()
                dot.destroy()
                space.destroy()
                Chat()
            btn_back=tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0,text="Back",font=("Verdana",12),command=back_frame1)
            btn_back.place(x=10, y=10)
        def speech():
            btn_textmain.destroy()
            btn_speechmain.destroy()
            btn_back_to_options.destroy()
            phototext.destroy()
            def reply():
                msgs_audio.insert(tk.END, "Say anything:")
                global Message_audio
                Message_audio = get_audio()
                Message_audio=Message_audio.lower()
                msgs_audio.insert(tk.END ,"You: "+Message_audio)
                # message=input("You: ")
                ints=predict_class(Message_audio, model)
                if ints==[]:
                    ints=[{'intent': 'internet_audio', 'probability': '0.9999997615814209'}]
                 
                res=getResponse_audio(ints, intents) 
                if 'time' in Message_audio:
                    time_audio()
                elif 'date' in Message_audio:
                    date_today_audio()
                msgs_audio.yview(tk.END)
            def enter_function_speech(event):
                btn_speech.invoke()
            def back_frame_speech():
                frame_chatbot_scrollbar.destroy()
                msgs_audio.destroy()
                btn_speech.destroy()
                photochatL.destroy()
                btn_back.destroy()
                Chat()
            img = tk.PhotoImage(file="charm.png")
            photochatL=tk.Label(window,image=img,bg='white')
            photochatL.image=img
            photochatL.pack(pady=10)
            frame_chatbot_scrollbar=tk.Scrollbar(window)
            global msgs_audio
            msgs_audio=tk.Listbox(window,width=90,height=15,yscrollcommand=frame_chatbot_scrollbar.set)
            frame_chatbot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            frame_chatbot_scrollbar.config(command=msgs_audio.yview)
            msgs_audio.pack(side=tk.LEFT, fill=tk.BOTH,  pady=10)
            msgs_audio.place(x=60,y=80)
            msgs_audio.insert(tk.END, "Welcome to the CHARM")
            window.bind('<Return>', enter_function_speech)
            btn_speech=tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0,text="Speak",font=("Verdana",12),command=reply)
            btn_speech.place(x=500, y=405)
            btn_back=tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0,text="Back",font=("Verdana",12),command=back_frame_speech)
            btn_back.place(x=10, y=10)



        img= tk.PhotoImage(file="charm.png")
        phototext=tk.Label(window,image=img,bg='white')
        phototext.image=img
        phototext.pack(pady=10)
        btn_textmain=tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0,text="Text",font=("Verdana",12),command=text)
        btn_textmain.place(x=250, y=250)
        btn_speechmain=tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0,text="Speech",font=("Verdana",12),command=speech)
        btn_speechmain.place(x=500, y=250)
        btn_back_to_options=tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0,text="Back",font=("Verdana",12),command=back_to_options)
        btn_back_to_options.place(x=10, y=10)
    def Suggestion_btn():
        btn_chatbot.destroy()
        btn_tour.destroy()
        btn_attendance.destroy()
        btn_suggestion.destroy()
        btn_information.destroy()
        btn_clubs.destroy()
        photoL.destroy()
        global window
        global exp
        exp = " "          # global variable 
# showing all data in display 

        def press(num):
            global exp
            exp=exp + str(num)
            name_equation.set(exp)  
# end 

# function clear button

        def clear():
            global exp
            exp = " "
            name_equation.set(exp)

        def send_suggestion():
            smtplibObj=smtplib.SMTP('smtp.gmail.com', 587)
            smtplibObj.ehlo()
            smtplibObj.starttls()
            smtplibObj.login("pavitnarang14@gmail.com" ,"glzininpfrxdjdao")
            smtplibObj.sendmail("pavitnarang14@gmail.com","pavitnarang0512@gmail.com",frame_suggestion_entry.get())
            smtplibObj.quit()
            frame_suggestion_entry.delete(0,tk.END)
        
        def enter_function_suggestion(event):
            frame_suggestion_btn.invoke()
            
        def back_frame2():
            frame_suggestion_entry.destroy()
            frame_suggestion_btn.destroy()   
            btn_back_2.destroy()
            photosuggestionL.destroy()
            q.destroy()
            w.destroy()
            E.destroy()
            R.destroy()
            T.destroy()
            Y.destroy()
            U.destroy()
            I.destroy()
            O.destroy()
            P.destroy()
            A.destroy()
            S.destroy()
            D.destroy()
            F.destroy()
            G.destroy()
            H.destroy()
            J.destroy()
            K.destroy()
            L.destroy()
            Z.destroy()
            X.destroy()
            C.destroy()
            V.destroy()
            B.destroy()
            N.destroy()
            M.destroy()
            cur.destroy()
            cur_c.destroy()
            back_slash.destroy()
            delete.destroy()
            semi_co.destroy()
            open_b.destroy()
            close_b.destroy()
            d_colon.destroy()
            left.destroy()
            right.destroy()
            q_mark.destroy()
            coma.destroy()
            slas.destroy()
            dot.destroy()
            space.destroy()
            Options()

        window.bind('<Return>',enter_function_suggestion) 
        frame_suggestion_btn= tk.Button(window,width=20,pady=7, text="Send",bg='#57a1f8', fg='white', border=0,command=send_suggestion)
        frame_suggestion_btn.place(x=300, y=450)    
        btn_back_2=tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0,text="Back",font=("Verdana",12),command=back_frame2)
        btn_back_2.place(x=10, y=10)
        img = tk.PhotoImage(file="charm.png")
        photosuggestionL=tk.Label(window,image=img,bg='white')
        photosuggestionL.image=img
        photosuggestionL.pack(pady=10)
        name_equation = tk.StringVar()
        frame_suggestion_entry=tk.Entry(window,width=82,font=("Verdana",12),textvariable = name_equation)
        frame_suggestion_entry.place(x=50,y=200)

# end entry box

# add all button line wise 

# First Line Button

        q = ttk.Button(window,text = 'Q' , width = 4, command = lambda : press('Q'))
        q.place(x=50,y=300)

        w = ttk.Button(window,text = 'W' , width = 4, command = lambda : press('W'))
        w.place(x=100,y=300)

        E = ttk.Button(window,text = 'E' , width = 4, command = lambda : press('E'))
        E.place(x=150,y=300)

        R = ttk.Button(window,text = 'R' , width = 4, command = lambda : press('R'))
        R.place(x=200,y=300)

        T = ttk.Button(window,text = 'T' , width = 4, command = lambda : press('T'))
        T.place(x=250,y=300)

        Y = ttk.Button(window,text = 'Y' , width = 4, command = lambda : press('Y'))
        Y.place(x=300,y=300)

        U = ttk.Button(window,text = 'U' , width = 4, command = lambda : press('U'))
        U.place(x=350,y=300)

        I = ttk.Button(window,text = 'I' , width = 4, command = lambda : press('I'))
        I.place(x=400,y=300)

        O = ttk.Button(window,text = 'O' , width = 4, command = lambda : press('O'))
        O.place(x=450,y=300)

        P = ttk.Button(window,text = 'P' , width = 4 ,command = lambda : press('P'))
        P.place(x=500,y=300)

        cur = ttk.Button(window,text = '{' , width = 4, command = lambda : press('{'))
        cur.place(x=550,y=300)

        cur_c = ttk.Button(window,text = '}' , width = 4, command = lambda : press('}'))
        cur_c.place(x=600,y=300)

        back_slash = ttk.Button(window,text = '\\' , width = 4, command = lambda : press('\\'))
        back_slash.place(x=650,y=300)

        delete = ttk.Button(window,text = 'Del' , width = 6, command = clear)
        delete.place(x=700,y=300)

# Second Line Button

        A = ttk.Button(window,text = 'A' , width = 4, command = lambda : press('A'))
        A.place(x=50,y=330)

        S = ttk.Button(window,text = 'S' , width = 4, command = lambda : press('S'))
        S.place(x=100,y=330)

        D = ttk.Button(window,text = 'D' , width = 4, command = lambda : press('D'))
        D.place(x=150,y=330)

        F = ttk.Button(window,text = 'F' , width = 4, command = lambda : press('F'))
        F.place(x=200,y=330)


        G = ttk.Button(window,text = 'G' , width = 4, command = lambda : press('G'))
        G.place(x=250,y=330)


        H = ttk.Button(window,text = 'H' , width = 4, command = lambda : press('H'))
        H.place(x=300,y=330)


        J = ttk.Button(window,text = 'J' , width = 4, command = lambda : press('J'))
        J.place(x=350,y=330)


        K = ttk.Button(window,text = 'K' , width = 4, command = lambda : press('K'))
        K.place(x=400,y=330)

        L = ttk.Button(window,text = 'L' , width = 4, command = lambda : press('L'))
        L.place(x=450,y=330)


        semi_co = ttk.Button(window,text = ';' , width = 4, command = lambda : press(';'))
        semi_co.place(x=500,y=330)


        d_colon = ttk.Button(window,text = '"' , width = 4, command = lambda : press('"'))
        d_colon.place(x=550,y=330)

        open_b = ttk.Button(window,text = '(' , width = 4, command = lambda : press('('))
        open_b.place(x= 600 , y=330 )

        close_b = ttk.Button(window,text = ')' , width = 4, command = lambda : press(')'))
        close_b.place(x = 650 , y = 330 )

# third line Button

        Z = ttk.Button(window,text = 'Z' , width = 4, command = lambda : press('Z'))
        Z.place(x=50,y=360)


        X = ttk.Button(window,text = 'X' , width = 4, command = lambda : press('X'))
        X.place(x=100,y=360)


        C = ttk.Button(window,text = 'C' , width = 4, command = lambda : press('C'))
        C.place(x=150,y=360)


        V = ttk.Button(window,text = 'V' , width = 4, command = lambda : press('V'))
        V.place(x=200,y=360)

        B = ttk.Button(window, text= 'B' , width = 4 , command = lambda : press('B'))
        B.place(x=250,y=360)


        N = ttk.Button(window,text = 'N' , width = 4, command = lambda : press('N'))
        N.place(x=300,y=360)


        M = ttk.Button(window,text = 'M' , width = 4, command = lambda : press('M'))
        M.place(x=350,y=360)


        left = ttk.Button(window,text = '<' , width = 4, command = lambda : press('<'))
        left.place(x=400,y=360)


        right = ttk.Button(window,text = '>' , width = 4, command = lambda : press('>'))
        right.place(x=450,y=360)


        slas = ttk.Button(window,text = '/' , width = 4, command = lambda : press('/'))
        slas.place(x=500,y=360)


        q_mark = ttk.Button(window,text = '?' , width = 4, command = lambda : press('?'))
        q_mark.place(x=550,y=360)


        coma = ttk.Button(window,text = ',' , width = 4, command = lambda : press(','))
        coma.place(x=600,y=360)

        dot = ttk.Button(window,text = '.' , width = 4, command = lambda : press('.'))
        dot.place(x=650,y=360)

#Fourth Line Button

        space = ttk.Button(window,text = '   Space   ' , width = 18,command = lambda : press(' '))
        space.place(x=300,y=390)
    def Information_btn():
        btn_chatbot.destroy()
        btn_tour.destroy()
        btn_information.destroy()
        btn_clubs.destroy()
        btn_attendance.destroy()
        btn_suggestion.destroy()
        photoL.destroy()
        def back_info():
            msgs_info.destroy()
            btn_back_info.destroy()
            photo_info.destroy()
            Options()

        img= tk.PhotoImage(file="charm.png")
        photo_info=tk.Label(window,image=img,bg='white')
        photo_info.image=img
        photo_info.pack(pady=10)
        msgs_info=tk.Listbox(window,width=120,height=20)
        msgs_info.pack(side=tk.LEFT, fill=tk.BOTH,  pady=10)
        msgs_info.place(x=50,y=100)
        msgs_info.insert(tk.END, "C.H.A.R.M (Computerised Hardware Automated Reception Manager) is a computerised robotic program that simulates and ") 
        msgs_info.insert(tk.END,"processes human conversation (either written or spoken), allowing us to interact with digital devices as if we are communicating with a")
        msgs_info.insert(tk.END," real person. It is built with the objective of monitoring the reception area of our Institute with  certain features that aim to have ")
        msgs_info.insert(tk.END,"conversation with individuals,  interact with them and display details of college as per their requirement. CHARM has been programmed ")
        msgs_info.insert(tk.END,"in such a sophisticated way that it will be charming and attractive enough with his facial expressions and voice that everyone is")
        msgs_info.insert(tk.END,"willing to see in a robot. CHARM is a combination of automated processes and various technologies in the field of Artificial Intelligence,")
        msgs_info.insert(tk.END," Machine Learning, Robotics, Electronics,Computer Science and Web Development which makes it facinating with all the features ")
        msgs_info.insert(tk.END,"that a robotic chatbot should have.")
        btn_back_info = tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0, text='Back',command=back_info)
        btn_back_info.place(x=10, y=10)
    def Clubs_btn():
        btn_chatbot.destroy()
        btn_tour.destroy()
        btn_information.destroy()
        btn_clubs.destroy()
        btn_attendance.destroy()
        btn_suggestion.destroy()
        photoL.destroy()
        def back_clubs():
            edc_label.destroy()
            gdsc_label.destroy()
            btn_back_clubs.destroy()
            photoclubs.destroy()
            edc_img.destroy()
            gdsc_img.destroy()
            Options()
        img= tk.PhotoImage(file="charm.png")
        photoclubs=tk.Label(window,image=img,bg='white')
        photoclubs.image=img
        photoclubs.pack(pady=10)
        btn_back_clubs = tk.Button(window,width=20,pady=7,bg='#57a1f8', fg='white', border=0, text='Back',command=back_clubs)
        btn_back_clubs.place(x=10, y=10)
        edc_label = tk.Label(window, text="EDC",bg='white', fg='black', font=('Helvetica', 12, 'bold'))
        edc_label.place(x=100, y=100)
        gdsc_label = tk.Label(window, text="GDSC",bg='white', fg='black', font=('Helvetica', 12, 'bold'))
        gdsc_label.place(x=500, y=100)
        edc_info_img= tk.PhotoImage(file="edc.png")
        edc_img=tk.Label(window,image=edc_info_img,bg='white')
        edc_img.image=edc_info_img
        edc_img.place(x=300,y=100)
        gdsc_info_img= tk.PhotoImage(file="gdsc.png")
        gdsc_img=tk.Label(window,image=gdsc_info_img,bg='white')
        gdsc_img.image=gdsc_info_img
        gdsc_img.place(x=700,y=100)




    btn_chatbot = tk.Button(window,width=35,pady=7, text="Chatbot",bg='#57a1f8', fg='white', border=0,command=Chat)
    btn_chatbot.place(x=100, y=205)
    btn_tour = tk.Button(window,width=35,pady=7,bg='#57a1f8', fg='white', border=0, text='Acropolis Virtual Tour')
    btn_tour.place(x=500, y=205)
    btn_information = tk.Button(window,width=5,pady=7,bg='#57a1f8', fg='white', border=0, text='i',command=Information_btn)
    btn_information.place(x=850, y=30)
    btn_clubs = tk.Button(window,width=35,pady=7,bg='#57a1f8', fg='white', border=0, text='Our Clubs',command=Clubs_btn)
    btn_clubs.place(x=100, y=405)
    btn_attendance = tk.Button(window, text='Attendance',width=35,pady=7,bg='#57a1f8', fg='white', border=0)
    btn_attendance.place(x=100, y=305)
    btn_suggestion = tk.Button(window, text='Suggestion Box',width=35,pady=7,bg='#57a1f8', fg='white', border=0,command=Suggestion_btn)
    btn_suggestion.place(x=500, y=305)
    img= tk.PhotoImage(file="charm.png")
    photoL=tk.Label(window,image=img,bg='white')
    photoL.image=img
    photoL.pack(pady=10)

Options()
window.mainloop()