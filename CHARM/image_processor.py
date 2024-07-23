def person():
    import cv2
    imcap = cv2.VideoCapture(0)  
    imcap.set(3, 640) 
    imcap.set(4, 480) # set height as 480
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_frontalface_default.xml" )



    while True:
        success, img = imcap.read() # capture frame from video
        # converting image from color to grayscale 
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  
    
        for (x, y, w, h) in faces:
            centre=x+w//2, y+h//2
            radius=w//2
            img = cv2.circle(img, centre,radius, (0, 255,  0), 3)
        

        cv2.imshow('face_detect', img)
        if len(faces)>0:
            print("Person Detected")
            return 1
        else:
            return 0
            
person()