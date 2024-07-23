import frame_viewer as frview
import image_processor as iprocess
while True:
        if iprocess.person()==1:
            frview.mediaplayerdefaultset()
        elif iprocess.person()==0:
            frview.mediaplayerset()  