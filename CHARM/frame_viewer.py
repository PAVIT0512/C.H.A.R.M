
import vlc
import time

# creating vlc media player object
media_player = vlc.MediaPlayer()
# media object
media = vlc.Media("Robotic Eye.mp4")
media1 = vlc.Media("frame.mp4")

# setting media to the media player
media_player.set_fullscreen(True)
def mediaplayerdefaultset():
        media_player.set_media(media)
        media_player.play()
        time.sleep(6)

def mediaplayerset():
        media_player.set_media(media1)
        media_player.play()
        time.sleep(6)


