import threading
import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name])

if __name__ == "__main__":
    script1_thread = threading.Thread(target=run_script, args=("final_processor5.py",))
    script2_thread = threading.Thread(target=run_script, args=("video_player.py",))

    script1_thread.start()
    script2_thread.start()

    

    print("Both scripts have finished executing.")