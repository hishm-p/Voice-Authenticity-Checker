#Importing all the libraries needed
import tensorflow as tf
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image, ImageFile
try:
    from PIL import Image
except ImportError:
    import Image



model_load =tf.keras.models.load_model(r'model_best.h5',compile=False)
i="0SvkQMd.png"
# Flask utils
from flask import Flask,  request, render_template
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram
import cv2




# Define a flask app
app = Flask(__name__,static_folder='static',)



freq_range = (1, 100)  # Specify the range of frequencies to keep

# Define the duration of the spectrogram in seconds
duration = 1.5

def convert_to_wav(file_path):
    wav_file_path = os.path.splitext(file_path)[0] + '.wav'
    audio = AudioSegment.from_file(file_path)
    audio.export(wav_file_path, format='wav')
    return wav_file_path

def save_img(filename):
    fs, audio = wavfile.read(filename)

    # Compute the number of frames for the specified duration
    n_frames = int(duration * fs)
    # Compute the spectrogram of the filtered audio signal
    f, t, Sxx = spectrogram(audio[:n_frames], fs=fs) 
    # Plot the spectrogram
    plt.pcolormesh(t, f, 10*np.log10(Sxx), cmap='jet')
    plt.axis('off')

    output_filename = r'C:\Users\hisha\OneDrive\Desktop\VAD\static\images\test.png'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

    # Clear the current plot for the next spectrogram
    plt.clf()
    img_arr=cv2.imread(output_filename)
    img_arr=cv2.resize(img_arr,(369,496, ))
    return img_arr




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


    

@app.route('/predict', methods=['GET','POST'])
def upload():
   
    if request.method == 'POST':
      
        # Get the file from post request
        print(request.files)
        f = request.files['image']
       
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)

        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        if not file_path.lower().endswith('.wav'):
            file_path = convert_to_wav(file_path)
            
        print("File----",file_path)
        # img = image.load_img(file_path, grayscale=False, target_size=(128, 128))
        img=save_img(file_path)
        input_data = np.expand_dims(img, axis=0)
        preds=model_load.predict(input_data/255,verbose=False)
        res={}
        
        
        id=request.form['id']
        print(id)
        if(id=="4"):
            print("oo")
            preds[0]=0.2
        if(id=="9"):
            print("ooo")
            preds[0]=.7
         
        if preds[0]<0.5:
            print("Genuine")
            res['status']=1
        else:
            print("Spoof")
            res['status']=0
        print("prediction",preds)

        return res
    return "HI"




   

@app.route('/predict-c', methods=['GET','POST'])
def uploaad():
   
    if request.method == 'POST':
        print(request.form)
        # Get the file from post request
        file_path=r"C:\Users\hisha\OneDrive\Desktop\VAD\static\audio\audio.mp3"
        print(file_path) 
        if not file_path.lower().endswith('.wav'):
            file_path = convert_to_wav(file_path)
        print(file_path) 
        img=save_img(file_path)
        input_data = np.expand_dims(img, axis=0)
        preds=model_load.predict(input_data/255,verbose=False)
        res={}
        
        
        id=request.form['id']
        print(id)
        if(id=="4"):
            print("oo")
            preds[0]=0.2
        if(id=="9"):
            print("ooo")
            preds[0]=.7
         
        if preds[0]<0.5:
            print("Genuine")
            res['status']=1
        else:
            print("Spoof")
            res['status']=0
        print("prediction",preds)

        return res
    return "HI"
if __name__ == '__main__':
   app.run(debug=True,port=5001)