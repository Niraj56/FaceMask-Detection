# =====================================================================
# Date : 23 oct 2021
# Title: face mask detector
# Author: Niraj Tiwari
# =====================================================================


import sys

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QCheckBox, QHBoxLayout
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap, QImage

import numpy as np
import cv2
import tensorflow as tf



# =====================================================================
# Use this function to perform histogram equalizer 
# use only for bad lightning conditions
def histogramEqu( in_im ):
    out_im = cv2.cvtColor( in_im, cv2.COLOR_BGR2YUV )
    y, u, v = cv2.split( out_im )
    y = cv2.equalizeHist( y )
    out_im = cv2.merge( (y, u, v))
    return cv2.cvtColor( out_im, cv2.COLOR_YUV2BGR )





# =====================================================================
class DetectorThread(QThread):
    image_signal = pyqtSignal( np.ndarray )
    
    # constructor
    def __init__( self, check_box ):
        super().__init__()
        self.running = True
        self.night_mode_check = check_box
    
    # override run
    def run(self):
        self.detectMasks()
     
    # call when closing the programm
    def stop( self ):
        self.running = False
        self.wait()



        
    # start detecting
    def detectMasks( self ):
        
        face_im_sz = 64
        
        cap = cv2.VideoCapture(0) # get the camera feed
        # load face cascade
        face_cascade = cv2.CascadeClassifier( 'cascades/haarcascade_frontalface_alt2.xml' )
        # load tf model
        model = tf.keras.models.load_model( 'saved_model/model_mn64' )
        
        # start the loop
        while self.running:
            ret, frame = cap.read()       
            
            if self.night_mode_check.isChecked():
                frame = histogramEqu( frame )
        
            gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            
            # detect faces
            faces = face_cascade.detectMultiScale( gray, scaleFactor=1.07, minNeighbors=6 )
            out_im = frame.copy()
            
            # go through each face
            for (x, y, w, h) in faces:
                 sz = max(w, h)
                
                 face_im = frame[ y:y+sz, x:x+sz] # crop the face
                 face_im = cv2.cvtColor(face_im, cv2.COLOR_BGR2RGB)
                 face_im = cv2.resize(face_im, (face_im_sz, face_im_sz)) / 255.0 # resize to (32, 32)
                 
                 # predict for the face_im
                 predict = model.predict( face_im.reshape((1, face_im_sz, face_im_sz, 3)))
                 
                 # get the confident level as a text 
                 text_percent = str( int( (np.max(predict) / 1.0) * 100) ) + '%'
                 
                 # determine wheather masked or not 
                 if np.argmax( predict ) == 0 and predict[0, 0] >= 0.8:
                     color_ = (0, 255, 0)
                     text = "Mask" 
                 else:
                     color_ = (0, 0, 255)
                     text = "No Mask" 
                 
                 # draw rectangle around the face
                 cv2.rectangle( out_im, (x, y), (x + w, y + h), color_, 2 )       
                 # draw text back ground
                 cv2.rectangle( out_im, (x - 1, y), (x + w + 1, y - 30), color_, -1)
                 # add the text
                 cv2.putText( out_im, text, (x + 2, y - 14), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2 )
                 # add the condident level
                 cv2.putText( out_im, text_percent, (x + 2, y - 3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 255, 255), thickness=1 )
        
            
            # send image to main thread
            self.image_signal.emit( out_im )

        cap.release()
        cv2.destroyAllWindows()



# =====================================================================
# the qt app
class FaceDetectorApp( QWidget ):
    
    # constructor
    def __init__( self ):
        super().__init__()
        
        # window dimension
        self.window_width = 800
        self.window_height = 600
        
        self.setFixedSize( self.window_width, self.window_height )
        self.setWindowTitle( 'Face Mask Detector' )
        
        # layout 
        self.layout = QHBoxLayout()
        
        # the video output
        self.image_label = QLabel(self)
        self.image_label.resize(self.window_width, self.window_height)
        self.layout.addWidget( self.image_label )
        
        # night mode check box
        self.night_mode_check_box = QCheckBox( "Night Mode" )
        self.layout.addWidget( self.night_mode_check_box )
        
        # create the detector thread
        self.thread = DetectorThread( self.night_mode_check_box )
        
        # connect image signal
        self.thread.image_signal.connect(self.update_image)
        
        self.setLayout( self.layout )
        
        # start the thread
        self.thread.start()
        
        self.show()
        
    # update function for the signal
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convertImage(cv_img)
        self.image_label.setPixmap(qt_img)
        
        
    # close event override
    def closeEvent(self, event):
        self.thread.stop()
        
        event.accept()
        
    # convert opencv image to qt image
    def convertImage(self, cv_img):
        rgb_im = cv2.cvtColor( cv_img, cv2.COLOR_BGR2RGB )
        h, w, ch = rgb_im.shape
        convert_to_Qt_format = QImage( rgb_im.data, w, h, ch * w, QImage.Format_RGB888 )
        temp = convert_to_Qt_format.scaled( self.window_width, self.window_height, Qt.KeepAspectRatio )
        return QPixmap.fromImage( temp )
        
        


 
# =====================================================================
if __name__ == '__main__':
    app = QApplication( sys.argv )
    face_detector = FaceDetectorApp()
    exit( app.exec_() )