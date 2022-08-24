# =====================================================================
# Date : 22 oct 2021
# Title: face mask detectro without qt
# Author: Niraj Tiwari
# =====================================================================


import cv2
import numpy as np
import tensorflow as tf


# Use this function to perform histogram equalizer 
# use only for bad lightning conditions
def histogramEqu( in_im ):
    out_im = cv2.cvtColor( in_im, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split( out_im )
    y = cv2.equalizeHist( y )
    out_im = cv2.merge( (y, u, v))
    return cv2.cvtColor( out_im, cv2.COLOR_YUV2BGR)      

def load_image( path ): 
    return cv2.imread( path )


if __name__ == '__main__':
    
    face_im_sz = 64
    
    cap = cv2.VideoCapture(0)
       
    face_cascade = cv2.CascadeClassifier( 'cascades/haarcascade_frontalface_alt2.xml' )
   
    model = tf.keras.models.load_model( "saved_model/model_mn64" )
    
    while (True):
        ret, frame = cap.read()      
       
        # only for bad lightning conditions
        #frame = histogramEqu()
        
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        faces = face_cascade.detectMultiScale( gray, scaleFactor=1.12, minNeighbors=5 )
        out_im = frame.copy()
        
        for (x, y, w, h) in faces:
             sz = max(w, h)
            
             face_im = frame[ y:y+sz, x:x+sz]
             face_im = cv2.cvtColor(face_im, cv2.COLOR_BGR2RGB)       
             face_im = cv2.resize(face_im, (face_im_sz, face_im_sz)) / 255.0
             predict = model.predict( face_im.reshape((1, face_im_sz, face_im_sz, 3)))
             
             text_percent = str( int( (np.max(predict) / 1.0) * 100) ) + '%'
             
             
             if np.argmax( predict ) == 0 and predict[0, 0] >= 0.8:
                 color_ = (0, 255, 0)
                 text = "Mask" 
             else:
                 color_ = (0, 0, 255)
                 text = "No Mask" 
                 
             
             
             cv2.rectangle( out_im, (x, y), (x + w, y + h), color_, 2 )
             cv2.rectangle( out_im, (x - 1, y), (x + w + 1, y - 30), color_, -1)
             cv2.putText( out_im, text, (x + 2, y - 14), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2 )
             cv2.putText( out_im, text_percent, (x + 2, y - 3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 255, 255), thickness=1 )
            
        cv2.putText( out_im, "Press Q to Close", (20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1 )
        cv2.imshow( 'Face Mask Detector', out_im)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
