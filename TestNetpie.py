import microgear.client as netpie
import time
import base64
from PIL import Image
from io import BytesIO
import zlib
import json
import requests

key = 'V9uOQBaAfxHM1YP'
secret = 'xGDHtxzG9kE71OrkVvldLjsWt'
app = 'SeniorXD'

netpie.create(key,secret,app,{'debugmode': True})
connected = False

def connection():
    global connected
    connected = True
    print("Connected")

def subscription(topic,msg):
    global this_role,ready_to_receive,img_status
    ask = msg[2:-1]
    #print(msg)
    if this_role == 'receiver' :
        if not ready_to_receive :
            if ask=='ruok' :
                print('Received sender message, sender is ready.')
                netpie.chat(those_name,'iamok')
                ready_to_receive = True
        else:
            if len(msg)>1000:
                print("...Receiving image data...")
                print('img len:',len(msg))
                decode_base64(msg, 'test.jpg') # don't need to save on disk
                print("Image received!")
                print("--------------------------")
                img_status = True

                files = {'file': open('test.jpg', 'rb')}
                url = "http://localhost:5000/uploadImage"
                requests.post(url, files=files)
                print("process is done")                
            elif img_status:
                print("...Receiving prediction data...")
                print(msg,len(msg))
                print("Data received!")
                print("==========================")
                print("==========================")
                img_status = False
                ready_to_receive = False
            else:
                ready_to_receive = False
    else :
        print('sdfsdfsf')
        print(topic+":"+msg)

def callback_error(msg) :
    print(msg)

def callback_reject(msg) :
    print (msg)
    print ("Script exited")
    exit(0) 

def encode_base64(img_data):
    encoded = None
    try:
  #compress it first.
        compressed_data = zlib.compress(img_data.getvalue(),9)
  #encode it to base64 string
        encoded = base64.b64encode(compressed_data)  
    except:
        pass 
  
    return encoded
  
def decode_base64(compressed_b64str=None, save_to_file=None): 
    try :
  #firstly, decode it
#         decoded = base64.decodestring(compressed_b64str)
        decoded = base64.b64decode(compressed_b64str[1:])
        decompr = zlib.decompress(decoded)
  #save it if is needed.
        if save_to_file is not None:
            with open(save_to_file,"wb") as fh:
                fh.write(decompr)
        else:
   #just display on screen
            w,h = 1024,768
            image = Image.open(BytesIO(decompr))
            image.show()
            time.sleep(3)
    except:
         print('error, cannot receive the image')   

this_name = 'n3a1'
those_name = 'n3a2'
this_role = 'receiver'
running = True
ready_to_receive = False 
img_status = False

netpie.setname(this_name)
netpie.on_reject = callback_reject
netpie.on_connect = connection
netpie.on_message = subscription
netpie.on_error = callback_error
netpie.subscribe("/test")
netpie.connect(False)
    
try :
    while running:
        while not ready_to_receive:
            print('No connection from sender...')
            time.sleep(2)
        pass
except KeyboardInterrupt :
    running=False 

