import cv2 as cv
import numpy as np

#onceden egitilmis ag net degiskenine atilir
net =cv.dnn.readNet('yolov3.weights','yolov3.cfg')

siniflar =[]
# YOLO sinif isimlerinin oldugu dosya okunarak siniflar dizisine atilir
with open('coco.names','r') as file:
    siniflar=file.read().splitlines()

#detection yapilacak olan fotograf okunur
img = cv.imread('koyun.jpg')
height, width,_= img.shape

# img, pre-processing
normalizasyon = cv.dnn.blobFromImage(img,1/255,(350,350),(0,0,0),swapRB=True,crop=False)

#pre-processing asamasindan sonra fotograf net degiskenine input olarak verilir
net.setInput(normalizasyon)
#output layer isimlerini döndürür.
output_layers_names= net.getUnconnectedOutLayersNames()
#döndürülen layer isimleri ağa gönderilir.
layerOutputs=net.forward(output_layers_names)


boxes=[]
confidences=[]
class_ids=[]

#tanimlanan objelerin sinirlarini belirlenir
for output in layerOutputs:
    for detection in output:
        #classes prediction
        scores =detection[5:]
        #highest class location
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # 0.5 olasiligindan buyuk olan degerin belirledigi sinirlar bulunur.
        if confidence > 0.5 :
            center_x= int(detection[0]*width)
            center_y = int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)

            x = int(center_x-w/2)
            y = int(center_y-h/2)
            # bulunan degerler dizilere eklenir
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-Maximum Suppression
indexes=cv.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

font = cv.FONT_HERSHEY_PLAIN
colors= np.random.uniform(0,255,size=(len(boxes),3))

#bulunan objelerin cizimi
for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(siniflar[class_ids[i]])
    confidence = str(round(confidences[i],2))
    color = colors[i]
    cv.rectangle(img, (x,y),(x+w,y+h),color,2)
    cv.putText(img,label+ " "+ confidence, (x,y+20),font,2,(255,255,255),2)


cv.imshow('Koyun',img)
cv.waitKey()
cv.destroyAllWindows()
