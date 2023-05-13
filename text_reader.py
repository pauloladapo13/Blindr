import cv2 #procesamiento imagenes
import pytesseract #interpretación palabras
from gtts import gTTS #interpretar palabras para convertirlas en sonido
from playsound import playsound #reproducir sonido

cuadro=100
anchocam,altocam=640,480
cap=cv2.VideoCapture(0)
cap.set(3,anchocam) #definimos un ancho y un alto definidos para siempre
cap.set(4,altocam)

#creamos las función para extraer el texto
def text(image):
    #creamosla funciónpara reproducir la voz
    def voz(archi_text,language,nom_archi):
        with open(archi_text, "r") as lec:
            lectura = lec.read()
        lect = gTTS(text = lectura, lang=language, slow=False )
        nombre = nom_archi
        lect.save(nombre)

    pytesseract.pytesseract.tesseract_cmd = r'd:\Users\Usuario\Desktop\HackUPC'
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texto =pytesseract.image_to_string(gris)
    print(texto)
    dire = open('Info.txt',"m")
    dire.write(texto)
    dire.close()
    voz("Info.txt","en","Voz.mp3")
    audio = "Voz.mp3"
    playsound(audio)



#si ret está mal, es decir, la lectura de la cámara está mal hacemos un break
#rectángulo donde procesamos la información que esté en él
#cogemos los píxeles de nuestro cuadrado y los almacenamos y guardamos 
while True:
    ret,frame =cap.read() #leemos la captura de vídeo y almacenamos los fps en frame
    if ret == False:break
    cv2.putText(frame, 'Coloque aquí el tecto que quiere leer', (158,80), cv2.FONT_HERSHEY_SIMPLEX, 0.71, (255,255,0),2)
    cv2.putText(frame, 'Coloque aquí el tecto que quiere leer', (160,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
    cv2.rectangle(frame,(cuadro,cuadro),(anchocam-cuadro,altocam-cuadro),(0,0,0),2)
    x1,y1 =cuadro,cuadro #extraemos las coordenadas de la esquina superior izquierda
    ancho,alto=(anchocam-cuadro)-x1,(altocam-cuadro)-y1
    x2,y2=x1+ancho,y1+alto
    doc = frame[y1:y2,x1:x2]
    cv2.imwrite("Imatext.jpg",doc)
    cv2.imshow("Lector Inteligente",frame)
    t=cv2.waitKey(1)

    if t == 27:
        break

    text(doc)
    cap.release()
    cv2.destroyAllWindows()    