import pyttsx3
import cv2
import pytesseract
from gtts import gTTS
cuadro =100
anchocam, altocam= 640,480
cap= cv2.VideoCapture(0)
cap.set(3, anchocam)
cap.set(4, altocam)
def text(image):
    #creamos la función para reproducir la voz
    def voz(archi_text,language,nom_archi):
        with open(archi_text, "r") as lec:
            lectura = lec.read()
        lect = gTTS(text = lectura, lang=language, slow=False )
        nombre = nom_archi
        lect.save(nombre)

    
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texto:str =pytesseract.image_to_string(gris, lang='eng')

    hayTexto = texto.replace("\n","")
    hayTexto = hayTexto.replace(" ","")
    if not hayTexto: return
    textoConvertido = texto.encode("utf-8","ignore").decode("utf-8")
    print(textoConvertido)

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[17].id)

    engine.say(textoConvertido)
    engine.runAndWait()
    


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