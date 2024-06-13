import cv2
from deepface import DeepFace
import numpy as np

# Dicionário de mapeamento de emoções de inglês para português
emotion_translation = {
    'angry': 'Raiva',
    'disgust': 'Nojo',
    'fear': 'Medo',
    'happy': 'Feliz',
    'sad': 'Tristeza',
    'surprise': 'Surpresa',
    'neutral': 'Neutro'
}

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Não foi possível capturar a imagem da câmera")
        break

    # Inverter a imagem horizontalmente para um espelho de visualização
    frame = cv2.flip(frame, 1)

    # Analisar a imagem usando DeepFace
    results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
    print(results)

    # Verificar se results é uma lista e extrair o primeiro dicionário
    if isinstance(results, list) and len(results) > 0:
        results = results[0]

    # Verificar se 'age', 'gender' e 'dominant_emotion' estão em results
    if 'age' in results and 'gender' in results and 'dominant_emotion' in results:
        # Extrair idade, gênero e emoção dos resultados
        age = results['age']
        
        # Extrair gênero
        #gender = results['gender']
        #gender = 'Mulher' if 'Woman' in gender else 'Homem' if 'Man' in gender else 'Desconhecido'
        
        # Extrair gênero e suas acurácias
        gender_confidences = results['gender']
        man_confidence = gender_confidences['Man']
        woman_confidence = gender_confidences['Woman']
        
        # Determinar gênero com base nas acurácias
        if man_confidence > woman_confidence:
            gender = 'Homem'
        else:
            gender = 'Mulher'
            
        # Converter gênero para português
        '''if 'Woman' in gender:
            gender = 'Mulher'
        elif 'Man' in gender:
            gender = 'Homem'
        else:
            gender = 'Desconhecido' '''
            
        
        # Converter emoção para português
        emotion = results['dominant_emotion']
        emotion = emotion_translation.get(emotion, emotion)

        # Desenhar idade, gênero e emoção na imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Idade: {age}", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Genero: {gender}", (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Emocao: {emotion}", (50, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar a imagem
    cv2.imshow('Age, Gender, and Emotion Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
