import os   #Navegar por carpetas del sistema operativo.
import json #Leer los archivos de datos en formato JSON.
import numpy as np #Para manipulación eficiente de los arreglos multidimensionales
#(tensonre)s y operaciones matemáticas.
import joblib #Para serializar y guardar objetos de Python en disco (el escalador y 
#la lista de categorias).
import matplotlib.pyplot as plt #Para generar las gráficas de entrenamiento y la matriz de 
#confusión.
import seaborn as sns #Para generar las gráficas de entrenamiento y la matriz de 
#confusión.
from sklearn.preprocessing import StandardScaler #Para estandarizar los datos
#(media 0, desviación estándar 1).
from sklearn.metrics import confusion_matrix, classification_report #Para evaluar el 
#rendimiento del modelo mediante métricas detalladas.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam #Importa los módulos necesarios para
#construir, compilar y entrenar la red neuronal profunda (capas LSTM, Dense, 
#Dropout, BatchNormalization, optimizador Adam y codificación to_categorical).

# --- CONFIGURACIÓN GLOBAL ---
#CATEGORIAS = ['sentadilla', 'estocada', 'tijera', 'escalador', 'caminar']
# Define las 5 clases de movimiento que el modelo aprenderá a distinguir.
CATEGORIAS = ['sentadilla', 'estocada', 'tijera', 'reposo', 'caminar'] 

# Crea un mapeo númerico ({sentadilla:0, estocada:1, tijera:2,...})
DICCIONARIO_ETIQUETAS = {cat: i for i, cat in enumerate(CATEGORIAS)}

# Configura una tasa de muestreo de 16 Hz y segmentos de análisis de 2 segundos.
FRECUENCIA = 62.5
VENTANA_SEGUNDOS = 2

# Calcula el tamaño de cada secuencia de entrada (62.5 * 2 = 125 pasos de tiempo).
WINDOW_SIZE = int(FRECUENCIA * VENTANA_SEGUNDOS)  # 125 muestras

# Establece un solapamiento del 50% (16 muestras) entre ventanas consecutivas 
#para aumentar la cantidad de datos de entrenamiento y no perder transiciones 
#entre movimientos.
OVERLAP = int(WINDOW_SIZE * 0.5) 

# Rutas a los directorios de datos
# Definen las rutas locales donde se almacenan los datos de entrenamiento y prueba.
RUTA_BASE = r"D:\Usuarios-PC\WebMaster\Documents\InteligenciaArtificial\Lab_Analisis"
PATH_TRAIN = os.path.join(RUTA_BASE, 'training')
PATH_TEST = os.path.join(RUTA_BASE, 'testing')

# FUNCIÓN
def cargar_dataset(directorio):
    """Carga y procesa los archivos JSON aplicando ventanas deslizantes."""
    X, y = [], []

    # Verifica que la carpeta exista para evitar que el programa falle abruptamente.
    if not os.path.exists(directorio):
        print(f"Advertencia: El directorio {directorio} no existe.")
        return np.array(X), np.array(y)
        
    # Filtra y lista únicamente los archivos con extensión .json.
    archivos = [f for f in os.listdir(directorio) if f.endswith('.json')]
    print(f"Procesando {len(archivos)} archivos en {directorio}...")
    
    # Itera sobre cada archivo e infiere su etiqueta buscando si alguna palabra 
    # de CATEGORIAS está contenida en el nombre del archivo (en minúsculas). 
    # Si no coincide con ninguna, lo ignora (continue).
    for nombre_archivo in archivos:
        etiqueta = None
        for cat in CATEGORIAS:
            if cat in nombre_archivo.lower():
                etiqueta = DICCIONARIO_ETIQUETAS[cat]
                break
        
        if etiqueta is None: 
            continue

        ruta_completa = os.path.join(directorio, nombre_archivo)
        with open(ruta_completa, 'r') as f:
            # Extrae la matriz de valores del sensor (ejes X, Y, Z) desde la estructura 
            #payload -> values del JSON.
            contenido = json.load(f)
            datos_crudos = np.array(contenido['payload']['values'])
            
            # Aplicar Ventana Deslizante
            #Recorre la serie temporal dando saltos de 16 muestras (OVERLAP), extrayendo bloques 
            #de 32 muestras (ventana) y guardándolos en X junto con su etiqueta en y.
            for i in range(0, len(datos_crudos) - WINDOW_SIZE + 1, OVERLAP):
                ventana = datos_crudos[i : i + WINDOW_SIZE]
                X.append(ventana)
                y.append(etiqueta)
                
    return np.array(X), np.array(y)

# Preparación y Normalización
def main():
    # 1. CARGA DE DATOS
    print("--- 1. CARGANDO DATASET ---")
    # Carga los conjuntos de entrenamiento y prueba.
    X_train_raw, y_train_id = cargar_dataset(PATH_TRAIN)
    X_test_raw, y_test_id = cargar_dataset(PATH_TEST)

    if len(X_train_raw) == 0:
        print("Error: No se encontraron datos de entrenamiento. Revisa las rutas configuradas.")
        return

    # 2. NORMALIZACIÓN
    print("\n--- 2. NORMALIZACIÓN DE DATOS ---")
    scaler = StandardScaler()
    
    # Aplana temporalmente el tensor 3D (muestras, 32, 3) a una matriz 2D (muestras * 32, 3)
    #porque StandardScaler solo acepta datos bidimensionales.
    shape_train = X_train_raw.shape
    X_train_flat = X_train_raw.reshape(-1, 3)

    # Calcula la media y desviación estándar únicamente con los datos de entrenamiento 
    #(evitando fuga de datos o data leakage).
    scaler.fit(X_train_flat)

    # Aplica la normalización y devuelve los datos a su forma original 3D. Se hace lo mismo
    #con X_test.
    X_train = scaler.transform(X_train_flat).reshape(shape_train)

    shape_test = X_test_raw.shape
    X_test_flat = X_test_raw.reshape(-1, 3)
    X_test = scaler.transform(X_test_flat).reshape(shape_test)

    # 3. CODIFICACIÓN DE ETIQUETAS
    y_train = to_categorical(y_train_id, num_classes=len(CATEGORIAS))

    # Convierte las etiquetas enteras (ej. 2) en vectores One-Hot (ej. [0, 0, 1, 0, 0]).
    y_test = to_categorical(y_test_id, num_classes=len(CATEGORIAS))
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test:  {X_test.shape}")

    # 4. DEFINICIÓN DEL MODELO
    print("\n--- 3. CONSTRUYENDO EL MODELO ---")
    model = Sequential([
        # Define la entrada de la red: 32 pasos de tiempo por 3 características
        #(ejes del sensor).
        Input(shape=(WINDOW_SIZE, 3)),
        
        # Primera capa recurrente con 64 unidades. return_sequences=True es obligatorio
        #porque apila otra capa LSTM encima y necesita pasarle toda la secuencia
        #temporal, no solo el último estado.
        LSTM(64, return_sequences=True),
        
        # Apaga aleatoriamente el 30% de las neuronas durante el entrenamiento para
        #prevenir el sobreajuste (overfitting).
        Dropout(0.3),
        
        # Segunda capa recurrente que condensa la secuencia en un único vector de
        #características de tamaño 32.
        LSTM(32),
        Dropout(0.3),

        # Capa de salida con 5 neuronas que entrega la probabilidad de pertenencia a 
        #cada clase (la suma de todas es 1).
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(len(CATEGORIAS), activation='softmax')
    ])

    # Configura el optimizador Adam, la función de pérdida para clasificación multiclase
    #(categorical_crossentropy) y entrena durante 50 épocas en lotes de 32 ventanas.
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 5. ENTRENAMIENTO
    print("\n--- 4. INICIANDO ENTRENAMIENTO ---")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 6. EVALUACIÓN Y MÉTRICAS
    print("\n--- 5. EVALUACIÓN DEL MODELO ---")

    # Obtiene las predicciones en probabilidad y selecciona el índice con 
    #el valor más alto como la clase predicha.
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nReporte de Clasificación:")

    # Imprime métricas de precisión, exhaustividad (recall) y F1-Score por clase.
    print(classification_report(y_true, y_pred, target_names=CATEGORIAS))

    # 7. GUARDADO DE ARTEFACTOS
    print("\n--- 6. GUARDANDO ARTEFACTOS ---")

    # Guarda el modelo entrenado, el escalador y las categorías. (Nota técnica: 
    #En el código las líneas de guardado están duplicadas en las secciones 7 y 9;
    #la sección 9 es redundante y puede eliminarse).
    model.save('modelo_movimiento.keras')
    joblib.dump(scaler, 'scaler_movimiento.pkl')
    joblib.dump(CATEGORIAS, 'categorias_movimiento.pkl')
    print("¡Archivos exportados exitosamente ('modelo_movimiento.keras', 'scaler_movimiento.pkl', 'categorias_movimiento.pkl')!")

    # 8. GRÁFICOS
    # Genera y guarda las curvas de aprendizaje (historial_entrenamiento.png) y 
    # la matriz de confusión (matriz_confusion.png).
    plt.figure(figsize=(12, 4))
    
    # Gráfico de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo (Accuracy)')
    plt.xlabel('Época')
    plt.legend()

    # Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo (Loss)')
    plt.xlabel('Época')
    plt.legend()
    
    # Guardar gráfico y matriz de confusión como imágenes
    plt.savefig('historial_entrenamiento.png', bbox_inches='tight')
    plt.close()
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIAS, yticklabels=CATEGORIAS)
    plt.title('Matriz de Confusión')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.savefig('matriz_confusion.png', bbox_inches='tight')
    plt.close()
    
    print("Gráficas guardadas como 'historial_entrenamiento.png' y 'matriz_confusion.png'.")
    
    # 9. Guardar el modelo y el escalador para su uso en predicciones futuras
    
    # 1. Guardar la arquitectura y los pesos de la red neuronal
    # El formato '.keras' es el estándar recomendado para TensorFlow/Keras
    model.save('modelo_movimiento.keras')
    
    # 2. Guardar el escalador y las categorías usando joblib
    joblib.dump(scaler, 'scaler_movimiento.pkl')
    
    # 3. Guardar las categorías para mapear índices a nombres de clases
    joblib.dump(CATEGORIAS, 'categorias_movimiento.pkl')
    

if __name__ == "__main__":
    main()