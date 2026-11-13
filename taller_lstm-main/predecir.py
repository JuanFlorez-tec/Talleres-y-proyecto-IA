import numpy as np # Manejar la ventana como arreglo
import joblib # Cargar el scaler y las categorías (.pkl)
import tensorflow as tf # Cargar el modelo y ejecutarlo

class ClasificadorMovimiento:
    def __init__(self, ruta_modelo, ruta_escalador, ruta_categorias):
        """Inicializa cargando los pesos, el escalador y las clases."""
        # Reconstruye la red completa: arquitectura, pesos aprendidos y
        #configuración de compilación. Después de esto el modelo queda listo para predict.
        self.model = tf.keras.models.load_model(ruta_modelo)
        
        # Carga el StandardScaler ya ajustado. Trae guardadas la media (scaler.mean_) 
        #y la desviación (scaler.scale_) de cada eje, calculadas solo con datos de train.
        #Esta es la pieza clave: si escalaras con estadísticas nuevas, la red recibiría
        #datos en “otra escala” y fallaría.
        self.scaler = joblib.load(ruta_escalador)
        
        # Carga la lista ['sentadilla', 'estocada', 'tijera', 'reposo', 'caminar']. 
        #Su orden debe coincidir con el que se usó al entrenar, porque el índice 0
        #de la salida softmax es “sentadilla”, el 1 es “estocada”, y así sucesivamente.
        self.categorias = joblib.load(ruta_categorias)
        
        # Parámetros esperados por la red
        #La forma de entrada que espera la red. Está escrita a mano. 
        #Cuando corrijas la ventana a 125 en el entrenamiento, este script seguirá esperando 32
        #y fallará. Es mejor leerla del propio modelo (ver mejoras).
        self.window_size = 32
        self.features = 3

    def predecir_movimiento(self, senal_ventana):
        """
        Toma un array de numpy de forma (32, 3) y devuelve la predicción.
        """
        # Validación de la forma de entrada
        # Validación defensiva: si llega una ventana con otra forma, el script falla con un
        #mensaje claro y no con un error críptico de TensorFlow.
        #Es buena práctica. Un detalle: si alguien pasa una lista de Python,
        #.shape no existe. Conviene hacer antes np.asarray(senal_ventana).
        if senal_ventana.shape != (self.window_size, self.features):
            raise ValueError(f"La ventana debe tener forma ({self.window_size}, {self.features}). "
                             f"Se recibió: {senal_ventana.shape}")

        # 1. Aplanar la señal temporalmente para aplicar el StandardScaler (Muestras, Ejes)
        #Pasa la ventana a formato (muestras, ejes). Como la ventana ya llega en (32, 3),
        #esta línea no cambia nada. Es inofensiva, y está ahí por simetría con el entrenamiento,
        #donde sí hacía falta aplanar de 3D a 2D.
        senal_flat = senal_ventana.reshape(-1, self.features)

        # 2. Escalar usando la media y varianza del entrenamiento
        #Aplica z-score a cada eje: (x − media_train) / desv_train. Usa transform, 
        #unca fit_transform. Correcto.
        senal_scaled = self.scaler.transform(senal_flat)

        # 3. Reformatear a 3D para la capa LSTM: (Batch_size=1, TimeSteps=32, Features=3)
        #Agrega la dimensión de batch. Keras siempre espera (batch, timesteps, features),
        #aunque sea una sola ventana. Resultado: (1, 32, 3).
        X_input = senal_scaled.reshape(1, self.window_size, self.features)

        # 4. Obtener las probabilidades de la capa Softmax
        #Pasa la ventana por LSTM → LSTM → Dense → Softmax. Devuelve un arreglo (1, 5)
        #con probabilidades que suman 1, por ejemplo [[0.02, 0.91, 0.05, 0.01, 0.01]].
        #verbose=0 oculta la barra de progreso.
        predicciones = self.model.predict(X_input, verbose=0)
        
        # 5. Extraer la clase ganadora y su probabilidad
        #Índice de la probabilidad más alta. El [0] saca el valor del batch de tamaño 1.
        clase_idx = np.argmax(predicciones, axis=1)[0]

        # Toma la probabilidad ganadora y traduce el índice a su nombre.
        probabilidad = predicciones[0][clase_idx]
        movimiento_predicho = self.categorias[clase_idx]

        # Devuelve una tupla, por ejemplo ('estocada', 0.91).
        return movimiento_predicho, probabilidad

# --- EJECUCIÓN DE PRUEBA ---
if __name__ == "__main__":
    # Instanciar la clase cargando los archivos generados por el notebook
    # Instancia la clase con rutas relativas. Los archivos tienen que estar en la 
    #carpeta desde donde ejecutas el script.
    clasificador = ClasificadorMovimiento(
        ruta_modelo='modelo_movimiento.keras',
        ruta_escalador='scaler_movimiento.pkl',
        ruta_categorias='categorias_movimiento.pkl'
    )

    # Simular una lectura cruda del acelerómetro (32 instantes de tiempo, ejes X,Y,Z)
    # En producción, esto vendrá de tu sensor embebido o archivo JSON
    #Genera ruido aleatorio uniforme entre −10 y 10 (unidades de m/s²) para simular el
    #sensor.
    senal_cruda_mock = np.random.uniform(low=-10.0, high=10.0, size=(32, 3))

    # Realizar predicción
    # Predice e imprime el resultado en mayúsculas y la confianza como porcentaje
    #con 2 decimales.
    movimiento, conf = clasificador.predecir_movimiento(senal_cruda_mock)
    
    print(f"Movimiento detectado: {movimiento.upper()}")
    print(f"Nivel de confianza: {conf * 100:.2f}%")