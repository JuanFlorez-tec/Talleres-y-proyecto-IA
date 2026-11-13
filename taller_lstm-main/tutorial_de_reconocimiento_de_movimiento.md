## Índice de Documentación
> Obtén el índice completo de la documentación en: https://docs.edgeimpulse.com/llms.txt
> Usa este archivo para descubrir todas las páginas disponibles antes de explorar más a fondo.

# Reconocimiento de movimiento con detección de anomalías

> Reconoce gestos a partir de datos del acelerómetro con un clasificador de red neuronal y detección de anomalías, implementado en un microcontrolador.

En este tutorial, usarás aprendizaje automático (machine learning) para construir un sistema de reconocimiento de gestos que se ejecuta en un microcontrolador. Esta es una tarea difícil de resolver usando programación basada en reglas, ya que las personas no realizan los gestos exactamente de la misma manera cada vez. Pero el aprendizaje automático puede manejar estas variaciones con facilidad. Aprenderás cómo recopilar datos de alta frecuencia de sensores reales, usar procesamiento de señales para limpiar los datos, construir un clasificador de red neuronal y cómo implementar tu modelo de vuelta en un dispositivo. Al final de este tutorial, tendrás una sólida comprensión de cómo aplicar el aprendizaje automático en dispositivos integrados usando Edge Impulse.

> [card] [Salta directamente al proyecto](https://studio.edgeimpulse.com/public/14299/latest)
>
> Omite la configuración: clona un proyecto de Edge Impulse listo para usar con el conjunto de datos, el impulso y el modelo entrenado.

También hay una versión en video de este tutorial:

[Reproductor de video de YouTube](https://www.youtube-nocookie.com/embed/FseGCn-oBA0)

### 1. Requisitos previos

Para este tutorial, necesitarás un [dispositivo compatible](https://docs.edgeimpulse.com/hardware.md).

Alternativamente, usa el SDK de [Data forwarder](https://docs.edgeimpulse.com/tools/clis/edge-impulse-cli/data-forwarder.md) o [Edge Impulse para Linux](https://docs.edgeimpulse.com/tools/libraries/sdks/inference/linux.md) para recopilar datos desde cualquier otra placa de desarrollo, o desde tu [teléfono móvil](https://docs.edgeimpulse.com/hardware/devices/mobile-phone.md).

Si tu dispositivo está conectado (punto verde) en **Devices** (Dispositivos) en el estudio, puedes proceder:

![La página de Dispositivos con la placa de desarrollo conectada a través de gestión remota](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-connected-device.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=20632be2b221f097207f28930586679e)

Pestaña de dispositivos con el dispositivo conectado a la interfaz de gestión remota.

> [!NOTE]
>
> **Ingesta de datos**
>
> Edge Impulse puede ingerir datos de muchas fuentes y cualquier dispositivo, incluyendo dispositivos integrados que ya tengas en producción. Consulta la documentación sobre [Adquisición de datos](https://docs.edgeimpulse.com/studio/projects/data-acquisition.md) para más información.

### 2. Recopilando tus primeros datos

Con tu dispositivo conectado, podemos recopilar algunos datos. En el estudio ve a la pestaña **Data acquisition** (Adquisición de datos). Este es el lugar donde se almacenan todos tus datos sin procesar y, si tu dispositivo está conectado a la API de gestión remota, donde puedes comenzar a muestrear nuevos datos.

En **Record new data** (Registrar nuevos datos), selecciona tu dispositivo, establece la etiqueta (label) como `updown`, la longitud de la muestra (sample length) en `10000`, el sensor en `Built-in accelerometer` (Acelerómetro integrado) y la frecuencia en `62.5Hz`. Esto indica que deseas registrar datos durante 10 segundos y etiquetar los datos registrados como `updown`. Más adelante puedes editar estas etiquetas si es necesario.

![El formulario para registrar nuevos datos configurado para el muestreo del acelerómetro](https://mintcdn.com/edgeimpulse/-WgXP515XHSZ7nhZ/.assets/images/record-new-data-form-configured-for-accelerometer.png?fit=max\&auto=format\&n=-WgXP515XHSZ7nhZ\&q=85\&s=328b67c18200d839857d7038e92775cb)

Pantalla de registrar nuevos datos.

Después de hacer clic en **Start sampling** (Iniciar muestreo), mueve tu dispositivo hacia arriba y hacia abajo en un movimiento continuo. En unos doce segundos, el dispositivo debería completar el muestreo y cargar el archivo de vuelta a Edge Impulse. Verás aparecer una nueva línea bajo 'Collected data' (Datos recopilados) en el estudio. Al hacer clic en ella, verás los datos en bruto graficados. Como el acelerómetro de la placa de desarrollo tiene tres ejes, notarás tres líneas diferentes, una para cada eje.

> [!WARNING]
>
> **Movimiento continuo**
>
> Es importante hacer movimientos continuos, ya que más adelante dividiremos los datos en ventanas más pequeñas.

![Forma de onda del acelerómetro para una muestra de movimiento hacia arriba y hacia abajo](https://mintcdn.com/edgeimpulse/Ab0DDFOcWoL7t2Wd/.assets/images/accelerometer-waveform-for-an-updown-movement-sample.png?fit=max\&auto=format\&n=Ab0DDFOcWoL7t2Wd\&q=85\&s=5ea734fb29f2334ac3a107b179898bbf)

Movimiento de arriba hacia abajo (updown) registrado desde el acelerómetro.

El aprendizaje automático funciona mejor con muchos datos, por lo que una sola muestra no será suficiente. Ahora es el momento de empezar a construir tu propio conjunto de datos. Por ejemplo, usa las siguientes cuatro clases y registra alrededor de 3 minutos de datos por clase:

- Idle (Inactivo) - simplemente reposando en tu escritorio mientras trabajas.
- Snake (Serpiente) - moviendo el dispositivo sobre tu escritorio como una serpiente.
- Wave (Ola/Ondear) - ondeando el dispositivo de izquierda a derecha.
- Updown (Arriba-abajo) - moviendo el dispositivo hacia arriba y hacia abajo.

> [!NOTE]
>
> **Variaciones**
>
> Asegúrate de realizar variaciones en los movimientos. Por ejemplo, haz movimientos lentos y rápidos, y varía la orientación de la placa. Nunca sabrás cómo usará el dispositivo tu usuario. Lo mejor es recopilar muestras de \~10 segundos cada una.

**Conjunto de datos preconstruido**

Alternativamente, puedes cargar un conjunto de prueba de ejemplo que tiene unos diez minutos de datos en estas clases (pero, ¿qué tan divertido es eso?). Consulta el [Conjunto de datos de gestos continuos](https://docs.edgeimpulse.com/datasets/time-series/continuous-motion-recognition.md) para más información.

### 3. Diseñando un impulso

Con el conjunto de entrenamiento listo, puedes diseñar un impulso. Un impulso toma los datos sin procesar, los divide en ventanas más pequeñas, usa bloques de procesamiento de señales para extraer características (features) y luego usa un bloque de aprendizaje para clasificar nuevos datos. Los bloques de procesamiento de señales siempre devuelven los mismos valores para la misma entrada y se usan para hacer que los datos en bruto sean más fáciles de procesar, mientras que los bloques de aprendizaje aprenden de experiencias pasadas.

Para este tutorial usaremos el bloque de procesamiento de señales 'Spectral analysis' (Análisis espectral). Este bloque aplica un filtro, realiza un análisis espectral en la señal y extrae datos de frecuencia y potencia espectral. Luego usaremos un bloque de aprendizaje 'Neural Network' (Red Neuronal), que toma estas características espectrales y aprende a distinguir entre las cuatro clases (idle, snake, wave, updown).

En el estudio, ve a **Create impulse** (Crear impulso), establece el tamaño de la ventana (window size) en `2000` (puedes hacer clic en el texto `2000 ms.` para ingresar un valor exacto), el incremento de la ventana (window increase) en `80`, y añade los bloques 'Spectral Analysis' (Análisis Espectral) y 'Classification (Keras)' (Clasificación). Luego haz clic en **Save impulse** (Guardar impulso).

![Página de crear impulso con entrada de series temporales, análisis espectral y bloques de clasificación](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-create-impulse.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=9f1c75962834180ce64f577516300eae)

Primer impulso, con un bloque de procesamiento y un bloque de aprendizaje.

#### Configurando el bloque de análisis espectral

Para configurar tu bloque de procesamiento de señales, haz clic en **Spectral features** (Características espectrales) en el menú de la izquierda. Esto te mostrará los datos en bruto en la parte superior de la pantalla (puedes seleccionar otros archivos a través del menú desplegable), y los resultados del procesamiento de la señal a través de gráficos a la derecha. Para el bloque de características espectrales verás los siguientes gráficos:

- Respuesta del filtro (Filter response) - Si has elegido un filtro (con orden distinto de cero), esto te mostrará la respuesta en todas las frecuencias. Es decir, te mostrará cuánto se atenuará cada frecuencia.
- Después del filtro (After filter) - la señal después de aplicar el filtro. Esto eliminará el ruido.
- Potencia espectral (Spectral power) - las frecuencias en las que se repite la señal (por ejemplo, hacer un movimiento de ola por segundo mostrará un pico a 1 Hz).

> [!NOTE]
>
> Consulta la página dedicada al bloque de preprocesamiento [Características espectrales (Spectral features)](https://docs.edgeimpulse.com/studio/projects/processing-blocks/blocks/spectral-analysis.md).

Un buen bloque de procesamiento de señales producirá resultados similares para datos similares. Si mueves la ventana deslizante (en el gráfico de datos en bruto), los gráficos deberían permanecer similares. Además, cuando cambies a otro archivo con la misma etiqueta, deberías ver gráficos similares, incluso si la orientación del dispositivo fue diferente.

> [!NOTE]
>
> **Ejercicio adicional: filtros**
>
> Intenta razonar sobre los parámetros del filtro. ¿Qué controla la frecuencia de corte? ¿Y qué ves si cambias de un filtro de paso bajo (low-pass) a uno de paso alto (high-pass)?

Configura el filtro en paso bajo (low pass) con los siguientes parámetros:

![Configuración de parámetros de análisis espectral para datos de movimiento continuo](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-spectral-features.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=b3ba7a3943890adea86f85c10626b599)

Parámetros de características espectrales

Una vez que estés satisfecho con el resultado, haz clic en **Save parameters** (Guardar parámetros). Esto te enviará a la pantalla 'Feature generation' (Generación de características). Aquí harás lo siguiente:

1. Dividir todos los datos en bruto en ventanas (basado en el tamaño y el incremento de la ventana).
2. Aplicar el bloque de características espectrales en todas estas ventanas.
3. Calcular la importancia de las características (feature importance). Usaremos esto más adelante para configurar la detección de anomalías.

Haz clic en **Generate features** (Generar características) para iniciar el proceso.

Después se cargará el 'Feature explorer' (Explorador de características). Este es un gráfico de todas las características extraídas contra todas las ventanas generadas. Puedes usar este gráfico para comparar tu conjunto de datos completo. Una buena regla general es que si puedes identificar visualmente algunos grupos (clústeres) por clases, entonces el modelo de aprendizaje automático también podrá hacerlo.

![Página de generar características para el bloque de análisis espectral](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-spectral-features-generate.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=4a52a6a27448d4c3cbb47e0162517b42)

Características espectrales - Generar características

#### Configurando la red neuronal

Con todos los datos procesados es hora de comenzar a entrenar una red neuronal. Las redes neuronales son un conjunto de algoritmos, modelados vagamente según el cerebro humano, que están diseñados para reconocer patrones. La red que estamos entrenando aquí tomará los datos del procesamiento de señales como entrada, e intentará mapearlos a una de las cuatro clases.

Entonces, ¿cómo sabe una red neuronal qué predecir? Una red neuronal consta de capas de neuronas, todas interconectadas, y cada conexión tiene un peso. Una de esas neuronas en la capa de entrada sería la altura del primer pico del eje X (del bloque de procesamiento de señales); y una de esas neuronas en la capa de salida sería `wave` (una de las clases). Al definir la red neuronal, todas estas conexiones se inicializan aleatoriamente, y por lo tanto la red neuronal hará predicciones aleatorias. Durante el entrenamiento, tomamos todos los datos en bruto, le pedimos a la red que haga una predicción y luego hacemos pequeñas alteraciones en los pesos dependiendo del resultado (por esto es importante etiquetar los datos en bruto).

De esta manera, después de muchas iteraciones, la red neuronal aprende; y eventualmente se volverá mucho mejor para predecir nuevos datos. Probemos esto haciendo clic en **NN Classifier** (Clasificador NN) en el menú.

> [!NOTE]
>
> Consulta la página dedicada para el bloque de aprendizaje [Clasificación (Keras)](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/classification.md).

Establece el 'Number of training cycles' (Número de ciclos de entrenamiento) en `1`. Esto limitará el entrenamiento a una sola iteración. Y luego haz clic en **Start training** (Iniciar entrenamiento).

![Resultados del entrenamiento de la red neuronal después de una iteración de entrenamiento](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-nn-1-epoch.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=9dd6aa3561ed649f37089deba3cf869f)

Rendimiento del entrenamiento después de una sola iteración. En la parte superior derecha, hay un resumen de la precisión de la red y, en el medio, una matriz de confusión. Esta matriz muestra cuándo la red tomó decisiones correctas e incorrectas. Ves que la clase inactiva (idle) es relativamente fácil de predecir. ¿Por qué crees que es así?

Ahora cambia el 'Number of training cycles' a `2` y verás que el rendimiento sube. Finalmente, cambia el 'Number of training cycles' a `30` y deja que termine el entrenamiento.

¡Acabas de entrenar tus primeras redes neuronales!

![Resultados del entrenamiento de la red neuronal después de treinta épocas](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-nn-20-epochs.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=54f170956f3df51239882b730d9f46be)

Red neuronal entrenada con 30 épocas

> [!WARNING]
>
> **100% de precisión**
>
> Podrías terminar con un 100% de precisión después de entrenar durante 100 ciclos de entrenamiento. Esto no es necesariamente algo bueno, ya que podría ser una señal de que la red neuronal está demasiado ajustada para el conjunto de prueba específico y podría tener un mal rendimiento con datos nuevos (sobreajuste o overfitting). La mejor manera de reducir esto es agregando más datos o reduciendo la tasa de aprendizaje (learning rate).

### 4. Clasificando nuevos datos

A partir de las estadísticas en el paso anterior sabemos que el modelo funciona frente a nuestros datos de entrenamiento, pero ¿qué tan bien funcionaría la red con nuevos datos? Haz clic en **Live classification** (Clasificación en vivo) en el menú para averiguarlo. Tu dispositivo debería (al igual que en el paso 2) mostrarse en línea bajo 'Classify new data' (Clasificar nuevos datos). Establece la 'Sample length' (Longitud de muestra) en `10000` (10 segundos), haz clic en **Start sampling** (Iniciar muestreo) y comienza a hacer movimientos. Después, obtendrás un informe completo sobre lo que la red pensó que hiciste.

![Resultado de clasificación en vivo con datos de movimiento sin procesar y características procesadas](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-live-classification.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=1eb20b49d42d027f481a8c8bcf807d11)

Resultado de la clasificación. Mostrando las conclusiones, los datos sin procesar y las características procesadas en una sola vista general.

Si la red funcionó bien, ¡genial! Pero, ¿qué pasa si funcionó mal? Podría haber una variedad de razones, pero las más comunes son:

1. No hay suficientes datos. Las redes neuronales necesitan aprender patrones en conjuntos de datos, y cuantos más datos, mejor.
2. Los datos no se parecen a otros datos que la red haya visto antes. Esto es común cuando alguien usa el dispositivo de una manera que no agregaste al conjunto de pruebas. Puedes agregar el archivo actual al conjunto de entrenamiento haciendo clic en `⋮`, y luego seleccionando **Move to training set** (Mover al conjunto de entrenamiento). Asegúrate de actualizar la etiqueta en 'Data acquisition' antes de entrenar.
3. El modelo no se ha entrenado lo suficiente. Aumenta el número de épocas a `200` y ve si el rendimiento aumenta (el archivo clasificado se almacena y puedes cargarlo a través de 'Classify existing validation sample').
4. El modelo está sobreajustado (overfitting) y, por lo tanto, tiene un rendimiento pobre con datos nuevos. Intenta reducir la tasa de aprendizaje o agregar más datos.
5. La arquitectura de la red neuronal no es la adecuada para tus datos. Juega con el número de capas y neuronas y ve si el rendimiento mejora.

Como puedes ver, todavía hay mucha prueba y error al construir redes neuronales, pero esperamos que las visualizaciones ayuden mucho. También puedes ejecutar la red frente al conjunto de validación completo a través de 'Model validation' (Validación del modelo). ¡Piensa en la página de validación del modelo como un conjunto de pruebas unitarias para tu modelo!

Con un modelo funcional implementado, podemos buscar los puntos donde nuestro impulso actual tiene un rendimiento deficiente.

### 5. Detección de anomalías

Las redes neuronales son geniales, pero tienen un gran defecto. Son terribles al lidiar con datos que nunca han visto antes (como un nuevo gesto). Las redes neuronales no pueden juzgar esto, ya que solo conocen los datos de entrenamiento. Si le das algo distinto a cualquier cosa que haya visto antes, aún lo clasificará como una de las cuatro clases.

Veamos cómo funciona esto en la práctica. Ve a 'Live classification' (Clasificación en vivo) y registra algunos datos nuevos, pero ahora agita tu dispositivo vívidamente. Échale un vistazo y mira cómo la red predecirá algo independientemente.

Entonces, ¿cómo podemos mejorar esto? Si observas el explorador de características (feature explorer), deberías poder separar visualmente los datos clasificados de los datos de entrenamiento. Podemos usar esto a nuestra ventaja entrenando una nueva (segunda) red que cree agrupaciones (clústeres) alrededor de los datos que hemos visto antes y compare los datos entrantes frente a estos clústeres. Si la distancia desde un clúster es demasiado grande, puedes marcar la muestra como una anomalía y no confiar en la red neuronal.

![Explorador de características mostrando el movimiento de agitar separado de las clases entrenadas](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-live-classification-explorer.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=f1531d11c99fa52c767f444f94273f0c)

Los datos de agitar (shake) se separan fácilmente de los datos de entrenamiento.

Para agregar este bloque, ve a **Create impulse** (Crear impulso), haz clic en **Add learning block** (Agregar bloque de aprendizaje) y selecciona 'Anomaly Detection (K-Means)' (Detección de Anomalías). Luego haz clic en **Save impulse** (Guardar impulso).

![Página de crear impulso con un bloque de detección de anomalías agregado](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-create-impulse-2.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=48672587732feba4850791a5931750a8)

Agregar el bloque de detección de anomalías a la pestaña Crear impulso

Para configurar el modelo de agrupamiento, haz clic en **Anomaly detection** en el menú. Aquí necesitamos especificar:

- El número de clústeres. Aquí usa `32`.
- Los ejes que queremos seleccionar durante el agrupamiento. Haz clic en el botón **Select suggested axes** (Seleccionar ejes sugeridos) para aprovechar los resultados de la salida de [importancia de las características (feature importance)](https://docs.edgeimpulse.com/studio/projects/processing-blocks.md#feature-importance). Alternativamente, si los datos se separan bien en los ejes accX RMS, accY RMS y accZ RMS, también puedes incluir estos ejes.

> [!NOTE]
>
> Consulta la página dedicada al bloque de aprendizaje de [detección de anomalías (K-means)](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/anomaly-detection-k-means.md). También proporcionamos el bloque de aprendizaje de [detección de anomalías (GMM)](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/anomaly-detection-gmm.md) que es compatible con este tutorial.

Haz clic en **Start training** (Iniciar entrenamiento) para generar los clústeres. Puedes cargar muestras de validación existentes en el explorador de anomalías con el menú desplegable.

![Explorador de anomalías con muestras de agitar fuera de los clústeres de movimiento conocidos](https://mintcdn.com/edgeimpulse/MguyU0DkkpALWBaW/.assets/images/tutorial-continuous-motion-anomaly-explorer.png?fit=max\&auto=format\&n=MguyU0DkkpALWBaW\&q=85\&s=9cfeb26536534dc67734d3a17c1be3e7)

Clústeres conocidos en azul, los datos de agitar en naranja. Están claramente fuera de cualquier clúster conocido y, por lo tanto, pueden etiquetarse como una anomalía.

> [!NOTE]
>
> **Ejes**
>
> El explorador de anomalías solo traza dos ejes al mismo tiempo. En 'average axis distance' (distancia promedio del eje) puedes ver qué tan lejos de cada eje se encuentra la muestra de validación. Usa los menús desplegables para cambiar de ejes.

Si ahora regresas a 'Live classification' y cargas tu última muestra, ahora debería haber etiquetado todo como anomalía. Este es un gran ejemplo de dónde el procesamiento de señales (para extraer características), las redes neuronales (para clasificación) y los algoritmos de agrupamiento (para detección de anomalías) pueden trabajar juntos.

### 6. Implementando de vuelta en el dispositivo

Con el impulso diseñado, entrenado y verificado, puedes implementar este modelo de vuelta en tu dispositivo. Esto permite que el modelo se ejecute sin conexión a internet, minimiza la latencia y funciona con un consumo mínimo de energía. Edge Impulse puede empaquetar el impulso completo, incluido el código de procesamiento de señales, los pesos de la red neuronal y el código de clasificación, en una sola biblioteca en C++ que puedes incluir en tu software embebido.

> [!NOTE]
>
> **Teléfono móvil**
>
> Tu teléfono móvil puede compilar y descargar el impulso compilado directamente desde el cliente móvil. Consulta 'Implementando de vuelta en el dispositivo' en la página de [Uso de tu teléfono móvil](https://docs.edgeimpulse.com/hardware/devices/mobile-phone.md).

Para exportar tu modelo, haz clic en **Deployment** (Implementación) en el menú. Luego, bajo 'Build firmware' (Construir firmware), selecciona tu placa de desarrollo y haz clic en **Build** (Construir). Esto exportará el impulso y construirá un binario que se ejecutará en tu placa de desarrollo en un solo paso. Una vez completada la construcción, se te pedirá que descargues un binario. Guárdalo en tu computadora.

#### Flasheando el dispositivo

Al hacer clic en el botón **Build**, verás una ventana emergente con instrucciones en texto y video sobre cómo implementar el binario en tu dispositivo en particular. Sigue estas instrucciones. Una vez que hayas terminado, estaremos listos para probar tu impulso.

#### Ejecutando el modelo en el dispositivo

Podemos conectarnos al firmware recién flasheado de la placa a través del puerto serie. Abre una terminal y ejecuta:

```
$ edge-impulse-run-impulse
```

> [!WARNING]
>
> **Demonio serie (Serial daemon)**
>
> Si el dispositivo no está conectado por WiFi, sino a través del demonio serie de Edge Impulse, necesitarás detener el demonio. Solo una aplicación puede conectarse a la placa de desarrollo a la vez.

Esto tomará muestras de datos del sensor, ejecutará el código de procesamiento de señales y luego clasificará los datos:

```
Starting inferencing in 2 seconds...
Sampling... Storing in file name: /fs/device-classification261
Tensor shape: 4
Predictions (DSP: 17 ms., Classification: 1 ms., Anomaly: 0 ms.):
    idle: 0.00004
    snake: 0.00012
    updown: 0.00009
    wave: 0.99976
    anomaly score: 0.032
Finished inferencing, raw data is stored in '/fs/device-classification261'. Use AT+UPLOADFILE to send back to Edge Impulse.
```

> [!WARNING]
>
> **Movimiento continuo**
>
> Entrenamos un modelo para detectar movimiento continuo en intervalos de 2 segundos. Por lo tanto, cambiar tu movimiento mientras se toman muestras arrojará resultados incorrectos. Asegúrate de haber comenzado tu movimiento cuando se imprima 'Sampling...'. Entre muestreos, tienes dos segundos para cambiar de movimientos.
>
> Para ejecutar el muestreo continuo, ejecuta el siguiente comando:
>
> ```
> $ edge-impulse-run-impulse --continuous
> ```

¡Victoria! Ahora has construido tu primer modelo de aprendizaje automático (machine learning) en el dispositivo.

![Una persona celebrando en una piscina de pelotas después de completar el modelo en el dispositivo](https://mintcdn.com/edgeimpulse/kATUgMpNijnV5Flg/.assets/images/person-celebrating-in-a-ball-pit-after-completing.gif?s=be3e4c2cc3102bb5f87e2eb87a1dde86)

### 7. Conclusión

¡Felicidades! Has utilizado Edge Impulse para entrenar un modelo de aprendizaje automático capaz de reconocer tus gestos y comprender cómo puedes construir modelos que clasifiquen datos de sensores o encuentren anomalías. Ahora que has entrenado tu modelo, puedes integrar tu impulso en el firmware de tu propio dispositivo integrado; consulta los tutoriales de [Implementaciones (Deployments)](https://docs.edgeimpulse.com/hardware/deployments.md). Hay ejemplos para Mbed OS, Arduino, STM32CubeIDE y cualquier otro destino que admita un compilador de C++.

O si estás interesado en más, consulta nuestros tutoriales sobre [Reconocimiento de sonido](https://docs.edgeimpulse.com/tutorials/end-to-end/sound-recognition.md) o [Clasificación de imágenes](https://docs.edgeimpulse.com/tutorials/end-to-end/image-classification.md). Si tienes una gran idea para un proyecto diferente, eso también está bien. Edge Impulse te permite capturar datos de cualquier sensor, construir [bloques de procesamiento personalizados](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-processing-blocks.md) para extraer características, y tienes total flexibilidad en tu canalización de Aprendizaje Automático con los bloques de aprendizaje.

¡Estamos ansiosos por ver lo que construirás! 🚀