# Desafios Procesamiento de Lenguaje Natural

# Desafio 1
### **Vectorización y Medición de Similitud entre Documentos**
- Se utilizó **TF-IDF** para vectorizar los documentos del dataset **20 Newsgroups**.
- Se seleccionaron **5 documentos al azar** y se calculó la **similitud coseno** entre cada documento y el resto del corpus.
- Se encontraron los **5 documentos más similares** para cada caso, analizando la relación según el contenido y la clase.

**Ejemplo de resultados**:
- Documento original: 
  *"Maybe not to you. But to those who stand on this base, He is precious."*
- Documentos más similares:
  - Similaridad: **0.2658** | Clase: `misc.forsale`
  - Similaridad: **0.2076** | Clase: `rec.sport.baseball`

### **Clasificación con Naïve Bayes y Optimización del F1-Score**
- Se entrenaron modelos **Naïve Bayes Multinomial** y **ComplementNB**.
- Se optimizó el hiperparámetro `alpha` utilizando **GridSearchCV**.
- Los resultados de clasificación (f1-score macro):
  - **MultinomialNB**: 0.6772
  - **ComplementNB**: 0.6823

### **Transposición de Matriz y Similitud entre Palabras**
- La matriz **documento-término** fue transpuesta a **término-documento**.
- Se calculó la **similitud coseno** entre palabras seleccionadas manualmente:

| Palabra           | Palabras Similares             |
|-------------------|--------------------------------|
| **computer**      | shopper, verlag, delicate, drive, hackers |
| **space**         | nasa, shuttle, exploration, aeronautics, cfa |
| **religion**      | religious, religions, crusades, christianity, categorized |
| **car**           | cars, dealer, civic, loan, owner |
| **science**       | cognitivists, behaviorists, scientific, empirical, sects |

---

# Desafio 2

## Descripción General
En este ejercicio trabajamos con el dataset **20 Newsgroups**, que contiene textos de noticias categorizados en distintos temas. Utilizamos el modelo **Word2Vec** de Gensim para generar **word embeddings** y exploramos las relaciones semánticas entre palabras a través de **similitudes** y **visualizaciones**.

---

## Proceso

### 1. **Carga y Preprocesamiento del Dataset**
- Se seleccionaron tres categorías del dataset:
  - `rec.autos`: Noticias relacionadas con autos.
  - `sci.space`: Noticias sobre ciencia y espacio.
  - `talk.politics.mideast`: Noticias sobre política y conflictos.

- Se realizó el preprocesamiento:
  - Tokenización del texto en palabras.
  - Eliminación de **stopwords** y **puntuación**.
  - Conversión a minúsculas.

### 2. **Entrenamiento del Modelo Word2Vec**
Se entrenó un modelo **Word2Vec** utilizando:
- **Algoritmo**: Skip-Gram (`sg=1`).
- **Dimensionalidad de embeddings**: 100.
- **Contexto**: Ventana de 5 palabras (`window=5`).
- **Frecuencia mínima**: Palabras con aparición >= 5 (`min_count=5`).

### 3. **Exploración de Palabras Similares**
Se analizaron palabras de interés en los dominios de **autos**, **espacio** y **política**:

| Palabra           | Palabras Similares             |
|-------------------|--------------------------------|
| **car**           | porsche, accord, ford, mazda   |
| **engine**        | ins, compartment, cam, v6      |
| **space**         | digest, nasp, station, wales   |
| **nasa**          | select, ames, goldin, dryden   |
| **government**    | endorsed, likud, leasing       |
| **war**           | gulf, undeclared, enemy, torch |

### 4. **Visualización de Embeddings**
- Los embeddings generados se redujeron a **2 dimensiones** utilizando **t-SNE**.
- La visualización muestra cómo palabras con significados similares se agrupan en el espacio:
  - **Autos**: `car` y `engine`.
  - **Espacio**: `nasa` y `space`.
  - **Política**: `government` y `war`.

![Visualización t-SNE](imagen.png)

---

## Conclusiones
1. **Relaciones semánticas capturadas**:
   - El modelo identificó correctamente la similitud entre palabras dentro de dominios específicos.

2. **Coherencia de la visualización**:
   - t-SNE permitió observar agrupamientos lógicos en el espacio de embeddings.

3. **Aplicaciones**:
   - Los embeddings generados pueden usarse en tareas como clasificación de texto, detección de temas o tests de analogías.

---

# Desafío 3

### **Generación de Secuencias con Modelos RNN**

En este desafío, se utilizó un nuevo dataset para implementar estrategias de **generación de secuencias** con modelos RNN. El objetivo fue entrenar un modelo que pueda predecir la siguiente palabra dada una secuencia de entrada (many-to-one).

### **Proceso**
1. **Preparación del Dataset**:
   - Se utilizó un corpus de texto extenso (ejemplo: obras de Shakespeare).
   - El texto fue tokenizado, convertido a secuencias numéricas y dividido en entradas (X) y salidas (y).

2. **Construcción del Modelo RNN**:
   - Se implementó un modelo basado en **LSTM** con las siguientes características:
     - Capa de **Embedding** con dimensión de 50.
     - Dos capas **LSTM** con 128 unidades cada una.
     - Capa densa final con activación **softmax** para predecir la siguiente palabra.

3. **Entrenamiento del Modelo**:
   - El modelo fue entrenado utilizando **Sparse Categorical Crossentropy** como función de pérdida.
   - Validación sobre un subconjunto de los datos.

4. **Resultados del Entrenamiento**:
   - El modelo muestra una **pérdida de entrenamiento** decreciente pero una **pérdida de validación** creciente, lo que indica posible **overfitting**.

   **Texto generado**:
   > "to be or not to be taken to sleep to be a month to his cheeks"

### **Conclusiones**
1. Los modelos **LSTM** capturan patrones básicos del lenguaje, pero el modelo mostró **overfitting** al final del entrenamiento.
2. La generación de texto es **coherente**, aunque limitada en creatividad debido al tamaño del dataset y la configuración de hiperparámetros.
3. Ajustar la **regularización** (Dropout) y usar un dataset más amplio puede mejorar el rendimiento y la generación de texto.

---

# Desafío 4

## Descripción General
En este desafío construimos un **QA Bot** (Question-Answer Bot) utilizando un modelo **Seq2Seq** basado en redes neuronales **LSTM**. Implementamos un **encoder-decoder** para procesar pares de preguntas y respuestas del **Cornell Movie Dialogues Dataset**.

Además, utilizamos embeddings preentrenados **FastText** para mejorar la representación de palabras y lograr un modelo eficiente de procesamiento de lenguaje natural (NLP).

---

## Proceso Paso a Paso

### 1. Dataset Utilizado
- **Cornell Movie Dialogues Dataset**: Dataset que contiene diálogos entre personajes de películas, ideal para construir bots de conversación.
- El dataset fue descargado desde la web oficial de la Universidad de Cornell:
  - `movie_lines.txt`: Contiene líneas individuales de diálogo.
  - `movie_conversations.txt`: Contiene las conversaciones entre personajes (pares de diálogos).

---

### 2. Procesamiento de Datos
1. **Limpieza de Texto**:
   - Convertimos todo el texto a minúsculas.
   - Eliminamos caracteres especiales, puntuación y espacios extra.
   - **Ejemplo**:
     - Entrada: `"What's your name?"`
     - Salida: `"whats your name"`

2. **Emparejamiento de Preguntas y Respuestas**:
   - Extraemos las líneas de diálogo desde `movie_lines.txt`.
   - Emparejamos preguntas y respuestas usando las referencias en `movie_conversations.txt`.
   - Añadimos **tokens especiales** a las respuestas:
     - `<sos>`: Indica el inicio de una respuesta.
     - `<eos>`: Indica el final de una respuesta.

   **Ejemplo de Pares Procesados**:
   - Pregunta: `"how are you"`
   - Respuesta: `"<sos> i am fine <eos>"`

3. **Tokenización y Padding**:
   - Tokenizamos las preguntas y respuestas con un vocabulario máximo de **8000 palabras**.
   - Aplicamos **padding** para asegurar que todas las secuencias tengan la misma longitud:
     - Encoder: `pre-padding` (al inicio).
     - Decoder: `post-padding` (al final).

---

### 3. Embeddings
- Utilizamos **FastText** preentrenado (`fasttext-wiki-news-subwords-300`) para representar las palabras en un espacio vectorial de **300 dimensiones**.
- Creamos una matriz de embeddings donde cada palabra del vocabulario tiene su representación correspondiente.
- Las palabras no encontradas en los embeddings preentrenados se inicializan con vectores de ceros.

---

### 4. Modelo Seq2Seq (Encoder-Decoder)
Implementamos un modelo **Encoder-Decoder** con redes **LSTM**:
1. **Encoder**:
   - Recibe las preguntas como entrada y procesa la secuencia utilizando una capa **Embedding** y una capa **LSTM**.
   - La salida de la LSTM son los **estados ocultos** (`state_h`, `state_c`), que sirven como contexto inicial para el decoder.

2. **Decoder**:
   - Recibe la secuencia de respuestas con el token `<sos>` como entrada.
   - Utiliza los estados del encoder como entrada inicial.
   - Predice palabra por palabra utilizando una **capa Dense** con activación `softmax`.

3. **Arquitectura**:

Encoder: Input -> Embedding -> LSTM -> Context 
Decoder: Input -> Embedding -> LSTM (con estados del Encoder) -> Dense -> Predicción

---

### 5. Entrenamiento del Modelo
- Utilizamos un modelo Seq2Seq con una arquitectura **Encoder-Decoder** basado en LSTMs y embeddings **FastText** preentrenados.
- **Parámetros clave del modelo**:
  - **Embeddings**: FastText de 300 dimensiones.
  - **Tamaño de vocabulario**: 11,059 palabras.
  - **Longitud máxima de secuencia**: 10 tokens.
  - **Unidades LSTM**: 256.
  - **Dropout**: 0.2.
- **Resumen de parámetros**:
  - **Total de parámetros**: 10,618,299 (40.51 MB).
  - **Parámetros entrenables**: 3,982,899 (15.19 MB).
  - **Parámetros no entrenables**: 6,635,400 (25.31 MB).

---

### 6. Resultados del Entrenamiento
- El modelo logró una **precisión en el entrenamiento** de aproximadamente **47.4%** y una **precisión de validación** de **44.0%**.
- **Pérdida final**:
  - **Entrenamiento**: 2.8390.
  - **Validación**: 4.1394.

### **Observaciones:**
- El modelo mostró **mejora progresiva** en la pérdida durante las primeras épocas.
- La pérdida de validación **se estabilizó** sin mejorar significativamente en las últimas épocas, indicando posible **overfitting**.

---

### 7. Resultados de Inferencia
A pesar del entrenamiento exitoso, las respuestas generadas por el modelo fueron **repetitivas** y carecieron de contenido significativo:

### **Ejemplos:**
| Pregunta                | Respuesta Generada   |
|-------------------------|----------------------|
| *"what's your name?"*   | *"i dont know"*      |
| *"do you like movies?"* | *"i dont know"*      |
| *"Do you read?"*        | *"i dont know"*      |
| *"Do you have any pet?"*| *"i dont know"*      |
| *"Where are you from?"* | *"i dont know"*      |

---

### 8. Posibles Causas del Bajo Rendimiento
1. **Calidad y cantidad de datos**:  
   - El dataset Cornell Movie Dialogues fue reducido a 10,000 pares de conversación.
   - Baja diversidad de datos limita la capacidad del modelo para generalizar.
   
2. **Restricción en la longitud de secuencias**:  
   - `MAX_LEN = 10` restringe oraciones más largas, dificultando la generación de respuestas completas.

3. **Capacidad del modelo**:  
   - La arquitectura simple con **una sola capa LSTM** (256 unidades) no es suficiente para capturar patrones complejos.

4. **Sobreajuste**:  
   - La pérdida de validación se estabilizó, lo cual indica que el modelo no mejora más con los datos disponibles.

---

### **9. Conclusión General**
El modelo logró entrenarse con éxito pero presenta limitaciones en la **inferencia** debido a problemas con los datos, la arquitectura y la capacidad de generalización. Quizas implementando algunas mejoras podemos notar algun cambio, igualmente se probo con otro conjunto de hiperparametros y siempre llegamos al mismo resultado.

---

## Tecnologías Utilizadas
- **Python**
- **NLTK**: Preprocesamiento de texto.
- **Gensim**: Entrenamiento de embeddings con Word2Vec.
- **Scikit-learn**: Reducción de dimensionalidad con t-SNE.
- **Matplotlib**: Visualización de embeddings.

---

## Ejecución
1. Instalar dependencias:
   ```bash
   pip install nltk gensim scikit-learn matplotlib
   ```

2. Ejecutar el script:
   ```bash
   python embeddings_exploration.py
   ```

3. Visualizar los resultados y la gráfica generada.
