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
