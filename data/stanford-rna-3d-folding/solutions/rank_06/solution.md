# Description of my solution for the competition / Descripción de mi solución para la competición

- **Author:** Rem1210
- **Date:** 2025-05-30T18:10:32.160Z
- **Topic ID:** 582377
- **URL:** https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/582377
---

I’m sharing the steps I followed to develop my solution (score: 0.484, 6th place at the end of the submission phase). This is an initial summary that I plan to expand in future updates.

---

**Considerations:**

- No external models were used.  
- No additional data was used beyond what was provided for the training phase.  
- The solution is based solely on the input RNA sequences (e.g., AAGGCCUU...), with no extra information.

**Note:**

Given the limited time and available resources (CPU i5-6500, 24 GB RAM, no GPU), I’m aware that each step can be significantly improved.

---

**Process overview:**

Briefly, the process consisted of the following steps:

1. Data acquisition, unification, and selection.  
2. TM-score calculation between each sequence and all others, grouped by sequence length to reduce computational cost.  
3. Generation of a distance matrix, clustering into n groups (classes), and selection of the most representative sequences in each group.  
4. Training of a model (Keras) to classify sequences into 5 groups based on the input sequence.  
5. Model inference. For each input sequence, the model predicts the most likely groups (top 5), and returns the 3D sequence of the most similar entry within each predicted group (based on sequence length and best TM-score within that group).

---

**Additional note:**

Step 2 is computationally expensive, especially for long sequences. I considered optimizing this step by incorporating PDB data or using data augmentation techniques, but discarded those options due to hardware limitations. These ideas could be implemented as future extensions.

---

It’s been a pleasure to take part in this competition — I’ve learned a lot and really enjoyed the process. Many thanks to the organizers and the Kaggle community for making this challenge possible. Best of luck to the other participants!



---

## Versión en español

### 1. Descripción de mi solución para la competición

Comparto los pasos que seguí para desarrollar mi solución (score: 0.484, 6.º puesto al cierre de la fase de envíos). Este es un resumen inicial que trataré de ir ampliando en futuras actualizaciones.

**Consideraciones:**

- No se utilizó ningún modelo externo.  
- No se emplearon datos adicionales más allá de los proporcionados para la fase de entrenamiento.  
- La solución se basa únicamente en las secuencias de ARN de entrada (ej. AAGGCCUU...), sin información extra.

**Nota:**

Dado el tiempo limitado y los recursos disponibles (CPU i5-6500, 24 GB de RAM, sin GPU), soy consciente de que cada uno de los pasos es claramente mejorable.

---

**Proceso seguido:**

De forma resumida, el proceso consistió en los siguientes pasos:

1. Obtención, unificación y selección de datos.  
2. Cálculo del TM-score entre cada secuencia y todas las demás, agrupando por tamaño de secuencia para tratar de reducir el coste computacional.  
3. Generación de la matriz de distancias, *clustering* (división en *n* grupos o clases) y selección de las secuencias más representativas de cada grupo.  
4. Entrenamiento de un modelo (Keras) para clasificar las secuencias en 5 grupos a partir de la secuencia de entrada.  
5. Ejecución del modelo. Para cada nueva secuencia, se predice su grupo (los 5 mejores) y se devuelve la secuencia 3D de la entrada más similar dentro de ese grupo (según el tamaño de secuencia y el mejor TM-score dentro de ese rango), para cada uno de los 5 mejores grupos predichos.

---

**Nota adicional:**

El paso 2 es computacionalmente muy costoso, especialmente para secuencias largas. Se exploró la posibilidad de optimizar este paso incorporando datos de la base PDB o mediante *data augmentation*, pero se descartó debido a las limitaciones de hardware. Estas mejoras podrían implementarse como una extensión futura.

---

Ha sido un placer participar en esta competición, he aprendido mucho y he disfrutado del proceso. Agradezco a los organizadores y a la comunidad de Kaggle por hacer posible este reto. ¡Mucha suerte al resto de participantes!
