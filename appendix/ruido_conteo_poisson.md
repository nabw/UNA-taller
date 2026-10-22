# Apéndice Teórico: Ruido de Conteo y Distribución de Poisson en Modelos Epidemiológicos

## 1. Limitaciones del ruido gaussiano aditivo

En los modelos fenomenológicos iniciales, es común simular la incertidumbre observacional añadiendo ruido gaussiano aditivo a las trayectorias deterministas:

$$I_{\text{obs}}(t) = I(t) + \epsilon(t), \quad \epsilon(t) \sim \mathcal{N}(0, \sigma^2)$$

Aunque este enfoque es útil desde el punto de vista pedagógico por su simplicidad analítica, presenta fallas fundamentales al modelar datos epidemiológicos reales basados en conteos poblacionales:

1. **Variables continuas sobre la recta real:** La distribución normal asigna probabilidad no nula a cualquier valor en $(-\infty, \infty)$. Sin embargo, los casos de infección son necesariamente cantidades enteras no negativas.
2. **Valores negativos:** Con desviaciones estándar moderadas o altas ($\sigma$), el modelo estocástico puede generar proporciones o conteos negativos, lo que obliga a aplicar truncamientos artificiales (por ejemplo, recortar en cero) que deforman la distribución estadística.
3. **Homocedasticidad artificial:** El ruido gaussiano estándar asume que la varianza $\sigma^2$ es constante e independiente de la magnitud del brote. En la práctica, los datos de conteo muestran mayor variabilidad absoluta en los peaks epidémicos y menor variabilidad en los períodos de baja incidencia.

---

## 2. El Proceso de Conteo y la Distribución de Poisson

Cuando observamos un sistema epidemiológico, los datos provienen de contar eventos discretos (nuevos casos detectados en un intervalo de tiempo $\Delta t$) en una población finita de tamaño $N_{\text{pop}}$. 

Sea $C_k$ la variable aleatoria que representa el número de casos observados en el intervalo $t_k$. Asumimos que $C_k$ sigue una distribución de Poisson:

$$C_k \sim \text{Poisson}(\lambda_k)$$

donde el parámetro de tasa $\lambda_k$ está determinado por la proporción de infectados predicha por el modelo determinista y el tamaño poblacional:

$$\lambda_k = N_{\text{pop}} I(t_k)$$

### Propiedades Fundamentales

- **Soporte:** El espacio muestral está compuesto exclusivamente por enteros no negativos: $\{0, 1, 2, \dots\}$.
- **Media y Varianza Iguales:** Una de las características distintivas de la distribución de Poisson es que su esperanza matemática coincide con su varianza:
  $$\mathbb{E}[C_k] = \lambda_k, \quad \text{Var}(C_k) = \lambda_k$$

Esta propiedad implica que la dispersión de las observaciones crece de manera proporcional a la media del brote, reflejando fielmente la realidad estocástica de los contagios.

---

## 3. Heterocedasticidad y Efecto Embudo en Proporciones

En los modelos ajustados a proporciones de incidencia observadas, definimos la proporción estocástica como:

$$I_{\text{obs}}(t_k) = \frac{C_k}{N_{\text{pop}}}$$

Aplicando las propiedades de la varianza para variables aleatorias escaladas por una constante ($1/N_{\text{pop}}$), obtenemos la varianza de la proporción observada:

$$\text{Var}(I_{\text{obs}}) = \text{Var}\left( \frac{C_k}{N_{\text{pop}}} \right) = \frac{1}{N_{\text{pop}}^2} \text{Var}(C_k) = \frac{\lambda_k}{N_{\text{pop}}^2} = \frac{N_{\text{pop}} \, I(t_k)}{N_{\text{pop}}^2} = \frac{I(t_k)}{N_{\text{pop}}}$$

### Implicación Estructural: El Efecto Embudo

La varianza de la proporción observada depende directamente del nivel de infección $I(t_k)$:
- **Peak de la epidemia:** Cuando $I(t_k)$ alcanza su valor máximo, la varianza absoluta es mayor.
- **Colas de la epidemia:** Cuando $I(t_k) \to 0$ (al inicio o al final del brote), la varianza se aproxima a cero.

Esta dependencia explícita de la varianza respecto a la media genera un comportamiento de "embudo" en los diagramas de dispersión temporal de los residuos, invalidando el supuesto de homocedasticidad requerido por los métodos de mínimos cuadrados ordinarios clásicos.

---

## 4. Implicaciones en la Estimación de Parámetros

### Mínimos Cuadrados Ordinarios (MCO)
El método de Mínimos Cuadrados Ordinarios minimiza la suma de errores cuadráticos:

$$\mathcal{L}_{\text{MCO}} = \sum_{k} \left( I_{\text{obs}}(t_k) - I_{\text{mod}}(t_k) \right)^2$$

Este enfoque otorga el mismo peso estadístico a todos los puntos de la serie temporal, tratando de igual manera los datos en las colas (con baja variabilidad y alta precisión relativa) y los puntos cercanos al peak (con mayor varianza natural). Esto genera estimaciones sesgadas de los parámetros epidemiológicos.

### Verosimilitud de Poisson y Deviance
Para reflejar correctamente la naturaleza estocástica de los conteos, se formula la función de verosimilitud basada en la distribución de Poisson. La función de log-verosimilitud negativa (o *Poisson Negative Log-Likelihood*), omitiendo constantes independientes de los parámetros, se expresa como:

$$\ell(\theta) = - \sum_{k} \left[ C_k \ln(\lambda_k(\theta)) - \lambda_k(\theta) \right]$$

> **Nota de implementación computacional:** Si $\lambda_k(\theta) \to 0$ al inicio del brote, el término $\ln(\lambda_k)$ tiende a $-\infty$. En código científico se aplica una regularización numérica acotando inferiormente la tasa (por ejemplo, con `np.maximum(lambda_k, 1e-12)` o sumando una constante pequeña $\epsilon > 0$).

Al maximizar esta verosimilitud (o minimizar la deviance estadística asociada), las observaciones con menores conteos (menor $\lambda_k$) reciben una ponderación inversa adecuada a su menor varianza, mientras que las discrepancias relativas en zonas de alta incidencia son evaluadas con el rigor probabilístico correcto, mejorando la robustez y la consistencia estadística de la calibración del modelo.
