# Apéndice Teórico: Análisis de Estabilidad Lineal de Métodos de Integración Temporal

> [!NOTE]
> Este apéndice se ofrece como lectura complementaria y de profundización teórica para el taller. Su objetivo es formalizar el comportamiento de los esquemas numéricos de integración temporal más allá de la intuición geométrica o los ejemplos prácticos.

---

## 1. Introducción y la Ecuación de Prueba de Dahlquist

Al resolver numéricamente ecuaciones diferenciales ordinarias (EDO) del tipo:

$$\dot{x} = f(t, x)$$

nos interesa no solo la **consistencia** y la **convergencia** del método, sino fundamentalmente su **estabilidad lineal**. Para estudiar esto de manera canónica, Dahlquist propuso analizar el comportamiento de los esquemas numéricos aplicados a la ecuación lineal escalar de prueba:

$$\dot{x} = \lambda x$$

donde $\lambda \in \mathbb{C}$ representa el autovalor (o la tasa linealizada) del sistema. En muchos problemas disipativos o de difusión, $\lambda = -C$ con $C > 0$, de modo que $\text{Re}(\lambda) < 0$. La solución analítica exacta es:

$$x(t) = x(0) e^{\lambda t}$$

la cual decae exponencialmente hacia cero cuando $t \to \infty$ si $\text{Re}(\lambda) < 0$. Un método numérico es **estable** si la solución discreta $x_n$ permanece acotada a medida que avanzan los pasos temporales $n \to \infty$ con un paso de tiempo constante $\Delta t$.

---

## 2. Factor de Amplificación $R(z)$ y Análisis de Métodos Clásicos

Al aplicar un método de un solo paso (como Runge-Kutta o esquemas de Euler) a la ecuación de prueba, la relación entre dos pasos consecutivos se puede escribir de la forma general:

$$x_{n+1} = R(z) x_n$$

donde se define la variable adimensional:

$$z = \lambda \Delta t$$

La función $R(z)$ se denomina **factor de amplificación** del método. La condición necesaria y suficiente de estabilidad lineal es que el módulo del factor de amplificación esté acotado por la unidad:

$$|R(z)| \le 1$$

### 2.1. Método de Euler Explícito

El esquema de Euler explícito discretiza la derivada temporal mediante diferencias hacia adelante:

$$\frac{x_{n+1} - x_n}{\Delta t} = \lambda x_n \implies x_{n+1} = (1 + \lambda \Delta t) x_n$$

Por lo tanto, el factor de amplificación es:

$$R(z) = 1 + z$$

Para garantizar la estabilidad, exigimos $|1 + z| \le 1$. Si consideramos un problema puramente disipativo real donde $\lambda = -C$ ($C > 0$), entonces $z = -C \Delta t$ es un número real negativo. La condición de estabilidad se reduce a:

$$|1 - C \Delta t| \le 1 \implies -1 \le 1 - C \Delta t \le 1 \implies 0 \le C \Delta t \le 2$$

De aquí se deriva la conocida restricción estricta del paso de tiempo para métodos explícitos:

$$\Delta t \le \frac{2}{C}$$

Si $\Delta t > \frac{2}{C}$, el factor de amplificación supera en valor absoluto a 1 ($|R(z)| > 1$), provocando que el error numérico crezca de manera exponencial y oscilatoria hasta divergir por completo.

### 2.2. Método de Euler Implícito

El esquema de Euler implícito evalúa el término derecho en el nuevo nivel temporal $n+1$:

$$\frac{x_{n+1} - x_n}{\Delta t} = \lambda x_{n+1} \implies (1 - \lambda \Delta t) x_{n+1} = x_n$$

Despejando $x_{n+1}$ obtenemos el factor de amplificación:

$$R(z) = \frac{1}{1 - z}$$

Queremos analizar la condición de estabilidad para cualquier $\lambda$ con $\text{Re}(\lambda) \le 0$ (es decir, $\text{Re}(z) \le 0$). Evaluamos el módulo al cuadrado:

$$|R(z)|^2 = \left| \frac{1}{1 - z} \right|^2 = \frac{1}{(1 - x)^2 + y^2}$$

donde $z = x + iy$ con $x = \text{Re}(z) \le 0$. Como $x \le 0$, tenemos que $(1 - x)^2 \ge 1$, por lo tanto:

$$|R(z)|^2 = \frac{1}{(1 - x)^2 + y^2} \le 1$$

Esta propiedad notable, donde el método es estable para **cualquier** tamaño de paso $\Delta t$ siempre que el sistema sea disipativo ($\text{Re}(z) \le 0$), se conoce como **A-estabilidad**. Los métodos A-estables son ideales para resolver sistemas rígidos (*stiff systems*), donde los procesos rápidos imponen restricciones prohibitivas de paso de tiempo a los métodos explícitos.

---

## 3. Esquemas IMEX (Implicit-Explicit)

En muchas aplicaciones científicas complejas (por ejemplo, dinámica de fluidos geofísicos, reacciones químicas con múltiples escalas de tiempo, o modelos climáticos), el lado derecho de la ecuación diferencial contiene procesos físicos de naturaleza muy diversa:

$$\dot{x} = f_{\text{exp}}(x) + f_{\text{imp}}(x)$$

- $f_{\text{exp}}(x)$: Términos no lineales o advectivos suaves, cuya limitación de paso de tiempo es moderada. Tratarlos de forma implícita requeriría la inversión costosa de sistemas no lineales grandes en cada paso.
- $f_{\text{imp}}(x)$: Términos lineales rígidos, disipativos o difusivos (por ejemplo, operadores de difusión espacial o términos de amortiguamiento rápido) que imponen restricciones severas de estabilidad mediante esquemas explícitos.

Los esquemas **IMEX** (*Implicit-Explicit*) combinan lo mejor de ambos mundos mediante una descomposición aditiva. Aplicando, por ejemplo, Euler explícito para los términos no lineales y Euler implícito para los términos rígidos:

$$\frac{x_{n+1} - x_n}{\Delta t} = f_{\text{exp}}(x_n) + f_{\text{imp}}(x_{n+1})$$

Esto permite mantener una alta eficiencia computacional (evitando resolver sistemas no lineales completos) al tiempo que se neutraliza la inestabilidad numérica inducida por los componentes más rígidos del modelo.

---

## 4. Conexión con la Ecuación Logística y el Origen de las Oscilaciones

Para ilustrar cómo la teoría lineal se manifiesta en modelos no lineales reales, consideremos la **ecuación logística adimensionalizada**:

$$\dot{x} = C x (1 - x)$$

donde $C > 0$ es la tasa intrínseca de crecimiento. Este sistema posee dos puntos de equilibrio (atractores/repulsores):
1. $x^* = 0$ (inestable)
2. $x^* = 1$ (estable)

Supongamos que el sistema se encuentra cerca del estado estacionario estable $x^* = 1$. Definimos una pequeña perturbación lineal u en torno al atractor:

$$x(t) = 1 + u(t) \quad \text{con } |u| \ll 1$$

Sustituyendo en la ecuación logística:

$$\frac{d}{dt}(1 + u) = C (1 + u) (1 - (1 + u)) \implies \dot{u} = C (1 + u) (-u) = -C u - C u^2$$

Ignorando los términos de orden superior $\mathcal{O}(u^2)$, obtenemos la ecuación linealizada de perturbación:

$$\dot{u} = -C u$$

Esta es exactamente la ecuación de prueba de Dahlquist con $\lambda = -C$. 

### Origen Numérico de las Oscilaciones Espurias

Si integramos esta perturbación mediante **Euler explícito** con un paso de tiempo $\Delta t$ que viole ligeramente la cota de estabilidad pero sin divergencia catastrófica inmediata (es decir, $2/C < \Delta t < 2.78$), el factor de amplificación evaluado es:

$$R(-C \Delta t) = 1 - C \Delta t$$

Si $C \Delta t = 2.5$, entonces $R = 1 - 2.5 = -1.5$. Su valor absoluto es $|R| = 1.5 > 1$ (inestable), pero al ser negativo, cada paso multiplica la perturbación por $-1.5$. Esto genera una oscilación de signo alternado (creciente en magnitud) alrededor del estado estacionario:

$$u_{n+1} = -1.5 \, u_n \implies u_n = (-1.5)^n u_0$$

Este análisis demuestra que **las oscilaciones numéricas observadas al usar un $\Delta t$ excesivamente grande no provienen de la física subyacente del modelo**, sino de un artefacto matemático del factor de amplificación del método explícito cuando cruza el semiplano negativo hacia la región de inestabilidad espiral/creciente.
