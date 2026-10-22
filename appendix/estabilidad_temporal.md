# Apéndice Teórico: Análisis de Estabilidad Lineal y Métodos Implícitos

## 1. Introducción y la Ecuación de Prueba de Dahlquist

Al resolver numéricamente ecuaciones diferenciales ordinarias (EDO) del tipo:

$$\dot{x} = f(t, x)$$

nos interesa no solo la consistencia y la convergencia del método, sino fundamentalmente su estabilidad lineal. Para estudiar esto de manera canónica, Dahlquist propuso analizar el comportamiento de los esquemas numéricos aplicados a la ecuación lineal escalar de prueba:

$$\dot{x} = \lambda x$$

donde $\lambda \in \mathbb{C}$ representa el autovalor del sistema. En problemas disipativos, $\lambda = -C$ con $C > 0$, de modo que $\text{Re}(\lambda) < 0$. La solución analítica exacta es:

$$x(t) = x(0) e^{\lambda t}$$

la cual decae exponencialmente hacia cero cuando $t \to \infty$ si $\text{Re}(\lambda) < 0$. Un método numérico es estable si la solución discreta $x_n$ permanece acotada a medida que avanzan los pasos temporales $n \to \infty$ con un paso de tiempo constante $\Delta t$.

---

## 2. Factor de Amplificación y Análisis de Estabilidad

Al aplicar un método de un solo paso a la ecuación de prueba, la relación entre dos pasos consecutivos se escribe como:

$$x_{n+1} = R(z) x_n$$

donde $z = \lambda \Delta t$ y $R(z)$ es el factor de amplificación. La condición necesaria y suficiente de estabilidad lineal es:

$$|R(z)| \le 1$$

### 2.1. Euler Explícito
El esquema de Euler explícito conduce a:

$$R(z) = 1 + z$$

Para un problema real disipativo con $\lambda = -C$ ($C > 0$), la condición de estabilidad $|1 - C \Delta t| \le 1$ impone la restricción estricta en el paso de tiempo:

$$\Delta t \le \frac{2}{C}$$

### 2.2. Euler Implícito
El esquema de Euler implícito evalúa el término derecho en el nuevo nivel temporal $n+1$, obteniendo el factor de amplificación:

$$R(z) = \frac{1}{1 - z}$$

Para cualquier $\lambda$ con $\text{Re}(\lambda) \le 0$, se cumple que $|R(z)| \le 1$ sin restricciones sobre $\Delta t$. Esta propiedad se denomina A-estabilidad, haciendo que el método sea adecuado para sistemas rígidos.

---

## 3. Deducción Exacta del Euler Implícito para el Modelo SIR

En los modelos epidemiológicos no lineales como el sistema SIR, la aplicación de esquemas implícitos requiere resolver sistemas algebraicos no lineales en cada paso temporal. Consideremos la discretización mediante Euler implícito para las tres variables de estado:

$$S_{n+1} = S_n - \Delta t \beta S_{n+1} I_{n+1} \implies S_{n+1} = \frac{S_n}{1 + \Delta t \beta I_{n+1}}$$
$$I_{n+1} = I_n + \Delta t (\beta S_{n+1} I_{n+1} - \gamma I_{n+1})$$
$$R_{n+1} = R_n + \Delta t \gamma I_{n+1}$$

Sumando las ecuaciones de $S$ e $I$ para desacoplar el término no lineal de contagio:

$$(S_{n+1} - S_n) + (I_{n+1} - I_n) + \Delta t \gamma I_{n+1} = 0 \implies S_{n+1} + I_{n+1}(1 + \Delta t \gamma) - (S_n + I_n) = 0$$

Sustituyendo $S_{n+1} = \frac{S_n}{1 + \Delta t \beta I_{n+1}}$:

$$\frac{S_n}{1 + \Delta t \beta I_{n+1}} + I_{n+1}(1 + \Delta t \gamma) - (S_n + I_n) = 0$$

Multiplicando toda la expresión por el denominador común $(1 + \Delta t \beta I_{n+1})$ y agrupando potencias de $I_{n+1}$, se obtiene de forma exacta la ecuación cuadrática:

$$A \, I_{n+1}^2 + B \, I_{n+1} - I_n = 0$$

donde los coeficientes están dados explícitamente por:

$$A = \Delta t \beta (1 + \Delta t \gamma) > 0$$
$$B = 1 + \Delta t \gamma - \Delta t \beta (S_n + I_n)$$
$$C_{\text{cte}} = -I_n \le 0$$

### Existencia Incondicional y Selección de la Raíz Física

El discriminante de la ecuación cuadrática es:

$$\Delta = B^2 - 4 A (-I_n) = B^2 + 4 A I_n$$

Dado que $A > 0$ e $I_n \ge 0$, se cumple de forma incondicional que:

$$\Delta = B^2 + 4 A I_n \ge B^2 \ge 0, \quad \forall \Delta t > 0$$

Por lo tanto, **el discriminante es siempre real y no negativo sin restricción sobre el paso $\Delta t$**. 

Puesto que el producto de las dos raíces es $C_{\text{cte}} / A = -I_n / A \le 0$, existe exactamente una raíz real no negativa (físicamente admisible), dada por:

$$I_{n+1} = \frac{-B + \sqrt{B^2 + 4 A I_n}}{2 A}$$

Para evitar la cancelación sustractiva en aritmética de punto flotante cuando $B \ge 0$, se utiliza la formulación equivalente estabilizada:

$$I_{n+1} = \frac{2 I_n}{B + \sqrt{B^2 + 4 A I_n}}$$

Una vez obtenido $I_{n+1}$, se actualizan de inmediato $S_{n+1} = \frac{S_n}{1 + \Delta t \beta I_{n+1}}$ y $R_{n+1} = R_n + \Delta t \gamma I_{n+1}$.

---

## 4. Conexión con la Ecuación Logística: Monotonicidad vs Estabilidad

Consideremos la ecuación logística adimensionalizada:

$$\dot{x} = C x (1 - x)$$

Linealizando en torno al estado estacionario estable $x^* = 1$ mediante la perturbación $x(t) = 1 + u(t)$ con $|u| \ll 1$, se obtiene la ecuación linealizada de error:

$$\dot{u} = -C u$$

Aplicando el método de Euler explícito: $u_{n+1} = (1 - C \Delta t) u_n = R(z) u_n$ con $z = -C \Delta t$. Se identifican tres regímenes cualitativos según el paso temporal:
1. **Régimen Monótono y Estable ($\Delta t < 1/C$):** $0 < R(z) < 1$. La perturbación decrece monótonamente hacia 0 sin oscilaciones, reproduciendo fielmente el comportamiento de la solución continua.
2. **Régimen con Oscilaciones Espurias Acotadas ($1/C < \Delta t < 2/C$):** $-1 < R(z) < 0$. El factor de amplificación alterna de signo en cada paso, produciendo sobrepasos transitorios de la capacidad de carga ($x > 1$) y oscilaciones numéricas que se amortiguan hacia el equilibrio.
3. **Régimen Inestable y Divergente ($\Delta t > 2/C$):** $R(z) < -1 \implies |R(z)| > 1$. La perturbación se amplifica geométricamente en cada paso con alternancia de signo, causando divergencia numérica y el colapso de la simulación.
