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

## 3. Deducción del Euler Implícito para el Modelo SIR

En los modelos epidemiológicos no lineales como el sistema SIR, la aplicación de esquemas implícitos requiere resolver sistemas algebraicos no lineales en cada paso temporal. Consideremos la discretización mediante Euler implícito para la ecuación de los infectados $I(t)$:

$$\frac{I_{n+1} - I_n}{\Delta t} = \beta S_{n+1} I_{n+1} - \gamma I_{n+1}$$

En una formulación totalmente implícita estándar, las variables del nuevo paso temporal están acopladas. Para ilustrar la estructura algebraica sin perder generalidad, supongamos que aproximamos la no linealidad del término de infección utilizando el estado susceptible acoplado o una linealización implícita local donde $S_{n+1}$ se aproxima o se mantiene consistente con la conservación de masa. 

Sustituyendo la relación de conservación de susceptibles $S_{n+1} = S_n - (I_{n+1} - I_n)$ o agrupando los términos no lineales del tipo $\beta I_{n+1}^2$, la ecuación discreta se reorganiza en la forma de una ecuación cuadrática general para la incógnita $I_{n+1}$:

$$A \, I_{n+1}^2 + B \, I_{n+1} - I_n = 0$$

donde los coeficientes $A$ y $B$ dependen de los parámetros del modelo ($\beta$, $\gamma$), el paso temporal $\Delta t$ y el estado previo $I_n$ (así como $S_n$). 

### Selección de la Raíz Física

Aplicando la fórmula general para resolver la ecuación cuadrática, obtenemos dos posibles raíces matemáticas:

$$I_{n+1} = \frac{-B \pm \sqrt{B^2 + 4 A I_n}}{2 A}$$

Dado que $I_{n+1}$ representa una proporción de individuos infectados, debe ser una cantidad real y no negativa. Analizando los signos de los coeficientes derivados de la dinámica epidemiológica (donde $A > 0$ y el término bajo la raíz garantiza existencia real para pasos de tiempo razonables), la rama con el signo negativo produce valores negativos o incongruentes, mientras que la **única raíz física admisible** es la que selecciona el signo positivo:

$$I_{n+1} = \frac{-B + \sqrt{B^2 + 4 A I_n}}{2 A}$$

Este detalle algebraico demuestra por qué la resolución de esquemas implícitos requiere un tratamiento analítico cuidadoso.

---

## 4. Conexión con la Ecuación Logística y Oscilaciones Espurias

Consideremos la ecuación logística adimensionalizada:

$$\dot{x} = C x (1 - x)$$

Linealizando en torno al estado estacionario estable $x^* = 1$ mediante la perturbación $x(t) = 1 + u(t)$ con $|u| \ll 1$, se obtiene la ecuación linealizada de error:

$$\dot{u} = -C u$$

Si se integra esta perturbación mediante Euler explícito con un paso de tiempo que supere la cota de estabilidad pero sin divergencia inmediata, el factor de amplificación negativo $R = 1 - C \Delta t < -1$ multiplica la perturbación alternando su signo en cada paso:

$$u_{n+1} = R \cdot u_n$$

Esto genera oscilaciones numéricas espurias crecientes que no corresponden a la dinámica biológica real del sistema, evidenciando los límites de los métodos explícitos.
