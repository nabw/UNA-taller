# Declaración sobre el Uso de Inteligencia Artificial (AI Disclosure)

Todo el contenido presente en este repositorio —incluyendo la arquitectura del proyecto, código fuente de soporte (`src/`), scripts de simulación y generación programática (`scripts/nb*_builder.py`, `generate_all_notebooks.py`), cuadernos de laboratorio de Jupyter (`notebooks/`), apéndices teóricos y documentación técnica— fue creado utilizando el modelo de inteligencia artificial **Gemini Flash 3.8** (Google DeepMind).

---

## Alcance del Contenido Generado

1. **Cuadernos de Laboratorio (Jupyter Notebooks):**
   - Construcción programática de las celdas de Markdown y código para las versiones de *estudiante* y *solución*.
   - Desarrollo de explicaciones teóricas, deducciones analíticas y formulación de ejercicios prácticos.
   - Implementación de esquemas numéricos de discretización (Euler explícito, Euler implícito, esquemas IMEX, Runge-Kutta de 4.º orden).
   - Generación de datos sintéticos, modelos de ruido observacional (Poisson / Gaussiano) y calibración de problemas inversos mediante optimización numérica (Nelder-Mead).

2. **Código Base y Librerías:**
   - Módulos en `src/` (`solvers.py`, `plotting.py`, `synthetic.py`, `inverse.py`).
   - Scripts ejecutables de demostración en `scripts/` (01 al 04) y herramientas de verificación (`verify_notebooks.py`).

3. **Documentación y Material Complementario:**
   - Documento principal `README.md` y guías de configuración de entorno.
   - Notas técnicas y apéndices matemáticos ubicados en el directorio `appendix/`.

---

## Contexto Pedagógico y Revisión

Este material fue concebido para el taller práctico de Modelamiento Matemático en la **Universidad Nacional (UNA)**, Costa Rica. La dirección temática, el alcance didáctico y las decisiones pedagógicas fueron coordinadas para asegurar el rigor matemático y la pertinencia formativa.
