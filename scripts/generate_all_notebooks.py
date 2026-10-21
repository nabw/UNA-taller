#!/usr/bin/env python3
"""
Generador automatizado de cuadernos de laboratorio para el curso.
Produce los 4 cuadernos en dos versiones (estudiante y solucion):
- 01_modelo_logistico.ipynb
- 02_modelo_sir_estabilidad.ipynb
- 03_datos_sinteticos_ruido.ipynb
- 04_calibracion_problema_inverso.ipynb

Requisitos estrictos:
- Cero emojis.
- Cero explicaciones de complejidad algoritmica (sin Big-O).
- Autocontenidos (importaciones completas, funciones auxiliares incluidas).
- Formato nbformat 4.5 valido.
"""

import os
import sys
import json
import re
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configurar matplotlib para renderizado headless
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Importar constructores de cada cuaderno
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.nb1_builder import build_nb1_cells
from scripts.nb2_builder import build_nb2_cells
from scripts.nb3_builder import build_nb3_cells
from scripts.nb4_builder import build_nb4_cells


def check_no_emojis(text: str) -> None:
    """Verifica que el texto no contenga emojis ni simbolos pictograficos."""
    for char in text:
        cp = ord(char)
        if (
            0x1F600 <= cp <= 0x1F64F or  # Emoticons
            0x1F300 <= cp <= 0x1F5FF or  # Misc Symbols and Pictographs
            0x1F680 <= cp <= 0x1F6FF or  # Transport and Map
            0x1F700 <= cp <= 0x1F77F or  # Alchemical
            0x1F780 <= cp <= 0x1F7FF or  # Geometric Shapes Extended
            0x1F800 <= cp <= 0x1F8FF or  # Supplemental Arrows-C
            0x1F900 <= cp <= 0x1F9FF or  # Supplemental Symbols and Pictographs
            0x1FA00 <= cp <= 0x1FA6F or  # Chess, symbols
            0x1FA70 <= cp <= 0x1FAFF or  # Symbols and Pictographs Extended-A
            0x2600 <= cp <= 0x26FF or    # Misc symbols
            0x2700 <= cp <= 0x27BF       # Dingbats
        ):
            raise ValueError(f"Emoji detectado: {char!r} (U+{cp:04X}) en texto:\n{text[:120]}")


def check_no_big_o(text: str) -> None:
    """Verifica que el texto no contenga notacion de complejidad asintotica Big-O."""
    pattern = r"(\\mathcal\{?O\}?|(?<![a-zA-Z0-9_])O)\s*\([a-zA-Z0-9_\\^+\-* /]+\)"
    match = re.search(pattern, text)
    if match:
        raise ValueError(f"Notacion Big-O detectada: '{match.group(0)}' en texto:\n{text[:120]}")


def raw_cells_to_nb_cells(raw_cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convierte la lista de especificaciones de celda en celdas estandar de Jupyter."""
    nb_cells = []
    for rc in raw_cells:
        cell_type = rc["type"]
        source = rc["source"]
        check_no_emojis(source)
        check_no_big_o(source)

        lines = [l + "\n" for l in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines[-1] = ""
        if lines and lines[-1] == "":
            lines.pop()
            if lines:
                lines[-1] = lines[-1].rstrip("\n")

        if cell_type == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines
            })
        elif cell_type == "code":
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            })
        else:
            raise ValueError(f"Tipo de celda desconocido: {cell_type}")
    return nb_cells


def execute_notebook_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ejecuta las celdas de codigo en un espacio de nombres compartido,
    capturando stdout y figuras de matplotlib en formato display_data PNG.
    """
    namespace: Dict[str, Any] = {}
    exec_count = 1
    executed_cells = []

    for cell in cells:
        if cell["cell_type"] != "code":
            executed_cells.append(cell)
            continue

        code_text = "".join(cell["source"])
        outputs: List[Dict[str, Any]] = []

        stdout_trap = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_trap

        plt.close("all")
        error_msg = None

        try:
            exec(code_text, namespace)
        except Exception as e:
            error_msg = str(e)
            import traceback
            tb_lines = traceback.format_exc().splitlines()
        finally:
            sys.stdout = old_stdout

        # Capturar stdout
        stdout_val = stdout_trap.getvalue()
        if stdout_val:
            lines = [l + "\n" for l in stdout_val.splitlines()]
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": lines
            })

        # Capturar figuras de matplotlib
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64_img = base64.b64encode(buf.read()).decode("ascii")
            outputs.append({
                "data": {
                    "image/png": b64_img,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
        plt.close("all")

        if error_msg:
            outputs.append({
                "ename": "Exception",
                "evalue": error_msg,
                "output_type": "error",
                "traceback": tb_lines
            })

        new_cell = dict(cell)
        new_cell["execution_count"] = exec_count
        new_cell["outputs"] = outputs
        executed_cells.append(new_cell)
        exec_count += 1

        if error_msg:
            raise RuntimeError(f"Error al ejecutar celda {exec_count-1}:\n{error_msg}\nCodigo:\n{code_text}")

    return executed_cells


def make_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Empaqueta las celdas en el esquema nbformat 4.5 de Jupyter Notebook."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    dir_estudiante = base_dir / "notebooks" / "estudiante"
    dir_solucion = base_dir / "notebooks" / "solucion"

    dir_estudiante.mkdir(parents=True, exist_ok=True)
    dir_solucion.mkdir(parents=True, exist_ok=True)

    notebook_configs = [
        ("01_modelo_logistico.ipynb", build_nb1_cells),
        ("02_modelo_sir_estabilidad.ipynb", build_nb2_cells),
        ("03_datos_sinteticos_ruido.ipynb", build_nb3_cells),
        ("04_calibracion_problema_inverso.ipynb", build_nb4_cells),
    ]

    print("=" * 80)
    print("GENERADOR DE CUADERNOS DE LABORATORIO (ESTUDIANTE Y SOLUCION)")
    print("=" * 80)

    for filename, builder_fn in notebook_configs:
        print(f"\nGenerando {filename}...")

        # 1. Version Estudiante
        raw_cells_est = builder_fn(is_solution=False)
        nb_cells_est = raw_cells_to_nb_cells(raw_cells_est)
        nb_dict_est = make_notebook(nb_cells_est)

        path_est = dir_estudiante / filename
        with open(path_est, "w", encoding="utf-8") as f:
            json.dump(nb_dict_est, f, indent=1, ensure_ascii=False)
        print(f"  [OK] Estudiante guardado en: {path_est.relative_to(base_dir)}")

        # 2. Version Solucion (ejecutada con captura de salidas)
        raw_cells_sol = builder_fn(is_solution=True)
        nb_cells_sol = raw_cells_to_nb_cells(raw_cells_sol)
        print(f"  Ejecutando celdas de solucion para {filename}...")
        executed_cells_sol = execute_notebook_cells(nb_cells_sol)
        nb_dict_sol = make_notebook(executed_cells_sol)

        path_sol = dir_solucion / filename
        with open(path_sol, "w", encoding="utf-8") as f:
            json.dump(nb_dict_sol, f, indent=1, ensure_ascii=False)
        print(f"  [OK] Solucion guardada en:   {path_sol.relative_to(base_dir)}")

    # 3. Remover cuadernos antiguos obsoletos
    old_notebooks = [
        base_dir / "notebooks" / "01_laboratorio_estudiante.ipynb",
        base_dir / "notebooks" / "01_laboratorio_solucion.ipynb"
    ]
    for old_nb in old_notebooks:
        if old_nb.exists():
            old_nb.unlink()
            print(f"  [REMOVIDO] Cuaderno antiguo eliminado: {old_nb.relative_to(base_dir)}")

    print("\n" + "=" * 80)
    print("PROCESO DE GENERACION COMPLETADO CON EXITO.")
    print("=" * 80)


if __name__ == "__main__":
    main()
