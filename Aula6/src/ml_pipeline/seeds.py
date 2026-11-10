"""Fixação de seeds para reprodutibilidade (Vídeo 6.2).

Uma única fonte de verdade para a seed global. Importar e chamar `set_global_seeds`
no início de qualquer script garante que treino, split e amostragens sejam
determinísticos entre execuções e máquinas.
"""
import os
import random

import numpy as np

SEED = 42


def set_global_seeds(seed: int = SEED) -> int:
    """Fixa as seeds de `random`, `numpy` e `PYTHONHASHSEED`.

    Args:
        seed: Valor da seed. Default: 42.

    Returns:
        A seed aplicada (útil para logar no MLflow).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
