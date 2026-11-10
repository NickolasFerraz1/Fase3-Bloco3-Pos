"""Config global dos testes.

No Windows, quando a saída de um subprocess é capturada (pipe), o Python usa
cp1252 e quebra ao imprimir emoji (UnicodeEncodeError). Forçamos UTF-8 para que
os subprocessos disparados pelos testes (ex.: src/train/train.py) funcionem
igual no Windows, Linux e Mac.
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")
