"""Perfilamento do treino com cProfile (Vídeo 6.3).

Roda o treino sob o profiler e imprime as funções que mais consomem tempo,
ordenadas por `cumulative`. Use para achar gargalos antes de otimizar no escuro.

Uso:
    python scripts/profile_training.py            # imprime top 15
    python scripts/profile_training.py --dump out.prof   # salva para snakeviz
"""
import argparse
import cProfile
import pstats
from pstats import SortKey

from src.ml_pipeline.seeds import set_global_seeds
from src.ml_pipeline.pipeline import train


def _workload():
    """O que será perfilado: um treino completo."""
    set_global_seeds()
    train(model_name="rf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perfila o treino com cProfile")
    parser.add_argument("--dump", help="caminho .prof para abrir no snakeviz")
    parser.add_argument("--top", type=int, default=15, help="linhas a exibir")
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    _workload()
    profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs().sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(args.top)

    if args.dump:
        stats.dump_stats(args.dump)
        print(f"✅ perfil salvo em {args.dump} (abra com: snakeviz {args.dump})")


if __name__ == "__main__":
    main()
