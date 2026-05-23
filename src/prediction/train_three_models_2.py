"""
Compatibility wrapper for the renamed RUL hybrid training pipeline.

Use this instead:
    uv run python src/prediction/train_rul_hybrid_v2.py ...
"""

from train_rul_hybrid_v2 import main


if __name__ == "__main__":
    print(
        "\033[1m\033[33m[WARN] train_three_models_2.py is deprecated; "
        "use train_rul_hybrid_v2.py instead.\033[0m"
    )
    main()
