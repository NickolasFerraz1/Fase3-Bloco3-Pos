"""Demo: overfitting vs modelo balanceado."""
from sklearn.tree import DecisionTreeClassifier

from train.split import get_train_test_split


def comparar_profundidades():
    X_train, X_test, y_train, y_test = get_train_test_split()

    print(f"{'max_depth':<12} {'train':>8} {'test':>8} {'gap':>8}")
    print("-" * 40)

    for depth in [1, 3, 5, None]:  # None = sem limite (overfitting!)
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        gap = train_score - test_score

        label = "None" if depth is None else str(depth)
        print(f"{label:<12} {train_score:>8.3f} {test_score:>8.3f} {gap:>8.3f}")


if __name__ == "__main__":
    comparar_profundidades()
