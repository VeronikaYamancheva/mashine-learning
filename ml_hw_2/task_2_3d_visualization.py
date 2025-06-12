import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import os


def load_and_prepare_data(file_path="AmesHousing.csv"):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Order', 'PID', 'Mo Sold', 'Yr Sold'], errors="ignore")
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    return X, y


def identify_columns_types(X):
    #categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = []
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return categorical_cols, numerical_cols


def remove_highly_correlated_features(X, numerical_cols, threshold=0.8):
    corr_matrix = X[numerical_cols].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    X = X.drop(columns=high_corr_features)
    numerical_cols = [col for col in numerical_cols if col not in high_corr_features]

    return X, numerical_cols


def create_preprocessor(categorical_cols, numerical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    #categorical_transformer = Pipeline(steps=[
    #    ("imputer", SimpleImputer(strategy="most_frequent")),
     #   ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    #])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_cols)
     #   , ("cat", categorical_transformer, categorical_cols)
    ])

    return preprocessor


def plot_3d_pca(X_pca, y, title="3D-визуализация данных", save_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap="viridis")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("SalePrice")
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def evaluate_lasso_model(X_train, X_test, y_train, y_test, alphas=np.logspace(-4, 3, 10)):
    rmse_values = []
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_values.append(rmse)
    return alphas, rmse_values


def plot_rmse_vs_alpha(alphas, rmse_values, title="Зависимость RMSE от alpha (Lasso)", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, rmse_values, marker="o")
    plt.xscale("log")
    plt.xlabel("Коэффициент регуляризации (alpha)")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def get_feature_names(preprocessor, categorical_cols, numerical_cols):
    feature_names = []

    feature_names.extend(numerical_cols)

    if categorical_cols:
        cat_transformer = preprocessor.named_transformers_["cat"]
        ohe = cat_transformer.named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names)

    return feature_names


def print_model_results(best_alpha, min_rmse, top_feature):
    print(f"Лучшая alpha: {best_alpha}")
    print(f"Минимальный RMSE: {min_rmse:.2f}")
    print("Признак с наибольшим влиянием:")
    print(top_feature)


def main():
    X, y = load_and_prepare_data()

    categorical_cols, numerical_cols = identify_columns_types(X)

    if numerical_cols:
        plt.figure(figsize=(12, 10))
        corr_matrix = X[numerical_cols].corr()
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, fmt=".2f")
        plt.title("Корреляционная матрица числовых признаков (до удаления)")
        plt.tight_layout()
        plt.show()

    X, numerical_cols = remove_highly_correlated_features(X, numerical_cols)

    if numerical_cols:
        plt.figure(figsize=(12, 10))
        corr_matrix = X[numerical_cols].corr()
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, fmt=".2f")
        plt.title("Ковариационная матрица числовых признаков (после удаления)")
        plt.tight_layout()
        plt.show()

    preprocessor = create_preprocessor(categorical_cols, numerical_cols)
    X_processed = preprocessor.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    plot_3d_pca(X_pca, y, save_path="plots/3d_pca_visualization.png")

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE (Linear Regression): {rmse:.2f}')

    alphas, rmse_values = evaluate_lasso_model(X_train, X_test, y_train, y_test)
    plot_rmse_vs_alpha(alphas, rmse_values, save_path="plots/rmse_vs_alpha.png")

    best_alpha = alphas[np.argmin(rmse_values)]
    final_model = Lasso(alpha=best_alpha, max_iter=5000)
    final_model.fit(X_train, y_train)

    encoded_feature_names = get_feature_names(preprocessor, categorical_cols, numerical_cols)
    coef = pd.Series(final_model.coef_, index=encoded_feature_names)
    top_n = 10
    important_features = coef[coef != 0].abs().sort_values(ascending=False).head(top_n)

    print(f"Лучшая alpha: {best_alpha}")
    print(f"Минимальный RMSE: {min(rmse_values):.2f}")
    print(f"\nТоп-{top_n} признаков по абсолютной важности:")
    for i, (feature, value) in enumerate(important_features.items(), 1):
        print(f"{i}. {feature}: {value:.4f}")


if __name__ == "__main__":
    main()
