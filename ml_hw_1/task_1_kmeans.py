import imageio.v2 as imageio
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA


# Сохраняем фактическое распределение данных для сравнения результата
def save_real_cluster_plot(data, real_labels, filename, save_dir='real_labels_cluster'):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=real_labels)
    plt.title('Фактическое распределение по кластерам')
    plt.savefig(file_path)
    plt.close()


# Определяет оптимальное кол-во кластеров ("Метод локтя")
def find_k_optimal(data, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    optimal_k = np.argmin(diff_r) + 2

    plt.figure(figsize=(8, 5))
    ks = range(1, max_k + 1)
    plt.plot(ks, wcss, marker='o')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Оптимальное k = {optimal_k}')
    plt.title('Метод локтя (Elbow method)')
    plt.xlabel('Количество кластеров')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.legend()
    plt.show()

    return optimal_k


# Инициализия центроидов - случайно выбранные данные
def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


# Присваиваем данные кластерам
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = np.linalg.norm(point - centroids, axis=1)
        clusters.append(np.argmin(distances))
    return np.array(clusters)


# Меняем положение центроидов
def update_centroids(data, clusters, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points = data[clusters == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids


# Отрисовка шага алгоритма kmeans
def plot_kmeans_step(data, centroids, clusters, iteration, save_dir='kmeans_images'):
    plt.figure(figsize=(8, 6))

    for cluster_idx in range(len(centroids)):
        plt.scatter(data[clusters == cluster_idx, 0], data[clusters == cluster_idx, 1],
                    label=f'Кластер {cluster_idx + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='X')
    plt.title(f'K-means Кластеризация - Итерация {iteration}')
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'step_{iteration}.png')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return file_path


# Фактическая логика kmeans
def custom_kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for iteration in range(max_iters):
        print(f"Шаг {iteration + 1}: Присваиваем кластеры и обновляем центроиды ...")

        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if np.all(centroids == new_centroids):
            yield centroids, clusters, iteration + 1, True
            break

        centroids = new_centroids
        yield centroids, clusters, iteration + 1, False


def run_custom_kmeans(data, k, max_iters=100):
    images = []

    for centroids, clusters, iteration, is_last_step in custom_kmeans(data, k, max_iters):
        print(f"Шаг {iteration}: Генерируем plot...")
        file_path = plot_kmeans_step(data, centroids, clusters, iteration)
        images.append(imageio.imread(file_path))
        plt.imshow(images[-1])
        plt.axis('off')
        plt.show(block=False)
        plt.pause(1)

        if is_last_step:
            plt.show()
        else:
            plt.close()


def main():
    data = load_iris()
    X = data.data
    y = data.target

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    save_real_cluster_plot(X_reduced, y, 'iris_real.png')

    print('1. Определяем оптимальное количество кластеров')
    #optimal_k = find_k_optimal(X_reduced)
    optimal_k = 3

    print('2. Алгоритм')
    run_custom_kmeans(X_reduced, optimal_k)


if __name__ == '__main__':
    main()
