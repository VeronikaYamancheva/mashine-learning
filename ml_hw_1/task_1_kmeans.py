import imageio.v2 as imageio
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from PIL import Image

import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA
from itertools import combinations


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
    original_data = load_iris().data if data.shape[1] == 2 else data

    for centroids, clusters, iteration, is_last_step in custom_kmeans(data, k, max_iters):
        print(f"Шаг {iteration}: Генерируем plot...")
        file_path = plot_kmeans_step(data, centroids, clusters, iteration)
        images.append(imageio.imread(file_path))
        plt.imshow(images[-1])
        plt.axis('off')
        plt.show(block=False)
        plt.pause(1)

        if is_last_step:
            final_clusters = clusters
            plt.show()

            if original_data is not None:
                plot_all_projections(original_data, final_clusters,
                                     "Финальные кластеры по всем признакам")
        else:
            plt.close()


# Если нужно склеить все изображения-графиков в одно
def combine_images_from_folder(folder_path, output_filename="combined_plot.png",
                               cols=3, figsize=(15, 10), dpi=100):

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort()

    if not image_files:
        print("В указанной папке нет изображений.")
        return
    rows = (len(image_files) // cols) + (1 if len(image_files) % cols else 0)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for i, (ax, img_file) in enumerate(zip(axes.flatten(), image_files)):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(os.path.splitext(img_file)[0])
        ax.axis('off')

    for j in range(i + 1, rows * cols):
        axes.flatten()[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Изображения успешно объединены и сохранены в {output_filename}")


def plot_all_projections(data, labels, title="Кластеры по всем парам признаков"):
    feature_names = ['Длина чашелистика', 'Ширина чашелистика',
                     'Длина лепестка', 'Ширина лепестка']

    feature_pairs = list(combinations(range(data.shape[1]), 2))
    n_pairs = len(feature_pairs)

    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))
    plt.suptitle(title, fontsize=16)
    scatter = None

    for plot_num, (i, j) in enumerate(feature_pairs, 1):
        plt.subplot(n_rows, n_cols, plot_num)
        scatter = plt.scatter(data[:, j], data[:, i], c=labels, cmap='tab10', s=20)
        plt.xlabel(feature_names[j])
        plt.ylabel(feature_names[i])
        plt.grid(True)

    if scatter is not None and len(np.unique(labels)) > 1:
        legend = plt.figlegend(*scatter.legend_elements(),
                               title='Кластеры',
                               loc='lower right',
                               bbox_to_anchor=(1, 0),
                               ncol=min(10, len(np.unique(labels))))
        plt.setp(legend.get_texts(), fontsize='small')

    plt.tight_layout()
    plt.show()


def main():
    data = load_iris()
    X = data.data
    y = data.target

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    save_real_cluster_plot(X_reduced, y, 'iris_real.png')

    print('1. Определяем оптимальное количество кластеров')
    # optimal_k = find_k_optimal(X_reduced)
    optimal_k = 3

    print('2. Алгоритм')
    run_custom_kmeans(X_reduced, optimal_k)

    combine_images_from_folder("kmeans_images", "kmeans_steps_combined.png", cols=4)


if __name__ == '__main__':
    main()
