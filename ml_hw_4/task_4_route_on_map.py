import json
import webbrowser
import openrouteservice
import folium
import random
from math import radians, sin, cos, sqrt, atan2

API_KEY = '5b3ce3597851110001cf62480a013535cc434b3187022d1f88cc527d'
TIME_LIMIT_MINUTES = 70
PROFILE = 'driving-car' # 'foot-walking', 'driving-car'

client = openrouteservice.Client(key=API_KEY)


def load_points_from_file(filepath: str):
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def approx_route_duration(coords):
    total_km = 0
    for i in range(len(coords) - 1):
        point1 = (coords[i][1], coords[i][0])
        point2 = (coords[i + 1][1], coords[i + 1][0])
        total_km += haversine(point1, point2)
    speed_kmh = 5 if PROFILE == 'foot-walking' else 20
    return (total_km / speed_kmh) * 60


def evaluate_path(path):
    if len(path) < 2:
        return 0
    coords = [(p["lon"], p["lat"]) for p in path]
    duration = approx_route_duration(coords)
    if duration > TIME_LIMIT_MINUTES:
        return 0
    return sum(p["priority"] for p in path)


def genetic_algorithm_with_subsets(points, generations=100, population_size=30, mutation_rate=0.2):
    def random_individual():
        size = random.randint(2, len(points))
        return random.sample(points, size)

    population = [random_individual() for _ in range(population_size)]

    for _ in range(generations):
        scored = [(p, evaluate_path(p)) for p in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        next_gen = [p for p, _ in scored[:5]]

        while len(next_gen) < population_size:
            parent1 = random.choice(scored[:10])[0]
            parent2 = random.choice(scored[:10])[0]
            child_set = list({p["name"]: p for p in parent1 + parent2}.values())
            random.shuffle(child_set)
            child = child_set[:random.randint(2, len(child_set))]

            if random.random() < mutation_rate:
                if random.random() < 0.5 and len(child) > 2:
                    child.pop(random.randint(0, len(child) - 1))
                else:
                    remaining = [p for p in points if p not in child]
                    if remaining:
                        child.append(random.choice(remaining))
                random.shuffle(child)

            next_gen.append(child)

        population = next_gen

    best, score = max(((p, evaluate_path(p)) for p in population), key=lambda x: x[1])
    return best, score


def build_map(best_path):
    map_center = [
        sum(p["lat"] for p in best_path) / len(best_path),
        sum(p["lon"] for p in best_path) / len(best_path)
    ]
    m = folium.Map(location=map_center, zoom_start=13)

    for p in best_path:
        folium.Marker(
            [p["lat"], p["lon"]],
            tooltip=f'{p["name"]} (priority={p["priority"]})'
        ).add_to(m)

    return m


def add_route_to_map(m, best_path):
    coords = [(p["lon"], p["lat"]) for p in best_path]
    try:
        route = client.directions(coordinates=coords, profile=PROFILE, format='geojson')
        folium.GeoJson(route, name="route").add_to(m)
    except Exception as e:
        print("Ошибка при запросе маршрута:", e)


def save_and_open_map(m, filename='best_route.html'):
    m.save(filename)
    webbrowser.open(filename)


def main():
    points = load_points_from_file('points.json')
    best_path, score = genetic_algorithm_with_subsets(points)
    print(f"Best route score: {score} (via {len(best_path)} points)")

    m = build_map(best_path)
    add_route_to_map(m, best_path)
    save_and_open_map(m)


if __name__ == "__main__":
    main()
