import numpy as np
import matplotlib.pyplot as plt


class map:
    def __init__(self, file: str) -> None:
        self.map = np.load(file)
        self.height = self.map.shape[0]
        self.width = self.map.shape[1]
        self.max_dist = ((self.height - 1) ** 2 + 1) * (self.width - 1)

    def calc_dist(self, path: np.ndarray) -> float:
        if len(path) != self.width:
            raise ValueError("Path length does not match map width")
        dist = self.map[path[0]][0]
        for i in range(1, len(path)):
            t_dist = abs(path[i] - path[i - 1])
            if t_dist < 2:
                dist += self.map[path[i]][i]
            else:
                dist += t_dist**2 + 1
        return dist

    def print_path(self, path: np.ndarray) -> None:
        if len(path) != self.width:
            raise ValueError("Path length does not match map width")

        # Display map
        plt.figure()
        plt.imshow(self.map)

        # Plot path
        # plt.plot(range(self.width), path, color="red")
        for i in range(1, len(path)):
            if abs(path[i] - path[i - 1]) < 2:
                plt.plot([i - 1, i], [path[i - 1], path[i]], color="white")
            else:
                plt.plot([i - 1, i], [path[i - 1], path[i]], color="red")

        plt.title("Path length: " + str(int(self.calc_dist(path))))

        # Display map
        plt.show()


def create_map(height: int, width: int, prob: list) -> None:
    height = height
    width = width
    map = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            map[i][j] = np.random.choice(len(prob), p=prob) + 1
    np.save("my_map.npy", map)


# prob = [0.7, 0.15, 0.09, 0.02, 0.02, 0.01, 0.01]
# create_map(30, 75, prob)
# mp = map("my_map.npy")
# path = np.random.randint(0, mp.height, mp.width)
# print(mp.calc_dist(path))
# mp.print_path(path)

print("pause")
