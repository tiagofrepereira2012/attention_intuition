# In this file we generate the shape dataset (triangles and rectangles) used on Francois Fleuret's lecture on the attention mechanism
# https://fleuret.org/dlc/materials/dlc-handout-13-2-attention-mechanisms.pdf
#
#
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset

np.random.seed(10)


def norm_0_1(X):
    return (X - X.min()) / (X.max() - X.min())


@dataclass
class ShapeLog:
    """
    A class to store the information about the shapes in the image
    """

    shape_type: str
    base_point: int
    width: int
    height: int


def generate_pair(size: int = 100, max_height: int = 50, noise_std: float = 0.3):
    """
    Generate a pair of images. One with the randomly generated sequence of triangles and/or rectangles
    and the other one with the average hights of the shapes in the first image.

    Args:
        size: The size of the image on the x axis
        max_hight: The maximum hight of the shapes

    Returns:
        A tuple of two images. The first one is the image with the shapes and the second one is the image with the average hight of the shapes
    """

    min_height = 5
    base_size_range = [3, 10]
    offset = 0
    min_step = 10

    shape_choices = ["triangle", "rectangle"]
    # shape_choices = ["triangle"]

    shape_logs = []

    input = np.zeros(size)

    def draw(input, shape_type, base_point, width, height):
        if shape_type == "triangle":
            pick_point = (base_point + (base_point + width)) // 2
            input[pick_point] = height

            # Computing the triangle's first ramp
            num_points = pick_point - base_point
            values = np.linspace(0, height, num=num_points, endpoint=False, dtype=int)
            input[base_point : (base_point + width // 2)] = values

            # Computing the triangle's second ramp
            num_points = (base_point + width) - pick_point
            values = np.linspace(height, 0, num=num_points, endpoint=False, dtype=int)
            input[(base_point + width // 2) : base_point + width] = values

        else:
            # Rectangle
            input[base_point : (base_point + width)] = height

        return input

    def average_shapelogs_per_type(shape_logs):
        averages = {}
        for shape in shape_choices:
            averages[shape] = np.mean([s.height for s in shape_logs if s.shape_type == shape])

        input_averages = np.zeros(size)
        for s in shape_logs:
            input_averages = draw(input_averages, s.shape_type, s.base_point, s.width, averages[s.shape_type])

        return input_averages

    while offset < size:
        # Picking a point based on the current offset and the max displacement allowed
        base_point = np.random.randint(offset, offset + base_size_range[1])

        # Picking the width of the shape
        width = np.random.randint(base_size_range[0], base_size_range[1])

        if base_point + width > size:
            break

        # Picking the height of the shape
        height = np.random.randint(min_height, max_height)

        # Picking the shape type
        shape_type = np.random.choice(shape_choices)

        shape_logs.append(ShapeLog(shape_type, base_point, width, height))

        input = draw(input, shape_type, base_point, width, height)

        offset = base_point + width + min_step

    input += np.random.normal(0, noise_std, size)

    target = average_shapelogs_per_type(shape_logs) + np.random.normal(0, noise_std, size)

    return norm_0_1(np.expand_dims(input, axis=0).astype(np.float32)), norm_0_1(
        np.expand_dims(target, axis=0).astype(np.float32)
    )


class ShapeDataset(Dataset):
    def __init__(self, max_samples=1000, size=100, max_height=50, noise_std=0.3):
        """
        Shape dataset.
        It generates a pair of images. One with the randomly generated sequence of triangles and/or rectangles
        and the other one with the average hights of the shapes in the first image.

        Args:
            max_samples: The number of samples to generate
            size: The size of the image on the x axis
            max_hight: The maximum hight of the shapes
            noise_std: The standard deviation of the noise to add to the images

        """
        self.images = [generate_pair() for _ in range(max_samples)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx][0], self.images[idx][1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    images = [generate_pair() for _ in range(3)]

    for original, target in images:
        # Plotting the dataset
        fig = plt.figure()

        plt.plot(range(len(original[0])), original[0])
        plt.plot(range(len(target[0])), target[0])

        plt.show()
