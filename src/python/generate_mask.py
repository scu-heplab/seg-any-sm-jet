import math
import numpy as np
from matplotlib import pyplot as plt


def calc_mass(particle):
    px, py, pz, e = np.split(particle[..., :4], 4, -1)

    m2 = e * e - px * px - py * py - pz * pz
    return np.sqrt(np.where(m2 >= 0, m2, -m2))


def calc_pt_rapidity_phi(particle):
    px, py, pz, e = np.split(particle[..., :4], 4, -1)

    pt = np.sqrt(px * px + py * py)
    rapidity = np.log(np.abs(e + pz) / (np.sqrt(np.square(calc_mass(particle)) + pt * pt) + 1e-3) + 1e-3)

    phi = np.arctan(py / np.where(px != 0, px, 1e-3))
    phi = np.where(np.logical_and(px < 0, py > 0), phi + np.pi, phi)
    phi = np.where(np.logical_and(px < 0, py < 0), phi + np.pi, phi)
    phi = np.where(np.logical_and(px > 0, py < 0), phi + 2 * np.pi, phi)
    return np.concatenate([pt, rapidity, phi], -1)


def calc_center(particle):
    pt_rapidity_phi = calc_pt_rapidity_phi(particle)
    maximum = pt_rapidity_phi[np.argsort(pt_rapidity_phi[:, 0])[-1]]

    delta_phi = pt_rapidity_phi[:, 2] - maximum[None, 2]
    pt_rapidity_phi[:, 2] += np.where(np.abs(delta_phi) > np.pi, np.where(delta_phi > 0, -2 * np.pi, 2 * np.pi), 0)

    weights = (pt_rapidity_phi[:, 0] + 1e-3) / (np.sum(pt_rapidity_phi[:, 0] + 1e-3))
    weight_center = np.sum(weights[:, None] * pt_rapidity_phi[:, 1:], 0)
    return weight_center


def calc_distance(ref_particle, cand_particle):
    center = calc_center(ref_particle)
    pt_rapidity_phi = calc_pt_rapidity_phi(cand_particle)

    delta_phi = pt_rapidity_phi[:, 2] - center[None, 1]
    pt_rapidity_phi[:, 2] += np.where(np.abs(delta_phi) > np.pi, np.where(delta_phi > 0, -2 * np.pi, 2 * np.pi), 0)

    # weights = np.sum(pt_rapidity_phi[:, 0]) / pt_rapidity_phi[:, 0]
    distance = np.sqrt(np.sum(np.square(pt_rapidity_phi[:, 1:] - center[None]), -1)) / np.log(np.e + pt_rapidity_phi[:, 0])
    return distance


def particle_cuts(particle):
    pt_rapidity_phi = calc_pt_rapidity_phi(particle)
    cuts = np.logical_and(pt_rapidity_phi[:, 0] > 0.1, np.abs(pt_rapidity_phi[:, 1]) < np.pi)
    return particle[cuts]


def particle_to_image(particle, resolution=(0.02, 0.02), detector_range=((-np.pi, np.pi), (0, 2 * np.pi))):
    particle = particle_cuts(particle)
    pt_rapidity_phi = calc_pt_rapidity_phi(particle)

    image_width = int(math.ceil((detector_range[0][1] - detector_range[0][0]) / resolution[0]))
    image_height = int(math.ceil((detector_range[1][1] - detector_range[1][0]) / resolution[1]))
    point = np.array((pt_rapidity_phi[:, 1:] - (detector_range[0][0], detector_range[1][0])) / resolution, np.int32)

    add_image = np.zeros((image_height, image_width, 5))
    maximum_image = np.zeros((image_height, image_width, 1))

    np.add.at(add_image, (point[:, 1], point[:, 0]), particle[..., :5])
    np.maximum.at(maximum_image, (point[:, 1], point[:, 0]), particle[..., 5:])

    return np.concatenate([add_image, maximum_image], -1)


def image_to_mask(image, expand=9, radius=20, n_neighbor=3, sigma=1.0):
    mask = np.stack(np.where(np.any(image != 0, -1)), -1)

    distance = np.sqrt(np.square(mask[:, np.newaxis, 0] - mask[np.newaxis, :, 0]) + np.square(mask[:, np.newaxis, 1] - mask[np.newaxis, :, 1]))
    mask = mask[np.logical_or(np.sum(np.where(distance < radius, 1, 0), -1) > n_neighbor, calc_pt_rapidity_phi(image)[mask[:, 0], mask[:, 1], 0] > 5)]

    if len(mask) == 0:
        return np.zeros(image.shape[:2], dtype=np.float32)

    weights = np.log(np.sqrt(np.sum(np.square(image[..., :2]), -1))[mask[:, 0], mask[:, 1]] + np.e)

    kernel_radius = (expand - 1) // 2
    gaussian_mask = np.zeros(image.shape[:2])

    y, x = np.ogrid[-kernel_radius:kernel_radius + 1, -kernel_radius:kernel_radius + 1]
    kernel_weights = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2) / (expand ** 2)) * (x ** 2 + y ** 2 <= kernel_radius ** 2)

    for (y, x), weight in zip(mask, weights):
        y_min = max(y - kernel_radius, 0)
        y_max = min(y + kernel_radius + 1, image.shape[0])
        x_min = max(x - kernel_radius, 0)
        x_max = min(x + kernel_radius + 1, image.shape[1])

        k_y_min = kernel_radius - (y - y_min)
        k_y_max = kernel_radius + (y_max - y - 1)
        k_x_min = kernel_radius - (x - x_min)
        k_x_max = kernel_radius + (x_max - x - 1)

        gaussian_mask[y_min:y_max, x_min:x_max] += weight * kernel_weights[k_y_min:k_y_max + 1, k_x_min:k_x_max + 1]
        gaussian_mask = np.where(gaussian_mask > 0, gaussian_mask, 0)

    if np.max(gaussian_mask) > 0:
        gaussian_mask /= np.max(gaussian_mask)

    return gaussian_mask


def main():
    data = np.loadtxt("test/ww.dat")

    mass = []
    for i in range(1000):
        index = data[np.logical_and(data[:, 0] == i, data[:, 2] == 24)][0, 1]
        wp_data = data[np.logical_and(data[:, 0] == i, data[:, 1] == index)][:, 3:]

        wp_image = particle_to_image(wp_data)

        wp_mask = image_to_mask(wp_image, 15)

        print("\rimages: %d" % (i + 1), end="")

        plt.imshow(wp_mask)
        plt.show()

    mass = np.array(mass)

    plt.hist(mass, 20, (50, 150))
    plt.xlim(50, 150)
    plt.show()


if __name__ == '__main__':
    main()
