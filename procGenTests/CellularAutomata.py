import pygame
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class ErosionCA:
    def __init__(self, width, height, cell_size=5, initial_water=0.01, erosion_rate=0.001):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.erosion_rate = erosion_rate

        # Generate heightmap using fractal noise
        self.terrain = self.generate_fractal_noise()

        # Initialize water layer
        self.water = np.full((height, width), initial_water)

        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("Erosion Cellular Automaton")

    def generate_fractal_noise(self):
        def noise2d(shape, res):
            return np.random.normal(0, 1, (res + 1, res + 1))

        def fractal_noise(shape, res, octaves=6, persistence=0.5):
            noise = np.zeros(shape)
            frequency = 1
            amplitude = 1
            for _ in range(octaves):
                noise += amplitude * self.interpolate(noise2d(shape, res * frequency), shape)
                frequency *= 2
                amplitude *= persistence
            return noise

        noise = fractal_noise((self.height, self.width), 16)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise

    def interpolate(self, top_left, shape):
        x = np.linspace(0, top_left.shape[0] - 1, top_left.shape[0])
        y = np.linspace(0, top_left.shape[1] - 1, top_left.shape[1])
        interpolator = RegularGridInterpolator((x, y), top_left)

        x_new = np.linspace(0, top_left.shape[0] - 1, shape[0])
        y_new = np.linspace(0, top_left.shape[1] - 1, shape[1])
        X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
        points = np.array([X_new.ravel(), Y_new.ravel()]).T
        return interpolator(points).reshape(shape)

    def step(self):
        new_water = np.copy(self.water)
        new_terrain = np.copy(self.terrain)

        for i in range(self.height):
            for j in range(self.width):
                total_height = self.terrain[i, j] + self.water[i, j]

                # Check neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = (i + di) % self.height, (j + dj) % self.width
                    neighbor_height = self.terrain[ni, nj] + self.water[ni, nj]

                    if total_height > neighbor_height:
                        # Water flows downhill
                        flow = min(self.water[i, j], (total_height - neighbor_height) / 2)
                        new_water[i, j] -= flow
                        new_water[ni, nj] += flow

                        # Erosion
                        erosion = flow * self.erosion_rate
                        new_terrain[i, j] -= erosion
                        new_terrain[ni, nj] += erosion

        self.water = new_water
        self.terrain = np.clip(new_terrain, 0, 1)

    def draw(self):
        for i in range(self.height):
            for j in range(self.width):
                color = self.height_to_color(self.terrain[i, j], self.water[i, j])
                pygame.draw.rect(self.screen, color,
                                 (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

    def height_to_color(self, terrain_height, water_depth):
        if water_depth > 0.01:  # Water
            blue = int(max(0, min(255, 150 + water_depth * 350)))
            return (0, 0, blue)
        else:  # Land
            if terrain_height < 0.3:  # Beach
                return (238, 214, 175)
            elif terrain_height < 0.5:  # Lowlands
                green = int(max(0, min(255, terrain_height * 255)))
                return (34, green, 34)
            elif terrain_height < 0.7:  # Hills
                return (102, 85, 0)
            else:  # Mountains
                gray = int(max(0, min(255, 100 + terrain_height * 155)))
                return (gray, gray, gray)

    def run_simulation(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Add water at mouse click location
                    x, y = pygame.mouse.get_pos()
                    self.water[y // self.cell_size, x // self.cell_size] += 0.5

            self.step()
            self.draw()
            clock.tick(30)  # Limit to 30 frames per second

        pygame.quit()


# Run the simulation
ca = ErosionCA(2048, 2048, cell_size=1)
ca.run_simulation()
