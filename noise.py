import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.ndimage import gaussian_filter
import multiprocessing as mp
from PIL import Image
import os
import OpenEXR
import Imath


def import_heightmap(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        with Image.open(file_path) as img:
            # Convert to grayscale if it's not already
            if img.mode != 'L':
                img = img.convert('L')

            # Convert to numpy array and normalize to 0-1 range
            heightmap = np.array(img).astype(float) / 255.0

            # Flip the image vertically to match the coordinate system usually used in heightmaps
            heightmap = np.flipud(heightmap)

            print(f"Heightmap imported successfully. Shape: {heightmap.shape}")

            return heightmap
    except Exception as e:
        print(f"Error importing heightmap: {str(e)}")
        return None


def export_to_hdr(data, file_path):
    # Assuming data is a numpy array normalized between 0 and 1
    data = data.astype(np.float32)
    header = OpenEXR.Header(data.shape[1], data.shape[0])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    
    header['channels'] = dict([(c, float_chan) for c in "RGB"])
    exr = OpenEXR.OutputFile(file_path, header)
    r = (data).tobytes()
    g = (data).tobytes()
    b = (data).tobytes()

    exr.writePixels({'R': r, 'G': g, 'B': b})
    exr.close()


def create_circular_gradient(size):
    y, x = np.ogrid[-size:size + 1, -size:size + 1]
    return size - np.sqrt(x * x + y * y)

def add_noise(array, scale=1.0):
    noise = np.random.normal(0, scale, array.shape)
    return array + noise

def normalize(array):
    return (array - array.min()) / (array.max() - array.min())

def erode_chunk(args):
    heightmap, start, end, inertia, erosion_rate, deposition_rate = args
    height, width = heightmap.shape
    for _ in range(start, end):
        # Random starting position
        x, y = np.random.randint(0, width), np.random.randint(0, height)

        # Initialize water and sediment
        water = 1.0
        sediment = 0.0

        # Initialize velocity
        dx = dy = 0.0

        for _ in range(30):  # Max 30 steps per droplet
            # Calculate gradient
            gx = heightmap[(y - 1) % height, x] - heightmap[(y + 1) % height, x]
            gy = heightmap[y, (x - 1) % width] - heightmap[y, (x + 1) % width]

            # Update velocity
            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)

            # Normalize velocity
            length = np.sqrt(dx * dx + dy * dy)
            if length != 0:
                dx /= length
                dy /= length

            # Move to new position
            new_x, new_y = x + dx, y + dy
            new_x = int(np.clip(new_x, 0, width - 1))
            new_y = int(np.clip(new_y, 0, height - 1))

            # Stop if we've gone off the map
            if new_x == x and new_y == y:
                break

            # Calculate height difference
            h_diff = heightmap[new_y, new_x] - heightmap[y, x]

            # Erosion and deposition
            if h_diff < 0:
                # Erode
                amount = min(-h_diff, water * erosion_rate)
                heightmap[y, x] -= amount
                sediment += amount
            else:
                # Deposit
                amount = min(h_diff, sediment * deposition_rate)
                heightmap[new_y, new_x] -= amount
                sediment -= amount

            # Update position
            x, y = new_x, new_y

            # Reduce water
            water *= 0.99

    return heightmap

    return heightmap

def parallel_erode(heightmap, num_iterations=50000, inertia=0.05, erosion_rate=0.3, deposition_rate=0.3):
    num_processes = mp.cpu_count()
    chunk_size = num_iterations // num_processes

    pool = mp.Pool(processes=num_processes)

    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else num_iterations
        chunks.append((heightmap.copy(), start, end, inertia, erosion_rate, deposition_rate))

    results = pool.map(erode_chunk, chunks)

    pool.close()
    pool.join()

    # Combine results
    final_heightmap = np.mean(results, axis=0)
    return final_heightmap

def erode(heightmap, num_iterations=50000, inertia=0.05, erosion_rate=0.3, deposition_rate=0.3):
    height, width = heightmap.shape
    for _ in range(num_iterations):
        # Random starting position
        x, y = np.random.randint(0, width), np.random.randint(0, height)

        # Initialize water and sediment
        water = 1.0
        sediment = 0.0

        # Initialize velocity
        dx = dy = 0.0

        for _ in range(30):  # Max 30 steps per droplet
            # Calculate gradient
            gx = heightmap[(y - 1) % height, x] - heightmap[(y + 1) % height, x]
            gy = heightmap[y, (x - 1) % width] - heightmap[y, (x + 1) % width]

            # Update velocity
            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)

            # Normalize velocity
            length = np.sqrt(dx * dx + dy * dy)
            if length != 0:
                dx /= length
                dy /= length

            # Move to new position
            new_x, new_y = x + dx, y + dy
            new_x = int(np.clip(new_x, 0, width - 1))
            new_y = int(np.clip(new_y, 0, height - 1))

            # Stop if we've gone off the map
            if new_x == x and new_y == y:
                break

            # Calculate height difference
            h_diff = heightmap[new_y, new_x] - heightmap[y, x]

            # Erosion and deposition
            if h_diff < 0:
                # Erode
                amount = min(-h_diff, water * erosion_rate)
                heightmap[y, x] -= amount
                sediment += amount
            else:
                # Deposit
                amount = min(h_diff, sediment * deposition_rate)
                heightmap[new_y, new_x] -= amount
                sediment -= amount

            # Update position
            x, y = new_x, new_y

            # Reduce water
            water *= 0.99

    return heightmap

def create_sunlight_map(heightmap, latitude=0, sun_angle=45):
    height, width = heightmap.shape
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)

    # Basic latitude effect
    base_sunlight = np.cos(np.arcsin(yy) - np.radians(latitude))

    # Calculate slope and aspect
    gy, gx = np.gradient(heightmap)
    slope = np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)

    # Adjust sunlight based on slope, aspect, and sun angle
    sun_direction = np.radians(sun_angle)
    adjusted_sunlight = base_sunlight * (np.cos(slope) * np.cos(sun_direction) +
                                         np.sin(slope) * np.sin(sun_direction) * np.cos(aspect - np.pi / 2))

    # Normalize and apply elevation bonus
    adjusted_sunlight = (adjusted_sunlight - adjusted_sunlight.min()) / (
                adjusted_sunlight.max() - adjusted_sunlight.min())
    elevation_bonus = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min()) * 0.2
    final_sunlight = np.clip(adjusted_sunlight + elevation_bonus, 0, 1)

    return final_sunlight

def create_wind_map(heightmap, latitude = 0, season =0):
    height, width = heightmap.shape
    y, x = np.ogrid[:height, :width]

    # Normalize coordinates
    y = 2 * (y - height / 2) / height
    x = 2 * (x - width / 2) / width

    # Basic planetary wind patterns (simplified model)
    wind_y = np.sin(3 * np.pi * y + season)  # Seasonal variation
    wind_x = np.cos(latitude) * np.ones_like(x)  # Prevailing winds

    # Add Coriolis effect
    coriolis = 0.1 * np.sin(latitude) * wind_x
    wind_y += coriolis

    # Terrain influence
    grad_y, grad_x = np.gradient(heightmap)
    terrain_influence = 1 / (1 + np.exp(5 * heightmap))  # Sigmoid function to reduce wind at high elevations
    wind_x -= grad_x * terrain_influence
    wind_y -= grad_y * terrain_influence

    # Normalize wind vectors
    magnitude = np.sqrt(wind_x ** 2 + wind_y ** 2)
    wind_x /= magnitude
    wind_y /= magnitude

    # Add some turbulence
    turbulence = ndimage.gaussian_filter(np.random.randn(height, width), sigma=5)
    wind_x += 0.1 * turbulence
    wind_y += 0.1 * turbulence

    return wind_x, wind_y

def create_turbulent_wind_map(heightmap, prevailing_direction=(1, 0), turbulence_strength=0.5, vorticity_strength=0.3):
    height, width = heightmap.shape
    grad_y, grad_x = np.gradient(heightmap)

    # Base wind from prevailing direction
    wind_x = np.full_like(heightmap, prevailing_direction[0])
    wind_y = np.full_like(heightmap, prevailing_direction[1])

    # Terrain influence
    terrain_influence = 0.7
    wind_x -= grad_x * terrain_influence
    wind_y -= grad_y * terrain_influence

    # Add turbulence
    turbulence_x = np.random.randn(height, width)
    turbulence_y = np.random.randn(height, width)

    # Smooth turbulence
    turbulence_x = gaussian_filter(turbulence_x, sigma=2)
    turbulence_y = gaussian_filter(turbulence_y, sigma=2)

    wind_x += turbulence_x * turbulence_strength
    wind_y += turbulence_y * turbulence_strength

    # Add vorticity
    vorticity = np.random.randn(height, width)
    vorticity = gaussian_filter(vorticity, sigma=5)
    wind_x += np.gradient(vorticity)[1] * vorticity_strength
    wind_y -= np.gradient(vorticity)[0] * vorticity_strength

    # Normalize wind vectors
    magnitude = np.sqrt(wind_x ** 2 + wind_y ** 2)
    wind_x /= magnitude
    wind_y /= magnitude

    # Reduce wind speed at higher elevations
    elevation_factor = 1 - heightmap
    wind_x *= elevation_factor
    wind_y *= elevation_factor

    return wind_x, wind_y


def create_realistic_wind_patterns(heightmap, latitude=0, season=0):
    height, width = heightmap.shape
    y, x = np.mgrid[:height, :width]

    # Normalize coordinates
    y_norm = 2 * (y - height / 2) / height
    x_norm = 2 * (x - width / 2) / width

    # Basic planetary wind patterns (more complex model)
    wind_y = np.sin(2 * np.pi * y_norm + season) * (1 - 0.5 * np.abs(y_norm))
    wind_x = np.cos(latitude) * np.ones_like(heightmap) + 0.3 * np.sin(3 * np.pi * x_norm)

    # Terrain influence
    grad_y, grad_x = np.gradient(heightmap)
    terrain_influence = 1 / (1 + np.exp(10 * (heightmap - np.mean(heightmap))))
    wind_x -= grad_x * terrain_influence * 2
    wind_y -= grad_y * terrain_influence * 2

    # Add vorticity
    vorticity = ndimage.gaussian_filter(np.random.randn(*heightmap.shape), sigma=10)
    wind_x += ndimage.sobel(vorticity, axis=0) * 0.2
    wind_y -= ndimage.sobel(vorticity, axis=1) * 0.2

    # Normalize wind vectors
    magnitude = np.sqrt(wind_x ** 2 + wind_y ** 2)
    wind_x /= np.maximum(magnitude, 1e-10)
    wind_y /= np.maximum(magnitude, 1e-10)

    # Scale wind strength based on elevation
    wind_strength = 1 - 0.5 * (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))
    wind_x *= wind_strength
    wind_y *= wind_strength

    return wind_x, wind_y

def create_ocean_currents(heightmap, wind_x, wind_y):
    sea_level = 0.3  # Adjust as needed
    ocean_mask = heightmap < sea_level

    # Create coastal mask
    coastal_mask = np.zeros_like(heightmap)
    coastal_mask[1:-1, 1:-1] = ((heightmap[1:-1, 1:-1] < sea_level) &
                                ((heightmap[:-2, 1:-1] >= sea_level) |
                                 (heightmap[2:, 1:-1] >= sea_level) |
                                 (heightmap[1:-1, :-2] >= sea_level) |
                                 (heightmap[1:-1, 2:] >= sea_level)))

    # Initialize currents based on wind
    currents_x = wind_x * ocean_mask
    currents_y = wind_y * ocean_mask

    # Adjust currents near coasts
    coast_influence = 5
    currents_x -= np.gradient(coastal_mask)[1] * coast_influence
    currents_y -= np.gradient(coastal_mask)[0] * coast_influence

    # Normalize and apply ocean mask
    magnitude = np.sqrt(currents_x ** 2 + currents_y ** 2)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    currents_x = (currents_x / magnitude) * ocean_mask
    currents_y = (currents_y / magnitude) * ocean_mask

    return currents_x, currents_y

def create_temperature_map(heightmap, sunlight_map, ocean_currents_x, ocean_currents_y, sea_level=0.3):
    base_temp = 20 + 30 * sunlight_map
    altitude_effect = (heightmap - sea_level).clip(0) * -20

    ocean_effect = (np.abs(ocean_currents_x) + np.abs(ocean_currents_y)) * 10
    ocean_effect *= (heightmap < sea_level)

    return base_temp + altitude_effect + ocean_effect

def create_precipitation_map(heightmap, wind_x, wind_y, temperature_map, sea_level=0.3):
    height, width = heightmap.shape
    precipitation = np.zeros_like(heightmap)

    # Initialize moisture over the ocean
    moisture = np.zeros_like(heightmap)
    moisture[heightmap < sea_level] = 1.0

    # Wind speed affects evaporation and moisture carrying capacity
    wind_speed = np.sqrt(wind_x ** 2 + wind_y ** 2)

    # Simulate moisture movement and precipitation
    for _ in range(50):  # Increase iterations for more diffusion
        # Advection of moisture
        moisture_x = ndimage.sobel(moisture, axis=1)
        moisture_y = ndimage.sobel(moisture, axis=0)
        moisture -= 0.1 * (wind_x * moisture_x + wind_y * moisture_y)

        # Evaporation from ocean
        evaporation = 0.05 * wind_speed * (heightmap < sea_level)
        moisture += evaporation

        # Orographic effect (precipitation due to elevation change)
        orographic = 0.05 * (wind_x * ndimage.sobel(heightmap, axis=1) +
                             wind_y * ndimage.sobel(heightmap, axis=0))
        orographic = np.maximum(orographic, 0)  # Only positive changes cause precipitation

        # Precipitation
        precip = moisture * orographic
        precipitation += precip
        moisture -= precip

        # Limit moisture
        moisture = np.clip(moisture, 0, 1)

    # Normalize precipitation
    precipitation = (precipitation - precipitation.min()) / (precipitation.max() - precipitation.min())

    # Apply temperature effect (more precipitation in warmer areas, generally)
    precipitation *= 0.5 + 0.5 * temperature_map

    return precipitation

def classify_biomes(temp, precip, heightmap, sea_level=0.3):
    biomes = np.zeros_like(temp, dtype=int)

    # Ocean
    biomes[heightmap < sea_level] = 0

    # Land biomes
    land = heightmap >= sea_level

    biomes[land & (temp < 0.2)] = 1  # Tundra
    biomes[land & (temp < 0.4) & (precip < 0.3)] = 2  # Cold Desert
    biomes[land & (temp < 0.4) & (precip >= 0.3)] = 3  # Taiga
    biomes[land & (temp >= 0.4) & (temp < 0.7) & (precip < 0.4)] = 4  # Temperate Grassland
    biomes[land & (temp >= 0.4) & (temp < 0.7) & (precip >= 0.4)] = 5  # Temperate Forest
    biomes[land & (temp >= 0.7) & (precip < 0.2)] = 6  # Hot Desert
    biomes[land & (temp >= 0.7) & (precip >= 0.2) & (precip < 0.6)] = 7  # Savanna
    biomes[land & (temp >= 0.7) & (precip >= 0.6)] = 8  # Tropical Rainforest

    return biomes

def create_seasonal_maps(heightmap, base_temp, base_precip, num_seasons=4):
    seasons = []
    for i in range(num_seasons):
        # Adjust temperature
        season_temp = base_temp + np.sin(2 * np.pi * i / num_seasons) * 0.2

        # Adjust precipitation
        season_precip = base_precip * (1 + 0.3 * np.sin(2 * np.pi * i / num_seasons + np.pi / 2))

        # Recalculate biomes for this season
        season_biomes = classify_biomes(season_temp, season_precip, heightmap)

        seasons.append((season_temp, season_precip, season_biomes))

    return seasons

def create_river_system(heightmap, precipitation_map, sea_level=0.3, river_threshold=0.02):
    # Calculate flow direction
    grad_y, grad_x = np.gradient(heightmap)
    flow_direction = np.arctan2(-grad_y, -grad_x)

    # Initialize flow accumulation
    flow_acc = precipitation_map.copy()

    # Iterate to accumulate flow
    for _ in range(10):  # Adjust number of iterations as needed
        flow_x = flow_acc * np.cos(flow_direction)
        flow_y = flow_acc * np.sin(flow_direction)
        flow_acc += ndimage.shift(flow_x, [0, 1], cval=0) - ndimage.shift(flow_x, [0, -1], cval=0) \
                    + ndimage.shift(flow_y, [1, 0], cval=0) - ndimage.shift(flow_y, [-1, 0], cval=0)

    # Create river mask
    rivers = (flow_acc > np.max(flow_acc) * river_threshold) & (heightmap >= sea_level)

    return rivers
def create_coastal_effect(heightmap, sea_level=0.3, coastal_range=0.1):
    coastal_mask = np.zeros_like(heightmap)

    land = heightmap >= sea_level
    sea = heightmap < sea_level

    for _ in range(int(coastal_range * heightmap.shape[0])):
        coastal_mask += ndimage.binary_dilation(land) & sea

    coastal_effect = np.exp(-coastal_mask / (coastal_range * heightmap.shape[0]))

    return coastal_effect

def simulate_erosion(heightmap, river_map, erosion_rate=0.01, deposition_rate=0.005, iterations=10):
    for _ in range(iterations):
        # Calculate flow
        flow = ndimage.gaussian_filter(river_map.astype(float), sigma=1)

        # Erosion
        erosion = erosion_rate * flow
        heightmap -= erosion

        # Transportation and deposition
        transported_sediment = ndimage.gaussian_filter(erosion, sigma=2)
        heightmap += deposition_rate * transported_sediment

        # Update river map based on new heightmap
        river_map = create_river_system(heightmap, precipitation_map)

    return heightmap, river_map


def calculate_vegetation_density(temperature, precipitation, elevation, land_mask):
    # Base vegetation potential from temperature and precipitation
    veg_potential = (1 - abs(temperature - 0.5)) * precipitation

    # Reduce vegetation at very high elevations
    elevation_factor = 1 - np.clip(elevation - 0.7, 0, 0.3) / 0.3

    vegetation_density = veg_potential * elevation_factor

    # Apply land mask
    vegetation_density = np.where(land_mask, vegetation_density, 0)

    return np.clip(vegetation_density, 0, 1)

def generate_weather_patterns(temperature, precipitation, wind_x, wind_y, num_days=30):
    weather_patterns = []
    for _ in range(num_days):
        # Generate random weather events
        cloud_cover = np.random.rand(*temperature.shape) * precipitation
        rain_intensity = np.where(np.random.rand(*temperature.shape) < precipitation, precipitation, 0)
        wind_speed = np.sqrt(wind_x ** 2 + wind_y ** 2)

        # Create storms where conditions are right
        storm_probability = precipitation * wind_speed * (1 - temperature)
        storms = np.random.rand(*temperature.shape) < storm_probability

        weather_patterns.append({
            'cloud_cover': cloud_cover,
            'rain_intensity': rain_intensity,
            'wind_speed': wind_speed,
            'storms': storms
        })
    return weather_patterns


def calculate_animal_population(vegetation_density, temperature, elevation, water_proximity, land_mask):
    # Base population from vegetation density
    population = vegetation_density.copy()

    # Reduce population in extreme temperatures
    temp_factor = 1 - 2 * abs(temperature - 0.5)
    population *= temp_factor

    # Reduce population at very high elevations
    elevation_factor = 1 - np.clip(elevation - 0.8, 0, 0.2) / 0.2
    population *= elevation_factor

    # Increase population near water sources
    population *= (1 + water_proximity)

    # Apply land mask
    population = np.where(land_mask, population, 0)

    return np.clip(population, 0, 1)

def create_land_mask(heightmap, sea_level=0.3):
    return heightmap > sea_level


def create_island_heightmap(size=256, noise_scale=0.5):
    # Create circular gradient
    heightmap = create_circular_gradient(size // 2)

    # Clip negative values to create island shape
    heightmap = np.clip(heightmap, 0, None)

    # Add noise
    heightmap = add_noise(heightmap, noise_scale)

    # Apply Gaussian filter to smooth
    heightmap = gaussian_filter(heightmap, sigma=3)

    # Normalize values between 0 and 1
    heightmap = normalize(heightmap)

    heightmap = erode(heightmap)
    #heightmap = parallel_erode(heightmap)

    return heightmap

def simulate_weather(current_conditions, time_step):
    # Implement basic weather simulation logic
    # Return updated weather conditions
    pass

def classify_sub_biomes(biome_map, temperature, precipitation, elevation):
    # Implement sub-biome classification
    pass

def model_water_table(heightmap, precipitation, soil_type):
    # Implement water table and aquifer modeling
    pass

def generate_soil_map(heightmap, climate_data, vegetation_data):
    # Generate soil type and quality maps
    pass

def calculate_solar_angle(latitude, day_of_year, time_of_day):
    # Calculate solar angle
    pass

# Create initial heightmap
size = 256
scale = size / 2.5
latitude = 30
season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
season = 1  # spring
#baseImage = './Images/Terrain/iceland_lowres.jfif'
baseImage = './Images/Terrain/iceland_heightmap.png'
export_path = './Images/Terrain/export.hdr'

imported_map = import_heightmap(baseImage)
if imported_map is not None:
    eroded_heightmap = imported_map.copy()
else:
    eroded_heightmap = create_island_heightmap(size, scale)

# 3. Create land mask
land_mask = create_land_mask(eroded_heightmap)

# 4. Generate soil map (depends on initial terrain)
#soil_map = generate_soil_map(eroded_heightmap,eroded_heightmap, land_mask)
# 5. Create sunlight map (depends on terrain)
sunlight_map = create_sunlight_map(eroded_heightmap)

# 6. Create wind patterns
wind_x, wind_y = create_realistic_wind_patterns(eroded_heightmap, latitude, season)

# 7. Create ocean currents (depends on wind and terrain)
ocean_currents_x, ocean_currents_y = create_ocean_currents(eroded_heightmap, wind_x, wind_y)

# 8. Create initial temperature map
temperature_map = create_temperature_map(eroded_heightmap, sunlight_map, ocean_currents_x, ocean_currents_y)
normalized_temp = normalize(temperature_map)

# 9. Create initial precipitation map
precipitation_map = create_precipitation_map(eroded_heightmap, wind_x, wind_y, temperature_map)
normalized_precip = normalize(precipitation_map)

# 10. Create coastal effect
coastal_effect = create_coastal_effect(eroded_heightmap)

# 11. Adjust temperature and precipitation for coastal effect
coastal_temp = normalize(temperature_map) * (1 - 0.2 * coastal_effect)
coastal_precip = normalize(precipitation_map) * (1 + 0.5 * coastal_effect)

# 12. Model water table and aquifers
#water_table, aquifers = model_water_table(eroded_heightmap, precipitation_map, soil_map)

# 13. Create river system (depends on water table and precipitation)
river_map = create_river_system(eroded_heightmap, precipitation_map)

# 14. Final erosion simulation
eroded_heightmap2, updated_river_map = simulate_erosion(eroded_heightmap, river_map)

# 15. Update land mask with eroded terrain
land_mask = create_land_mask(eroded_heightmap2)

# 16. Classify main biomes
biome_map = classify_biomes(coastal_temp, coastal_precip, eroded_heightmap2)

# 17. Classify sub-biomes
sub_biome_map = classify_sub_biomes(biome_map, temperature_map, precipitation_map, eroded_heightmap2)

# 18. Calculate vegetation density
vegetation_density = calculate_vegetation_density(coastal_temp, coastal_precip, eroded_heightmap2, land_mask)

# 19. Calculate water proximity for animal distribution
water_proximity = 1 - eroded_heightmap2 + coastal_effect

# 20. Calculate animal population
animal_population = calculate_animal_population(vegetation_density, coastal_temp, eroded_heightmap2, water_proximity, land_mask)

'''
# 21. Initialize ecosystem state
ecosystem_state = initialize_ecosystem(
    eroded_heightmap2, temperature_map, precipitation_map, wind_x, wind_y,
    ocean_currents_x, ocean_currents_y, biome_map, sub_biome_map,
    vegetation_density, animal_population, water_table, aquifers,
    soil_map, river_map, coastal_effect
)

# 22. Main simulation loop
for time_step in range(simulation_duration):
    current_solar_angle = calculate_solar_angle(latitude, time_step)
    ecosystem_state = update_ecosystem(ecosystem_state, time_step, current_solar_angle)

    if time_step % export_interval == 0:
        export_state_to_hdr(ecosystem_state, time_step)
'''
# 23. Create seasonal maps (could also be done within the simulation loop)
seasonal_maps = create_seasonal_maps(eroded_heightmap2, coastal_temp, coastal_precip)



# Create a color map for biomes
biome_colors = ['#000080', '#FFFFFF', '#FFFF00', '#008000', '#90EE90', '#228B22', '#A52A2A', '#F4A460', '#006400']
biome_cmap = colors.ListedColormap(biome_colors)

# Define plot configurations
plot_configs = [
    ('Heightmap', eroded_heightmap2, 'gray', 'imshow'),
    ('Sunlight Map', sunlight_map, 'YlOrRd', 'imshow'),
    ('Turbulent Wind Map', (wind_x, wind_y), None, 'quiver'),
    ('Ocean Currents', (ocean_currents_x, ocean_currents_y), None, 'quiver'),
    ('Temperature Map', normalized_temp, 'coolwarm', 'imshow'),
    ('Precipitation Map', normalized_precip, 'Blues', 'imshow'),
    ('Biome Map', biome_map, biome_cmap, 'imshow'),
    ('River System', river_map,'Blues', 'imshow'),
    ('Coastal Effect', coastal_effect, 'YlOrRd', 'imshow'),
    ('Vegetation Density', (vegetation_density, land_mask), ('Greens', 'Blues'), 'masked_imshow'),
    ('Animal Population', (animal_population, land_mask), ('YlOrBr', 'Blues'), 'masked_imshow'),
]

# Calculate the number of rows needed
num_plots = len(plot_configs)
num_seasons = len(season_names)
total_plots = num_plots + num_seasons
num_rows = (total_plots + 3) // 4  # +3 to round up

# Create subplots
fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5*num_rows))

# Flatten the axs array for easier indexing
axs = axs.flatten()

# Plot maps
for i, (title, data, cmap, plot_type) in enumerate(plot_configs):
    if i < len(axs):
        ax = axs[i]
        if plot_type == 'imshow':
            ax.imshow(data, cmap=cmap)
        elif plot_type == 'quiver':
            y, x = np.mgrid[:data[0].shape[0], :data[0].shape[1]]
            ax.quiver(x[::10, ::10], y[::10, ::10], data[0][::10, ::10], data[1][::10, ::10],
                      scale=50, color='white', alpha=0.8)
            ax.imshow(eroded_heightmap2, cmap='terrain', alpha=0.7)  # Add terrain background
        elif plot_type == 'double_imshow':
            ax.imshow(data[0], cmap=cmap[0])
            ax.imshow(data[1], cmap=cmap[1], alpha=0.5)
        elif plot_type == 'masked_imshow':
            land_data = np.ma.masked_where(~data[1], data[0])
            water_data = np.ma.masked_where(data[1], np.ones_like(data[0]))
            ax.imshow(water_data, cmap=cmap[1])  # Water in blue
            ax.imshow(land_data, cmap=cmap[0])  # Land data with specified colormap
        ax.set_title(title)
        ax.axis('off')



# Plot seasonal maps
for i, (temp, precip, biomes) in enumerate(seasonal_maps):
    if num_plots + i < len(axs):
        ax = axs[num_plots + i]
        ax.imshow(biomes, cmap=biome_cmap)
        ax.set_title(f'{season_names[i]} Biomes')
        ax.axis('off')

# Remove extra subplots
for i in range(total_plots, len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()

print(biome_map.shape)
#biome_map_normalized = ((biome_map - biome_map.min()) / (biome_map.max() - biome_map.min()) * 255).astype('uint8')
colormap = plt.colormaps.get_cmap('viridis')
colored_biome_map = colormap(biome_map / 255.0)
colored_biome_map = (colored_biome_map[:, :, :3] * 255).astype('uint8')


newdata = Image.fromarray(colored_biome_map)
newdata.save('biome_map.png')


#export_to_hdr(newdata,export_path)