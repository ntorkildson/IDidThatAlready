import random

class Civilization:
    def __init__(self, name, power):
        self.name = name
        self.power = power

class HistoricalFigure:
    def __init__(self, name, civilization):
        self.name = name
        self.civilization = civilization

class Event:
    def __init__(self, year, description):
        self.year = year
        self.description = description

class WorldHistory:
    def __init__(self):
        self.civilizations = []
        self.figures = []
        self.events = []
        self.current_year = 0

    def add_civilization(self, civ):
        self.civilizations.append(civ)

    def add_figure(self, figure):
        self.figures.append(figure)

    def generate_event(self):
        event_types = [
            "war", "discovery", "natural_disaster", "cultural_achievement"
        ]
        event_type = random.choice(event_types)

        if event_type == "war":
            civ1, civ2 = random.sample(self.civilizations, 2)
            winner = civ1 if civ1.power > civ2.power else civ2
            loser = civ2 if winner == civ1 else civ1
            description = f"{winner.name} defeated {loser.name} in a war"
        elif event_type == "discovery":
            civ = random.choice(self.civilizations)
            discoveries = ["new land", "technology", "resource"]
            discovery = random.choice(discoveries)
            description = f"{civ.name} discovered {discovery}"
        elif event_type == "natural_disaster":
            civ = random.choice(self.civilizations)
            disasters = ["earthquake", "flood", "volcanic eruption"]
            disaster = random.choice(disasters)
            description = f"{civ.name} suffered a {disaster}"
        else:  # cultural_achievement
            figure = random.choice(self.figures)
            achievements = ["great work of art", "philosophical treatise", "architectural wonder"]
            achievement = random.choice(achievements)
            description = f"{figure.name} of {figure.civilization.name} created a {achievement}"

        self.events.append(Event(self.current_year, description))

    def simulate_year(self):
        self.current_year += 1
        self.generate_event()

    def run_simulation(self, years):
        for _ in range(years):
            self.simulate_year()

    def print_history(self):
        for event in self.events:
            print(f"Year {event.year}: {event.description}")

# Usage
world = WorldHistory()
world.add_civilization(Civilization("Dwarves", 10))
world.add_civilization(Civilization("Elves", 8))
world.add_figure(HistoricalFigure("Urist McBuilder", world.civilizations[0]))
world.add_figure(HistoricalFigure("Leafy Greenwood", world.civilizations[1]))

world.run_simulation(100)
world.print_history()