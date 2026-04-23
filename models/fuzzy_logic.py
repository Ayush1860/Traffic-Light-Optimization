import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyController:
    def __init__(self):
        # Antecedents (Inputs)
        self.density = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'density')
        self.queue = ctrl.Antecedent(np.arange(0, 50, 1), 'queue')

        # Consequent (Output)
        self.congestion = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'congestion')

        # Membership Functions - Density
        self.density['low'] = fuzz.trimf(self.density.universe, [0, 0, 0.5])
        self.density['medium'] = fuzz.trimf(self.density.universe, [0, 0.5, 1])
        self.density['high'] = fuzz.trimf(self.density.universe, [0.5, 1, 1])

        # Membership Functions - Queue
        self.queue['short'] = fuzz.trimf(self.queue.universe, [0, 0, 15])
        self.queue['medium'] = fuzz.trimf(self.queue.universe, [0, 20, 40])
        self.queue['long'] = fuzz.trapmf(self.queue.universe, [20, 40, 50, 50])

        # Membership Functions - Congestion Score
        self.congestion['low'] = fuzz.trimf(self.congestion.universe, [0, 0, 0.5])
        self.congestion['medium'] = fuzz.trimf(self.congestion.universe, [0, 0.5, 1])
        self.congestion['high'] = fuzz.trimf(self.congestion.universe, [0.5, 1, 1])

        # Rules
        self.rule1 = ctrl.Rule(self.density['low'] & self.queue['short'], self.congestion['low'])
        self.rule2 = ctrl.Rule(self.density['low'] & self.queue['medium'], self.congestion['medium'])
        self.rule3 = ctrl.Rule(self.density['medium'] & self.queue['short'], self.congestion['medium'])
        self.rule4 = ctrl.Rule(self.density['medium'] & self.queue['medium'], self.congestion['medium'])
        self.rule5 = ctrl.Rule(self.density['high'] | self.queue['long'], self.congestion['high'])

        # Control System
        self.congestion_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4, self.rule5])
        self.simulation = ctrl.ControlSystemSimulation(self.congestion_ctrl)

    def get_congestion_score(self, density_val, queue_val):
        """
        Compute congestion score given density (0-1) and queue length (vehicles).
        """
        # Clip inputs to range
        density_val = np.clip(density_val, 0, 1)
        queue_val = np.clip(queue_val, 0, 49)

        self.simulation.input['density'] = density_val
        self.simulation.input['queue'] = queue_val
        
        try:
            self.simulation.compute()
            return self.simulation.output['congestion']
        except Exception as e:
            print(f"Fuzzy logic error: {e}")
            return 0.5 # Default fallback
