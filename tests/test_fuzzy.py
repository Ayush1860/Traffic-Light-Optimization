import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models.fuzzy_logic import FuzzyController
    SKIP_TEST = False
except ImportError:
    print("Skipping Fuzzy Test: Dependencies not installed.")
    SKIP_TEST = True

class TestFuzzy(unittest.TestCase):
    def setUp(self):
        if not SKIP_TEST:
            self.fc = FuzzyController()

    def test_low_congestion(self):
        if SKIP_TEST: return
        score = self.fc.get_congestion_score(0.1, 5)
        print(f"Density: 0.1, Queue: 5 -> Score: {score}")
        self.assertLess(score, 0.5)

    def test_high_congestion(self):
        if SKIP_TEST: return
        score = self.fc.get_congestion_score(0.9, 45)
        print(f"Density: 0.9, Queue: 45 -> Score: {score}")
        self.assertGreater(score, 0.5)

if __name__ == "__main__":
    unittest.main()
