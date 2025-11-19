from .base_agent import BaseAgent

class BusinessAgent(BaseAgent):
    """
    Optimized for professionals who need connectivity, transport,
    and low noise for remote work.
    """

    def __init__(self):
        super().__init__("business")

    def score_property(self, prop):
        score = 0.0

        # Internet speed (mbps)
        internet = prop.get("internet_speed", 100)
        score += (min(internet, 1000) / 1000) * 0.40

        # Noise level: quiet helps
        noise = prop.get("noise_level", 5)
        if noise <= 3:
            score += 0.25
        elif noise <= 5:
            score += 0.10

        # Proximity to center (km)
        d = prop.get("distance_to_center_km", 10)
        score += (max(0, 10 - d) / 10) * 0.25

        return score
