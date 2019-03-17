class AreaLight:
    def __init__(self, shape_id, intensity, two_sided = False, hide_shape = False):
        self.shape_id = shape_id
        self.intensity = intensity
        self.two_sided = two_sided
        self.hide_shape = hide_shape
