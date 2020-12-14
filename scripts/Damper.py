class Damper:
    def __init__(self):
        self.damper_force0 = 0

    def d_damper_force(self, force, action):
        damper_force = force * -action # AIが決定するパラメータで、与えられた力に対して、どんな割合で力を返すかを決める値
        d_damper_force = damper_force - self.damper_force0
        self.damper_force0 = damper_force
        return d_damper_force
