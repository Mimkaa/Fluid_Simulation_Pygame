import pygame as pg
import math
from settings import *
vec = pg.Vector2

class Agent:
    def __init__(self,fluid, pos, dir, movement_angle, color):
        self.pos = vec(pos) * fluid.cells_scale
        self.pos_fluid = vec(pos)
        self.dir = vec(dir)
        self.dir_original = self.dir.copy()
        self.fluid = fluid
        self.color = color

        self.angle = math.atan2(self.dir.y, self.dir.x)
        self.original_angle = math.atan2(self.dir_original.y, self.dir_original.x)
        self.movement_angle = movement_angle
        self.angle_dir = 1

    def update(self, speed, scale_vel, scale_dens, dt):
        self.angle += speed * dt * self.angle_dir
        self.dir = vec(math.cos(self.angle), math.sin(self.angle))


        for i in range(-1, 1):
            for j in range(-1, 1):
                self.fluid.add_velocity(self.pos_fluid.x + i, self.pos_fluid.y + j , self.dir * scale_vel)
                self.fluid.add_density(self.pos_fluid.x + i, self.pos_fluid.y + j , scale_dens, self.color)



        if abs(math.atan2(self.dir.y, self.dir.x) - self.original_angle) > self.movement_angle:
            self.angle_dir *= -1

    def draw(self, surf, scale):
        pg.draw.line(surf, RED, self.pos, self.pos + (self.dir * scale))





