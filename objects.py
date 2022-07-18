import math
import random

import pygame as pg
import numpy as np
from gaus_seidel import Gaus_Seidel
from settings import *
vec = pg.Vector2

def translate(val, start_first_range, end_first_range, start_second_range, end_second_range):
    # find length of each range
    first_range_length = end_first_range - start_first_range
    second_range_length = end_second_range - start_second_range

    # how far val is in the first range and how much it is in the second range
    in_second_range = ((val - start_first_range) / first_range_length) * second_range_length

    return start_second_range + in_second_range

# class Mouse_Tracker:
#     def __init__(self):
#         self.prev_coords = vec(0, 0)
#         self.curr_coords = vec(0, 0)
#         self.dir_vec = vec(0, 0)



class Cell:
    def __init__(self, pos):
        self.density = 0
        self.density_prev = 0
        self.vel = vec(0, 0)
        self.vel_prev = vec(0, 0)
        self.pos = vec(pos)

class Fluid:
    def __init__(self, size, viscosity):
        self.size = size
        self.cells = [Cell(self.index_2d(i)) for i in range(size * size)]
        self.viscosity = viscosity
        self.cells_scale = 16
        self.cells_images = [pg.Surface((self.cells_scale, self.cells_scale)) for i in range(size * size)]

    def add_density(self,x,y,amount):
        self.cells[self.index(x, y)].density_prev += amount

    def add_velocity(self,x,y,vel):
        self.cells[self.index(x, y)].vel_prev += vel

    def update(self, dt):

        # diffuse vel
        vels_prev = [c.vel_prev.copy() for c in self.cells]
        vels = [c.vel.copy() for c in self.cells]
        prev_x_sep = [v.x for v in vels_prev]
        prev_y_sep = [v.y for v in vels_prev]
        x_sep = [v.x for v in vels]
        y_sep = [v.y for v in vels]

        self.diffuse(prev_x_sep, x_sep, dt + self.viscosity, 4)
        self.diffuse(prev_y_sep, y_sep, dt + self.viscosity, 4)
        for i in range(len(vels)):
            self.cells[i].vel = vec(x_sep[i], y_sep[i])
            self.cells[i].vel_prev = self.cells[i].vel.copy()

        # self.clear_divergence()

        # diffuse density
        dens_prev = [c.density_prev for c in self.cells]
        dens = [c.density for c in self.cells]
        self.diffuse(dens_prev, dens, dt + self.viscosity, 4)

        for i in range(len(dens)):
            self.cells[i].density = dens[i]
            self.cells[i].density_prev = dens[i]

        # advect density, velocity
        dens_prev = [c.density_prev for c in self.cells]
        new_densities = self.advection(dens_prev, dt)

        vels_prev = [c.vel_prev.copy() for c in self.cells]
        prev_x_sep = [v.x for v in vels_prev]
        prev_y_sep = [v.y for v in vels_prev]

        new_vel_x = self.advection(prev_x_sep, dt)
        new_vel_y = self.advection(prev_y_sep, dt)

        for i in range(len(self.cells)):
            self.cells[i].density_prev = new_densities[i]
            self.cells[i].vel_prev = vec(new_vel_x[i], new_vel_y[i])


        # clearing the divergence
        self.clear_divergence()


        for i in range(len(self.cells_images)):
            color = translate(self.cells[i].density_prev, 0, 1, 0, 255)
            if color > 255:
                color = 255
            if color < 0 :
                color = 0
            self.cells_images[i].fill((color, color, color))

        # fade
        for cell in self.cells:
            cell.density_prev *= 0.9



    def index(self, x ,y):
        y = int((y + self.size) % self.size)
        x = int((x + self.size) % self.size)
        return x + (y * self.size)

    def index_2d(self, i):
        return (i % self.size, i // self.size)

    def diffuse(self, values_prev, values, k, iterations):

        for itr in range(iterations):
            for x in range(self.size):
                for y in range(self.size):
                    values[self.index(x, y)] = (values_prev[self.index(x, y)] + k *((values[self.index(x + 1, y)]+
                                                                                   values[self.index(x - 1, y)]+
                                                                                   values[self.index(x, y + 1)]+
                                                                                   values[self.index(x, y - 1)]) / 4)) / (1 + k)


    def lerp(self, a, b, k):
        return a + k * (b - a)


    def advection(self, params,  dt):
        new_params = []
        for cell in self.cells:
            come_from = cell.pos - (cell.vel_prev * dt)
            come_from_int = vec(int(come_from.x + self.size) % self.size , int(come_from.y + self.size) % self.size )
            come_from_fract = vec(come_from.x - int(come_from.x), come_from.y - int(come_from.y))
            lerp_dens_up = self.lerp(params[self.index(come_from_int.x, come_from_int.y)],
                                     params[self.index(come_from_int.x + 1 , come_from_int.y)], come_from_fract.x)

            lerp_dens_down = self.lerp(params[self.index(come_from_int.x, come_from_int.y + 1)],
                                     params[self.index(come_from_int.x + 1 , come_from_int.y + 1)], come_from_fract.x)

            new = self.lerp(lerp_dens_up, lerp_dens_down, come_from_fract.y)

            new_params.append(new)

        return new_params

    def gaus_seidel_divergence(self, prev_vals, answ, iterations):
        for itr in range(iterations):
            for x in range(self.size):
                for y in range(self.size):
                    summ = (answ[self.index(x + 1, y)] + answ[self.index(x - 1, y)] + answ[self.index(x, y + 1)] + answ[self.index(x, y - 1)])
                    answ[self.index(x, y)] = (summ - prev_vals[self.index(x, y)]) / 4


    def clear_divergence(self):
        divergence = [0] * len(self.cells)
        for x in range(self.size):
            for y in range(self.size):
                # get divergence in each cell
                diff_x = (self.cells[self.index(x + 1, y)].vel_prev.x - self.cells[self.index(x - 1, y)].vel_prev.x)
                diff_y = (self.cells[self.index(x, y + 1)].vel_prev.y - self.cells[self.index(x, y - 1)].vel_prev.y)
                divergence[self.index(x, y)] = (diff_x + diff_y) * 0.5

        # gradients of vec field with no curl
        scalars = [0] * len(self.cells)
        self.gaus_seidel_divergence(divergence, scalars, 4)
        for x in range(self.size):
            for y in range(self.size):
                x_val = ((scalars[self.index(x + 1, y)] - scalars[self.index(x - 1, y)])) * 0.5
                y_val = ((scalars[self.index(x, y + 1)] - scalars[self.index(x, y - 1)])) * 0.5

                # clearing the divergence
                self.cells[self.index(x, y)].vel_prev -= vec(x_val, y_val)


    def draw(self, surf):
        for x in range(self.size):
            for y in range(self.size):
                surf.blit(self.cells_images[self.index(x, y)], (x * self.cells_scale, y * self.cells_scale))
        # for c in self.cells:
        #     pg.draw.line(surf, RED, c.pos * self.cells_scale, c.pos * self.cells_scale + c.vel_prev)



