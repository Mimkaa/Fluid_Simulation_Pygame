import math
import random

import pygame as pg
import numpy as np
from numba import njit
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


@njit(fastmath=True)
def index(size, x, y):
    y = int((y + size) % size)
    x = int((x + size) % size)
    return x + (y * size)


@njit(fastmath=True)
def diffuse(size, values_prev, values, k, iterations):
    for itr in range(iterations):
        for x in range(size):
            for y in range(size):
                values[index(size, x, y)] = (values_prev[index(size, x, y)] + k * ((values[index(size, x + 1, y)] +
                                                                                    values[index(size, x - 1, y)] +
                                                                                    values[index(size, x, y + 1)] +
                                                                                    values[index(size, x, y - 1)]) / 4)) / (1 + k)


@njit(fastmath=True)
def lerp(a, b, k):
    return a + k * (b - a)


@njit(fastmath=True)
def advection(size, cell_pos, cell_vel, params, dt):
    come_from = cell_pos - (cell_vel * dt)
    come_from_int = np.array([int(come_from[0] + size) % size, int(come_from[1] + size) % size])
    come_from_fract = np.array([come_from[0] - int(come_from[0]), come_from[1] - int(come_from[1])])
    lerp_dens_up = lerp(params[index(size, come_from_int[0], come_from_int[1])],
                        params[index(size, come_from_int[0] + 1, come_from_int[1])], come_from_fract[0])

    lerp_dens_down = lerp(params[index(size, come_from_int[0], come_from_int[1] + 1)],
                          params[index(size, come_from_int[0] + 1, come_from_int[1] + 1)], come_from_fract[0])

    new = lerp(lerp_dens_up, lerp_dens_down, come_from_fract[1])

    return new


@njit(fastmath=True)
def gaus_seidel_divergence(size, prev_vals, answ, iterations):
    for itr in range(iterations):
        for x in range(size):
            for y in range(size):
                summ = (answ[index(size, x + 1, y)] + answ[index(size, x - 1, y)] + answ[index(size, x, y + 1)] + answ[
                    index(size, x, y - 1)])
                answ[index(size, x, y)] = (summ - prev_vals[index(size, x, y)]) / 4


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
        self.cells_scale = 32
        self.cells_images = [pg.Surface((self.cells_scale, self.cells_scale)) for i in range(size * size)]

    def add_density(self, x, y, amount):
        self.cells[self.index(x, y)].density_prev += amount

    def add_velocity(self, x, y, vel):
        self.cells[self.index(x, y)].vel_prev += vel

    def update(self, dt):

        # diffuse vel
        vels_prev = [c.vel_prev.copy() for c in self.cells]
        vels = [c.vel.copy() for c in self.cells]
        prev_x_sep = np.array([v.x for v in vels_prev])
        prev_y_sep = np.array([v.y for v in vels_prev])
        x_sep = np.array([v.x for v in vels])
        y_sep = np.array([v.y for v in vels])

        diffuse(self.size, prev_x_sep, x_sep, dt + self.viscosity, 4)
        diffuse(self.size, prev_y_sep, y_sep, dt + self.viscosity, 4)
        for i in range(len(vels)):
            self.cells[i].vel = vec(float(x_sep[i]), float(y_sep[i]))


        self.clear_divergence()


        vels = [c.vel.copy() for c in self.cells]
        x_sep = np.array([v.x for v in vels])
        y_sep = np.array([v.y for v in vels])
        for c in self.cells:
            cell_pos = np.array(tuple(c.pos))
            cell_vel = np.array(tuple(c.vel))
            new_vel_x = advection(self.size, cell_pos, cell_vel, x_sep, dt)
            new_vel_y = advection(self.size, cell_pos, cell_vel, y_sep, dt)
            c.vel = vec(float(new_vel_x), float(new_vel_y))


        self.clear_divergence()

        # diffuse density
        dens_prev = np.array([c.density_prev for c in self.cells])
        dens = np.array([c.density for c in self.cells])
        diffuse(self.size, dens_prev, dens, dt + self.viscosity, 4)
        for i in range(len(dens)):
            self.cells[i].density = float(dens[i])

        # advect density, velocity
        dens = np.array([c.density for c in self.cells])
        for c in self.cells:
            cell_pos = np.array(tuple(c.pos))
            cell_vel = np.array(tuple(c.vel))
            c.density = float(advection(self.size, cell_pos, cell_vel, dens, dt))





        for i in range(len(self.cells_images)):
            color = translate(self.cells[i].density_prev, 0, 1, 0, 255)
            if color > 255:
                color = 255
            if color < 0:
                color = 0
            self.cells_images[i].fill((color, color, color))

        # fade
        for cell in self.cells:
            cell.density *= 0.993
            cell.vel *= 0.993

        # previous_vals -> current vals
        for c in self.cells:
            c.vel_prev = c.vel.copy()
            c.density_prev = c.density

    def clear_divergence(self):
        divergence = np.zeros(len(self.cells))
        for x in range(self.size):
            for y in range(self.size):
                # get divergence in each cell
                diff_x = (self.cells[index(self.size, x + 1, y)].vel.x - self.cells[
                    index(self.size, x - 1, y)].vel.x)
                diff_y = (self.cells[index(self.size, x, y + 1)].vel.y - self.cells[
                    index(self.size, x, y - 1)].vel.y)
                divergence[index(self.size, x, y)] = (diff_x + diff_y) * 0.5

        # gradients of vec field with no curl
        scalars = np.zeros(len(self.cells))
        gaus_seidel_divergence(self.size, divergence, scalars, 4)

        for x in range(self.size):
            for y in range(self.size):
                x_val = ((scalars[index(self.size, x + 1, y)] - scalars[index(self.size, x - 1, y)])) * 0.5
                y_val = ((scalars[index(self.size, x, y + 1)] - scalars[index(self.size, x, y - 1)])) * 0.5

                self.cells[index(self.size, x, y)].vel -= vec(x_val, y_val)

    def index(self, x, y):
        y = int((y + self.size) % self.size)
        x = int((x + self.size) % self.size)
        return x + (y * self.size)

    def index_2d(self, i):
        return (i % self.size, i // self.size)

    def draw(self, surf):
        for x in range(self.size):
            for y in range(self.size):
                surf.blit(self.cells_images[self.index(x, y)], (x * self.cells_scale, y * self.cells_scale))
        # for c in self.cells:
        #     pg.draw.line(surf, RED, c.pos * self.cells_scale, c.pos * self.cells_scale + c.vel_prev)
