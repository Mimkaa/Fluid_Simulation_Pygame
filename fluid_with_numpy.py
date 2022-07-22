import math
import random

import pygame as pg
import numpy as np
from numba import njit
from gaus_seidel import Gaus_Seidel
from settings import *

vec = pg.Vector2

@njit(fastmath=True)
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
def index_2d(size, i):
        return np.array([i % size, i // size])


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
def advection(size, cell_pos, cell_vel, params, ind, dt):
    come_from = cell_pos - (cell_vel * dt)
    come_from_int = np.array([int(come_from[0] + size) % size, int(come_from[1] + size) % size])
    come_from_fract = np.array([come_from[0] - int(come_from[0]), come_from[1] - int(come_from[1])])
    lerp_dens_up = lerp(params[index(size, come_from_int[0], come_from_int[1])],
                        params[index(size, come_from_int[0] + 1, come_from_int[1])], come_from_fract[0])

    lerp_dens_down = lerp(params[index(size, come_from_int[0], come_from_int[1] + 1)],
                          params[index(size, come_from_int[0] + 1, come_from_int[1] + 1)], come_from_fract[0])

    new = lerp(lerp_dens_up, lerp_dens_down, come_from_fract[1])

    params[ind] = new


@njit(fastmath=True)
def gaus_seidel_divergence(size, prev_vals, answ, iterations):
    for itr in range(iterations):
        for x in range(size):
            for y in range(size):
                summ = (answ[index(size, x + 1, y)] + answ[index(size, x - 1, y)] + answ[index(size, x, y + 1)] + answ[
                    index(size, x, y - 1)])
                answ[index(size, x, y)] = (summ - prev_vals[index(size, x, y)]) / 4


@njit(fastmath=True)
def linear_color_gradient(start_color, end_color, t):
    # t is how far we are in the linear interpolation
    return np.array([start_color[j] + t * (end_color[j] - start_color[j]) for j in range(3)])

@njit(fastmath=True)
def linear_color_gradient_mul(colors, t, len_colors):
        length_of_each = 1 / len_colors
        grad = np.array([i * length_of_each for i in range(len_colors)])

        # find the closest value(lower then our t) and the index of it
        closest = 9999
        closest_index = 0
        rev_grad = grad[:: -1]
        for i in range(len(rev_grad)):
            if rev_grad[i] <= t:
                closest = rev_grad[i]
                closest_index = (len_colors - 1) - i
                break

        to_go = (t - closest) / length_of_each

        return linear_color_gradient(colors[closest_index], colors[(closest_index + 1) % len_colors], to_go)





class Fluid:
    def __init__(self, size, viscosity, colors):
        self.size = size

        self.vels_x = np.zeros(size * size)
        self.vels_y = np.zeros(size * size)
        self.vels_prev_x = np.zeros(size * size)
        self.vels_prev_y = np.zeros(size * size)

        self.density = np.zeros(size * size)
        self.density_prev = np.zeros(size * size)
        self.viscosity = viscosity

        self.cells_scale = 16
        self.cells_images = [pg.Surface((self.cells_scale, self.cells_scale), pg.SRCALPHA) for i in range(size * size)]
        self.colors = np.array([np.array(c) for c in colors])
        self.max_density = 1
        self.full_size = size * size

    def add_density(self, x, y, amount):
        self.density_prev[index(self.size, x, y)] += amount

    def add_velocity(self, x, y, add_x, add_y):
        self.vels_prev_x[index(self.size, x, y)] += add_x
        self.vels_prev_y[index(self.size, x, y)] += add_y

    def update(self, dt):

        # diffuse vel
        diffuse(self.size, self.vels_prev_x, self.vels_x, dt + self.viscosity, 4)
        diffuse(self.size, self.vels_prev_y, self.vels_y, dt + self.viscosity, 4)

        self.clear_divergence()

        # advect velocity
        for n, _ in enumerate(self.vels_x):
            advection(self.size, index_2d(self.size, n), np.array([self.vels_x[n], self.vels_y[n]]), self.vels_x, n, dt)
            advection(self.size, index_2d(self.size, n), np.array([self.vels_x[n], self.vels_y[n]]), self.vels_y, n, dt)

        self.clear_divergence()

        # diffuse density
        diffuse(self.size, self.density_prev, self.density, dt + self.viscosity, 4)

        # advect density, velocity
        for n, _ in enumerate(self.density):
            advection(self.size, index_2d(self.size, n), np.array([self.vels_x[n], self.vels_y[n]]), self.density, n, dt)


        # color updating
        max_density = max(self.density)
        if self.max_density < max_density and int(max_density) > 0:
            self.max_density = max(1.0, max_density)

        for n, _ in enumerate(self.density):
            dens_trans = translate(self.density[n], 0, self.max_density, 0, 1)
            color = linear_color_gradient_mul(self.colors, dens_trans, len(self.colors))
            #color = np.append(color, translate(self.cells[i].density, 0, self.max_density, 0, 255))
            #color = (int(255 * dens_trans), int(255 * dens_trans), int(255 * dens_trans))
            self.cells_images[n].fill(color)

        # fade
        for n in range(self.full_size):
            self.density[n] *= 0.993
            self.vels_x[n] *= 0.993
            self.vels_y[n] *= 0.993

        # previous_vals -> current vals
        self.density_prev = np.copy(self.density)
        self.vels_prev_x = np.copy(self.vels_x)
        self.vels_prev_y = np.copy(self.vels_y)

    def clear_divergence(self):
        divergence = np.zeros(self.full_size)
        for x in range(self.size):
            for y in range(self.size):
                # get divergence in each cell
                diff_x = (self.vels_x[index(self.size, x + 1, y)] - self.vels_x[
                    index(self.size, x - 1, y)])
                diff_y = (self.vels_y[index(self.size, x, y + 1)] - self.vels_y[
                    index(self.size, x, y - 1)])
                divergence[index(self.size, x, y)] = (diff_x + diff_y) * 0.5

        # gradients of vec field with no curl
        scalars = np.zeros(self.full_size)
        gaus_seidel_divergence(self.size, divergence, scalars, 4)

        for x in range(self.size):
            for y in range(self.size):
                x_val = ((scalars[index(self.size, x + 1, y)] - scalars[index(self.size, x - 1, y)])) * 0.5
                y_val = ((scalars[index(self.size, x, y + 1)] - scalars[index(self.size, x, y - 1)])) * 0.5

                self.vels_x[index(self.size, x, y)] -= x_val
                self.vels_y[index(self.size, x, y)] -= y_val


    def draw(self, surf):
        for x in range(self.size):
            for y in range(self.size):
                surf.blit(self.cells_images[index(self.size, x, y)], (x * self.cells_scale, y * self.cells_scale))
        # for c in self.cells:
        #     pg.draw.line(surf, RED, c.pos * self.cells_scale, c.pos * self.cells_scale + c.vel_prev)

