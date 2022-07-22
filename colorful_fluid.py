import math
import random

import pygame as pg
import numpy as np
from numba import njit
from gaus_seidel import Gaus_Seidel
from settings import *

vec = pg.Vector2

@njit(fastmath=True)
def linear_color_gradient(start_color, end_color, t):
    # t is how far we are in the linear interpolation
    return np.array([start_color[j] + t * (end_color[j] - start_color[j]) for j in range(3)])


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
def index_color(size, x, y):
    y = int((y + size) % size)
    x = int((x + size) % size)
    return (x + (y * size)) * 3


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
def advection_color(size, cell_pos, cell_vel, params, dt):
    come_from = cell_pos - (cell_vel * dt)
    come_from_int = np.array([int(come_from[0] + size) % size, int(come_from[1] + size) % size])
    come_from_fract = np.array([come_from[0] - int(come_from[0]), come_from[1] - int(come_from[1])])
    lerp_color_up = linear_color_gradient(params[index(size, come_from_int[0], come_from_int[1])],
                                          params[index(size, come_from_int[0] + 1, come_from_int[1])], come_from_fract[0])

    lerp_color_down = linear_color_gradient(params[index(size, come_from_int[0], come_from_int[1] + 1)],
                                          params[index(size, come_from_int[0] + 1, come_from_int[1] + 1)], come_from_fract[0])


    new = linear_color_gradient(lerp_color_up, lerp_color_down, come_from_fract[1])

    return new


@njit(fastmath=True)
def gaus_seidel_divergence(size, prev_vals, answ, iterations):
    for itr in range(iterations):
        for x in range(size):
            for y in range(size):
                summ = (answ[index(size, x + 1, y)] + answ[index(size, x - 1, y)] + answ[index(size, x, y + 1)] + answ[
                    index(size, x, y - 1)])
                answ[index(size, x, y)] = (summ - prev_vals[index(size, x, y)]) / 4




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




class Cell:
    def __init__(self, pos):
        self.density = 0
        self.density_prev = 0
        self.vel = vec(0, 0)
        self.vel_prev = vec(0, 0)
        self.pos = vec(pos)
        self.color = list(BLACK)


class Fluid:
    def __init__(self, size, viscosity, colors, width):
        self.size = size
        self.cells = [Cell(self.index_2d(i)) for i in range(size * size)]
        self.viscosity = viscosity
        self.cells_scale = width // self.size
        self.width = width
        self.colors = np.array([np.array(c) for c in colors])
        self.max_density = 1
        self.image = pg.Surface((self.size * self.cells_scale, self.size * self.cells_scale))

    def add_density(self, x, y, amount, color):
        self.cells[index(self.size, x, y)].density_prev += amount
        self.cells[index(self.size, x, y)].color = color

    def add_velocity(self, x, y, vel):
        self.cells[index(self.size, x, y)].vel_prev += vel

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
        for n, _ in enumerate(vels):
            self.cells[n].vel = vec(float(x_sep[n]), float(y_sep[n]))

        # advect  velocity
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

        # advect colors
        colors = np.array([np.array(c.color) for c in self.cells])

        for c in self.cells:
            cell_pos = np.array(tuple(c.pos))
            cell_vel = np.array(tuple(c.vel))
            c.color = advection_color(self.size, cell_pos, cell_vel, colors, dt)
            c.color = [max(0, i) if i < 0 else min(255, i) for i in c.color]

        # fade
        for cell in self.cells:
            cell.density *= 0.9993
            cell.vel *= 0.9993
            cell.color = [i * 0.9993 for i in cell.color]

        # make velocity to be 0 at borders
        for x in range(self.size):
            for y in range(self.size):
                if x == 0 or y == 0 or y == self.size - 1 or x == self.size - 1:
                    self.cells[index(self.size, x, y)].vel *= 0

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
                pg.draw.rect(self.image, self.cells[index(self.size, x, y)].color, (x * self.cells_scale, y * self.cells_scale, self.cells_scale, self.cells_scale))

        surf.blit(self.image, (0, 0))
                #surf.blit(self.cells_images[self.index(x, y)], (x * self.cells_scale, y * self.cells_scale))
        # for c in self.cells:
        #     pg.draw.line(surf, RED, c.pos * self.cells_scale, c.pos * self.cells_scale + c.vel_prev)

