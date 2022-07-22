import math
import pygame as pg
import sys
from settings import *
from colorful_fluid import *
from agent import Agent
from os import path
class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.load_data()

    def load_data(self):
        self.font=path.join("PixelatedRegular-aLKm.ttf")
    def draw_text(self, text, font_name, size, color, x, y, align="nw"):
        font = pg.font.Font(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "nw":
            text_rect.topleft = (x, y)
        if align == "ne":
            text_rect.topright = (x, y)
        if align == "sw":
            text_rect.bottomleft = (x, y)
        if align == "se":
            text_rect.bottomright = (x, y)
        if align == "n":
            text_rect.midtop = (x, y)
        if align == "s":
            text_rect.midbottom = (x, y)
        if align == "e":
            text_rect.midright = (x, y)
        if align == "w":
            text_rect.midleft = (x, y)
        if align == "center":
            text_rect.center = (x, y)
        self.screen.blit(text_surface, text_rect)
        return text_rect

    def new(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.fluid = Fluid(30, 0.00001, (BLACK, PURPULISH1, PURPULISH2, PURPULISH3, PURPULISH_BODY2), 640)
        self.angle = 0
        self.agent = Agent(self.fluid, (15, 15), (1, 1), math.pi/4, YELLOW)
        self.agent1 = Agent(self.fluid, (18, 15), (1, 1), math.pi/4, BLUE)
        self.agent2 = Agent(self.fluid, (15, 15), (1, 1), math.pi/4, RED)
        self.agent3 = Agent(self.fluid, (12, 15), (1, 1), math.pi/4, GREEN)

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()

        params = (20, 1)
        #self.agent.update(0.4, params[0], params[1], self.dt)
        self.agent1.update(0.4, params[0], params[1], self.dt)
        self.agent2.update(0.4, params[0], params[1], self.dt)
        self.agent3.update(0.4, params[0], params[1], self.dt)


        self.fluid.update(self.dt)




    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.fluid.draw(self.screen)
        # self.agent.draw(self.screen, 100)
        # self.agent1.draw(self.screen, 100)
        # self.agent2.draw(self.screen, 100)
        # self.agent3.draw(self.screen, 100)
        # fps
        self.draw_text(str(int(self.clock.get_fps())), self.font, 40, WHITE, 50, 50, align="center")
        pg.display.flip()

    def events(self):
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()




# create the game object
g = Game()
g.new()
g.run()
