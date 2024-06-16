import pygame

import src.local.game.assets as assets
from src.local.game import configs
import random

from src.local.game.layer import Layer


class Pipe(pygame.sprite.Sprite):
    def __init__(self,x=0,*groups):
        self._layer = Layer.OBSTACLE
        self.gap = 120

        self.sprite = assets.get_sprite("pipe")
        self.sprite_rect = self.sprite.get_rect()

        self.pipe_bottom = self.sprite
        self.pipe_bottom_rect = self.pipe_bottom.get_rect(topleft=(0, self.sprite_rect.height + self.gap))


        self.pipe_top = pygame.transform.flip(self.sprite, False, True)
        self.pipe_top_rect = self.pipe_top.get_rect(topleft=(0, 0))

        self.image = pygame.surface.Surface((self.sprite_rect.width, self.sprite_rect.height * 2 + self.gap),
                                            pygame.SRCALPHA)
        self.image.blit(self.pipe_bottom, self.pipe_bottom_rect)
        self.image.blit(self.pipe_top, self.pipe_top_rect)
        print(x)
        self.rect = self.image.get_rect(midleft=(configs.SCREEN_WIDTH + x, random.uniform(150,400)))
        self.mask = pygame.mask.from_surface(self.image)

        self.passed = False

        super().__init__(*groups)

    def update(self):
        self.rect.x -= 3

        if self.rect.right <= 0:
            self.kill()

    def is_passed(self):
        if self.rect.x < 50 and not self.passed:
            self.passed = True
            return True
        return False
