import pygame

import src.local.game.assets as assets
from src.local.game import configs
from src.local.game.layer import Layer


class Ground(pygame.sprite.Sprite):
    def __init__(self,index, *groups):

        self._layer = Layer.GROUND
        self.image = assets.get_sprite("ground")
        self.rect = self.image.get_rect(topleft=(configs.SCREEN_WIDTH * index, configs.SCREEN_HEIGHT - self.image.get_height() / 2))
        self.mask = pygame.mask.from_surface(self.image)
        super().__init__(*groups)
    def update(self):
        self.rect.x -= 3

        if self.rect.right <= 0:
            self.rect.x = configs.SCREEN_WIDTH