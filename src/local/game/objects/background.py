import pygame
import src.local.game.assets as assets
import src.local.game.configs as configs
from src.local.game.layer import Layer


class Background(pygame.sprite.Sprite):
    def __init__(self,index,*groups):
        self._layer = Layer.BACKGROUND
        self.image = assets.get_sprite("bg")
        self.rect = self.image.get_rect(topleft = (configs.SCREEN_WIDTH * index,-100))
        self.mask = pygame.mask.from_surface(self.image)
        super().__init__(*groups)
    def update(self):
        self.rect.x -= 1

        if self.rect.right <= 0:
            self.rect.x = configs.SCREEN_WIDTH
