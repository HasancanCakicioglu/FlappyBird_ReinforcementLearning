import pygame

import src.local.game.assets as assets
from src.local.game import configs
from src.local.game.layer import Layer
from src.local.game.objects.ground import Ground
from src.local.game.objects.pipe import Pipe


class Bird(pygame.sprite.Sprite):
    def __init__(self, *groups):
        self._layer = Layer.PLAYER

        self.images = [assets.get_sprite("bird1"), assets.get_sprite("bird2"), assets.get_sprite("bird3")]
        self.image = self.images[0]
        self.rect = self.image.get_rect(topleft=(100,0))
        self.velocity = 0

        self.mask = pygame.mask.from_surface(self.image)
        self.flap = 0

        super().__init__(*groups)

    def update(self):
        self.images.insert(0, self.images.pop())
        self.image = self.images[0]

        self.flap += configs.GRAVITY
        self.rect.y += self.flap

    def handle_event(self,event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.flap = 0
            self.flap -= 8

    def check_collision(self, sprites):
        for sprite in sprites:
            if (type(sprite) is Pipe or type(sprite) is Ground) and sprite.mask.overlap(self.mask, (self.rect.x - sprite.rect.x, self.rect.y - sprite.rect.y)):
                return True
        return False
