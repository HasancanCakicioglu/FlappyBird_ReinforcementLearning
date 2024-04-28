import pygame

import src.local.game.configs
import src.local.game.assets as assets
from src.local.game.objects.background import Background
from src.local.game.objects.bird import Bird
from src.local.game.objects.pipe import Pipe
from src.local.game.objects.ground import Ground

pygame.init()

screen = pygame.display.set_mode((src.local.game.configs.SCREEN_WIDTH, src.local.game.configs.SCREEN_HEIGHT))
pygame.display.set_caption(src.local.game.configs.SCREEN_TITLE)
clock = pygame.time.Clock()
pipe_create_event = pygame.USEREVENT
running = True

assets.load_sprite()

sprites = pygame.sprite.LayeredUpdates()

Background(0,sprites)
Background(1,sprites)

Ground(0,sprites)
Ground(1,sprites)

Pipe(0,sprites)
Pipe(300,sprites)
Pipe(600,sprites)
Pipe(900,sprites)


bird = Bird(sprites)

def check_pipe_count():
    """sprites içindeki Pipe nesnelerinin sayısını kontrol et."""
    pipe_count = sum(1 for sprite in sprites.sprites() if isinstance(sprite, Pipe))
    if pipe_count < 4:  # Eğer 4'ten az Pipe varsa yeni bir Pipe ekleyin
        Pipe(300,sprites)



while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        bird.handle_event(event)

    check_pipe_count()
    screen.fill(0)
    sprites.draw(screen)
    sprites.update()

    if bird.check_collision(sprites):
        running = False


    pygame.display.flip()
    clock.tick(src.local.game.configs.SCREEN_FPS)

