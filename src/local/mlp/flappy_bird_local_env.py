import os
import time

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import random
from enum import IntEnum,auto

class Layer(IntEnum):
    BACKGROUND = auto()
    OBSTACLE = auto()
    GROUND = auto()
    PLAYER = auto()
    UI = auto()


def check_pipe_count(screen_width,sprites_asset,sprites):
    """sprites içindeki Pipe nesnelerinin sayısını kontrol et."""
    pipe_count = sum(1 for sprite in sprites.sprites() if isinstance(sprite, Pipe))
    if pipe_count < 4:  # Eğer 4'ten az Pipe varsa yeni bir Pipe ekleyin
        Pipe(screen_width,sprites_asset,300,sprites)


def load_sprite(sprites_asset):
    path = os.path.join("../../../assets")
    for file in os.listdir(path):
        sprites_asset[file.split(".")[0]] = pygame.image.load(os.path.join(path, file))
    return sprites_asset

def get_sprite(sprites_asset,name):
    return sprites_asset[name]

class Background(pygame.sprite.Sprite):
    def __init__(self,index,screen_width,sprites_asset,*groups):
        self._layer = Layer.BACKGROUND
        self.image = get_sprite(sprites_asset,"bg")
        self.SCREEN_WIDTH = screen_width
        self.rect = self.image.get_rect(topleft = (self.SCREEN_WIDTH * index,-100))
        self.mask = pygame.mask.from_surface(self.image)
        super().__init__(*groups)
    def update(self):
        self.rect.x -= 1

        if self.rect.right <= 0:
            self.rect.x = self.SCREEN_WIDTH


class Bird(pygame.sprite.Sprite):
    def __init__(self,gravity,sprites_asset,*groups):
        self._layer = Layer.PLAYER
        self.images = [get_sprite(sprites_asset,"bird1"),get_sprite(sprites_asset,"bird2"), get_sprite(sprites_asset,"bird3")]
        self.image = self.images[0]
        self.rect = self.image.get_rect(topleft=(100,200))
        self.GRAVITY = gravity

        self.mask = pygame.mask.from_surface(self.image)
        self.flap = 0

        super().__init__(*groups)

    def update(self):
        self.images.insert(0, self.images.pop())
        self.image = self.images[0]

        self.flap += self.GRAVITY
        self.rect.y += self.flap

    def handle_event(self,event):
        #if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        #    self.flap = 0
        #    self.flap -= 8
        if event==0:
            self.flap = 0
            self.flap -= 8

    def check_collision(self, sprites):
        for sprite in sprites:
            if (type(sprite) is Pipe or type(sprite) is Ground) and sprite.mask.overlap(self.mask, (self.rect.x - sprite.rect.x, self.rect.y - sprite.rect.y)):
                return True
        return False

class Ground(pygame.sprite.Sprite):
    def __init__(self,index,screen_width,screen_height,sprites_asset ,*groups):

        self._layer = Layer.GROUND
        self.image = get_sprite(sprites_asset,"ground")
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.rect = self.image.get_rect(topleft=(self.SCREEN_WIDTH * index, self.SCREEN_HEIGHT - self.image.get_height() / 2))
        self.mask = pygame.mask.from_surface(self.image)
        super().__init__(*groups)
    def update(self):
        self.rect.x -= 3

        if self.rect.right <= 0:
            self.rect.x = self.SCREEN_WIDTH


class Pipe(pygame.sprite.Sprite):
    def __init__(self,screen_width,sprites_asset,x=0,*groups):
        self._layer = Layer.OBSTACLE
        self.gap = 150

        self.SCREEN_WIDTH = screen_width
        self.sprite = get_sprite(sprites_asset,"pipe")
        self.sprite_rect = self.sprite.get_rect()

        self.pipe_bottom = self.sprite
        self.pipe_bottom_rect = self.pipe_bottom.get_rect(topleft=(0, self.sprite_rect.height + self.gap))

        self.pipe_top = pygame.transform.flip(self.sprite, False, True)
        self.pipe_top_rect = self.pipe_top.get_rect(topleft=(0, 0))

        self.image = pygame.surface.Surface((self.sprite_rect.width, self.sprite_rect.height * 2 + self.gap),
                                            pygame.SRCALPHA)
        self.image.blit(self.pipe_bottom, self.pipe_bottom_rect)
        self.image.blit(self.pipe_top, self.pipe_top_rect)

        self.rect = self.image.get_rect(midleft=(self.SCREEN_WIDTH + x, random.uniform(150,400)))
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

    def get_data(self):
        return self.pipe_bottom_rect.x,self.pipe_bottom_rect.y,self.pipe_top_rect.y

class FlappyBirdMlpLocalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "FlappyBirdMlpLocalEnv_v0"}
    def __init__(self):
        super().__init__()

        # 83x100
        self.observation_space = spaces.Box(low=-50, high=1000, shape=(4,), dtype=np.float16)

        # 0: flap, 1: do nothing
        self.action_space = spaces.Discrete(2)

        # Configurations
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.SCREEN_SIZE = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.SCREEN_TITLE = "Flappy Bird"
        self.SCREEN_FPS = 60
        self.GRAVITY = 0.4
        self.GAP_BETWEEN_PIPE = 300

        # Assets
        self.sprites_asset = {}

        #  Render
        self.screen = None
        self.clock = None
        self.background = None
        self.bird = None
        self.pipes = None
        self.grounds = None
        self.score = 0
        self.game_over = False


        pygame.init()
        self.sprites_asset = load_sprite(self.sprites_asset)

        self.sprites = pygame.sprite.LayeredUpdates()

        Background(0,self.SCREEN_WIDTH,self.sprites_asset,self.sprites)
        Background(1,self.SCREEN_WIDTH,self.sprites_asset,self.sprites)

        Ground(0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT,self.sprites_asset,self.sprites)
        Ground(1, self.SCREEN_WIDTH, self.SCREEN_HEIGHT,self.sprites_asset,self.sprites )

        Pipe(self.SCREEN_WIDTH, self.sprites_asset,0, self.sprites)
        Pipe(self.SCREEN_WIDTH, self.sprites_asset,300,self.sprites)
        Pipe(self.SCREEN_WIDTH, self.sprites_asset,600,self.sprites)
        Pipe(self.SCREEN_WIDTH, self.sprites_asset,900,self.sprites)

        self.bird = Bird(self.GRAVITY,self.sprites_asset,self.sprites)


    def reset(self,seed=None):
        """
        Resets the environment and returns the initial observation.

        :return:
            observation (np.ndarray): The initial observation.
        """
        self.game_over = False
        self.score = 0
        self.sprites.empty()


        self.sprites = pygame.sprite.LayeredUpdates()

        Background(0, self.SCREEN_WIDTH, self.sprites_asset, self.sprites)
        Background(1, self.SCREEN_WIDTH, self.sprites_asset, self.sprites)

        Ground(0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.sprites_asset, self.sprites)
        Ground(1, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.sprites_asset, self.sprites)

        Pipe(self.SCREEN_WIDTH, self.sprites_asset, 0, self.sprites)
        Pipe(self.SCREEN_WIDTH, self.sprites_asset, 300, self.sprites)
        Pipe(self.SCREEN_WIDTH, self.sprites_asset, 600, self.sprites)
        Pipe(self.SCREEN_WIDTH, self.sprites_asset, 900, self.sprites)

        self.bird = Bird(self.GRAVITY, self.sprites_asset, self.sprites)

        return self.get_observation() , {}

    def step(self, action):

        #for event in pygame.event.get():
        #    if event.type == event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        #        self.bird.handle_event(0)
        #    else:
        #        self.bird.handle_event(1)

        self.bird.handle_event(action)

        check_pipe_count(self.SCREEN_WIDTH,self.sprites_asset,self.sprites)
        self.sprites.update()
        reward = 0
        for sprite in self.sprites:
            if isinstance(sprite,Pipe):
                if not sprite.is_passed():
                    next_pipe_reward = sprite
                    reward = self.calculate_score(self.bird.rect.y,next_pipe_reward.rect.center[1])
                    break


        if self.bird.check_collision(self.sprites) or self.bird.rect.y < 0 :
            self.game_over = True
            reward = -10

        return self.get_observation(), reward, self.game_over, False , {}

    def get_observation(self):
        next_pipe = None
        for sprite in self.sprites:
            if isinstance(sprite,Pipe):
                if not sprite.is_passed():
                    next_pipe = sprite
                    break
        return np.array([
            self.bird.rect.y,
            format(self.bird.flap, ".2f"),
            next_pipe.rect.x - self.bird.rect.x,
            self.bird.rect.y-next_pipe.rect.center[1]
        ],dtype=np.float16)

    def render(self,mode="human"):
        if mode == "human":
            if self.screen is None:
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption(self.SCREEN_TITLE)
                self.clock = pygame.time.Clock()

            self.screen.fill(0)
            self.sprites.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(self.SCREEN_FPS)

    def calculate_score(self,bird_y, pipe_center_y):
        # Mesafeyi hesapla
        distance = abs(bird_y - pipe_center_y)

        # Burada max_distance değeri, en uzak mesafe olarak kabul edilecek değeri belirler.
        # max_distance ne kadar büyükse, mesafeye göre puan o kadar düşük olur.
        max_distance = 250  # Bu değeri ihtiyacınıza göre ayarlayabilirsiniz.

        # Mesafe ne kadar küçükse, puan o kadar yüksek olur.
        # Mesafe büyükse, puan küçük olur.
        # Mesafe sıfırsa (yani bird_y == pipe_center_y), puan maksimum olur.
        score = max(0, (max_distance - distance) / max_distance * 100) /100

        return score


    def close(self):
        pygame.quit()




if __name__ == "__main__":
    env = FlappyBirdMlpLocalEnv()
    env.render()
    obs = env.reset()
    for i in range(10000):
        #action = env.action_space.sample()
        if i % 45 == 0:
            action = 0
        else:
            action = 1
        env.render()
        obs, reward, done, _ , info = env.step(action)
        time.sleep(0.1)
        if done:
            print("Game Over")
            env.reset()
