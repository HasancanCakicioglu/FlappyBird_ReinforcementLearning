import os
import pygame

sprites = {}

def load_sprite():
    path = os.path.join("../../../assets")
    for file in os.listdir(path):
        sprites[file.split(".")[0]] = pygame.image.load(os.path.join(path, file))


def get_sprite(name):
    return sprites[name]