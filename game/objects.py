import pygame_sdl2

pygame_sdl2.import_as_pygame()

import pygame as pg


class GameObject(pg.sprite.Sprite):
    """Generic game object as a rectangle filled with color."""

    def __init__(self, color, width, height):
        pg.sprite.Sprite.__init__(self)

        self.image = pg.Surface((width, height), pg.SRCALPHA)
        self.image.fill(color)
        self.rect = self.image.get_rect()

    def move(self, x: int, y: int) -> None:
        self.rect = self.rect.move(x, y)

    def move_to(self, x: int, y: int) -> None:
        self.rect = pg.Rect(x, y, self.rect.width, self.rect.height)


class Stone(GameObject):
    """Stone game object. Just a circle of 'color'."""

    def __init__(self, color, width, height, margin):
        GameObject.__init__(self, (0, 0, 0, 0), width, height)
        self.rect = pg.draw.circle(
            self.image,
            color,
            self.rect.center,
            width // 2 - margin,
        )
