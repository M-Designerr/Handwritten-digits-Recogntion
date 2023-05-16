import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
pygame.init()

BOUNDARY = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

MODEL = load_model('best_model.h5')

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three",
          4: "Four", 5: "Five", 6: "Six", 7: "Seven",
          8: "Eight", 9: "Nine"}

FONT = pygame.font.SysFont("Arial", 30)
# Set up the window
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
# WHILE_INT = DISPLAYSURF.mp_rgb(WHITE)
pygame.display.set_caption('Board')

iswriting = False
number_xc = []
number_yc = []
img_cnt = 1
PREDICT = True
while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xc, yc = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xc, yc), 4, 0)
            number_xc.append(xc)
            number_yc.append(yc)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xc = sorted(number_xc)
            number_yc = sorted(number_yc)

            rect_min_x, rect_max_x = max(
                number_xc[0] - BOUNDARY, 0), min(WINDOWSIZEX, number_xc[-1] + BOUNDARY)

            rect_min_y, rect_max_y = max(
                number_yc[0] - BOUNDARY, 0), min(WINDOWSIZEY, number_yc[-1] + BOUNDARY)

            number_xc = []
            number_yc = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[
                rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("img.png", img_arr)
                img_cnt += 1

            if PREDICT:

                img = cv2.resize(img_arr, (28, 28))
                img = np.pad(img, (10, 10), 'constant', constant_values=0)
                img = cv2.resize(img, (28, 28))/255

                label = str(
                    LABELS[np.argmax(MODEL.predict(img.reshape(1, 28, 28, 1)))])

                text = FONT.render(label, True, RED, WHITE)
                textRect = text.get_rect()
                textRect.left, textRect.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(text, textRect)

        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
