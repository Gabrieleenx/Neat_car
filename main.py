import pygame
import numpy as np
import copy
from Neat import Neat

class Car(object):

    def __init__(self, start_pos_x, start_pas_y, start_direction):
        self.pos_x = start_pos_x
        self.pos_y = start_pas_y
        self.direction = start_direction
        self.velocity = 0
        self.vector_offset = 0
        self.mass = 1000
        self.V_1 = 0

    def drive(self, throttle, steer_angle, t):
        V_1 = self.velocity*np.cos(self.vector_offset)
        V_2 = self.velocity*np.sin(self.vector_offset)
        F_1 = throttle * 8000
        F_2 = V_1 * 200

        a_1 = (F_1-F_2)/self.mass

        radius = (1 + np.tan(abs(self.vector_offset))*V_1 * 2)*np.tan((np.pi/2)-steer_angle)*2
        F_3 = (self.mass*V_1**2)/radius
        if V_2 != 0:
            F_4 = np.sign(V_2)*4000
        elif abs(F_3) > 2000:
            F_4 = np.sign(F_3)*2000
        else:
            F_4 = F_3

        a_2 = (F_3-F_4)/self.mass

        self.direction += (steer_angle + self.vector_offset*0.2) * (V_1 - abs((0.2*V_1)**2)) * 0.6*t
        V_2n = V_2
        V_1 += a_1 * t
        V_2 += a_2 * t
        V_1 -= abs(V_2*0.02)

        if V_1 < 0:
            V_1 = 0
        if abs(V_2n) > 0.01 and np.sign(V_2n) != np.sign(V_2):
            V_2 = 0

        self.velocity = np.sqrt(V_1**2 + V_2**2)
        if V_1 > 0.1:
            self.vector_offset = np.arctan(V_2/V_1)
        else:
            self.vector_offset = np.arctan(V_2/0.01)
        self.V_1 = V_1

        self.pos_y += self.velocity*np.cos(self.direction-self.vector_offset)/2
        self.pos_x -= self.velocity*np.sin(self.direction-self.vector_offset)/2

    def output(self):
        return self.pos_x, self.pos_y, self.direction, self.V_1


def activation_function(wh):
    # write activation function here
    return 1/(1 + np.exp(-wh))

def intersect(p1, p2, p3, p4):

    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False


def on_segment(p1, p2, p):
    return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])


def cross_product(p1, p2):
    return p1[0] * p2[1] - p2[0] * p1[1]


def direction(p1, p2, p3):
    return cross_product([p3[0]-p1[0], p3[1]-p1[1]], [p2[0]-p1[0], p2[1]-p1[1]])


def rotate(x1, y1, x2, y2, angle):

    qx = x1 + np.cos(angle) * (x2 - x1) - np.sin(angle) * (y2 - y1)
    qy = y1 + np.sin(angle) * (x2 - x1) + np.cos(angle) * (y2 - y1)

    return int(qx), int(qy)

def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def sight(car, map, win):
    car_ = car.output()
    car_x = [car_[0]+37, car_[0] + 200+37, car_[0] + 200+37, car_[0] + 200+37, car_[0]+37, car_[0] - 200+37, car_[0] -
             200+37, car_[0] - 200+37]
    car_y = [720 - car_[1] - 400 +37, 720 - car_[1] - 200+37, 720 - car_[1]+37, 720 - car_[1] + 200+37, 720 - car_[1] +
             200+37, 720 - car_[1] + 200+37, 720 - car_[1]+37, 720 - car_[1] - 200+37]
    sight_list = [2.0, 1.415, 1.0, 1.415, 1.0, 1.415, 1.0, 1.415]
    for i in range(8):
        x1 = car_[0] + 37
        y1 = 720 - car_[1] + 37
        x2, y2 = rotate(car_[0] + 37, 720 - car_[1] + 37, car_x[i], car_y[i], -car_[2])

        # pygame.draw.line(win, (0, 0, 0), (x1, y1), (x2, y2), 1)

        for k in range(len(map[0])):
            if intersect([x1, y1], [x2, y2], [map[0][k - 1], map[1][k - 1]], [map[0][k], map[1][k]]):
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([map[0][k - 1], map[1][k - 1]])
                p4 = np.array([map[0][k], map[1][k]])
                in_point = seg_intersect(p1, p2, p3, p4)
                # pygame.draw.circle(win, (0,0,0), (int(in_point[0]), int(in_point[1])), 4)
                dist = 0.005*np.sqrt((x1-in_point[0])**2 + (y1-in_point[1])**2)
                if dist < sight_list[i]:
                    sight_list[i] = dist
        for k in range(len(map[2])):
            if intersect([x1, y1], [x2, y2], [map[2][k - 1], map[3][k - 1]], [map[2][k], map[3][k]]):
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([map[2][k - 1], map[3][k - 1]])
                p4 = np.array([map[2][k], map[3][k]])
                in_point = seg_intersect(p1, p2, p3, p4)
                dist = 0.005 * np.sqrt((x1 - in_point[0]) ** 2 + (y1 - in_point[1]) ** 2)
                # pygame.draw.circle(win, (0,0,0), (int(in_point[0]), int(in_point[1])), 4)

                if dist < sight_list[i]:
                    sight_list[i] = dist

    return sight_list

class Map_build(object):
    def __init__(self, mouse, win):
        self.mouse = mouse
        self.win = win
        self.check = 0
        self.checkp = 0
        self.track = 0
        self.delete = 0
        self.x1 = np.array([], dtype=int)
        self.y1 = np.array([], dtype=int)
        self.x2 = np.array([], dtype=int)
        self.y2 = np.array([], dtype=int)

    def level_editor(self):
        mouse = self.mouse.mouser(self)
        keys = pygame.key.get_pressed()
        for key in keys:
            if keys[pygame.K_m]:
                self.track = 1
            if keys[pygame.K_n] and self.check == 0:

                self.delete = 1
                self.check = 1
            if keys[pygame.K_n] == False:
                self.check = 0
            if keys[pygame.K_p] and self.checkp == 0:
                #write_data(self.x1, self.y1, self.x2, self.y2)
                self.checkp = 1

        if mouse[2] == 1:

            print(mouse[0], mouse[1])


        if self.track == 0:
            if self.delete == 1:
                self.x1 = np.delete(self.x1, -1)
                self.y1 = np.delete(self.y1, -1)
        else:
            if self.delete == 1:
                self.x2 = np.delete(self.x2, -1)
                self.y2 = np.delete(self.y2, -1)
        self.delete = 0
        if len(self.x1) >= 2:
            for i in range(len(self.x1)):
                pygame.draw.line(self.win, (0, 0, 0), (self.x1[i-1], self.y1[i-1]), (self.x1[i], self.y1[i]), 1)
        if len(self.x2) >= 2:
            for i in range(len(self.x2)):
                pygame.draw.line(self.win, (0, 0, 0), (self.x2[i-1], self.y2[i-1]), (self.x2[i], self.y2[i]), 1)


def read_data():
    map = np.load('map.npz')
    return map['arr_0'], map['arr_1'], map['arr_2'], map['arr_3']


def write_data(x1, y1, x2, y2):
    np.savez('map.npz', x1, y1, x2, y2)


class Mouse(object):

    def __init__(self):
        self.click = 0

    def mouser(self):
        clicker = 0
        pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if click[0] == 0:
            self.click = 0

        if self.click == 0:
            clicker = click[0]
            if clicker == 1:
                self.click = 1

        return pos[0], pos[1], clicker


def render(win, car, car_img, map):

    pos_x, pos_y, direction, speed = car.output()
    #pygame.draw.rect(win, (200, 200, 200), (pos_x, 720-pos_y, 25, 50))


    for i in range(len(map[0])):
        pygame.draw.line(win, (0, 0, 0), (map[0][i-1], map[1][i-1]), (map[0][i], map[1][i]), 1)
    for i in range(len(map[2])):
        pygame.draw.line(win, (0, 0, 0), (map[2][i-1], map[3][i-1]), (map[2][i], map[3][i]), 1)

    image = rot_center(car_img, direction*180/np.pi)

    win.blit(image, (pos_x, 720-pos_y))

    pygame.display.update()


def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


def human_driving():
    keys = pygame.key.get_pressed()
    trottle = 0
    steer_angle = 0
    for key in keys:
        if keys[pygame.K_w]:
            trottle = 1
        elif keys[pygame.K_s]:
            trottle = -1
        if keys[pygame.K_d]:
            steer_angle = -40*np.pi/180
        elif keys[pygame.K_a]:
            steer_angle = 40*np.pi/180
    return trottle, steer_angle


def game_loop(car, win, clock, car_img, map_build, game_iteration, neat):
    fps = 30
    map = read_data()
    game_on = True
    throttle = 0
    steer_angle = 0
    too_slow = 0
    x1_score = [13, 7, 19, 232, 508,  389, 550, 725, 1061, 1224, 561, 252]
    y1_score = [391, 308, 175, 5, 182, 478,520, 446, 201, 412, 530, 502]
    x2_score = [198, 190, 207, 250, 307, 508, 540, 628, 1138, 1050, 560, 54]
    y2_score = [398, 310, 242, 186, 213, 341, 341, 272, 19, 344, 686, 608]

    score = 0
    i_ss = 0
    ittr = 0

    '''

    out = neat.evaluate(network_nr=49, net_input=[1.1, 2.3, 0.5], recursion=3)
    neat.update_fitness(3, 9)
    neat.update_fitness(1, 4)
    neat.update_fitness(2, 5)
    neat.update_fitness(5, 8)
    neat.update_fitness(7, 1)

    neat.train(0.2, 0.06, 0.5)
    throttle = wh[0]*2-1
    steer_angle = wh[1]*1.39626-0.6981
'''
    while game_on:

        win.fill((100, 100, 100))
        #throttle, steer_angle = human_driving()
        car.drive(throttle, steer_angle, 1/fps)
        for i in range(len(x2_score)):
            pygame.draw.line(win, (0,0,0), (x1_score[i], y1_score[i]),(x2_score[i], y2_score[i]),1)
        #map_build.level_editor()

       # collision(car, map, win)
        vision = sight(car, map, win)
        for i in range(8):
            if vision[i] < 0.075:
                game_on = False
                neat.update_fitness(game_iteration, score)
                return score

        input_layer = np.append(vision, car.output()[3])
        out = neat.evaluate(network_nr=game_iteration, net_input=input_layer, recursion=3)
        throttle = out[0]*2-1
        steer_angle = out[1]*1.39626-0.6981
        i_s = i_ss % len(x1_score)
        score_lines = [[x1_score[i_s], x2_score[i_s]], [y1_score[i_s], y2_score[i_s]], [x1_score[i_s], x2_score[i_s]], [y1_score[i_s], y2_score[i_s]]]
        score_col = sight(car, score_lines, win)
        ittr += 1
        if ittr > 6000:
            game_on = False
            neat.update_fitness(game_iteration, score)
            return score




        if score_col[0] < 0.075:
            score += 5
            i_ss += 1
            #print(ittr)


        render(win, car, car_img, map)

        if car.output()[3] <= 0.3:
            too_slow += 1
        if too_slow >= 100:
            neat.update_fitness(game_iteration, score)
            return score

            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        #clock.tick(fps)


def main():
    win = pygame.display.set_mode((1280, 720))
    car_load = pygame.image.load("car.png").convert_alpha()
    car_img = pygame.transform.scale(car_load, (75, 75))
    mouse = Mouse
    map_build = Map_build(mouse, win)
    clock = pygame.time.Clock()
    game_iteration = 0

    game_gen = 0
    score = 0
    neat = Neat(population_size=50, generations_to_extinct=10, c1=1, c2=1, c3=0.4, delta_species=0.4, input_size=9,
                output_size=2)
    neat.initial_population()
    '''
    neat = Neat(population_size=50, generations_to_extinct=8, c1=1, c2=1, c3=0.4, delta_species=0.4, input_size=3,
                output_size=1)
    neat.initial_population()

    out = neat.evaluate(network_nr=49, net_input=[1.1, 2.3, 0.5], recursion=3)
    neat.update_fitness(3, 9)
    neat.update_fitness(1, 4)
    neat.update_fitness(2, 5)
    neat.update_fitness(5, 8)
    neat.update_fitness(7, 1)

    neat.train(0.2, 0.06, 0.5)
    throttle = wh[0]*2-1
    steer_angle = wh[1]*1.39626-0.6981
'''
    while True:

        car = Car(100, 300, 0)

        score_ = game_loop(car, win, clock, car_img, map_build, game_iteration, neat)
        print('score',neat.population[str(game_iteration)].fitness)
        if score_ > score:
            score = score_
        game_iteration += 1
        if game_iteration == 50:
            neat.train(0.2, 0.26, 0.5)
            game_iteration = 0
            game_gen += 1

            score = 0
            print('gen',game_gen)
            print(neat.species_fitness)


main()

