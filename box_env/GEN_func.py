import numpy as np
import gym
# from main import model_trained
from shapely.geometry import LineString, LinearRing, box, Point, Polygon
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pickle
import math as mt
import matplotlib.pyplot as plt
import random, csaps, scipy, math
from scipy import misc
from scipy import interpolate
# from gym.envs.registration import register



def model_trained(speed, theta, t_old, theta_init):    
    name = 'lay1_wt_four_bowl_64.pk1'
    with open(name, 'rb') as weight:
        T  = pickle.load(weight)
        weight.close()

    name2 = 'lay2_wt_four_bowl_48.pk1'
    with open(name2, 'rb') as weight2:
        TT = pickle.load(weight2)
        weight2.close()

    trj_hd_resp, _ = HD(theta, theta_init)
    PI1d, theta_int = PI(trj_hd_resp, speed, t_old)
    ot1 = all_plot(T, PI1d)#, 0.70) #, obj)
    ot2 = all_plot(TT, ot1)#, 0.95) #, obj)
    # ot2 = ot2.T 
    return ot2, theta_int, trj_hd_resp.flatten()
    

def HD(t, theta_init):
    with open('hd_som_wt2.pk1', 'rb') as k:
        wt2 = pickle.load(k)

    k = theta_init
    X1 = [mt.cos(mt.radians(k)), mt.sin(mt.radians(k))]
    X2 = [mt.cos(mt.radians(t)), mt.sin(mt.radians(t))]
    s1 = X2[0]*X1[1] - X1[0]*X2[1]
    s2 = X2[0]*X1[0] + X1[1]*X2[1]
    X = [s1, s2]
    y_p = repsom2dlinear(X, wt2)
    # print("HD response computed")
    # print(y_p)
    return y_p, y_p.flatten()


def PI(resp, s, t_old):
    bf = 6*2*mt.pi
    dt = 0.01
    betaa = 40

    y_q = resp
    inp1d = np.reshape(np.transpose(y_q),(100,1))
    theta_dot = [(bf + betaa * s * k[0] * 10) for k in inp1d]
    theta_dot_int = [x*dt for x in theta_dot]
    thet = [i+j for i,j in zip(t_old, theta_dot_int)]

    thet = np.transpose(np.asarray(thet))
    Xarr = np.cos(thet)

    PI1d = Xarr
    return PI1d, thet


def all_plot(T, PI1d):
    PI1d = np.asarray(PI1d)
    PI1d = PI1d.reshape((PI1d.shape[0], 1))

    ot = np.matmul(T, PI1d)

    return ot #the output is (1,35)


def repsom2dlinear(x1, wt):
    sz_wt = list(wt.shape)
    y1 = np.zeros((sz_wt[0], sz_wt[1]))
    if(sz_wt[2] != len(x1)):
        print('Invalid input size in repsom2d()\n')
        return

    for i in range(sz_wt[0]):
        for j in range(sz_wt[1]):
            v = wt[i][j].reshape(sz_wt[2], 1)    
            # print(v)
            y1[i][j] =  np.dot(x1,v)
    return y1


def mod(theta, t_old):
    speed = (1/15)
    out, theta_sum = model_trained(speed, theta, t_old)
    # print(out)
    return out, theta_sum


def rew(x,y,cir):
    p = Point((x,y))
    R = 0
    square = box(-2,-2,2,2)
    if not square.contains(p):
        R = -1
    if cir[0].contains(p):
        R = 1
    if cir[2].contains(p):
        R = 1
    if cir[1].contains(p):
        R = 0.5
    if cir[3].contains(p):
        R = 0.5     
    return R
    
def cir_check(circs, x, y):
    p1 = Point((x,y))
    for k in circs:
        if k.contains(p1):
            check = True
            return check, circs.index(k)
            break
        else:
            check = False
    return check, None


def sq_chk(out_boundary, xchk, ychk):
    p2 = Point((xchk, ychk))
    k2 = LinearRing(out_boundary)
    if k2.contains(p2):
        return False
    else:
        return True

def env_boundary(x2, y2):
    if x2 <= -size:
        x2 = -1.99
    if x2 >= size:
        x2 = 1.99
    if y2 >= size:
        y2 = 1.99
    if y2 <= -size:
        y2 = -1.99
    
    return x2, y2


def speedpos(x, y):
    dist = np.sqrt((x[-1] - x[-2])**2 + (y[-1] - y[-2])**2)
    return dist


def speedinit(x, y):
    dist = []
    for i in range(1,len(x)):
        dist.append(np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2))
    return dist


def hd(x2, y2):
    # calculating theta for given coords
    tgn = myfrenet(x2, y2) # tangent vectors at all the points
    ang = [] 
    p = tgn.shape
   
    # adjusting angle according to different quadrants
    for i in range(p[0]):
        temp = math.degrees(math.atan(tgn[i, 1]/tgn[i, 0]))
        if tgn[i, 0] >= 0:
            if tgn[i,1] >= 0:
                # quadrant 1
                ang.insert(i, temp)
            else:
                #quadrant 4
                ang.insert(i, (temp+360))
        else:
            if (tgn[i, 1] >= 0) or (tgn[i, 1] <= 0):
                # quadrant 2, 3
                ang.insert(i, (temp+180))
    return ang


def myfrenet(x1, y1):
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    sz = x1.shape

    # calculate derivatives of the curve using spline method
    X = csaps.CubicSmoothingSpline(range(sz[0]), x1, smooth = 1.021)
    Y = csaps.CubicSmoothingSpline(range(sz[0]), y1, smooth = 1.021)
    p = np.asarray(range(sz[0]))
    mx = scipy.misc.derivative(X, p)
    my = scipy.misc.derivative(Y, p)
    A = [mx, my] # list of all derivative vectors at given points of curve
    
    # discard bad points
    j1 = np.sqrt(np.sum(np.multiply(A,A),axis=0))
    ind = np.nonzero(j1)
    data_x =[mx[i]for i in ind[0]]
    data_y =[my[i]for i in ind[0]]
    data = np.transpose(np.array([data_x, data_y])) # array with non-zero derivative values

    # Normalize Tangents
    T_1 = np.sqrt(np.sum(np.multiply(data,data),axis=1))
    T = np.divide(data,np.column_stack((T_1, T_1)))

    # finding derivatives at given trajectory inputs
    fx = scipy.interpolate.interp1d(ind[0], np.transpose(T), axis=1, fill_value="extrapolate")
    T = np.transpose(fx(p))
    
    # Normalize all the tangent vectors
    Tang_1 = np.sqrt(np.sum(np.multiply(T,T),axis=1))
    Tang = np.divide(T,np.column_stack((Tang_1,Tang_1)))
    
    return Tang

def new_circle(flag, obj, c):
    if obj == False:
        #creating a rectangular box
        if flag == 4:
            sq = box(-1,-1,1,1)
    
    elif obj == True: # if there is an object in the environment
        # creating a circular object
        if flag == 1:
            r = 0.25
            p = Point(c)
            sq = p.buffer(r)
    # seperating cords    
    k = list(sq.exterior.coords)
    xi = [m[0] for m in k ]
    yi = [m[1] for m in k ]
    
    #plotting both the boundaries
    plt.plot(xi, yi)
    # plt.pause(0.001)
    return k


def mirrorImage(square, pts): 
    square = LinearRing(square)
    line = LineString(pts)
    k = square.intersection(line)

    if k.is_empty:
        x, y = pts[1][0], pts[1][1]

    else:
        x_cord = k.x 
        y_cord= k.y

        # print(x_cord)

        x1, y1 = pts[1][0], pts[1][1]
        if x_cord == 1:
            a, b, c = 1, 0, -1
        if x_cord == -1:
            a, b, c = 1, 0, 1
        if y_cord == 1:
            a, b, c = 0, 1, -1
        if y_cord == -1:
            a, b, c = 0, 1, 1

        temp = -2 * (a * x1 + b * y1 + c) /(a * a + b * b) 
        x = temp * a + x1 
        y = temp * b + y1  
    return x, y 


def updatestep(dif, delV, Vhigh, Vlow):
    k3, k4, sigma = 0.1, 0.1, 0.3
    big = delV - Vhigh
    small = delV - Vlow
    es = (delV**2)/(sigma**2)
    mid = np.exp(-es)*np.random.normal(0,0.08)

    final = (logsig(k3*big) - logsig(k4*small))*dif + mid
    return final



def logsig(t):
    return 1/(1 + np.exp(-t))






'''
    def step(action):
        if steps == 0:
            theta = init
            obs, t_sum = mod(theta, [0]*100)
            initial = [x, y]
        else:
            obs, t_sum = mod(theta, t_sum)
        # print(steps)
        steps += 1

        # print(x)
        if action == 0:
            x += (1/15) #10**(-2)
            theta = 0
        if action == 1:
            x -= (1/15) #10**(-2)
            theta = 180
        if action == 2:
            y += (1/15) #10**(-2)
            theta = 90
        if action == 3:
            y -= (1/15)
            theta = -90

        reward = rew(x,y, circles)
        rew_epis += reward
        x, y = env_boundary(x,y)

        xhist.append(x)
        yhist.append(y)
        # print(len(pathx))

        
        obs_new, t_sum = mod(theta, t_sum)
        # print(t_sum)
        # print('next')
        # tot_steps += 1
        if rew_epis >= 10 : #steps > 40000: # or reward 
            done = True
            episodes += 1
            print('episode ' + str(episodes) + ' done')
            print(steps)
            tot_rew.append(rew_epis)
            tot_steps.append(steps)
            rew_epis = 0
            steps = 0
        else:
            done = False
        # print(obs_new)
        return obs_new, reward, done, {}  


    def new_circle(flag, obj, c):
        if obj == False:
            #creating a rectangular box
            if flag == 4:
                sq = box(-2,-2,2,2)

        elif obj == True: # if there is an object in the environment
            # creating a circular object
            if flag == 1:
                r = 0.50
                p = Point(c)
                sq = p.buffer(r)

        # seperating cords    
        k = list(sq.exterior.coords)
        xi = [m[0] for m in k ]
        yi = [m[1] for m in k ]

        #plotting both the boundaries
        plt.plot(xi, yi)
        # plt.pause(0.001)
        return k


    def plot_episode(x0, y0, xf, yf, xtraj, ytraj):
        square = new_circle(flag=4, obj=False, c=(0,0))
        for kk in bowl_cen:
            bowl = new_circle(flag=1, obj=True, c =kk)

        plt.plot(x0, y0, 'bo')
        plt.plot(xf, yf, 'ro')
        plt.plot(xtraj, ytraj)
        fol = 'training_9/'
        fig.savefig(fol + 'test ' + str(episodes))
        # plt.show()

        return None


    def reset(self):
        theta = random.choice(all_action_thetas)
        init = theta 
        steps = 0
        pathx = xhist
        pathy = yhist
        final = [x, y]
        fig = plt.figure()
        plot_episode(initial[0], initial[1], final[0], final[1], pathx, pathy)
        plt.close(fig)
        xhist = []
        yhist = []
        
        if random_start:
            x = np.random.uniform(-size,size)
            y = np.random.uniform(-size,size)
            check_in = cir_check(circles, x, y)
            while check_in:
                x = np.random.uniform(-size,size)
                y = np.random.uniform(-size,size)
                check_in = cir_check(circles, x, y)    

        else:
            x = 0
            y = 0
        # initial = [x, y]
        obs, t_sum = mod(theta, [0]*100)
        return obs #, [x, y]
    
    # def var(self):
        # return x, y

    def render(self):
        return pathx, pathy, initial, final, [tot_steps, tot_rew, episodes]
'''
