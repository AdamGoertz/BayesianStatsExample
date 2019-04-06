import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

TESTS = 100
GRID_SIZE = 50
    
def update_bayes(prob_vector, guess, direction):
    A_prob = prob_vector[guess] # Prior probability of the ball being at the given x or y coordinate
    B_prob = 0.5 - (1 / GRID_SIZE)

    if direction in ('R', 'D'):
        # Probability that 'guess' is to the right of, or below, an unknown spot        
        B_given_A =  guess / GRID_SIZE

    elif direction in ('U', 'L'):
        B_given_A = (GRID_SIZE - guess - 1) / GRID_SIZE

    else:
        return A_prob

    res = ((B_given_A) * A_prob) / B_prob
    return res

# Set the location we are going to try to guess
ball_location = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))


# Create the grid
X = np.arange(0, GRID_SIZE, 1)
Y = np.arange(0, GRID_SIZE, 1)
XX, YY = np.meshgrid(X, Y)

# Set Prior Probabilities (Each grid location has equal probability of 1 / GRID_SIZE**2)
X_probs = np.full(GRID_SIZE, 1 / GRID_SIZE)
Y_probs = np.full(GRID_SIZE, 1 / GRID_SIZE)

fig = plt.figure()
ax = Axes3D(fig)
plt.show(block=False)


for _ in range(TESTS):
    # Draw the surface
    Z = np.matmul(Y_probs.reshape((GRID_SIZE, 1)), X_probs.reshape((1, GRID_SIZE))) 
    ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=cm.viridis)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.pause(0.001)
    fig.canvas.flush_events()
    plt.cla()

    guess_throw = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))

    if guess_throw[0] < ball_location[0]:   # Ball is Right
        X_direction = 'R'
    elif guess_throw[0] > ball_location[0]: # Ball is Left
        X_direction = 'L'
    else:                                   # Ball is directly Up/Down
        X_direction = None

    if guess_throw[1] < ball_location[1]:   # Ball is Down
        Y_direction = 'D'
    elif guess_throw[1] > ball_location[1]: # Ball is Up
        Y_direction = 'U'
    else:                                   # Ball is directly Right/Left
        Y_direction = None

    for i in range(GRID_SIZE):
        # Update priors
        X_probs[i] = update_bayes(X_probs, i, X_direction)
        Y_probs[i] = update_bayes(Y_probs, i, Y_direction)


estimation = (np.argmax(X_probs), np.argmax(Y_probs))
error = ((ball_location[0] - estimation[0])**2 + (ball_location[1] - estimation[1])**2)**0.5
print(f"Actual Location: {ball_location}")
print(f"Result: {estimation}")
print(f"Error: {error:.3f}")
