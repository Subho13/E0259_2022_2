import numpy as np

import datetime
from scipy.optimize import minimize

INFTY = 1_000_000

def MarsEquantModel(C, r, e1, e2, z, s, times, oppositions):
  center = (np.cos(np.radians(C)), np.sin(np.radians(C)))
  equant = (e1 * np.cos(np.radians(e2)), e1 * np.sin(np.radians(e2)))

  equant_angles = ((times * s) + z) % 360
  slopes = np.tan(np.radians(equant_angles))

  a = 1 + (slopes ** 2)
  b = (-2 * center[0]) + (2 * slopes * (equant[1] - center[1] - (equant[0] * slopes)))
  c = (equant[1] - center[1] - equant[0] * slopes) ** 2 + (center[0] ** 2) - (r ** 2)
  discriminants = np.sqrt((b ** 2) - (4 * a * c))
  x1 = (-b - discriminants) / (2 * a)
  x2 = (-b + discriminants) / (2 * a)
  y1 = equant[1] + (x1 - equant[0]) * slopes
  y2 = equant[1] + (x2 - equant[0]) * slopes

  # q23 = (equant_angles > 90) & (equant_angles < 270)
  # q14 = ~q23

  intersections_x = []
  intersections_y = []
  for i in range(x1.shape[0]):
    # if np.isnan(x1[i]):
    #   print('nan,', end='')
    if (0 <= equant_angles[i] <= 90) or (270 <= equant_angles[i] <= 360):
      if x1[i] < x2[i]:
        intersections_x.append(x2[i])
        intersections_y.append(y2[i])
      else:
        intersections_x.append(x1[i])
        intersections_y.append(y1[i])
    else:
      if x1[i] < x2[i]:
        intersections_x.append(x1[i])
        intersections_y.append(y1[i])
      else:
        intersections_x.append(x2[i])
        intersections_y.append(y2[i])

  intersections_x = np.array(intersections_x)
  intersections_y = np.array(intersections_y)
  a1 = np.degrees(np.arctan2(intersections_y, intersections_x))
  a2 = np.array([i if i <= 180 else i - 360 for i in oppositions])

  errors = a1 - a2
  errors[np.isnan(errors)] = INFTY
  max_error = np.max(np.abs(errors))

  return errors, max_error

def maxError(x, r, s, times, oppositions):
  c, e1, e2, z = x
  _, max_error = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
  return np.abs(max_error)

def bestOrbitInnerParams(r, s, times, oppositions):
  # Random initial guesses
  c = 150
  e1 = 1
  e2 = 150

  for i in range(8):
    # z_guesses = range(360)
    z_guesses = np.linspace(0, 359.5, 720)
    errors = [maxError((c, e1, e2, z_guess), r, s, times, oppositions) for z_guess in z_guesses]
    z = z_guesses[np.argmin(errors)]

    e1_guesses = np.linspace(0, 3, 150)
    errors = [maxError((c, e1_guess, e2, z), r, s, times, oppositions) for e1_guess in e1_guesses]
    e1 = e1_guesses[np.argmin(errors)]

    # e2_guesses = range(360)
    e2_guesses = np.linspace(0, 359.5, 720)
    errors = [maxError((c, e1, e2_guess, z), r, s, times, oppositions) for e2_guess in e2_guesses]
    e2 = e2_guesses[np.argmin(errors)]

    # c_guesses = range(360)
    c_guesses = np.linspace(0, 359.5, 720)
    errors = [maxError((c_guess, e1, e2, z), r, s, times, oppositions) for c_guess in c_guesses]
    c = c_guesses[np.argmin(errors)]
  
  params = (r, s, times, oppositions)
  bounds = [(0, 360), (0, r), (0, 360), (0, 360)]
  temp = minimize(maxError, np.array([c, e1, e2, z]), args=params, method="L-BFGS-B", bounds=bounds)

  c, e1, e2, z = temp.x
  errors, _ = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
  return c, e1, e2, z, errors, temp.fun

def bestS(r, times, oppositions):
  global_max_error = INFTY
  for s in np.linspace(0.5, 0.6, 250):
    _, _, _, _, errors, max_error = bestOrbitInnerParams(r, s, times, oppositions)
    if max_error < global_max_error:
      global_errors = errors
      global_max_error = max_error
      best_s = s
  
  return best_s, global_errors, global_max_error

def bestR(s, times, oppositions):
  global_max_error = INFTY
  for r in np.linspace(5, 8, 120):
    _, _, _, _, errors, max_error = bestOrbitInnerParams(r, s, times, oppositions)
    if max_error < global_max_error:
      global_errors = errors
      global_max_error = max_error
      best_r = r
  
  return best_r, global_errors, global_max_error

def bestMarsOrbitParams(times, oppositions):
    r = 8
    s = 360/687
    err = INFTY
    # To limit number of iterations
    count = 0
    while err > (1) and count < 4:
        s, _, err = bestS(r, times, oppositions)
        r, _, err = bestR(s, times, oppositions)
        count += 1
        print(err, r, s)
    
    c, e1, e2, z, errors, max_error = bestOrbitInnerParams(r, s, times, oppositions)
    return r, s, c, e1, e2, z, errors, max_error

import matplotlib.pyplot as plt
def plot(C, r, e1, e2, z, s, times, oppositions):
    fig = plt.gcf()
    ax = fig.gca()

    center = (np.cos(np.radians(C)), np.sin(np.radians(C)))
    equant = (e1 * np.cos(np.radians(e2)), e1 * np.sin(np.radians(e2)))

    equant_angles = ((times * s) + z) % 360
    slopes = np.tan(np.radians(equant_angles))
    slopes2 = np.tan(np.radians(oppositions))

    a = 1 + (slopes ** 2)
    b = (-2 * center[0]) + (2 * slopes * (equant[1] - center[1] - (equant[0] * slopes)))
    c = (equant[1] - center[1] - equant[0] * slopes) ** 2 + (center[0] ** 2) - (r ** 2)
    a2 = 1 + (slopes2 ** 2)
    b2 = -2 * ((slopes2 * center[1]) + center[0])
    c2 = (center[1] ** 2) + (center[0] ** 2) - (r ** 2)
    discriminants = np.sqrt((b ** 2) - (4 * a * c))
    discriminants2 = np.sqrt((b2 ** 2) - (4 * a2 * c2))
    x1 = (-b - discriminants) / (2 * a)
    x2 = (-b + discriminants) / (2 * a)
    x12 = (-b2 - discriminants2) / (2 * a2)
    x22 = (-b2 + discriminants2) / (2 * a2)
    y1 = equant[1] + (x1 - equant[0]) * slopes
    y2 = equant[1] + (x2 - equant[0]) * slopes
    y12 = x12 * slopes2
    y22 = x22 * slopes2

    # q23 = (equant_angles > 90) & (equant_angles < 270)
    # q14 = ~q23

    for i in range(x12.shape[0]):
        if (0 <= oppositions[i] <= 90) or (270 <= oppositions[i] <= 360):
            if x12[i] < x22[i]:
                plt.plot([0, x22[i]], [0, y22[i]], 'k')
                plt.plot(x22[i], y22[i], marker='o')
            else:
                plt.plot([0, x12[i]], [0, y12[i]], 'k')
                plt.plot(x12[i], y12[i], marker='o')
        else:
            if x12[i] < x22[i]:
                plt.plot([0, x12[i]], [0, y12[i]], 'k')
                plt.plot(x12[i], y12[i], marker='o')
            else:
                plt.plot([0, x22[i]], [0, y22[i]], 'k')
                plt.plot(x22[i], y22[i], marker='o')

    for i in range(x1.shape[0]):
        # if np.isnan(x1[i]):
        #   print('nan,', end='')
        if (0 <= equant_angles[i] <= 90) or (270 <= equant_angles[i] <= 360):
            if x1[i] < x2[i]:
                plt.plot([equant[0], x2[i]], [equant[1], y2[i]], 'b--')
                # plt.plot(*equant, x2[i], y2[i])
                # plt.axline(equant, (x2[i], y2[i]))
            else:
                plt.plot([equant[0], x1[i]], [equant[1], y1[i]], 'b--')
                # plt.plot(*equant, x1[i], y1[i])
                # plt.axline(equant, (x1[i], y1[i]))
        else:
            if x1[i] < x2[i]:
                plt.plot([equant[0], x1[i]], [equant[1], y1[i]], 'b--')
                # plt.plot(*equant, x1[i], y1[i])
                # plt.axline(equant, (x1[i], y1[i]))
            else:
                plt.plot([equant[0], x2[i]], [equant[1], y2[i]], 'b--')
                # plt.plot(*equant, x2[i], y2[i])
                # plt.axline(equant, (x2[i], y2[i]))

    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    ax.add_patch(plt.Circle(center, r, color='black', fill=False))
    plt.plot(0, 0, marker='o', markerfacecolor="yellow", markeredgecolor="orange")
    plt.plot(*center, marker='o', markerfacecolor="black", markeredgecolor="grey")
    plt.plot(*equant, marker='o', markerfacecolor="blue", markeredgecolor="black")
    
    plt.show()

def get_times(data):
    time0 = data[0]
    diffs = []
    t0 = datetime.datetime(time0[0], time0[1], time0[2], time0[3], time0[4])
    
    for i in range(12):
        t1 = datetime.datetime(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4])
        timed = t1 - t0
        diffs.append(timed.days + (timed.seconds / (60*60*24)))

    return np.array(diffs)

def get_oppositions(data):
    return data[:,5] * 30 + data[:,6] + (data[:,7] / 60) + (data[:,8] / 3600)

if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )

    # Extract times from the data in terms of number of days.
    # "times" is a numpy array of length 12. The first time is the reference
    # time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    # Extract angles from the data in degrees. "oppositions" is
    # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"

    # Call the top level function for optimization
    # The angles are all in degrees
    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(abs(errors)) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {0:2.4f}".format(maxError))
