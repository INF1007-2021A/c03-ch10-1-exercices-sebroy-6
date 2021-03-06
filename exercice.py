#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cmath
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def linear_values_exo1(min: float = -1.3, max: float = 2.5, reps: int = 64) -> np.ndarray:

    return np.linspace(min, max, reps)


def coordinate_conversion_exo2(cartesian_coordinates: np.ndarray) -> np.ndarray:

    return np.array([cmath.polar(c) for c in cartesian_coordinates])


def find_closest_index_exo3(values: np.ndarray, number: float) -> int:

    return np.abs(values - number).argmin()


def afficher_graph(title: str = "Aucun titre entré"):
    plt.title(title)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()


def graph_exo4(intervale: tuple = (-1, 1), nb_points: int = 250):
    x = np.linspace(intervale[0], intervale[1], nb_points)
    y = x**2 * np.sin(1/(x**2)) + x

    plt.scatter(x, y)
    afficher_graph("Exercice #4")


def estimer_pi_exo5(nb_points: int = 5000):
    points_interieur, points_exterieur = list(), list()

    for i in range(nb_points):
        x, y = np.random.random(), np.random.random()

        if (x**2 + y**2)**(1/2) <= 1:
            points_interieur.append([x, y])
        else:
            points_exterieur.append([x, y])

    plt.scatter([points_interieur[i][0] for i in range(len(points_interieur))],
                [points_interieur[i][1] for i in range(len(points_interieur))])

    plt.scatter([points_exterieur[i][0] for i in range(len(points_exterieur))],
                [points_exterieur[i][1] for i in range(len(points_exterieur))])

    afficher_graph("Exercice #5")



def integrale(intervale: tuple = (-4, 4), nb_points :int = 500):
    x = np.linspace(intervale[0], intervale[1], nb_points)
    y = [integrate.quad(lambda x: np.e**(-x**2), 0, value)[0] for value in x]

    plt.scatter(x, y)
    afficher_graph("Exercice #6")

    integral_tuple = integrate.quad(lambda x: np.e**(-x**2), -np.inf, np.inf)

    return integral_tuple[0] - integral_tuple[1]



if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print("Exercice 1 - linear_values:\n", linear_values_exo1(), "\n")
    #print("Fonction coordinate_conversion:", coordinate_conversion_exo2(np.array([(0, 0), (10, 10), (2, -1)])))
    print("Exercice 3 - find_closest_index:", find_closest_index_exo3(np.array([1, 3, 8, 10]), 9))
    graph_exo4()
    estimer_pi_exo5()
    print("Exercice 6 - integrale:", integrale())

