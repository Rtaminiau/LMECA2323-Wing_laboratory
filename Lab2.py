import numpy as np
import matplotlib.pyplot as plt


def extraction(file):
    txt = np.loadtxt(file)
    zero = txt[0, :]
    return np.array(txt[1:, 0]), np.array(txt[1:, 1])-zero[1], np.array(txt[1:, 2])-zero[2], np.array(txt[1:, 3])-zero[3]


def lift_or_drag_coefficient(l,d, r, u,s):
    """
    Non dimensionalise the mesure from the lab for the lift and drag forces
    :param l: lift mesured in the laboratory
    :param d: drag mesured in the laboratory
    :param r: air density (rho)
    :param u: upstream velocity U_infinity (m/s)
    :param s: Surface of the wing
    :return: cl the lift coefficient or cd the drag coefficient of the wing
    """
    coef = (0.5*r*(u**2)*s)
    return l/coef, d/coef


def plot_force(force, aoa, name):
    plt.figure()
    plt.plot(aoa, force)
    plt.xlabel("angle d'attaque")
    plt.ylabel("$"+name+"$")
    plt.title("coefficient de lift " if name == "C_l" else "coefficient de drag")


def mean_pressure(p):
    return np.mean(p)

def plot_polar(l,d):
    plt.figure()
    plt.plot(d, l, label="donées expérimentale")
    plt.xlabel("$C_d$")
    plt.ylabel("$C_l$")
    plt.legend()
    plt.title("Polaire de l'aile")



if __name__ == '__main__':

    rho = 1.204  # [kg/m3]
    u_infinity = 15
    # Data from pdf
    type = "NACA0012"
    b = 0.3  # wingspan
    c_bar = 0.05  # mean chord
    surface = b*c_bar  # the surface of the wing “as seen from above”


    # Data from laboratory
    angle, drag, lift, p_dyn = extraction('Data.txt')

    # mean dynamic pressure
    p_mean = mean_pressure(p_dyn)

    cl, cd = lift_or_drag_coefficient(lift, drag, rho, u_infinity, surface)
    plot_force(lift, angle, "C_l")
    plot_force(drag, angle, "C_d")

    plot_polar(lift, drag)


    plt.show()
