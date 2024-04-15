import numpy as np
import matplotlib.pyplot as plt


def extraction(file):
    txt = np.loadtxt(file)
    zero = txt[0, :]
    return (np.array(txt[1:, 0]), np.array(txt[1:, 1]),
            np.array(txt[1:, 2]), np.array(txt[1:, 3]))


def lift_or_drag_coefficient(l, d, r, u, s):
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


def regression(l, d):
    return np.poly1d(np.polyfit(d, l, 2))


def mean_pressure(p):
    return np.mean(p)

def Create_model_force(X,Y):
    return np.poly1d(np.polyfit(X,Y,1))


def plot_polar(l, d):
    plt.figure()
    plt.plot(d, l, label="donées expérimentale")
    model = regression(l, d)
    ax = np.linspace(0.2,0.7, 100)
    plt.plot(ax, model(ax), label="interpolation")
    plt.xlabel("$C_d$")
    plt.ylabel("$C_l$")
    plt.legend()
    plt.title("Polaire de l'aile")

def plot_Coeff_aoa(force,name):
    aoa = np.linspace(-6,20,len(force))
    C = force /(0.5 * rho * u_infinity**2 * surface)
    plt.figure()
    plt.plot(aoa,C)
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel(name)
    plt.title("coefficient de lift " if name == "C_l" else "coefficient de drag")


if __name__ == '__main__':


    # Data from laboratory
    angle, drag_voltage, lift_voltage, p_dyn = extraction('Data.txt')

    # Data from pdf
    type = "NACA0012"
    b = 0.3  # wingspan
    c_bar = 0.05  # mean chord
    surface = b * c_bar  # the surface of the wing “as seen from above”

    rho = 1.204  # [kg/m3]
    g = 9.81
    mu = 1.8e-5
    u_infinity = np.sqrt(2 * rho * mean_pressure(p_dyn))
    Re = rho * u_infinity * c_bar/mu

    # Calibration data

    LIFT_V = np.array([-0.2,-0.16,-0.1,-0.04,0.02,0.14]) + (-0.209 + 0.2) # mets l'offset sur les tension de calibrage
    LIFT_F = np.array([0,0.06816,0.181558,0.294956,0.408354,0.521752]) * g

    DRAG_V = np.array([0.71,1.02,1.54,2.07,2.59,3.14,4.2,5.24]) + (0.429 - 0.71)
    DRAG_F = np.array([0,0.6816,0.181558,0.294956,0.40854,0.521752,0.748548,0.975344])* g

    model_Lift = Create_model_force(LIFT_V,LIFT_F)
    model_drag = Create_model_force(DRAG_V,DRAG_F)

    lift_force = model_Lift(lift_voltage)
    drag_force = model_drag(drag_voltage)

    plot_Coeff_aoa(lift_force,'C_l')
    plot_Coeff_aoa(drag_force,'C_d')


    # mean dynamic pressure
    # p_mean = mean_pressure(p_dyn)
    #
    # cl, cd = lift_or_drag_coefficient(lift, drag, rho, u_infinity, surface)
    # plot_force(lift, angle, "C_l")
    # plot_force(drag, angle, "C_d")
    #
    # plot_polar(lift, drag)

    plt.show()
