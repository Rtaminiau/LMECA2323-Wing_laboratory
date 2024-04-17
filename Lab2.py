import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


def extraction(file):
    txt = np.loadtxt(file)
    zero = txt[0, :]
    return (np.array(txt[1:, 0]), np.array(txt[1:, 1]),
            np.array(txt[1:, 2]), np.array(txt[1:, 3]))

def plot_force(force, aoa, name):
    plt.figure()
    plt.plot(aoa, force)
    plt.xlabel("angle d'attaque")
    plt.ylabel("$"+name+"$")
    plt.title("coefficient de lift " if name == "C_l" else "coefficient de drag")


def mean_pressure(p):
    return np.mean(p)

def Create_model_force(X,Y):
    return np.poly1d(np.polyfit(X,Y,1))

def plot_Coeff_aoa(force,name):
    aoa = angle
    C = force /(0.5 * rho * u_infinity**2 * surface)
    if name == 'C_d':
        C = C - Cd_arms
    plt.figure()
    plt.plot(aoa,C)
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel(name)
    plt.title("coefficient de lift " if name == "C_l" else "coefficient de drag")

def compute_max_lift_drag_ratio(lift,drag):
    max = 0
    max_index = 0
    for i in range(len(lift)):
        if lift[i]/drag[i] > max:
            max = lift[i]/drag[i]
            max_index = i
    angle = np.linspace(-6, 20, len(lift))[max_index]
    return angle,max

def linear_function(x,m,p):
    return m*(x - p)

def compute_sensitivity():
    """

    :return: kb lift , kb drag
    """
    LIFT_V = np.array([-0.2, -0.17, -0.11, -0.05, 0.01, 0.07, 0.13, 0.19, 0.25, 0.31, 0.57])
    LIFT_F = np.array([0, 0.0682, 0.181558, 0.294956, 0.4084, 0.5218, 0.6352, 0.7485, 0.861946, 0.9753, 1.4289]) * g

    DRAG_V = np.array([0.38, 0.7, 1.23, 1.76, 2.3, 2.82, 3.89, 4.9])
    DRAG_F = np.array([0, 0.06816, 0.181558, 0.294956, 0.408354, 0.521752, 0.748548, 0.975344]) * g

    model_Lift = Create_model_force(LIFT_V, LIFT_F)
    model_drag = Create_model_force(DRAG_V, DRAG_F)

    return model_Lift.coefficients[0],model_drag.coefficients[0]

def fit_func_polar(Cl,C_d0,k):
    return C_d0 + k * Cl**2

if __name__ == '__main__':


    # Data from laboratory
    angle, drag_voltage, lift_voltage, p_dyn = extraction('Data.txt')

    # Data from pdf
    type = "NACA0012"
    b = 0.3  # wingspan
    c_bar = 0.05  # mean chord
    surface = b * c_bar  # the surface of the wing “as seen from above”
    Cd_arms = 0.06433

    rho = 1.204  # [kg/m3]
    g = 9.81
    nu = 1.5e-5
    u_infinity = np.sqrt(2 * mean_pressure(p_dyn)/rho)
    print("U inf",u_infinity)
    Re = u_infinity * c_bar/nu
    print("Re : ",Re)

    # Drag et Lift forces

    K_b_lift,K_b_drag = compute_sensitivity()
    zero_lift_tension = -0.209
    zero_drag_tension = 0.429
    lift_force = linear_function(lift_voltage,K_b_lift, zero_lift_tension)
    drag_force = linear_function(drag_voltage,K_b_drag,zero_drag_tension)

    #Pour plot les calibrations et les données du labo
    # LIFT_V = np.array([-0.2, -0.17, -0.11, -0.05, 0.01, 0.07, 0.13, 0.19, 0.25, 0.31, 0.57])
    # LIFT_F = np.array([0, 0.0682, 0.181558, 0.294956, 0.4084, 0.5218, 0.6352, 0.7485, 0.861946, 0.9753, 1.4289]) * g
    #
    # DRAG_V = np.array([0.38, 0.7, 1.23, 1.76, 2.3, 2.82, 3.89, 4.9])
    # DRAG_F = np.array([0, 0.06816, 0.181558, 0.294956, 0.408354, 0.521752, 0.748548, 0.975344]) * g
    #
    # plt.figure()
    # plt.plot(DRAG_V,DRAG_F,label="Calibration")
    # plt.plot(drag_voltage,drag_force,label="Labo valeur")
    # plt.legend()

    #Adimensionnalisation de Cd et Cl
    coeff = 0.5 * rho * u_infinity ** 2 * surface
    Cd = drag_force/coeff - Cd_arms
    Cl = lift_force/coeff

    plt.figure()
    plt.plot(Cd,Cl)
    plt.xlabel("C_d")
    plt.ylabel("C_l")
    plt.title("polar")

    plt.figure()
    plt.plot(angle,Cd)
    plt.title("Cd/angle graphe")

    plt.figure()
    plt.plot(angle,Cl)
    plt.title("Cl/angle graphe")

    print("Angle and max lift to drag ratio : ",compute_max_lift_drag_ratio(Cl,Cd))

    plt.figure()
    plt.title("Polar with fit")
    C_l_data = Cl[1:6]
    C_d_data = Cd[1:6]
    popt,pcov = scipy.optimize.curve_fit(fit_func_polar,C_l_data,C_d_data)
    C_l_fit = np.linspace(-0.2,0.3,100)
    plt.plot(fit_func_polar(C_l_fit,*popt),C_l_fit,color="orange")
    plt.plot(C_d_data,C_l_data,color="blue")
    print(popt)
    print(f"C_D0 = {popt[0]} , k = {popt[1]} ")

    print("e : ",1/((b/c_bar) * np.pi * popt[1]))

    plt.show()
