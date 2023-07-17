import numpy as np
import matplotlib.pyplot as plt

def generateTrajectoryWithNoise(er: np.ndarray, ep: np.ndarray, t: float, noise: float = 0.5, A: float = 1, omega: float = 1, phi: float = 0):
    """
    return position at time t
    Args:
        er (np.ndarray): unit vector of motion direction
        ep (np.ndarray): displacement vector of the motion origin
        A (float, optional): attitude of motion distiance. Defaults to 1.
        t (float, optional): current time of the motion. Defaults to 0.
        phi (float, optional): phase of sinusoidal motion. Defaults to 0.
    """
    assert er.shape == (3,), "er must be a 3D vector"
    assert ep.shape == (3,), "ep must be a 3D vector"
    r = A*np.sin(omega*t+phi) + np.random.normal(0, noise, size=t.shape)
    return r.reshape(-1, 1)*er.reshape(1, -1) + ep.reshape(1, -1)


if __name__ == "__main__":

    # ===== Definition of the trajectory =====
    er = np.array([1, 1, 1])
    er = er/np.linalg.norm(er)
    ep = np.array([0, 1, 3])
<<<<<<< HEAD
    A = 2.5
=======
    A = 20
>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
    OMEGA = 1
    PERIOD = 2*np.pi/OMEGA
    PHI = 30
    T = 3*PERIOD
    dT = PERIOD / 20
    t = np.linspace(0, T, int(T/dT)+1)
    trajectory = generateTrajectoryWithNoise(er, ep, t,
                                             noise=0.2, A=A, omega=OMEGA, phi=PHI)
    print("dt: ", dT)
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "*")
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(t, trajectory[:, 0], "*-")
    ax.set_title("Trajectory")
    ax.set_ylabel("x")
    ax.grid()
    ax = fig.add_subplot(312)
    ax.plot(t, trajectory[:, 1], "*-")
    ax.set_ylabel("y")
    ax.grid()
    ax = fig.add_subplot(313)
    ax.plot(t, trajectory[:, 2], "*-")
    ax.set_ylabel("z")
    ax.set_xlabel("t")
    ax.grid()
    plt.show()

    # save to file
    np.savetxt("build/trajectory.txt", trajectory, fmt="%.5f", delimiter=" ")
<<<<<<< HEAD
=======
    np.savetxt("build/Release/trajectory.txt", trajectory, fmt="%.5f", delimiter=" ")
>>>>>>> 4728431d2e1897746d51fcfe6f99021943799068
    np.savetxt("trajectory.txt", trajectory, fmt="%.5f", delimiter=" ")

    print("Done!")
