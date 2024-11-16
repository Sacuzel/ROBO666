########################################
# Name: tutorial_computed_torque_control.agxPy
# Description:
# Shows how to implement computed torque control from the Panda that tracks a given trajectory.
# The controller is subject to external disturbance along the trajectory, from which it
# recovers nicely.
########################################


from typing import List, Callable, Tuple
import numpy as np

import agx
import agxOSG
import agxSDK
import agxModel
import agxCollide
import agxPython
import agxRender

from agxPythonModules.utils.environment import simulation, init_app, application, root
from agxPythonModules.robots.panda import Panda


# Build robot scene
def build_scene():
    # Set a good camera angle
    eye = agx.Vec3(1.74, -1.83, 1.74)
    center = agx.Vec3(-1.72E-01, 1.46E-01, 4.15E-01)
    up = agx.Vec3(-0.34, 0.26, 0.89)
    application().setCameraHome(eye, center, up)

    sim = simulation()

    # Information text via SceneDecorator
    sd = application().getSceneDecorator()
    sd.setBackgroundColor(agxRender.Color.DimGray())
    sd.setText(0, ("Computed torque control subject to disturbance"))

    # Create the robot, using the Franka Emika Panda loaded from urdf.
    panda = Panda(simulation())
    chain = agxModel.SerialKinematicChain(simulation(), panda.base, panda.ee)
    # Visualize the Axes of the end-effector
    #
    # Normally we would want to use:
    #   agxOSG.createAxes(panda.ee, agx.AffineMatrix4x4(), root(), 0.2)
    # but the panda_link8 has no geometry so the axes will not work as we want them too.
    # panda_link7 and panda_link8 are locked together and panda_link7 has all the geometry.
    #
    # This is due to what we get from the urdf-file.
    #
    agxOSG.createAxes(
        panda.assembly.getRigidBody("panda_link7"),
        panda.assembly.getRigidBody("panda_link8").getFrame().getMatrix().inverse() *
        panda.assembly.getRigidBody("panda_link7").getFrame().getMatrix(),
        root(), 0.2)

    # Set the panda in its ready pose
    _, transforms = chain.computeForwardKinematicsAll(panda.q_ready)
    for transform, link in zip(transforms, panda.links):
        link.getFrame().setLocalMatrix(transform)

    # Get the transform corresonding to the end-effector
    X_ready = panda.ee.getFrame().getLocalMatrix()

    # Choose a desired reachable position in task-space
    x_ready = X_ready.getTranslate()
    x_d = x_ready + agx.Vec3(-0.9, -0.5, -0.2)
    # Want to make the z-axis point in the negative x-direction
    rot_d = agx.Quat(-np.pi / 2, agx.Vec3.Y_AXIS())
    # This is the desired tranform
    X_d = agx.AffineMatrix4x4(rot_d, x_d)
    # Visualize desired task-space transform
    agxOSG.createAxes(X_d, root(), 0.3)

    # Use inverse kinematics to compute corresponding joint angles.
    _, q_d = chain.computeInverseKinematics(X_d, panda.q_ready)
    q_end = list(q_d)
    # Create trajectory with duration.
    h: float = sim.getTimeStep()
    trajectory = Trajectory(panda.q_ready, q_end, h, duration=7)

    # The controller gains have not been tuned. We just choose something so that the controller
    # is not overly stiff and damped. The integral term is often chosen as zero.
    n_joints: int = chain.getNumJoints()
    kp = n_joints * [1 / h]
    kd = n_joints * [1 / (10 * h)]
    ki = n_joints * [0]
    # Create the controller and add it to simulation
    ctc = ComputedTorqueControl(panda, chain, trajectory, kp, kd, ki)
    sim.add(ctc)

    # Add a box during the motion to create a distrubance
    sim.add(BoxEvent(panda.ee, 0.4 * trajectory.duration))

    # The ground is just for visual purposes.
    ground = agxCollide.Geometry(agxCollide.Box(3, 3, 0.1))
    ground.setPosition(agx.Vec3(0, 0, -0.1))
    simulation().add(ground)
    ground_node = agxOSG.createVisual(ground, root())
    agxOSG.setTexture(ground, root(), "textures/grid.png")
    agxOSG.setDiffuseColor(ground_node, agxRender.Color.WhiteSmoke())


class ComputedTorqueControl(agxSDK.StepEventListener):
    """
    Implements the computed torque controller or inverse dynamics controller
    """
    def __init__(self,
                 panda: Panda,
                 chain: agxModel.SerialKinematicChain,
                 trajectory: Callable,
                 kp: List,
                 kd: List,
                 ki: List):

        super().__init__()

        self._chain = chain
        self._panda = panda
        self._panda.enable_motors(True)

        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.ki = np.array(ki)
        self.e_int = 0

        self.trajectory = trajectory

    def pre(self, time):
        h = self.getSimulation().getTimeStep()

        # Get the desired values from the trajectory
        q_d, qd_d, qdd_d = self.trajectory(time)

        # Current state
        q = self._panda.get_joint_positions()
        qd = self._panda.get_joint_velocities()

        # Error terms
        e = q_d - q
        e_dot = qd_d - qd
        self.e_int += e * h

        # The desired acceleration contains a feed-forward term from the trajectory and
        # feedback from the sensed joint angles and velocities.
        qdd_ff_fb = qdd_d + self.kp * e + self.kd * e_dot + self.ki * self.e_int

        # Calculate and apply torques
        status, torque = self._chain.computeInverseDynamics(list(qdd_ff_fb))
        assert status == agxModel.KinematicChain.SUCCESS, \
            f"Inverse dynamics computation failed: {status}"

        for tau, joint in zip(torque, self._panda.joints):
            joint.motor.setForceRange(tau, tau)

    def post(self, time):
        '''
        Visualize the motion of the end-effector
        '''
        sphere = agxCollide.Geometry(agxCollide.Sphere(0.005))
        sphere.setPosition(self._panda.ee.getPosition())
        sphere.setEnableCollisions(False)
        simulation().add(sphere)
        sphere_node = agxOSG.createVisual(sphere, root())
        agxOSG.setDiffuseColor(sphere_node, agxRender.Color.Red())
        agxOSG.setAlpha(sphere_node, 0.4)


class Trajectory():
    '''
    Trajectory as a straight-line path in joint space with a time scaling s(t).
    We use a cosine time scaling, which in practise can be substituted by any function
    s:[0, T] -> [0, 1], e.g. a quintic polynomial or trapezoidal.
    '''
    def __init__(self,
                 start: List[float],
                 final: List[float],
                 step: float,
                 duration=5.0):

        self.duration = duration

        q_start = np.array(start)
        q_final = np.array(final)

        # The trajectory callables
        self.q_d, self.qd_d, self.qdd_d = self.interp(q_start, q_final, step, self.time_scale)

    def __call__(self, time):
        return self.q_d(time), self.qd_d(time), self.qdd_d(time)

    def interp(self,
               q_start: np.ndarray,
               q_end: np.ndarray,
               h: float,
               s_func: Callable) -> Tuple[Callable, Callable, Callable]:
        '''
        Defines a straight-line path in joint space.
        '''

        def q_d(t: float) -> np.ndarray:
            return q_start + (q_end - q_start) * s_func(t)

        # The derivatives just use central differences
        def qd_d(t: float) -> np.ndarray:
            return (q_d(t + h) - q_d(t - h)) / (2 * h)

        def qdd_d(t: float) -> np.ndarray:
            return (q_d(t + h) - 2 * q_d(t) + q_d(t - h)) / (h ** 2)

        return q_d, qd_d, qdd_d

    def time_scale(self, time: float) -> float:
        '''
        Time scale using cosine.
        '''
        # This linear interpolation of time also clips it at [0, 1].
        t = np.interp(time, [0, self.duration], [0, 1])

        return (1.0 - np.cos(t * np.pi)) * 0.5


class BoxEvent(agxSDK.StepEventListener):
    '''
    Add a box at certain time that collides with a body, symbolizing an external disturbance.
    '''

    def __init__(self, tracked_body, event_time):
        super().__init__()
        self.event_time = event_time
        self.body = tracked_body

    def post(self, time):
        if time > self.event_time:
            g_box = agxCollide.Geometry(agxCollide.Box(0.06, 0.06, 0.06))
            box = agx.RigidBody(g_box)
            # Get the position of the body we want to collide with
            x_body = self.body.getPosition()
            box.setPosition(agx.Vec3(x_body.x(), x_body.y(), x_body.z() + 0.6))
            simulation().add(box)
            box_node = agxOSG.createVisual(box, root())
            agxOSG.setDiffuseColor(box_node, agxRender.Color.Black())

            # We are done
            simulation().remove(self)


# This function is called if agxViewer is used
def buildScene():
    filename = application().getArguments().getArgumentName(1)
    application().addScene(filename, "build_scene", ord('1'))
    build_scene()


# Entry point when this script is started with python executable
init = init_app(
    name=__name__,
    scenes=[(build_scene, "1")],
    autoStepping=False,  # Default: False
    onInitialized=lambda app: print("App successfully initialized."),
    onShutdown=lambda app: print("App successfully shut down."),
)
