########################################
# Name: tutorial_differential_kinematics.agxPy
# Description:
# Shows how to implement position based servoing to track a moving object.
# The controller receives the joint angles of the robot along with the 6D pose of the object.
# Based on the error between the current end-effector pose and the object pose, it computes joint
# velocities using the pseudo inverse of the manipulator Jacobian.
########################################


import numpy as np

import agx
import agxSDK
import agxOSG
import agxPython
import agxCollide
import agxRender
import agxModel

from agxPythonModules.utils.environment import simulation, init_app, application, root
from agxPythonModules.robots.panda import Panda


# Build robot scene
def build_scene():
    sim = simulation()
    sim.setTimeStep(0.01)

    # Information text via SceneDecorator
    sd = application().getSceneDecorator()
    sd.setText(0, ("Position based servoing to track a moving taget."))

    # Set camera angle
    eye = agx.Vec3(2.1600266566935877E+00, -9.4980901226186532E-01, 8.2567670074709199E-01)
    center = agx.Vec3(1.8790158629417419E-01, 2.0381560921669006E-01, 4.7483745217323303E-01)
    up = agx.Vec3(-0.1501, 0.0438, 0.9877)
    application().setCameraHome(eye, center, up)

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

    # Use a different ready pose than panda.q_ready to demonstrate an alternative initial state.
    q_ready = [0, -0.25 * np.pi, 0, -0.75 * np.pi, 0, 0.5 * np.pi, 0.25 * np.pi]
    # Set the panda in the ready pose
    _, transforms = chain.computeForwardKinematicsAll(q_ready)
    for transform, link in zip(transforms, panda.links):
        link.getFrame().setLocalMatrix(transform)

    # Set the position target as a geometry that we move around
    target = agxCollide.Geometry(agxCollide.Box(0.05, 0.05, 0.01))
    target.setEnableCollisions(False)
    node = agxOSG.createVisual(target, root())
    agxOSG.setDiffuseColor(node, agxRender.Color.Black())
    agxOSG.setAlpha(node, 0.4)
    sim.add(target)
    # visualize desired task-space transform
    agxOSG.createAxes(target, agx.AffineMatrix4x4(), root(), 0.1)

    # Add controller that does position based servoing. For the gains we just use ones for both
    # the translational and rotational part.
    gains = np.ones(6)
    pbs = PositionBasedServoing(panda, chain, gains, target)
    sim.add(pbs)


class PositionBasedServoing(agxSDK.StepEventListener):
    '''
    Implements position based servoing, which uses differential kinematics to get the
    end-effector to track a moving target.
    '''
    def __init__(self,
                 panda: Panda,
                 chain: agxModel.SerialKinematicChain,
                 gain: np.ndarray,
                 target: agxCollide.Geometry,
                 err_tol: float = 0.001,
                 duration: float = 12.0):

        super().__init__()

        self._chain = chain
        self._panda = panda
        self.k = gain

        # We will use velocity control and set desired values to the 1D motors of the hinge joints.
        self._panda.enable_motors(True)

        # Get the initial pose and move the target in relation.
        _, self.X_0 = chain.computeForwardKinematics(list(panda.get_joint_positions()))
        self.target = target
        _ = self.move_target(0)

        self.err_tol = err_tol
        self.duration = duration
        self.done = False

        # Stores the geometries we use to visualize the motion of the end-effector.
        self.visual_trace = []

    def pre(self, time: float):
        # From feedback: get current joints angles
        q = self._panda.get_joint_positions()

        # Compute the estimated pose of the end-effector given the current joint angles.
        # It is an estimate because it assumes that the robot and its joints are completely rigid.
        # This is almost true here, making the estimate accurate.
        _, X = self._chain.computeForwardKinematics(list(q))

        # We let the target move around for a while, then make it static.
        if time < self.duration:
            X_d = self.move_target(time)
        else:
            X_d = self.target.getFrame().getMatrix()

        # Calculate the error in position and orientation using angle axis.
        error = self.angle_axis_error(X, X_d)
        self.done = np.sum(np.abs(error)) < self.err_tol

        # Calculate desired velocity of end-effector
        v_d = np.diag(self.k) @ error

        # Compute the manipulator Jacobian in the base frame
        _, J_vec = self._chain.computeManipulatorJacobian(list(q))

        # The J here is a agx.RealVector with 6 x n_joints elements. Reshape it to correct
        # shape in a numpy array
        J_list = list(J_vec)
        J = np.array(J_list).reshape(6, self._chain.getNumJoints())

        # For the redundant Panda with 7 dofs, the Jacobian will be of shape 6 X 7.
        # Therefore, we use the pseudoinverse to solve for joint velocities qd_d in v_d = J qd_d.
        J_pinv = np.linalg.pinv(J)
        qd_d = J_pinv @ v_d
        qd_d = qd_d.squeeze()

        # Apply the velocites
        if not self.done:
            self._panda.set_joint_velocities(qd_d)
        else:
            application().getSceneDecorator().setText(
                1, f"Target reached within error tolerance {self.err_tol}")
            self._panda.set_joint_velocities(np.zeros_like(qd_d))

    def move_target(self, time, radius=0.3, omega=0.3 * np.pi):
        '''
        Translate the target in a circle while also rotating it.
        '''
        x_0 = self.X_0.getTranslate()
        x = x_0.x()
        y = x_0.y() + radius * np.cos(omega * time)
        z = x_0.z() + radius * np.sin(omega * time)
        x_next = agx.Vec3(x, y, z)

        r_next = self.X_0.getRotate() * agx.Quat(0.3 * omega * time, agx.Vec3.X_AXIS())

        # Apply the next pose to the target geometry for visualization.
        X_next = agx.AffineMatrix4x4(r_next, x_next)
        self.target.getFrame().setMatrix(X_next)

        return X_next

    def post(self, time):
        '''
        Visualize the motion of the end-effector
        '''
        h = simulation().getTimeStep()
        if int(time / h) % 5 == 0:

            sphere = agxCollide.Geometry(agxCollide.Sphere(0.005))
            sphere.setPosition(self._panda.ee.getPosition())
            sphere.setEnableCollisions(False)
            simulation().add(sphere)
            sphere_node = agxOSG.createVisual(sphere, root())
            agxOSG.setDiffuseColor(sphere_node, agxRender.Color.Black())
            agxOSG.setAlpha(sphere_node, 0.3)
            self.visual_trace.append(sphere)

        if len(self.visual_trace) > 200:
            s = self.visual_trace.pop(0)
            simulation().remove(s)

    @staticmethod
    def angle_axis_error(X: agx.AffineMatrix4x4, X_d: agx.AffineMatrix4x4):
        '''
        Computes the 6-vector error between two poses. Translational is straight-forward and
        orientation uses quaternions to angle-axis error.
        '''
        # Position error
        x_d = np.array([*X_d.getTranslate()])
        x = np.array([*X.getTranslate()])
        x_err = x_d - x

        # orientation error using angle axis
        quat = agx.Quat(X)
        quat_d = agx.Quat(X_d)
        quat_err = quat.inverse() * quat_d

        angle_err = quat_err.getAngle()
        angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi  # get into [-pi, pi]
        axis_err = np.array([*quat_err.getUnitVector()])

        r_err = axis_err * angle_err

        # stack as 6-vector
        return np.hstack((x_err, r_err))


# This function is called if agxViewer is used
def buildScene():
    build_scene()


# Entry point when this script is started with python executable
init = init_app(
    name=__name__,
    scenes=[(build_scene, "1")],
    autoStepping=False,  # Default: False
    onInitialized=lambda app: print("App successfully initialized."),
    onShutdown=lambda app: print("App successfully shut down."),
)
