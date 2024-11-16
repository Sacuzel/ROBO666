########################################
# Name: tutorial_pick_and_place.agxPy
# Description:
# This tutorial shows how a robot can be controlled to pick up some boxes
# and place them on top of each other.
#
# The tutorial uses two "controllers" for driving the robot motion:
# - A high level controller that handles tasks such as open/close gripper and
#   computing the trajectory that the robot should follow when moving.
# - A low level controller that is responsible for tracking target joint angles.
#   - Target angles are received from the high level controller.
#   - The controller has access to a "force/torque sensor" to estimate the load from what the tool picks up
#   - The controller has feedback (can read current joint angles and joint velocities)
#
########################################

# AGX Dynamics imports
import agx
import agxCollide
import agxSDK
import agxModel
import agxOSG
import agxRender
import agxUtil

from agxPythonModules.utils.environment import simulation, init_app, application, root
from agxPythonModules.robots.panda import Panda  # noqa

import math
import numpy as np

from collections import namedtuple


class LowLevelController(agxSDK.StepEventListener):
    """
    A controller that makes the robot follow joint angles that are provided externally.
    """

    def __init__(self, panda, chain):
        super().__init__()
        self._sd = application().getSceneDecorator()
        self._chain = chain
        self._robot = panda

        # Initial target angles, will be fed new values from the high level task controller
        self._target_angles = panda.q_ready

        # Configure the lock joint between the last link of the robot and the first part of the tool
        # so it can act as if we have a force/torque sensor
        #
        self._j = panda.assembly.getConstraint("panda_hand_joint")
        self._j.setEnableComputeForces(True)

        # The "sensor" will feel the effect from the tool bodies and we'll take this into account
        self._tool_mass = panda.assembly.getRigidBody('panda_hand').getMassProperties().getMass() + \
            panda.assembly.getRigidBody('panda_leftfinger').getMassProperties().getMass() + \
            panda.assembly.getRigidBody('panda_rightfinger').getMassProperties().getMass()

    def pre(self, t):
        # We create the chain with a relative transform at the end, hence we have one body less then the chain thinks
        num_links = self._chain.getNumLinks() - 1

        h = self.getSimulation().getTimeStep()
        g = self.getSimulation().getGravityField().calculateGravity(self._robot.ee.getPosition())

        # "Read" values from the force/torque sensor.
        lf1 = agx.Vec3()
        lf2 = agx.Vec3()
        self._j.getLastForce(0, lf1, lf2, True)
        estimated = max(lf1[2] / g.length() - self._tool_mass, 0)

        # Approximated external force due to what the gripper does/holds
        ext_force = np.zeros((6 * num_links, 1))
        for idx in range(3):
            ext_force[(num_links - 1) * 6 + idx] = -g[idx] * estimated

        # Compute desired joint accelerations from the current robot state
        current_q = self._robot.get_joint_positions()
        current_q_dot = self._robot.get_joint_velocities()

        desired_q_dot = (self._target_angles - current_q) / h
        desired_q_dotdot = (desired_q_dot - current_q_dot) / h

        # Perform inverse dynamics calculations to compute the joint
        # torques required to reach the desired joint accelerations
        status, torque = self._chain.computeInverseDynamics(list(desired_q_dotdot), list(ext_force))

        assert status == agxModel.KinematicChain.SUCCESS, "Inverse dynamics computation failed: %s" % status

        # Apply computed torques as force limits to joint motors
        for t, joint in zip(torque, self._robot.joints):
            joint.motor.setEnable(True)
            joint.motor.setForceRange(t, t)

        self._sd.setText(1, "Low level controller: estimated extra load: %4.2f" % estimated)


Operation = namedtuple('Operation', ['name', 'start_time', 'end_time', 'start_transform', 'end_transform'])


class TaskController(agxSDK.StepEventListener):
    """
    A high level controller that handles tasks such as
    open or close gripper or compute and feed trajectory data
    to the low level controller.
    """

    def __init__(self, panda, chain, lowlevel):
        super().__init__(agxSDK.StepEventListener.PRE_COLLIDE)
        self._sd = application().getSceneDecorator()
        self._chain = chain
        self._robot = panda
        self._llc = lowlevel

        # List of operations that should be performed.
        self.plan = []

    def preCollide(self, t):
        active = []

        for item in self.plan:
            if item.start_time <= t and t <= item.end_time:
                # Current time is in the range for when task should be active
                active.append(item.name)

                task_length = item.end_time - item.start_time
                local_time = t - item.start_time

                # These statements relate to if we want to open or close the gripper, or move the
                # end-effector.
                if item.name == "open":
                    joint = self._robot.assembly.getConstraint('panda_finger_joint1').asPrismatic()
                    joint.getMotor1D().setEnable(True)
                    joint.getMotor1D().setSpeed(0.05)
                    joint.getLock1D().setEnable(False)

                if item.name == "close":
                    joint = self._robot.assembly.getConstraint('panda_finger_joint1').asPrismatic()
                    joint.getMotor1D().setEnable(True)
                    joint.getMotor1D().setSpeed(-0.05)
                    joint.getLock1D().setEnable(False)

                if item.name == "move":
                    start_pos = item.start_transform.getTranslate()
                    start_q = item.start_transform.getRotate()
                    start_aa = start_q.getAngle() * start_q.getUnitVector()

                    target_pos = item.end_transform.getTranslate()
                    target_q = item.end_transform.getRotate()
                    target_aa = target_q.getAngle() * target_q.getUnitVector()

                    # Linear interpolation, kept simple on purpose. Smoother velocity profile
                    # can be achieved by changing how this step is done.
                    # Handle position and rotation separately:
                    pos = start_pos + (target_pos - start_pos) * (local_time / task_length)
                    aa = start_aa + (target_aa - start_aa) * (local_time / task_length)

                    angle = aa.length()
                    if math.isclose(angle, 0):
                        axis = agx.Vec3(0, 0, 1)
                    else:
                        axis = aa / angle

                    # The desired transform at local_time
                    xf = agx.AffineMatrix4x4(agx.Quat(angle, axis), pos)

                    # Compute joint angles we want and feed them to low-level-controller
                    status, target_angles = self._chain.computeInverseKinematics(xf, list(self._robot.get_joint_positions()))

                    assert status == agxModel.KinematicChain.SUCCESS, "Pose could not be found"

                    self._llc._target_angles = target_angles

        self._sd.setText(0, "High level Controller, tasks = %s" % ", ".join(active))


def set_robot_initial_pose(panda, chain, pose):

    # Compute transforms for each link relative the base give the provided joint angles
    # The second argument clamps any joint angles within its limits.
    #
    # The tool fingers are not part of the chain and need to be positioned manually.
    #
    # The fingers and hand are all part of the same assembly and hence have that assemblys frame as
    # frame parent. Therefore we can work with the local matrix for hand/fingers when computing and
    # updating where the fingers are relative the hand.
    # Updating the frame hierarchy this way saves some computations compared using getMatrix / setMatrix.
    #
    hand = panda.assembly.getRigidBody('panda_hand')
    hand_matrix_inverse = hand.getFrame().getLocalMatrix().inverse()
    fingers = [panda.assembly.getRigidBody(name) for name in ['panda_rightfinger', 'panda_leftfinger']]
    finger_relative_transforms = [finger.getFrame().getLocalMatrix() * hand_matrix_inverse for finger in fingers]

    status, transforms = chain.computeForwardKinematicsAll(pose, True)

    if status is not agxModel.KinematicChain.SUCCESS:
        print(f"Error. Forward computation failed with status: {status}")
    else:
        # Set the local model frame matrix for each link/body.
        # Using local matrix to handle if entire robot was moved by e.g. setting a translation
        # in the robot-assembly-frame.
        for transform, link in zip(transforms, panda.links):
            link.getFrame().setLocalMatrix(transform)

    # The tool/finger positioning, maintain same relative transform to the hans as prior to the robot was repositioned.
    hand_matrix = hand.getFrame().getLocalMatrix()
    for finger, rel_transform in zip(fingers, finger_relative_transforms):
        finger.getFrame().setLocalMatrix(rel_transform * hand_matrix)


def create_task_plan(controller, observer, box_stack1_pos, box_stack2_pos, box_size):

    # Move the two boxes located at "box_stack1_pos" and place them
    # on top of the box at "box_stack2_pos" to for a 3 high box stack.
    #
    pickup_offset = 0.05
    drop_offset = 0.08

    # Target orientations we'll use when when creating target transforms
    #   q1: z-axis points straight downwards
    #   q2: z-axis downwards, x-axis pointing to the right instead of forward
    #   q3: z-axis forward, x-axis downwards
    #
    q1 = agx.Quat(agx.Vec3.Z_AXIS(), -agx.Vec3.Z_AXIS())
    q2 = agx.Quat(math.pi, agx.Vec3(1, 0, 0), math.pi * 0.5, agx.Vec3(0, 0, 1), 0, agx.Vec3(0, 0, 1))
    q3 = agx.Quat(math.pi, agx.Vec3(1, 0, 0), -math.pi * 0.5, agx.Vec3(0, 1, 0), math.pi, agx.Vec3(1, 0, 0))

    box_z = box_size[2] * 2
    yz_diff = box_size[2] - box_size[1]

    # To keep this simple, robot base transform and world frame are the same.
    # If the entire robot is repositioned, by e.g. changing the robot assembly frame, then
    # these pX transforms should be updated to be relative the base and not in world coords.
    p0 = observer.getFrame().getMatrix()

    p1 = agx.AffineMatrix4x4(q1, box_stack1_pos + agx.Vec3(0, 0, box_z + pickup_offset))
    p2 = agx.AffineMatrix4x4(q1, box_stack1_pos + agx.Vec3(0, 0, box_z))

    p3 = agx.AffineMatrix4x4(q2, box_stack2_pos + agx.Vec3(0, 0, box_z + drop_offset))
    p4 = agx.AffineMatrix4x4(q2, box_stack2_pos + agx.Vec3(0, 0, box_z))

    p5 = agx.AffineMatrix4x4(q1, box_stack1_pos + agx.Vec3(0, 0, pickup_offset))
    p6 = agx.AffineMatrix4x4(q1, box_stack1_pos)

    p7 = agx.AffineMatrix4x4(q3, box_stack2_pos + agx.Vec3(0, 0, box_z * 2 + yz_diff + drop_offset))
    p8 = agx.AffineMatrix4x4(q3, box_stack2_pos + agx.Vec3(0, 0, box_z * 2 + yz_diff))

    # These transforms can be visualized with e.g:
    # agxOSG.createAxes(p7, root())

    controller.plan.append(Operation("open", 0.0, 1.0, None, None))
    controller.plan.append(Operation("move", 0.0, 1.0, p0, p1))
    controller.plan.append(Operation("move", 1.0, 2.0, p1, p2))
    controller.plan.append(Operation("close", 2.0, 3.0, None, None))
    controller.plan.append(Operation("move", 3.0, 4.0, p2, p1))
    controller.plan.append(Operation("move", 4.0, 6.0, p1, p3))
    controller.plan.append(Operation("move", 6.0, 7.0, p3, p4))
    controller.plan.append(Operation("open", 7.0, 7.5, None, None))
    controller.plan.append(Operation("move", 7.5, 9.0, p4, p3))
    controller.plan.append(Operation("move", 9.0, 11.0, p3, p5))
    controller.plan.append(Operation("move", 11.0, 12.0, p5, p6))
    controller.plan.append(Operation("close", 12.0, 13.0, None, None))
    controller.plan.append(Operation("move", 13.0, 14.0, p6, p5))
    controller.plan.append(Operation("move", 14.0, 16.0, p5, p7))
    controller.plan.append(Operation("move", 16.0, 17.0, p7, p8))
    controller.plan.append(Operation("open", 17.0, 18.0, None, None))
    controller.plan.append(Operation("move", 18.0, 19.0, p8, p7))
    controller.plan.append(Operation("move", 19.0, 20.0, p7, p0))


# Build robot scene
def buildScene():

    # Materials and ContactMaterials
    finger_material = agx.Material("GripperMaterial")
    box_material = agx.Material("BoxMaterial")

    mm = simulation().getMaterialManager()

    box_finger = mm.getOrCreateContactMaterial(box_material, finger_material)
    fm1 = agx.ScaleBoxFrictionModel(agx.FrictionModel.DIRECT)
    box_finger.setFrictionModel(fm1)
    box_finger.setFrictionCoefficient(0.9)

    box_box = mm.getOrCreateContactMaterial(box_material, box_material)
    fm2 = agx.ScaleBoxFrictionModel(agx.FrictionModel.DIRECT)
    box_box.setFrictionModel(fm2)
    box_box.setFrictionCoefficient(0.5)

    # Create the scene:
    # - a box as ground
    # - another box as a table/working area
    # - 3 boxes with different mass on the table
    # - a panda robot with a gripper tool
    #
    ground = agxCollide.Geometry(agxCollide.Box(3, 3, 0.1))
    ground.setPosition(agx.Vec3(0, 0, -0.1))
    simulation().add(ground)
    ground_node = agxOSG.createVisual(ground, root())
    agxOSG.setDiffuseColor(ground_node, agxRender.Color.Black())

    # Table/working area
    #
    table_size = agx.Vec3(0.3, 0.3, 0.3)

    table = agxCollide.Geometry(agxCollide.Box(table_size))
    table.setPosition(agx.Vec3(0.5, 0, -0.2))
    simulation().add(table)
    agxOSG.createVisual(table, root())
    agxOSG.setTexture(table, root(), "textures/grid.png")

    # The boxes on the table
    #
    box_size = agx.Vec3(0.04, 0.02, 0.03)

    box_stack1_pos = table.getPosition() + agx.Vec3(-0.5 * table_size[0], -0.5 * table_size[1], table_size[2] + box_size[2])
    box_stack2_pos = table.getPosition() + agx.Vec3(0.3 * table_size[0], 0.4 * table_size[1], table_size[2] + box_size[2])

    box_colors = [agxRender.Color.Yellow(), agxRender.Color.Orange(), agxRender.Color.Red()]
    box_positions = [box_stack1_pos, box_stack1_pos + agx.Vec3(0, 0, 0.05999), box_stack2_pos]

    for pos, color, mass in zip(box_positions, box_colors, [0.40, 0.50, 0.60]):
        box = agx.RigidBody()
        box.add(agxCollide.Geometry(agxCollide.Box(0.04, 0.02, 0.03)))
        box.setPosition(pos)
        box.getMassProperties().setMass(mass)

        simulation().add(box)
        box_node = agxOSG.createVisual(box, root())
        agxOSG.setDiffuseColor(box_node, color)

        agxUtil.setBodyMaterial(box, box_material)

    # The robot
    #
    panda = Panda(simulation(), use_tool=True)

    # We add a relative transform to the end of the chain and will get a chain with
    # one additional link. This is so that we can make the position between the gripper fingers
    # the end of the chain. This simplifies which part of the robot we control.
    #
    chain = agxModel.SerialKinematicChain(simulation(), panda.base, panda.ee, panda.tooltip_offset)

    # Verify that a valid chain was found.
    if not chain.isValid():
        print("Error: a chain could not be found")
        application().stop()

    # Initial pose for the robot.
    set_robot_initial_pose(panda, chain, panda.q_ready)

    # Set material on the gripper fingers
    for name in ['panda_rightfinger', 'panda_leftfinger']:
        agxUtil.setBodyMaterial(panda.assembly.getRigidBody(name), finger_material)

    # Low level controller:
    #
    llc = LowLevelController(panda, chain)
    simulation().add(llc)

    # High level controller:
    #
    hlc = TaskController(panda, chain, llc)
    simulation().add(hlc)
    create_task_plan(hlc, panda.tooltip, box_stack1_pos, box_stack2_pos, box_size)

    # Position camera
    eye = agx.Vec3(1.93, -2.78, 0.93)
    center = agx.Vec3(0.312, 0, 0.497)
    up = agx.Vec3(-0.0251, 0.1424, 0.9895)
    application().setCameraHome(eye, center, up)


# Entry point when this script is started with python executable
init = init_app(
    name=__name__,
    scenes=[(buildScene, "1")],
    autoStepping=False,  # Default: False
    onInitialized=lambda app: print("App successfully initialized."),
    onShutdown=lambda app: print("App successfully shut down."),
)
