"""
Example scene with the E85 excavator exposing its actuators using the ROS2ControlInterface to be controlled
using the topics /joint_commands and publishing its states on /joint_states

Start this simulation first and then the controller script excavator_E85_controller.py in another terminal
"""


import agx
import agxOSG
import agxROS2
import agxRender
import agxCollide
import agxTerrain

import argparse

from agxPythonModules.utils.environment import simulation, root, \
    application, init_app

from agxPythonModules.models.excavators.excavatorE85 import ExcavatorE85 as Excavator


def height_function(x, y):
    return 0.0


def createHeightfield(resolution, size):
    # Create a heightfield that will be used to define the terrain surface
    hf = agxCollide.HeightField(resolution[0], resolution[1], size[0], size[1])

    # Helper functions for constructing the height field surface
    def hf_index_to_pos(index, dim):
        return (index / float(resolution[dim]) - 0.5) * size[dim]

    # Set the heights of the height field
    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            height = height_function(hf_index_to_pos(i, 0),
                                     hf_index_to_pos(j, 0))
            hf.setHeight(i, j, height)
    return hf


def setupCamera(app):
    camera_data = app.getCameraData()
    camera_data.eye = agx.Vec3(10.1412, -12.2315, 7.1398)
    camera_data.center = agx.Vec3(2.1634, 2.4046, 0.8891)
    camera_data.up = agx.Vec3(-0.1693, 0.3076, 0.9363)
    camera_data.nearClippingPlane = 0.1
    camera_data.farClippingPlane = 5000
    app.applyCameraData(camera_data)


def buildScene1():

    arguments = application().getArguments()
    argument_string = [arguments.getArgumentName(a)
                       for a in range(2, arguments.getNumArguments())]

    # Handle argument
    ap = argparse.ArgumentParser(argument_string)

    ap.add_argument("--automatic_digging", action="store_true",
                    help="If specified the excavator will \
                          start an automatic digging sequence")

    args, unknown = ap.parse_known_args(argument_string)
    args = vars(args)

    # Create the terrain from a height field, set maximum depth to
    # 5m and add it to the simulation
    res = (400, 400)
    size = (30, 30)
    terrain = agxTerrain.Terrain.createFromHeightField(
        createHeightfield(res, size), 5.0)

    simulation().add(terrain)

    # Load a material from the material library, this sets
    #   - Bulk material
    #   - Particle material
    #   - Terrain material
    #   - Particle-Particle contact material
    #   - Particle-Terrain contact material
    #   - Aggregate-Terrain contact material
    # WARNING:  Changing ANY material, terrain material or contact material
    #           retrieved from Terrain will invalidate these settings!
    terrain.loadLibraryMaterial("SAND_1")
    terrain.getTerrainMaterial().getCompactionProperties().setAngleOfReposeCompactionRate(24.0)
    terrain.getTerrainMaterial().getBulkProperties().setYoungsModulus(1e6)
    # Setup a renderer for the terrain.
    renderer = agxOSG.TerrainVoxelRenderer(terrain, root())
    # Render the particles as textured meshes.
    renderer.setRenderSoilParticlesMesh(True)
    simulation().add(renderer)

    # Create an excavator
    excavator = Excavator()

    excavator.setRotation(terrain.getRotation())
    excavator.setPosition(0, 0, 0)
    simulation().add(excavator)

    joint_names = [
        'ArticulatedArm_Prismatic',
        'ArmPrismatic',
        'StickPrismatic',
        'TiltPrismatic',
        'BucketPrismatic',
        'BladePrismatic1',
        'BladePrismatic2',
        'CabinHinge',
        'LeftSprocketHinge',
        'RightSprocketHinge',
    ]

    excavator_ros2_interface = agxROS2.ROS2ControlInterface(
        "joint_commands",
        "joint_states",
    )
    for name in joint_names:
        excavator_ros2_interface.addJoint(excavator.getConstraint1DOF(name), agxROS2.ROS2ControlInterface.VELOCITY)
    simulation().add(excavator_ros2_interface)
    ros2_clock = agxROS2.ROS2ClockPublisher()
    simulation().add(ros2_clock)

    # Configure the contact material between the tracks and the ground(terrain)
    excavator.configure_track_ground_contact_materials(terrain)

    # Set contact materials of the terrain and bucket
    particle_material = terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE)
    bucket_particle_cm = simulation().getMaterialManager().getOrCreateContactMaterial(particle_material, excavator.bucket_material)

    bucket_particle_cm.setYoungsModulus(1e9)
    bucket_particle_cm.setRestitution(0.0)
    bucket_particle_cm.setFrictionCoefficient(0.7)
    bucket_particle_cm.setRollingResistanceCoefficient(0.7)

    # Create the Shovel object using the cutting and top edge
    terrain_shovel = agxTerrain.Shovel(excavator.bucket_body,
                                       excavator.top_edge,
                                       excavator.cutting_edge,
                                       excavator.forward_cutting_vector)
    # Set the maximum distance to solid terrain for leveling using this shovel
    terrain_shovel.setVerticalBladeSoilMergeDistance(0.0)
    # Set a margin around the bounding box of the shovel where particles are not to be merged
    terrain_shovel.setNoMergeExtensionDistance(0.1)

    terrain_shovel.setContactRegionThreshold(0.05)
    terrain_shovel.setContactRegionVerticalLimit(0.2)
    terrain_shovel.setAlwaysRemoveShovelContacts(False)
    terrain_shovel.getExcavationSettings(agxTerrain.Shovel.ExcavationMode_DEFORM_BACK).setEnable(False)

    # Add the shovel to the terrain
    terrain.add(terrain_shovel)

    # Create the Shovel object using the cutting and top edge
    blade_shovel = agxTerrain.Shovel(excavator.blade_body,
                                     excavator.blade_top_edge,
                                     excavator.blade_cutting_edge,
                                     excavator.blade_forward_cutting_vector)
    # Set the maximum distance to solid terrain for leveling using this shovel
    blade_shovel.setVerticalBladeSoilMergeDistance(0.0)
    # Set a margin around the bounding box of the shovel where particles are not to be merged
    blade_shovel.setNoMergeExtensionDistance(0.1)

    terrain_shovel.setContactRegionThreshold(0.05)
    blade_shovel.setAlwaysRemoveShovelContacts(False)
    # Add the shovel to the terrain
    terrain.add(blade_shovel)

    # Set the contact stiffness multiplier for the generated contacts between the soil aggregates <-> terrain for excavation
    # and deformation. The final Young's modulus value that will be used in the contact material thus becomes:
    #       YM_final = BulkYoungsModulus * stiffnessMultiplier
    terrain.getTerrainMaterial().getExcavationContactProperties().setAggregateStiffnessMultiplier(5e-5)

    # Sets the maximum volume (m3) of active zone wedges that should wake particles.
    terrain.getProperties().setMaximumParticleActivationVolume(2)

    agx.setNumThreads(0)
    n = int(agx.getNumThreads() / 2) - 1
    agx.setNumThreads(n)

    application().getSceneDecorator().setBackgroundColor(agxRender.Color.SkyBlue(), agxRender.Color.DodgerBlue())

    # Enable warm starting of contacts
    simulation().getDynamicsSystem().setEnableContactWarmstarting(True)

    setupCamera(application())


def buildScene():
    buildScene1()


init = init_app(name=__name__,
                scenes=[
                    (buildScene1, '1')
                ],
                autoStepping=True)
