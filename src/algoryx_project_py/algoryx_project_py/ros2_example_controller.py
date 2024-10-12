"""
Controller script for the example scene with the E85 excavator

Controlling the excavator:
    The simulated execavator exposes its actuators using the ROS2ControlInterface class.
    It publishes all the current actuator states (position, velocity) using a ROS2 sensorMsgs/JointState
    and subscribes to velocity commands coming as senosrMsgs/JointState messages.

    First start the excavator simulation excavator_E85_terrain_ros2.agxPy, then start this controller script
    in a different terminal.
"""

import agxROS2

import sys
import math
import time
from enum import Enum


class ExcavatorE85ROS2Controller():
    '''
    Implements the controller of the excavator sending and receiving ROS2 sensorMsgs/JointState messages to
    control the E85 excavator.
    '''

    class Stage(Enum):
        DIG = 1
        HOME = 2
        DRIVING = 3

    def __init__(self):
        self.joint_names = [
            'ArticulatedArm_Prismatic',
            'ArmPrismatic',
            'StickPrismatic',
            'TiltPrismatic',
            'BucketPrismatic',
            'BladePrismatic1',
            'BladePrismatic2',
            'CabinHinge',
            'LeftSprocketHinge',
            'RightSprocketHinge'
        ]

        # Subscribes on the clock. The simulation publishes the time at every timestep
        self.clock_sub = agxROS2.SubscriberRosgraphMsgsClock("clock", agxROS2.createClockQOS())
        self.clock_msg = agxROS2.RosgraphMsgsClock()
        self.current_time = 0

        # Frequence to run the controller loop in
        self.freq = 120

        # Publisher to publish commands
        self.joint_command_msg = agxROS2.SensorMsgsJointState()
        self._pub_qos = agxROS2.QOS()
        self._pub_qos.historyDepth = 1
        self._pub_qos.history = agxROS2.QOS_HISTORY_KEEP_LAST_HISTORY_QOS
        self._pub_qos.reliability = agxROS2.QOS_RELIABILITY_RELIABLE
        self._pub_qos.durability = agxROS2.QOS_DURABILITY_VOLATILE
        self.joint_command_publisher = agxROS2.PublisherSensorMsgsJointState("joint_commands", self._pub_qos)

        # Subscriber to receive joint states
        self.joint_state_msg = agxROS2.SensorMsgsJointState()
        self.joint_state_subscriber = agxROS2.SubscriberSensorMsgsJointState("joint_states", agxROS2.createSensorDataQOS())

        # The sequence of velocities on joint for a given time to perform one digging and emptying trajectory
        self.digging_sequence_commands = [
            {
                'time': 1.9,
                'StickPrismatic': -0.15,
                'BucketPrismatic': -0.15,
            },
            {
                'time': 2,
                'ArmPrismatic': -0.07
            },
            {
                'time': 6.6,
                'StickPrismatic': 0.08,
                'BucketPrismatic': 0.06,
                'ArmPrismatic': 0.03
            },
            {
                'time': 2.0,
                'CabinHinge': 0.35,
            },
            {
                'time': 3.0,
                'StickPrismatic': -0.2,
                'BucketPrismatic': -0.2,
            },
            {
                'time': 1.0,
            }
        ]
        # The current sequence
        self.sequence = 0
        # The current stage
        self.stage = ExcavatorE85ROS2Controller.Stage.DIG
        # The maximum number of Dig, Home, Drive cycles to repeat before stopping the controller (this script)
        self.nb_max_cycles = 3
        self.nb_cycles = 0



        ######################################################################################################
        # Control algortihm values
        self.k = 0.3 # Proportional gain
        self.error = 0 # Distance error used in linear control 
        self.max_error = 5 # Maximum error treshold
        self.d = 5 # Guessed wheel distance
        self.r = 1 # Guessed wheel radius
        self.x_d = (5,7) # Goal (destination)


    def send_command(self, command):
        '''
        Send one command message
        '''
        sec = math.floor(self.current_time)
        nanosec = round((self.current_time - sec) * 1e9)
        self.joint_command_msg.header.stamp.sec = sec
        self.joint_command_msg.header.stamp.nanosec = nanosec
        default_cmd = [0 for _ in self.joint_names]
        velocity = []
        names = []
        for j, cs in zip(self.joint_names, default_cmd):
            if j in command:
                velocity.append(command[j])
            else:
                velocity.append(cs)
            names.append(j)
        self.joint_command_msg.name = names
        self.joint_command_msg.velocity = velocity
        self.joint_command_publisher.sendMessage(self.joint_command_msg)

    def get_joint_state(self):
        '''
        Receive the current joint states
        '''
        self.joint_state_subscriber.receiveMessage(self.joint_state_msg)

    def get_sim_time(self):
        '''
        Receive the current simulation time
        '''
        recieved_time = self.clock_sub.receiveMessage(self.clock_msg)
        self.current_time = self.clock_msg.clock.sec + 1e-9 * self.clock_msg.clock.nanosec
        return self.current_time, recieved_time

    def calculate_going_home_commands(self):
        '''
        Calculate the velocities to set at all the joints to go from any position to the home (0.0) position
        '''
        # The velocity to set at this timestep is scaled with the current position error
        def calculate_velocity_from_current_pos(pos):
            v = (0 - pos) / 3
            if v < -1e3:
                v = min(v, -0.05)
            elif v > 1e3:
                v = max(v, 0.05)
            return v
        command = {}
        accumulated_pos_error = 0
        for name, current_pos in zip(self.joint_state_msg.name, self.joint_state_msg.position):
            # The tracks should not move. Only the arm.
            if "Sprocket" not in name:
                command[name] = calculate_velocity_from_current_pos(current_pos)
                accumulated_pos_error += abs(current_pos)

        stage = ExcavatorE85ROS2Controller.Stage.HOME
        # If the accumulated error is smaller than this than we hare officially home
        if accumulated_pos_error < 0.05:
            stage = ExcavatorE85ROS2Controller.Stage.DRIVING
        return command, stage

    def check_if_stopped(self):
        '''
        Checks if all the joints have close to zero current speed
        '''
        close_to_zero = True
        for current_vel in self.joint_state_msg.velocity:
            if abs(current_vel) > 1e-4:
                close_to_zero = False
                break
        return close_to_zero
    
    # This should be the main control algorithm
    def controlAlgorithm(self):
        x = self.joint_state_msg.position # Excavator position vector
        
        v_r = self.r*x[7] # Right track
        v_l = self.r*x[6] # Left track
        
        v_x = 0.5*(v_r+v_l) # Object velocity
        
        self.error = math.sqrt((self.x_d[0]-x[7])**2 + (self.x_d[1] - x[7])**2)

        self.u = self.k*self.error

        commands = {}

        commands = {
            "LeftSprocketHinge": self.u,
            "RightSprocketHinge": self.u
        }

        return commands
        
    # Test control algorithm    
    def testControl(self):
        # Calculate the error
        target_position = self.x_d # Goal (destination)
        position = self.joint_state_msg.position # Excavator position vector

        error_x = target_position[0] - position[7]
        error_y = target_position[1] - position[7]
        distance_error = math.sqrt(error_x**2 + error_y**2)

        # Simple proportional control gains
        k_linear = 0.5
        k_angular = 0.5

        # Calculate desired angle and error
        desired_angle = math.atan2(error_y, error_x)
        angle_error = desired_angle - position[6]

        # Normalize the angle error
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        # Compute wheel velocities
        left_velocity = k_linear * distance_error - k_angular * angle_error * (self.d/ 2)
        right_velocity = k_linear * distance_error + k_angular * angle_error * (self.d / 2)

        self.error = math.sqrt((self.x_d[0]-position[7])**2 + (self.x_d[1] - position[7])**2) # Check error


        commands = {}

        commands = {
            "LeftSprocketHinge": left_velocity,
            "RightSprocketHinge": right_velocity
        }

        return commands



    def run(self):
        '''
        The controller loop
        '''
        running = True
        commands = {}

        controller_time_now = time.time()
        controller_time_prev = time.time()
        controller_update_time = 1 / self.freq

        start_time, received_time = self.get_sim_time()
        print("Connecting...")
        time.sleep(0.25)
        while not received_time:
            start_time, received_time = self.get_sim_time()
            time.sleep(0.001)

        print("Connected to Excavator!")

        print("Running loading sequence: ")
        print("\tDigging")

        self.stage == ExcavatorE85ROS2Controller.Stage.DRIVING

        while running:
            self.get_joint_state()
            current_time, _ = self.get_sim_time()

            if self.stage == ExcavatorE85ROS2Controller.Stage.DIG:
                if current_time - start_time > self.digging_sequence_commands[self.sequence]['time']:
                    self.sequence += 1
                    start_time = current_time
                    if self.sequence >= len(self.digging_sequence_commands):
                        self.stage = ExcavatorE85ROS2Controller.Stage.HOME
                        self.sequence = 0
                        print("\tDigging done, going home.")
                        continue
                commands = self.digging_sequence_commands[self.sequence]
            elif self.stage == ExcavatorE85ROS2Controller.Stage.HOME:
                commands, self.stage = self.calculate_going_home_commands()
                if self.stage != ExcavatorE85ROS2Controller.Stage.HOME:
                    start_time, _ = self.get_sim_time()
                    print("\tGoing home done, start driving.")

            elif self.stage == ExcavatorE85ROS2Controller.Stage.DRIVING:

                while(True):

                    commands = self.testControl() # Call the test control algorithm
                    self.send_command(commands) # Publish the commands for moving


                    if self.error < self.max_error: # If error is small enough, stop
                        print("Target reached!")
                        break
                
                

                if current_time - start_time > 6:
                    self.nb_cycles += 1
                    commands = {}
                    self.stage = ExcavatorE85ROS2Controller.Stage.DIG
                    start_time, _ = self.get_sim_time()
                    print("\tDriving done, start digging.")
            
            if self.nb_cycles >= self.nb_max_cycles:
                running = False

            # Sleep so the controller runs at the set frequency
            controller_time_now = time.time()
            time_to_sleep = controller_time_prev + controller_update_time - controller_time_now
            controller_time_prev = controller_time_now
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        # Make sure that the simulation got the stop command and that the machine stopped before exiting
        while not self.check_if_stopped():
            self.get_joint_state()
            self.send_command({})


def main(argv):
    excavator_controller = ExcavatorE85ROS2Controller()
    excavator_controller.run()


if __name__ == '__main__':
    main(sys.argv[1:])
