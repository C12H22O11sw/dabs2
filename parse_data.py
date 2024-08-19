import numpy as np
import scipy
from scipy.interpolate import LinearNDInterpolator, CubicSpline
from scipy.constants import lbf, inch, mph, mach
import matplotlib.pyplot as plt
import scipy.optimize
import yaml


class APCPropeller:
    def __init__(self, filename: str, display_name = "unknown propeller",  use_metric: bool=True, is_pusher: bool=False):
        self.is_pusher = is_pusher

        self.display_name = display_name

        self._prop_data = self._parse_prop_apc_data(filename)
        self._thrust_curve = self._interpolate_prop_apc_curve(self._prop_data, use_metric)

        self.max_rpm = max(self._prop_data.keys())
        self.max_vel = max(self._prop_data[self.max_rpm].keys())

    def get_thrust(self, rpm: int | float, velocity: int | float):
        return self._thrust_curve(rpm, velocity)[0]
    
    def get_torque(self, rpm: int | float, velocity: int | float):
        if self.is_pusher:
            return -self._thrust_curve(rpm, velocity)[3]
        else:
            return self._thrust_curve(rpm, velocity)[3]
    
    def get_wrench(self, rpm: int | float, velocity: int | float):
        return self._thrust_curve(rpm, velocity)

    def get_power_draw(self, rpm: int | float, velocity: int | float):
        return (2 * np.pi * rpm / 60) * self.get_torque(rpm, velocity)

    def _interpolate_prop_apc_curve(self, prop_data, use_metric):
        points = []
        values = []
        for rpm, table in prop_data.items():
            for velocity, entrees in table.items():
                if use_metric:
                    thrust = entrees['Thrust(N)']
                    torque = entrees['Torque(N-m)']
                    velocity *= mph
                else:
                    thrust = entrees['Thrust(Lbf)']
                    torque = entrees['Torque(In-Lbf)']
            
                points.append([rpm, velocity])
                values.append(np.array([thrust, 0, 0, torque, 0, 0]))

        return LinearNDInterpolator(points, values, 0)

    def _parse_prop_apc_data(self, filename: str):

        prop_data = dict()

        with open(filename) as file:
            tables = file.read().split('PROP RPM =')
            for table in tables[1:]:

                rows = table.split('\n')

                rpm = int(table.split()[0])

                prop_data[rpm] = dict()

                header = rows[2].split()
                units = rows[3].split()
                # add units to the end of duplicate column names
                for i, unit in enumerate(units):
                    if unit != '-':
                        header[i] += unit

                for row in rows[4:-4]:
                    entrees = row.split()
                    if len(entrees) != len(header):
                        continue
                    speed = float(entrees[0])
                    prop_data[rpm][speed] = dict()
                    for measurement, value in zip(header[1:], entrees[1:]):
                        prop_data[rpm][speed][measurement] = float(value)

        return prop_data

class Airframe:
    def __init__(self, filename:str, display_name:str = 'unknown airframe'):

        self.display_name = display_name

        file = open(filename, 'r')
        data = yaml.load(file, Loader=yaml.SafeLoader)
        file.close()

        self.mass = data['mass']
        self.diameter = data['diameter'] * 1e-3
        self.area = (np.pi / 4) * self.diameter**2
        self._drag_curve = self._parse_cd_data(data)

    def get_cd(self, velocity: int | float):
        return self._drag_curve(velocity / mach)
    
    def get_drag(self, velocity: int | float, rho=1.225):
        return 0.5 * rho * velocity**2 * self.area * self.get_cd(velocity)
    
    def _parse_cd_data(self, data):
        points = list(map( lambda p : p[0] , data['cd_curve'] )) 
        values = list(map( lambda p : p[1] , data['cd_curve'] )) 
        return CubicSpline(points, values)    

class DCMotor:

    def __init__(self, max_rpm: int|float, max_power: int|float, display_name:str='unknown dc motor'):
        self.max_rpm = max_rpm
        self.max_power = max_power
        self.display_name = display_name
        self.max_torque = 2 * max_power / (np.pi * max_rpm / 60)



    def get_torque(self, rpm: float|int):
        return self.max_torque * (1 - rpm / self.max_rpm)
    
    def get_power(self, rpm: float|int):
        return self.get_torque(rpm) * (2 * np.pi * rpm / 60)

def plot_function(func, begin, end, n=100, ax=plt, label=None, linestyle='solid', show=True):
    points = list(np.linspace(begin, end, n))
    values = list(map(func, points))

    ax.plot(points, values, label=label, linestyle=linestyle)

    if label is not None:
        ax.legend()
    if show:
        plt.show()

def main():

    # build thrust curves
    apc_3_5x75 = APCPropeller('data/apc_props/PER3_5x75E.dat', 'APC 5x75') # 5x75E, 230
    apc_3_525x8E = APCPropeller('data/apc_props/PER3_525x8E.dat', 'APC 525x8E') # 525x8E, 231
    apc_3_55x70 = APCPropeller('data/apc_props/PER3_55x70.dat', 'APC 55x70') # 55x70, 259
    apc_3_7x9 = APCPropeller('data/apc_props/PER3_7x9.dat', 'APC 7x9') # 7x9, 250
    apc_3_7x10 = APCPropeller('data/apc_props/PER3_7x10.dat', 'APC 7x10') # 7x10, 246
    apc_3_7x11E = APCPropeller('data/apc_props/PER3_7x11E.dat', 'APC 7x11E') # 7x11E, 251
    apc_3_8x10 = APCPropeller('data/apc_props/PER3_8x10.dat', 'APC 8x10') # 8x10, 242
    apc_3_11x14 = APCPropeller('data/apc_props/PER3_11x14.dat', 'APC 11x14') # 11x14, 240

    simple_airframe_75mm = Airframe('./data/test_airframe_75mm.yaml', 'Simple 75mm Airframe')
    simple_airframe_98mm = Airframe('./data/test_airframe_98mm.yaml', 'Simple 98mm Airframe')
    simple_airframe_156mm = Airframe('./data/test_airframe_156mm.yaml', 'Simple 156mm Airframe')
    simple_airframe_203mm = Airframe('./data/test_airframe_203mm.yaml', 'Simple 203mm Airframe')

    scorpion_IS_2840_5100KV = DCMotor(42000, 880, 'Scorpion-IS-2840-5100KV')
    scorpion_IS_3435_2600KV = DCMotor(42000, 2050, 'Scorpion-IS-3435-2600KV')

    '''
    Plausable propeller selection by airframe
     - 75mm: 
        - 5x75: 113.2m/2, 0.76kW
        - 525x8E: 
        - 55x70: 
     - 98mm: everything
     - 156mm: 
        - 11x14: 115m/s, 3.6kW
        - 8x10: 113m/s: 3.4kW
        - 7x9: 
        - 7x11E:
        - 7x10E:
        - 55x70: 
     - 203mm:
        - 11x14: 108m/s, 4.4kW
        - 8x10: 104m/s, 4.3kW 
        - 7x9: 101m/s, 4.0kW
    
    '''


    def get_max_thrust_curve(prop: APCPropeller):
        return lambda velocity : 4 * prop.get_thrust(prop.max_rpm, velocity)
    
    def get_power_curve(prop: APCPropeller):
        return lambda velocity : prop.get_power_draw(prop.max_rpm, velocity)
    
    max_vel = 160

    if False:

        fig, ax_force = plt.subplots()

        ax_force.set_ylabel('force (N)')
        ax_force.set_xlabel('velocity (m/s)')
        ax_force.set_title('Forces and Power Consumption vs Velocity @ Max RPM (75mm, 98mm)')

        ax_power = ax_force.twinx()
        ax_power.set_ylabel('power required per motor (W)')



        for prop_name, prop in prop_dict.items():
            plot_function(get_max_thrust_curve(prop), 0, max_vel, ax = ax_force, show=False, label=f'thrust {prop_name} prop')
            plot_function(get_power_curve(prop), 0, max_vel, ax = ax_power, show=False, linestyle='dashed', label=f'power consumption {prop_name} prop')
            
        for airframe_name, airframe in airframe_dict.items():
            plot_function(airframe.get_drag, 0, max_vel, ax = ax_force, show=False, label=f'drag {airframe_name} rocket')

        plt.show()

    if False:

        fig, ax_velocity = plt.subplots()

        ax_velocity.set_ylabel('Maximum Velocity (m/s)')
        ax_velocity.set_xlabel('Coefficient of Drag')
        ax_velocity.set_title('Max Velocity and Power Consumption vs CD @ Max RPM (98mm)')

        ax_power = ax_velocity.twinx()
        ax_power.set_ylabel('power required per motor (W)')

        cd_list = []
        power_list = []
        vel_list = []
        for cd in np.linspace(0, 1, 30):

            prop = apc_3_5x75

            net_force = lambda v : 4 * prop.get_thrust(prop.max_rpm, v[0]) \
                - 0.5 * 1.225 * v[0]**2 * (np.pi / 4 * 0.098**2) * cd
                        
            vel = scipy.optimize.fsolve(net_force, 100)[0]
            vel_list.append(vel)
            cd_list.append(cd)
            power_list.append(prop.get_power_draw(prop.max_rpm, vel))

        #vel_cd_curve = CubicSpline()

        ax_velocity.plot(cd_list, vel_list, label='Maximum Velocity')
        ax_power.plot(cd_list, power_list, label='Power Draw/Motor', linestyle='dashed')
        ax_velocity.legend()
        ax_power.legend()
        plt.show()

    if False:
        vel = 100
        prop = apc_3_5x75
        airframe = simple_airframe_98mm

        fig, ax_rpm = plt.subplots()

        ax_rpm.set_ylabel('Required Propeller RPM')
        ax_rpm.set_xlabel('Coefficient of Drag')
        ax_rpm.set_title('Required RPM and Power Consumption vs CD @ 100m/s(98mm)')

        ax_power = ax_rpm.twinx()
        ax_power.set_ylabel('Power Required per Motor (W)')

        rpm_list = []
        power_list = []
        cd_list = np.linspace(0, 1, 30)

        for cd in cd_list:
            func = lambda rpm, : 4 * prop.get_thrust(rpm[0], vel) \
                - 0.5 * 1.225 * vel**2 * (np.pi / 4 * 0.098**2) * cd

            rpm = scipy.optimize.fsolve(func, prop.max_rpm*0.9)[0]
            rpm_list.append(rpm)

            power = prop.get_power_draw(rpm, vel)
            power_list.append(power)

        ax_rpm.plot(cd_list, rpm_list, label='propeller rpm')
        ax_power.plot(cd_list, power_list, label='power draw (W)', linestyle='dashed')

        ax_rpm.legend()
        ax_power.legend()

        plt.show()

    if False:
        prop = apc_3_5x75
        airframe = simple_airframe_98mm

        fig, ax_torque = plt.subplots()

        ax_torque.set_ylabel('Torque (Nm)')
        ax_torque.set_xlabel('Propeller RPM')
        ax_torque.set_title('Torque and Velocity vs RPM')

        ax_vel = ax_torque.twinx()
        ax_vel.set_ylabel('Velocity (m/s)')


        rpm_list = np.linspace(0, prop.max_rpm, 30)

        for cd in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

            vel_list = []
            torque_list = []
            for rpm in rpm_list:
                func = lambda vel, : 4 * prop.get_thrust(rpm, vel[0]) \
                    - 0.5 * 1.225 * vel[0]**2 * (np.pi / 4 * airframe.area**2) * cd

                vel = scipy.optimize.fsolve(func, 0)[0]
                vel_list.append(vel)

                torque = prop.get_torque(rpm, vel)
                torque_list.append(torque)

                print()

            ax_torque.plot(rpm_list, torque_list, label=f'Propeller Torque (cd={cd})')
            ax_vel.plot(rpm_list, vel_list, label=f'Velocity (cd={cd})', linestyle='dashed')

        for motor in [scorpion_IS_2840_5100KV, scorpion_IS_3435_2600KV]:
            motor_torque_list = [motor.get_torque(rpm) for rpm in rpm_list]
            ax_torque.plot(rpm_list, motor_torque_list, label=f'Motor Torque ({motor.display_name})')      

        ax_torque.legend(loc='upper left')
        ax_vel.legend(loc='upper center')
        plt.show()        

    if True:
        motor_list = [scorpion_IS_2840_5100KV, scorpion_IS_3435_2600KV]
        airframe = simple_airframe_98mm
        prop = apc_3_5x75

        fig, ax_force = plt.subplots()

        ax_force.set_ylabel('Force')
        ax_force.set_xlabel('Velocity (m/s)')
        ax_force.set_title('Thrust and Drag vs Velocity @ Max Voltage, '
                           +f'Propeller = {prop.display_name}, '
                           +f'Airframe Diameter = {int(airframe.diameter*1000)}mm')
        
        ax_rpm = ax_force.twinx()
        ax_rpm.set_xlabel('rpm')

        #ax_torque = ax_force.twinx()
        #ax_torque.set_xlabel('torque')

        vel_list = np.linspace(0, max_vel, 30)
        for cd in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            drag_list = [ 0.5 * 1.225 * vel**2 * cd * airframe.area \
                         for vel in vel_list]
            ax_force.plot(vel_list, drag_list, label=f'Drag Curve (cd={cd})')

        for motor in motor_list:
            thrust_list = []
            for vel in vel_list:
                func = lambda rpm : prop.get_torque(rpm[0], vel) - motor.get_torque(rpm[0])
                rpm = min(scipy.optimize.fsolve(func, prop.max_rpm*0.9)[0], prop.max_rpm)
                thrust = 4 * prop.get_thrust(rpm, vel)
                thrust_list.append(thrust)

            ax_force.plot(vel_list, thrust_list, label=f'Thrust Curve ({prop.display_name}, {motor.display_name})')

        ax_force.legend()
        ax_rpm.legend()
            
        plt.show()


if __name__ == "__main__":
    main()