import numpy as np
import rospy
from dynamic_reconfigure.server import Server

class PIDController:
    def __init__(self, pid_constants, imax, derivative_low_pass=1.0, integral_leak=1.0, tuning=False):
        self.Kp, self.Kd, self.Ki = pid_constants
        
        # Configuration
        self.derivative_low_pass = derivative_low_pass
        self.integral_leak = integral_leak
        self.imax = imax

        # State Variables
        self.last_errx = 0
        self.last_erry = 0
        self.last_errz = 0
        self.integral_dx = 0
        self.integral_dy = 0
        self.integral_dz = 0
        self.dy_filtered = 0
        self.dx_filtered = 0
        self.dz_filtered = 0
        self.last_time = rospy.Time.now().to_sec()

        if tuning is True:
            self.srv = Server(CenteringPIDConfig, self.cfg_callback)

    def PID(self, error):
        now = rospy.Time.now().to_sec()
        dt = now - self.last_time
        self.last_time = now

        # Avoid divide-by-zero on first callback
        if dt == 0:
            dt = 1e-6
        ex, ey, ez = error
        # Compute derivatives
        dx = (ex - self.last_errx) / dt
        dy = (ey - self.last_erry) / dt
        dz = (ez - self.last_errz) / dt

        # low pass filter
        alpha = self.derivative_low_pass
        self.dx_filtered = alpha * self.dx_filtered + (1-alpha) * dx
        self.dy_filtered = alpha * self.dy_filtered + (1-alpha) * dy
        self.dz_filtered = alpha * self.dz_filtered + (1-alpha) * dz

        # limit response
        self.dx_filtered = max(min(self.dx_filtered, self.imax), -self.imax)
        self.dy_filtered = max(min(self.dx_filtered, self.imax), -self.imax)
        self.dz_filtered = max(min(self.dz_filtered, self.imax), -self.imax)

        # Compute Integrals
        leak = 1
        self.integral_dx = leak * self.integral_dx + (ex) * dt
        self.integral_dy = leak * self.integral_dy + (ey) * dt
        self.integral_dz = leak * self.integral_dz + (ez) * dt

        # Windup Clamp
        self.integral_dx = max(min(self.integral_dx, self.imax), -self.imax)
        self.integral_dy = max(min(self.integral_dy, self.imax), -self.imax)
        self.integral_dz = max(min(self.integral_dz, self.imax), -self.imax)

        self.last_errx = ex
        self.last_erry = ey
        self.last_errz = ez

        # rospy.loginfo(
        # f"[Centering] dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f} | "
        # f"vx={vx:.3f}, vy={vy:.3f} | "
        # f"ddx={self.dx_filtered:.3f}, ddy={self.dy_filtered:.3f} | dt={dt:.4f}"
        # f"intx={self.integral_dx:.3f}, inty={self.integral_dy:.3f}"
        # )

        x = self.Kp * ex - self.Kd * self.dx_filtered + self.Ki * self.integral_dx
        y = self.Kp * ey - self.Kd * self.dy_filtered + self.Ki * self.integral_dy
        z = self.Kp * ez - self.Kd * self.dz_filtered + self.Ki * self.integral_dz

        return x, y, z

    def cfg_callback(self, config, level):
        self.Kp = config.Kp
        self.Kd = config.Kd
        self.Ki = config.Ki
        self.imax = config.imax
        return config
    