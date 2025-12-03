
import rospy
from rospy import Duration
import constants
from std_msgs.msg import String

class WatchDog():
    def __init__(self, timer_duration, publisher):
        self.timer_period = Duration(timer_duration)
        self.timer = None
        self.publisher = publisher        

    def feed(self):
        if self.timer is not None:
            self.timer.shutdown()
        self.timer = rospy.Timer(self.timer_period, self.hungry_dog, oneshot=True)

    def hungry_dog(self, error):
        self.publisher.publish(str(constants.TEAM_ID + ',' + constants.TEAM_PASSWORD + ",-1,xxxx"))

