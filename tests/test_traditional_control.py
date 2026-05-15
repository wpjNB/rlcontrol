import numpy as np
import sys
sys.path.insert(0, '.')

from traditional_control.kinematics import Go2LegKinematics, HOME_QPOS


def test_fk_ik_roundtrip():
    """FK(IK(pos)) should recover the original position."""
    kin = Go2LegKinematics()
    target = np.array([0.0, 0.0, -0.4])
    joints = kin.solve('FL', target)
    recovered = kin.forward('FL', joints)
    np.testing.assert_allclose(recovered, target, atol=1e-4)


def test_ik_standing_pose():
    """IK for a straight-down foot should give thigh~0, calf~negative."""
    kin = Go2LegKinematics()
    target = np.array([0.0, 0.0, -(kin.L1 + kin.L2)])
    joints = kin.solve('FL', target)
    assert abs(joints[0]) < 0.01
    assert abs(joints[1]) < 0.01
    assert abs(joints[2]) < 0.01


def test_ik_bent_knee():
    """IK for a closer foot should give negative calf angle (bent)."""
    kin = Go2LegKinematics()
    target = np.array([0.0, 0.0, -0.3])
    joints = kin.solve('FL', target)
    assert joints[2] < 0


def test_clamp_joints():
    """clamp_joints should enforce joint limits."""
    kin = Go2LegKinematics()
    joints = np.array([5.0, 5.0, -5.0])
    clamped = kin.clamp_joints('FL', joints)
    limits = kin.get_limits('FL')
    for i in range(3):
        assert limits[i][0] <= clamped[i] <= limits[i][1]


def test_fk_standing():
    """FK of HOME_QPOS should give a reasonable foot position."""
    kin = Go2LegKinematics()
    joints = HOME_QPOS[0:3]
    pos = kin.forward('FL', joints)
    assert pos[2] < 0
    assert -0.5 < pos[2] < -0.2


from traditional_control.gait_phase import GaitPhaseScheduler


def test_phase_trot_diagonal_sync():
    """Trot: FR/RL in phase, FL/RR in phase, offset by pi."""
    sched = GaitPhaseScheduler(freq=2.0)
    phases = sched.step(0.0, 0.001)
    assert abs(phases['FR']['phase'] - 0.0) < 0.01
    assert abs(phases['FL']['phase'] - np.pi) < 0.01
    assert abs(phases['RL']['phase'] - np.pi) < 0.01
    assert abs(phases['RR']['phase'] - 0.0) < 0.01


def test_phase_swing_stance():
    """Phase should correctly identify swing vs stance."""
    sched = GaitPhaseScheduler(freq=2.0, duty_cycle=0.6)
    phases = sched.step(0.0, 0.001)
    assert phases['FR']['is_swing'] == True
    phases = sched.step(0.25, 0.001)
    assert phases['FR']['is_swing'] == False


def test_step_length_forward():
    """Step length should scale with vx."""
    sched = GaitPhaseScheduler(freq=2.0)
    step = sched.get_step_length(vx=1.0, yaw_rate=0.0, leg='FL')
    assert abs(step - 0.5) < 0.01


def test_step_length_turning():
    """Turning should create differential step lengths."""
    sched = GaitPhaseScheduler(freq=2.0)
    left = sched.get_step_length(vx=0.0, yaw_rate=1.0, leg='FL')
    right = sched.get_step_length(vx=0.0, yaw_rate=1.0, leg='FR')
    assert left > 0
    assert right < 0


if __name__ == '__main__':
    test_fk_ik_roundtrip()
    test_ik_standing_pose()
    test_ik_bent_knee()
    test_clamp_joints()
    test_fk_standing()
    test_phase_trot_diagonal_sync()
    test_phase_swing_stance()
    test_step_length_forward()
    test_step_length_turning()
    print("All traditional control tests passed!")
