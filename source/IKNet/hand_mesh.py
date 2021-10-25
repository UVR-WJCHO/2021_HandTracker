import numpy as np
from transforms3d.quaternions import quat2mat

import sys
sys.path.append("C:/Research/2021_HandTracker/source/IKNet/")
from config import *
from kinematics import *
from utils import *


_FLOAT_EPS = np.finfo(np.float).eps

class HandMesh():
  """
  Wrapper for the MANO hand model.
  """
  def __init__(self, model_path):
    """
    Init.

    Parameters
    ----------
    model_path : str
      Path to the MANO model file. This model is converted by `prepare_mano.py`
      from official release.
    """

    params = load_pkl(model_path)
    self.verts = params['verts']
    self.faces = params['faces']
    self.weights = params['weights']
    self.joints = params['joints']

    ##773 verts
    #1538 face

    self.n_verts = self.verts.shape[0]
    self.n_faces = self.faces.shape[0]

    self.ref_pose = []
    self.ref_T = []

    for j in range(MANOHandJoints.n_joints):
      parent = MANOHandJoints.parents[j]
      if parent is None:
        self.ref_T.append(self.verts)
        self.ref_pose.append(self.joints[j])
      else:
        self.ref_T.append(self.verts - self.joints[parent])
        self.ref_pose.append(self.joints[j] - self.joints[parent])
    self.ref_pose = np.expand_dims(np.stack(self.ref_pose, 0), -1)
    self.ref_pose_tensor = torch.tensor(self.ref_pose, dtype=torch.float32).cuda()

    self.ref_T = np.expand_dims(np.stack(self.ref_T, 1), -1)


  def set_abs_quat(self, quat):
    """
    Set absolute (global) rotation for the hand.

    Parameters
    ----------
    quat : np.ndarray, shape [J, 4]
      Absolute rotations for each joint in quaternion.

    Returns
    -------
    np.ndarray, shape [V, 3]
      Mesh vertices after posing.
    """
    mats = []
    for j in range(MANOHandJoints.n_joints):
      mats.append(quat2mat(quat[j]))
    mats = np.stack(mats, 0)

    pose = np.matmul(mats, self.ref_pose)
    joint_xyz = [None] * MANOHandJoints.n_joints
    for j in range(MANOHandJoints.n_joints):
      joint_xyz[j] = pose[j]
      parent = MANOHandJoints.parents[j]
      if parent is not None:
        joint_xyz[j] += joint_xyz[parent]
    joint_xyz = np.stack(joint_xyz, 0)[..., 0]

    T = np.matmul(np.expand_dims(mats, 0), self.ref_T)[..., 0]
    self.verts = [None] * MANOHandJoints.n_joints
    for j in range(MANOHandJoints.n_joints):
      self.verts[j] = T[:, j]
      parent = MANOHandJoints.parents[j]
      if parent is not None:
        self.verts[j] += joint_xyz[parent]
    self.verts = np.stack(self.verts, 1)
    self.verts = np.sum(self.verts * self.weights, 1)

    return self.verts.copy()

  def set_abs_xyz(self, quat):
    """
    Set absolute (global) rotation for the hand.

    Parameters
    ----------
    quat : np.ndarray, shape [J, 4]
      Absolute rotations for each joint in quaternion.

    Returns
    -------
    np.ndarray, shape [V, 3]
      Mesh vertices after posing.
    """
    mats = []
    for j in range(MANOHandJoints.n_joints):
      mats.append(quat2mat(quat[j]))
    mats = np.stack(mats, 0)

    pose = np.matmul(mats, self.ref_pose)
    joint_xyz = [None] * MANOHandJoints.n_joints
    for j in range(MANOHandJoints.n_joints):
      joint_xyz[j] = pose[j]
      parent = MANOHandJoints.parents[j]
      if parent is not None:
        joint_xyz[j] += joint_xyz[parent]
    joint_xyz = np.stack(joint_xyz, 0)[..., 0]

    return joint_xyz.copy()

  def set_abs_xyz_torch(self, quat):
    """
    Set absolute (global) rotation for the hand.

    Parameters
    ----------
    quat : np.ndarray, shape [J, 4]
      Absolute rotations for each joint in quaternion.

    Returns
    -------
    np.ndarray, shape [V, 3]
      Mesh vertices after posing.
    """
    mats = []
    for j in range(MANOHandJoints.n_joints):
      tmp = self.quat2mat_torch(quat[j])
      mats.append(tmp)
    mats = torch.stack(mats, 0)

    pose = torch.matmul(mats, self.ref_pose_tensor)
    joint_xyz = [None] * MANOHandJoints.n_joints
    for j in range(MANOHandJoints.n_joints):
      joint_xyz[j] = pose[j]
      parent = MANOHandJoints.parents[j]
      if parent is not None:
        joint_xyz[j] += joint_xyz[parent]
    joint_xyz = torch.stack(joint_xyz, 0)[..., 0]

    return joint_xyz

  def quat2mat_torch(self, q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.

    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z

    #mat = [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
    #        [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
    #        [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]]

    m1 = [ 1.0-(yY+zZ), xY-wZ, xZ+wY ]
    m2 = [ xY+wZ, 1.0-(xX+zZ), yZ-wX ]
    m3 = [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]
    m1 = torch.stack(m1, 0)
    m2 = torch.stack(m2, 0)
    m3 = torch.stack(m3, 0)

    mat = torch.stack((m1, m2, m3), 0)

    return mat
