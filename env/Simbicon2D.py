"""
A 2D bouncing ball environment

"""


import sys, os, random, time
from math import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode
# from twisted.protocols import stateful
import copy

def sign(x):
    """Returns 1.0 if x is positive, -1.0 if x is negative or zero."""
    if x > 0.0: return 1.0
    else: return -1.0

def len3(v):
    """Returns the length of 3-vector v."""
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def neg3(v):
    """Returns the negation of 3-vector v."""
    return (-v[0], -v[1], -v[2])

def add3(a, b):
    """Returns the sum of 3-vectors a and b."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub3(a, b):
    """Returns the difference between 3-vectors a and b."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def mul3(v, s):
    """Returns 3-vector v multiplied by scalar s."""
    return (v[0] * s, v[1] * s, v[2] * s)

def div3(v, s):
    """Returns 3-vector v divided by scalar s."""
    return (v[0] / s, v[1] / s, v[2] / s)

def dist3(a, b):
    """Returns the distance between point 3-vectors a and b."""
    return len3(sub3(a, b))

def norm3(v):
    """Returns the unit length 3-vector parallel to 3-vector v."""
    l = len3(v)
    if (l > 0.0): return (v[0] / l, v[1] / l, v[2] / l)
    else: return (0.0, 0.0, 0.0)

def dot3(a, b):
    """Returns the dot product of 3-vectors a and b."""
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

def cross(a, b):
    """Returns the cross product of 3-vectors a and b."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0])

def project3(v, d):
    """Returns projection of 3-vector v onto unit 3-vector d."""
    return mul3(v, dot3(norm3(v), d))

def acosdot3(a, b):
    """Returns the angle between unit 3-vectors a and b."""
    x = dot3(a, b)
    if x < -1.0: return pi
    elif x > 1.0: return 0.0
    else: return acos(x)

def rotate3(m, v):
    """Returns the rotation of 3-vector v by 3x3 (row major) matrix m."""
    return (v[0] * m[0] + v[1] * m[1] + v[2] * m[2],
        v[0] * m[3] + v[1] * m[4] + v[2] * m[5],
        v[0] * m[6] + v[1] * m[7] + v[2] * m[8])

def invert3x3(m):
    """Returns the inversion (transpose) of 3x3 rotation matrix m."""
    return (m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])

def zaxis(m):
    """Returns the z-axis vector from 3x3 (row major) rotation matrix m."""
    return (m[2], m[5], m[8])

def calcRotMatrix(axis, angle):
    """
    Returns the row-major 3x3 rotation matrix defining a rotation around axis by
    angle.
    """
    cosTheta = cos(angle)
    sinTheta = sin(angle)
    t = 1.0 - cosTheta
    return (
        t * axis[0]**2 + cosTheta,
        t * axis[0] * axis[1] - sinTheta * axis[2],
        t * axis[0] * axis[2] + sinTheta * axis[1],
        t * axis[0] * axis[1] + sinTheta * axis[2],
        t * axis[1]**2 + cosTheta,
        t * axis[1] * axis[2] - sinTheta * axis[0],
        t * axis[0] * axis[2] - sinTheta * axis[1],
        t * axis[1] * axis[2] + sinTheta * axis[0],
        t * axis[2]**2 + cosTheta)

def makeOpenGLMatrix(r, p):
    """
    Returns an OpenGL compatible (column-major, 4x4 homogeneous) transformation
    matrix from ODE compatible (row-major, 3x3) rotation matrix r and position
    vector p.
    """
    return (
        r[0], r[3], r[6], 0.0,
        r[1], r[4], r[7], 0.0,
        r[2], r[5], r[8], 0.0,
        p[0], p[1], p[2], 1.0)

def getBodyRelVec(b, v):
    """
    Returns the 3-vector v transformed into the local coordinate system of ODE
    body b.
    """
    return rotate3(invert3x3(b.getRotation()), v)


# rotation directions are named by the third (z-axis) row of the 3x3 matrix,
#   because ODE capsules are oriented along the z-axis
rightRot = (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
leftRot = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
upRot = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
downRot = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
bkwdRot = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# axes used to determine constrained joint rotations
rightAxis = (1.0, 0.0, 0.0)
leftAxis = (-1.0, 0.0, 0.0)
upAxis = (0.0, 1.0, 0.0)
downAxis = (0.0, -1.0, 0.0)
bkwdAxis = (0.0, 0.0, 1.0)
fwdAxis = (0.0, 0.0, -1.0)

UPPER_ARM_LEN = 0.30
FORE_ARM_LEN = 0.25
HAND_LEN = 0.13 # wrist to mid-fingers only
FOOT_LEN = 0.18 # ankles to base of ball of foot only
HEEL_LEN = 0.05

BROW_H = 1.68
MOUTH_H = 1.53
NECK_H = 1.50
SHOULDER_H = 1.37
CHEST_H = 1.35
HIP_H = 0.86
KNEE_H = 0.48
ANKLE_H = 0.08

SHOULDER_W = 0.41
CHEST_W = 0.36 # actually wider, but we want narrower than shoulders (esp. with large radius)
LEG_W = 0.28 # between middles of upper legs
PELVIS_W = 0.25 # actually wider, but we want smaller than hip width

R_SHOULDER_POS = (-SHOULDER_W * 0.5, SHOULDER_H, 0.0)
L_SHOULDER_POS = (SHOULDER_W * 0.5, SHOULDER_H, 0.0)
R_ELBOW_POS = sub3(R_SHOULDER_POS, (UPPER_ARM_LEN, 0.0, 0.0))
L_ELBOW_POS = add3(L_SHOULDER_POS, (UPPER_ARM_LEN, 0.0, 0.0))
R_WRIST_POS = sub3(R_ELBOW_POS, (FORE_ARM_LEN, 0.0, 0.0))
L_WRIST_POS = add3(L_ELBOW_POS, (FORE_ARM_LEN, 0.0, 0.0))
R_FINGERS_POS = sub3(R_WRIST_POS, (HAND_LEN, 0.0, 0.0))
L_FINGERS_POS = add3(L_WRIST_POS, (HAND_LEN, 0.0, 0.0))

R_HIP_POS = (-LEG_W * 0.5, HIP_H, 0.0)
L_HIP_POS = (LEG_W * 0.5, HIP_H, 0.0)
R_KNEE_POS = (-LEG_W * 0.5, KNEE_H, 0.0)
L_KNEE_POS = (LEG_W * 0.5, KNEE_H, 0.0)
R_ANKLE_POS = (-LEG_W * 0.5, ANKLE_H, 0.0)
L_ANKLE_POS = (LEG_W * 0.5, ANKLE_H, 0.0)
R_HEEL_POS = sub3(R_ANKLE_POS, (0.0, 0.0, HEEL_LEN))
L_HEEL_POS = sub3(L_ANKLE_POS, (0.0, 0.0, HEEL_LEN))
R_TOES_POS = add3(R_ANKLE_POS, (0.0, 0.0, FOOT_LEN))
L_TOES_POS = add3(L_ANKLE_POS, (0.0, 0.0, FOOT_LEN))


class RagDoll(object):
    def __init__(self, world, space, density, offset = (0.0, 0.0, 0.0)):
        #Creates a ragdoll of standard size at the given offset.

        self.world = world
        self.space = space
        self.density = density
        self.bodies = []
        self.geoms = []
        self.joints = []
        self.totalMass = 0.0

        self.offset = offset
        self.rot = (0.0, 1.0, 0.0)

        self.chest = self.addBody((-CHEST_W * 0.5, CHEST_H, 0.0),
            (CHEST_W * 0.5, CHEST_H, 0.0), 0.13)
        self.belly = self.addBody((0.0, CHEST_H - 0.1, 0.0),
            (0.0, HIP_H + 0.1, 0.0), 0.125)
        self.midSpine = self.addFixedJoint(self.chest, self.belly)
        self.pelvis = self.addBody((-PELVIS_W * 0.5, HIP_H, 0.0),
            (PELVIS_W * 0.5, HIP_H, 0.0), 0.125)
        self.lowSpine = self.addFixedJoint(self.belly, self.pelvis)

        self.head = self.addBody((0.0, BROW_H, 0.0), (0.0, MOUTH_H, 0.0), 0.11)
        self.neck = self.addBallJoint(self.chest, self.head,
            (0.0, NECK_H, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), pi * 0.25,
            pi * 0.25, 80.0, 40.0)

        self.rightUpperLeg = self.addBody(R_HIP_POS, R_KNEE_POS, 0.11)
        self.rightHip = self.addUniversalJoint(self.pelvis, self.rightUpperLeg,
            R_HIP_POS, bkwdAxis, rightAxis, -0.1 * pi, 0.3 * pi, -0.15 * pi,
            0.75 * pi)
        self.leftUpperLeg = self.addBody(L_HIP_POS, L_KNEE_POS, 0.11)
        self.leftHip = self.addUniversalJoint(self.pelvis, self.leftUpperLeg,
            L_HIP_POS, fwdAxis, rightAxis, -0.1 * pi, 0.3 * pi, -0.15 * pi,
            0.75 * pi)

        self.rightLowerLeg = self.addBody(R_KNEE_POS, R_ANKLE_POS, 0.09)
        self.rightKnee = self.addHingeJoint(self.rightUpperLeg,
            self.rightLowerLeg, R_KNEE_POS, leftAxis, 0.0, pi * 0.75)
        self.leftLowerLeg = self.addBody(L_KNEE_POS, L_ANKLE_POS, 0.09)
        self.leftKnee = self.addHingeJoint(self.leftUpperLeg,
            self.leftLowerLeg, L_KNEE_POS, leftAxis, 0.0, pi * 0.75)

        self.rightFoot = self.addBody(R_HEEL_POS, R_TOES_POS, 0.09)
        self.rightAnkle = self.addHingeJoint(self.rightLowerLeg,
            self.rightFoot, R_ANKLE_POS, rightAxis, -0.1 * pi, 0.05 * pi)
        self.leftFoot = self.addBody(L_HEEL_POS, L_TOES_POS, 0.09)
        self.leftAnkle = self.addHingeJoint(self.leftLowerLeg,
            self.leftFoot, L_ANKLE_POS, rightAxis, -0.1 * pi, 0.05 * pi)

        self.rightUpperArm = self.addBody(R_SHOULDER_POS, R_ELBOW_POS, 0.08)
        self.rightShoulder = self.addBallJoint(self.chest, self.rightUpperArm,
            R_SHOULDER_POS, norm3((-1.0, -1.0, 4.0)), (0.0, 0.0, 1.0), pi * 0.5,
            pi * 0.25, 150.0, 100.0)
        self.leftUpperArm = self.addBody(L_SHOULDER_POS, L_ELBOW_POS, 0.08)
        self.leftShoulder = self.addBallJoint(self.chest, self.leftUpperArm,
            L_SHOULDER_POS, norm3((1.0, -1.0, 4.0)), (0.0, 0.0, 1.0), pi * 0.5,
            pi * 0.25, 150.0, 100.0)

        self.rightForeArm = self.addBody(R_ELBOW_POS, R_WRIST_POS, 0.075)
        self.rightElbow = self.addHingeJoint(self.rightUpperArm,
            self.rightForeArm, R_ELBOW_POS, downAxis, 0.0, 0.6 * pi)
        self.leftForeArm = self.addBody(L_ELBOW_POS, L_WRIST_POS, 0.075)
        self.leftElbow = self.addHingeJoint(self.leftUpperArm,
            self.leftForeArm, L_ELBOW_POS, upAxis, 0.0, 0.6 * pi)

        self.rightHand = self.addBody(R_WRIST_POS, R_FINGERS_POS, 0.075)
        self.rightWrist = self.addHingeJoint(self.rightForeArm,
            self.rightHand, R_WRIST_POS, fwdAxis, -0.1 * pi, 0.2 * pi)
        self.leftHand = self.addBody(L_WRIST_POS, L_FINGERS_POS, 0.075)
        self.leftWrist = self.addHingeJoint(self.leftForeArm,
            self.leftHand, L_WRIST_POS, bkwdAxis, -0.1 * pi, 0.2 * pi)
        
    def getPosition(self):
        return self.pelvis.getPosition()

    def addBody(self, p1, p2, radius):
        
        # Adds a capsule body between joint positions p1 and p2 and with given
        # radius to the ragdoll.
        

        p1 = add3(p1, self.offset)
        p2 = add3(p2, self.offset)

        # cylinder length not including endcaps, make capsules overlap by half
        #   radius at joints
        cyllen = dist3(p1, p2) + radius
        """
        body = ode.Body(self.world)
        m = ode.Mass()
        m.setCappedCylinder(self.density, 3, radius, cyllen)
        body.setMass(m)

        # set parameters for drawing the body
        body.shape = "capsule"
        body.length = cyllen
        body.radius = radius

        # create a capsule geom for collision detection
        geom = ode.GeomCCylinder(self.space, radius, cyllen)
        geom.setBody(body)
        """     
        # Create body
        body = ode.Body(self.world)
        m = ode.Mass()
        m.setBox(self.density, radius, radius, cyllen)
        body.setMass(m)
    
        # Set parameters for drawing the body
        body.shape = "rectangle"
        body.boxsize = (radius, radius, cyllen)
    
        # Create a box geom for collision detection
        geom = ode.GeomBox(self.space, lengths=body.boxsize)
        geom.setBody(body)

        # define body rotation automatically from body axis
        za = norm3(sub3(p2, p1))
        if (abs(dot3(za, (1.0, 0.0, 0.0))) < 0.7): xa = (1.0, 0.0, 0.0)
        else: xa = (0.0, 1.0, 0.0)
        ya = cross(za, xa)
        xa = norm3(cross(ya, za))
        ya = cross(za, xa)
        rot = (xa[0], ya[0], za[0], xa[1], ya[1], za[1], xa[2], ya[2], za[2])

        body.setPosition(mul3(add3(p1, p2), 0.5))
        body.setRotation(rot)

        self.bodies.append(body)
        self.geoms.append(geom)
        
        self.totalMass += body.getMass().mass

        return body
    
    def addFixedJoint(self, body1, body2):
        joint = ode.FixedJoint(self.world)
        joint.attach(body1, body2)
        joint.setFixed()

        joint.style = "fixed"
        self.joints.append(joint)

        return joint

    def addHingeJoint(self, body1, body2, anchor, axis, loStop = -ode.Infinity,
        hiStop = ode.Infinity):

        anchor = add3(anchor, self.offset)

        joint = ode.HingeJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis(axis)
        joint.setParam(ode.ParamLoStop, loStop)
        joint.setParam(ode.ParamHiStop, hiStop)

        joint.style = "hinge"
        self.joints.append(joint)

        return joint

    def addUniversalJoint(self, body1, body2, anchor, axis1, axis2,
        loStop1 = -ode.Infinity, hiStop1 = ode.Infinity,
        loStop2 = -ode.Infinity, hiStop2 = ode.Infinity):

        anchor = add3(anchor, self.offset)

        joint = ode.UniversalJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis1(axis1)
        joint.setAxis2(axis2)
        joint.setParam(ode.ParamLoStop, loStop1)
        joint.setParam(ode.ParamHiStop, hiStop1)
        joint.setParam(ode.ParamLoStop2, loStop2)
        joint.setParam(ode.ParamHiStop2, hiStop2)

        joint.style = "univ"
        self.joints.append(joint)

        return joint

    def addBallJoint(self, body1, body2, anchor, baseAxis, baseTwistUp,
        flexLimit = pi, twistLimit = pi, flexForce = 0.0, twistForce = 0.0):

        anchor = add3(anchor, self.offset)

        # create the joint
        joint = ode.BallJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)

        # store the base orientation of the joint in the local coordinate system
        #   of the primary body (because baseAxis and baseTwistUp may not be
        #   orthogonal, the nearest vector to baseTwistUp but orthogonal to
        #   baseAxis is calculated and stored with the joint)
        joint.baseAxis = getBodyRelVec(body1, baseAxis)
        tempTwistUp = getBodyRelVec(body1, baseTwistUp)
        baseSide = norm3(cross(tempTwistUp, joint.baseAxis))
        joint.baseTwistUp = norm3(cross(joint.baseAxis, baseSide))

        # store the base twist up vector (original version) in the local
        #   coordinate system of the secondary body
        joint.baseTwistUp2 = getBodyRelVec(body2, baseTwistUp)

        # store joint rotation limits and resistive force factors
        joint.flexLimit = flexLimit
        joint.twistLimit = twistLimit
        joint.flexForce = flexForce
        joint.twistForce = twistForce

        joint.style = "ball"
        self.joints.append(joint)

        return joint
    

    def update(self):
        for j in self.joints:
            if j.style == "ball":
                # determine base and current attached body axes
                baseAxis = rotate3(j.getBody(0).getRotation(), j.baseAxis)
                currAxis = zaxis(j.getBody(1).getRotation())

                # get angular velocity of attached body relative to fixed body
                relAngVel = sub3(j.getBody(1).getAngularVel(),
                    j.getBody(0).getAngularVel())
                twistAngVel = project3(relAngVel, currAxis)
                flexAngVel = sub3(relAngVel, twistAngVel)

                # restrict limbs rotating too far from base axis
                angle = acosdot3(currAxis, baseAxis)
                if angle > j.flexLimit:
                    # add torque to push body back towards base axis
                    j.getBody(1).addTorque(mul3(
                        norm3(cross(currAxis, baseAxis)),
                        (angle - j.flexLimit) * j.flexForce))

                    # dampen flex to prevent bounceback
                    j.getBody(1).addTorque(mul3(flexAngVel,
                        -0.01 * j.flexForce))

                # determine the base twist up vector for the current attached
                #   body by applying the current joint flex to the fixed body's
                #   base twist up vector
                baseTwistUp = rotate3(j.getBody(0).getRotation(), j.baseTwistUp)
                base2current = calcRotMatrix(norm3(cross(baseAxis, currAxis)),
                    acosdot3(baseAxis, currAxis))
                projBaseTwistUp = rotate3(base2current, baseTwistUp)

                # determine the current twist up vector from the attached body
                actualTwistUp = rotate3(j.getBody(1).getRotation(),
                    j.baseTwistUp2)

                # restrict limbs twisting
                angle = acosdot3(actualTwistUp, projBaseTwistUp)
                if angle > j.twistLimit:
                    # add torque to rotate body back towards base angle
                    j.getBody(1).addTorque(mul3(
                        norm3(cross(actualTwistUp, projBaseTwistUp)),
                        (angle - j.twistLimit) * j.twistForce))

                    # dampen twisting
                    j.getBody(1).addTorque(mul3(twistAngVel,
                        -0.01 * j.twistForce))


def createCapsule(world, space, density, length, radius):
    """Creates a capsule body and corresponding geom."""

    # create capsule body (aligned along the z-axis so that it matches the
    #   GeomCCylinder created below, which is aligned along the z-axis by
    #   default)
    body = ode.Body(world)
    M = ode.Mass()
    M.setCappedCylinder(density, 3, radius, length)
    body.setMass(M)

    # set parameters for drawing the body
    body.shape = "capsule"
    body.length = length
    body.radius = radius

    # create a capsule geom for collision detection
    geom = ode.GeomCCylinder(space, radius, length)
    geom.setBody(body)

    return body, geom

# create_box
def createBox(world, space, density, lx, ly, lz):
    """Create a box body and its corresponding geom."""

    # Create body
    body = ode.Body(world)
    M = ode.Mass()
    M.setBox(density, lx, ly, lz)
    body.setMass(M)

    # Set parameters for drawing the body
    body.shape = "rectangle"
    body.boxsize = (lx, ly, lz)

    # Create a box geom for collision detection
    geom = ode.GeomBox(space, lengths=body.boxsize)
    geom.setBody(body)

    return body, geom

def createSphere(world, space, density, radius):
    """Creates a capsule body and corresponding geom."""

    # create capsule body (aligned along the z-axis so that it matches the
    #   GeomCCylinder created below, which is aligned along the z-axis by
    #   default)
    body = ode.Body(world)
    M = ode.Mass()
    M.setSphere(density, radius)
    body.setMass(M)

    # set parameters for drawing the body
    body.shape = "sphere"
    body.radius = radius

    # create a capsule geom for collision detection
    geom = ode.GeomSphere(space, radius)
    geom.setBody(body)

    return body, geom

def near_callback(args, geom1, geom2):
    """
    Callback function for the collide() method.

    This function checks if the given geoms do collide and creates contact
    joints if they do.
    """

    if (ode.areConnected(geom1.getBody(), geom2.getBody())):
        return

    # check if the objects collide
    contacts = ode.collide(geom1, geom2)

    # create contact joints
    world, contactgroup = args
    for c in contacts:
        c.setBounce(0.2)
        c.setMu(500) # 0-5 = very slippery, 50-500 = normal, 5000 = very sticky
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())


# polygon resolution for capsule bodies
CAPSULE_SLICES = 16
CAPSULE_STACKS = 12

def draw_body(body):
    """Draw an ODE body."""
    glColor3f(0.8, 0.3, 0.3)
    rot = makeOpenGLMatrix(body.getRotation(), body.getPosition())
    glPushMatrix()
    glMultMatrixd(rot)
    if body.shape == "capsule":
        cylHalfHeight = body.length / 2.0
        glBegin(GL_QUAD_STRIP)
        for i in range(0, CAPSULE_SLICES + 1):
            angle = i / float(CAPSULE_SLICES) * 2.0 * pi
            ca = cos(angle)
            sa = sin(angle)
            glNormal3f(ca, sa, 0)
            glVertex3f(body.radius * ca, body.radius * sa, cylHalfHeight)
            glVertex3f(body.radius * ca, body.radius * sa, -cylHalfHeight)
        glEnd()
        glTranslated(0, 0, cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
        glTranslated(0, 0, -2.0 * cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
    elif body.shape == "sphere":
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
    elif body.shape == "rectangle":
        sx,sy,sz = body.boxsize
        glScalef(sx, sy, sz)
        glutSolidCube(1)
    glPopMatrix()
    
    
def generateTerrainVerts(terrainData_, translateX):
    
    terrainScale=0.1
    verts=[]
    i=0
    verts.append([1.0, terrainData_[i], (i*terrainScale)+translateX])
    verts.append([-1.0, terrainData_[i], (i*terrainScale)+translateX]) 
    faces=[]
    for i in range(1, len(terrainData_)):
        verts.append([1.0, terrainData_[i], (i*terrainScale)+translateX])
        verts.append([-1.0, terrainData_[i], (i*terrainScale)+translateX])
        j=2*i
        face=[]
        face.append(j-2)
        face.append(j-1)
        face.append(j)
        faces.append(face)
        
        face=[]
        face.append(j)
        face.append(j-1)
        face.append(j+1)
        faces.append(face) 
        
    return verts, faces


def drawTerrain(terrainData, translateX):
    
    terrainScale=0.1
    glColor3f(0.6, 0.6, 0.9)
    verts, faces = generateTerrainVerts(terrainData, translateX)
    glBegin(GL_TRIANGLES)
    for face in faces:
        # j=i*3
        # glNormal3f(0, 1.0, 0)
        v0 = verts[face[0]]
        glVertex3f(v0[0],v0[1],v0[2]) #;//triangle one first vertex
        v0 = verts[face[1]]
        glVertex3f(v0[0],v0[1],v0[2]) #;//triangle one first vertex
        v0 = verts[face[2]]
        glVertex3f(v0[0],v0[1],v0[2]) #;//triangle one first vertex
    glEnd()
    
    # draw side of terrain
    glColor3f(0.4, 0.4, 0.8)
    glBegin(GL_QUAD_STRIP)
    for i in range(0, len(terrainData)):

        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-1.0, terrainData[i], (i*terrainScale)+translateX)
        glVertex3f(-1.0, -10, (i*terrainScale)+translateX)
    glEnd()
    

class BallGame2D(object):
    def __init__(self, settings):
        """Creates a ragdoll of standard size at the given offset."""
        self._game_settings=settings
        # initialize GLUT
        if self._game_settings['render']:
            glutInit([])
            glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
        
        self._terrainScale=self._game_settings["terrain_scale"]
        self._terrainParameters=self._game_settings['terrain_parameters']
        
        # create the program window
        if self._game_settings['render']:
            x = 0
            y = 0
            width = 800
            height = 480
            glutInitWindowPosition(x, y);
            glutInitWindowSize(width, height);
            glutCreateWindow("PyODE 2DBallGame Simulation")
        
        # create an ODE world object
        self._world = ode.World()
        self._world.setGravity((0.0, -9.81, 0.0))
        self._world.setERP(0.1)
        self._world.setCFM(1E-4)
        
        # create an ODE space object
        self._space = ode.Space()
        
        # create an infinite plane geom to simulate a floor
        # self._floor = ode.GeomPlane(self._space, (0, 1, 0), 0)
        """
        self._terrainStartX=0.0
        self._terrainStripIndex=-1
        self._terrainData = self.generateValidationTerrain(12)
        verts, faces = generateTerrainVerts(self._terrainData)
        self._terrainMeshData = ode.TriMeshData()
        self._terrainMeshData.build(verts, faces)
        """
        self._terrainMeshData = None
        self._floor = None
        self._terrainData = []
        
        # create a list to store any ODE bodies which are not part of the ragdoll (this
        #   is needed to avoid Python garbage collecting these bodies)
        self._bodies = []
        
        # create a joint group for the contact joints generated during collisions
        #   between two bodies collide
        self._contactgroup = ode.JointGroup()
        
        # set the initial simulation loop parameters
        self._fps = 60
        self._dt = 1.0 / self._fps
        self._stepsPerFrame = 16
        self._SloMo = 1.0
        self._Paused = False
        self._lasttime = time.time()
        self._numiter = 0
        
        # create the ragdoll
        self._ragdoll = RagDoll(self._world, self._space, 500, (0.0, 0.1, 2.0))
        print ("total mass is %.1f kg (%.1f lbs)" % (self._ragdoll.totalMass , self._ragdoll.totalMass *2.2 ))
        # self._bodies.append(self._ragdoll)
        
        # set GLUT callbacks
        # glutKeyboardFunc(onKey)
        # glutDisplayFunc(onDraw)
        # glutIdleFunc(onIdle)
        
        # enter the GLUT event loop
        # Without this there can be no control over the key inputs
        # glutMainLoop()
        
        self._ballRadius=0.05
        self._ballEpsilon=0.1
        self._state_num=0
        self._state_num_max=10
        self._num_points=self._game_settings['num_terrain_samples']
        
        # create an obstacle (world, space, density, height, radius)
        # self._obstacle, self._obsgeom = createSphere(self._world, self._space, 100, self._ballRadius)
        pos = (0.0, self._ballRadius+self._ballEpsilon, 0.0)
        #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
        # self._obstacle.setPosition(pos)
        # self._obstacle.setRotation(rightRot)
        # self._bodies.append(self._obstacle)
        # print ("obstacle created at %s" % (str(pos)))
        # print ("total mass is %.4f kg" % (self._obstacle.getMass().mass))
        
        
    def finish(self):
        pass
    
    def init(self):
        pass
    
    def initEpoch(self):
        pos = (0.0, self._ballRadius+self._ballEpsilon, 0.0)
        #pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
        # self._obstacle.setPosition(pos)
        # self._obstacle.setRotation(rightRot)
        # self._terrainData = []
        # self._terrainStartX=0.0
        # self._terrainStripIndex=0
        
        
        self._state_num=0
        self._end_of_Epoch_Flag=False
        
        self._validating=False
        
        # self.generateTerrain()
    
    def getEvaluationData(self):
        """
            The best measure of improvement for this environment is the distance the 
            ball reaches.
        """
        pos = self._obstacle.getPosition()
        return [pos[0]]
    
    def clear(self):
        pass
    
    def addAnchor(self, _anchor0, _anchor1, _anchor2):
        pass 
    
    def generateValidationEnvironmentSample(self, seed):
        # Hacky McHack
        self._terrainStartX=0.0
        self._terrainStripIndex=0
        self._validating=False
        self.generateValidationTerrain(seed)
        
    def generateEnvironmentSample(self):
        # Hacky McHack
        self._terrainStartX=0.0
        self._terrainStripIndex=0
        
        self.generateTerrain()
        
    def endOfEpoch(self):
        pos = self._obstacle.getPosition()
        start = (pos[0]/self._terrainScale)+1 
        # assert start+self._num_points+1 < (len(self._terrainData)), "Ball is exceeding terrain length %r after %r actions" % (start+self._num_points+1, self._state_num)
        if (self._end_of_Epoch_Flag):
            return True 
        else:
            return False  
                    
    def prepare_GL(self):
        """Setup basic OpenGL rendering with smooth shading and a single light."""
    
        glClearColor(0.8, 0.8, 0.9, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
    
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective (45.0, 1.3333, 0.2, 20.0)
    
        glViewport(0, 0, 640, 480)
    
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
        glLightfv(GL_LIGHT0,GL_POSITION,[0, 0, 1, 0])
        glLightfv(GL_LIGHT0,GL_DIFFUSE,[1, 1, 1, 1])
        glLightfv(GL_LIGHT0,GL_SPECULAR,[1, 1, 1, 1])
        glEnable(GL_LIGHT0)
    
        glEnable(GL_COLOR_MATERIAL)
        glColor3f(0.8, 0.8, 0.8)
    
        pos = self._ragdoll.getPosition()
        gluLookAt(-8.0, 0.0, pos[2], 0.0, 1.0, pos[2], 0.0, 1.0, 0.0)
        
    def actContinuous(self, action, bootstrapping=False):
        # print ("Action: ", action)
        
        pos = self._ragdoll.getPosition()
        # self._ragdoll.setLinearVel((action[0],action[1],0.0))
        contact = False
        while ( ( pos[1] > (self._ballRadius - (self._ballRadius*0.5))) and (not contact)):
            contact = self.simulateAction()
            pos = self._ragdoll.getPosition()
            
        reward = self.calcReward(bootstrapping=bootstrapping)
        # print (pos)
        pos = (pos[0], self._ballRadius+self._ballEpsilon, 0.0)
        # self._obstacle.setPosition(pos)
        # self._terrainData = self.generateTerrain()
        self._state_num=self._state_num+1
        # state = self.getState()
        # print ("state length: " + str(len(state)))
        # print (state)
        return reward
        # obstacle.addForce((0.0,100.0,0.0))
        
    def hitWall(self):
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        if (pos[1] > (self._ballRadius + (self._ballRadius*0.5))) or (vel[1] > 0): # fell in a hole
            # print ("Hit a wall")
            return True

        
    def calcReward(self, bootstrapping=False):
        """
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        contacts = ode.collide(self._floor, self._obsgeom)
        if (len(contacts)> 0):
                # print ("Num contacts: " + str(len(contacts)))
            contactInfo = contacts[0].getContactGeomParams()
            # print ("Constact info: ", contacts[0].getContactGeomParams())
            contactNormal = contactInfo[1]
        # print ("Ball velocity:", vel, " Ball position: ", pos)
        if ((pos[1] < (self._ballRadius - (self._ballRadius*0.5))) or self.hitWall() or
            (contactNormal[1] > -0.999999)
            ): # fell in a hole
        # if (pos[1] < (0.0)): # fell in a hole
            # print ("Ball Fell in hole: ", pos[1])
            # if (not bootstrapping):
            self._end_of_Epoch_Flag=True # kind of hacky to end Epoch after the ball falls in a hole.
            return 0
        if ((vel[0] < 0) or (pos[0] < 0.0)): # Something really odd happened
            self._end_of_Epoch_Flag=True # kind of hacky to end Epoch after the ball falls in a hole.
            return 0
        
        targetVel = 2.0
        
        vel_error = vel[0] - targetVel
        reward = exp((vel_error*vel_error)*(-0.75))
        
        # reward = 1.0 - (fabs(vel[0] - targetVel)/targetVel)
        return reward
        """
        return 0
    def onKey(c, x, y):
        """GLUT keyboard callback."""
    
        global SloMo, Paused
    
        # set simulation speed
        if c >= '0' and c <= '9':
            SloMo = 4 * int(c) + 1
        # pause/unpause simulation
        elif c == 'p' or c == 'P':
            Paused = not Paused
        # quit
        elif c == 'q' or c == 'Q':
            sys.exit(0)
    
    
    def onDraw(self):
        """GLUT render callback."""
        self.prepare_GL()
    
        for b in self._bodies:
            draw_body(b)
        for b in self._ragdoll.bodies:
            draw_body(b)
        drawTerrain(self._terrainData, self._terrainStartX)
    
        glutSwapBuffers()
    
        
    def simulateAction(self):
        """
            Returns True if a contact was detected
        
        """
        if self._Paused:
            return
        t = self._dt - (time.time() - self._lasttime)    
        if self._game_settings['render']:
            if (t > 0):
                time.sleep(t)
            
        for i in range(self._stepsPerFrame):
            # Detect collisions and create contact joints
            self._space.collide((self._world, self._contactgroup), near_callback)
    
            # Simulation step (with slow motion)
            self._world.step(self._dt / self._stepsPerFrame / self._SloMo)
    
            self._numiter += 1
    
            # apply internal ragdoll forces
            self._ragdoll.update()
            # pos = self._obstacle.getPosition()
            # print ("Ball pos: ", pos)
                
            # contacts = ode.collide(self._floor, self._obsgeom)
            # print ("Num contacts: " + str(len(contacts)))
            #if (len(contacts)> 0):
                # print ("Num contacts: " + str(len(contacts)))
                # print ("Constact info: ", contacts[0].getContactGeomParams())
            #    return True
            
            # Remove all contact joints
            # for joint_ in self._contactgroup:
            #     print ("Joint: " + str(joint_))
            self._contactgroup.empty()
            
        if self._game_settings['render']:
            glutPostRedisplay()
            self.onDraw()
        return False
        
        
    def generateTerrain(self):
        """
            If this is the first time this is called generate a new strip of terrain and use it.
            The second time this is called and onward generate a new strip and add to the end of the old strip.
            Also remove the begining half of the old strip
        """
        # print ("Generating more terrain")
        terrainData_=[]
        if (self._terrainStripIndex == 0):
            print ("Generating NEW terrain, translateX: ", self._terrainStartX)
            terrainData_ = self._generateTerrain(self._terrainParameters['terrain_length'])
            self._terrainStartX=0
        elif (self._terrainStripIndex > 0):
            print ("Generating more terrain, translateX: ", self._terrainStartX)
            terrainData_ = self._generateTerrain(self._terrainParameters['terrain_length']/2)
            self._terrainStartX=self._terrainStartX+((len(self._terrainData)*self._terrainScale)/2.0)
            terrainData_ = np.append(self._terrainData[len(self._terrainData)/2:], terrainData_)
        else:
            print ("Why is the strip index < 0???")
            sys.exit()    
        
        self.setTerrainData(terrainData_)
        self._terrainStripIndex=self._terrainStripIndex+1
        
    def _generateTerrain(self, length):
        """
            Generate a single strip of terrain
        """
        terrainLength=length
        terrainData=np.zeros((terrainLength))
        # gap_size=random.randint(2,7)
        gap_size=self._terrainParameters['gap_size']
        # gap_start=random.randint(2,7)
        gap_start=self._terrainParameters['gap_start']
        next_gap=self._terrainParameters['distance_till_next_gap']
        for i in range(terrainLength/next_gap):
            gap_start= gap_start+np.random.random_integers(self._terrainParameters['random_gap_start_range'][0],
                                                self._terrainParameters['random_gap_start_range'][1])
            gap_size=np.random.random_integers(self._terrainParameters['random_gap_width_range'][0],
                                    self._terrainParameters['random_gap_width_range'][1])
            terrainData[gap_start:gap_start+gap_size] = self._terrainParameters['terrain_change']
            gap_start=gap_start+next_gap
            
        return terrainData
    
    def generateValidationTerrain(self, seed):
        """
            If this is the first time this is called generate a new strip of terrain and use it.
            The second time this is called and onward generate a new strip and add to the end of the old strip.
            Also remove the beginning half of the old strip
        """
        if (not self._validating):
            self._validating=True
            random.seed(seed)
        # print ("Generating more terrain")
        terrainData_=[]
        if (self._terrainStripIndex == 0):
            print ("Generating NEW validation terrain, translateX: ", self._terrainStartX)
            terrainData_ = self._generateValidationTerrain(self._terrainParameters['terrain_length'], seed)
            self._terrainStartX=0
        elif (self._terrainStripIndex > 0):
            print ("Generating more validation terrain, translateX: ", self._terrainStartX)
            terrainData_ = self._generateValidationTerrain(self._terrainParameters['terrain_length']/2, seed)
            self._terrainStartX=self._terrainStartX+((len(self._terrainData)*self._terrainScale)/2.0)
            terrainData_ = np.append(self._terrainData[len(self._terrainData)/2:], terrainData_)
        else:
            print ("Why is the strip index < 0???")
            sys.exit()    
        
        
        self.setTerrainData(terrainData_)
        self._terrainStripIndex=self._terrainStripIndex+1
        
    def _generateValidationTerrain(self, length, seed):
        terrainLength=length
        terrainData=np.zeros((terrainLength))
        # gap_size=random.randint(2,7)
        gap_size=self._terrainParameters['gap_size']
        # gap_start=random.randint(2,7)
        gap_start=self._terrainParameters['gap_start']
        next_gap=self._terrainParameters['distance_till_next_gap']
        for i in range(terrainLength/next_gap):
            gap_start= gap_start+random.randint(self._terrainParameters['random_gap_start_range'][0],
                                                self._terrainParameters['random_gap_start_range'][1])
            gap_size=random.randint(self._terrainParameters['random_gap_width_range'][0],
                                    self._terrainParameters['random_gap_width_range'][1])
            terrainData[gap_start:gap_start+gap_size] = self._terrainParameters['terrain_change']
            gap_start=gap_start+next_gap
        return terrainData
    
    def setTerrainData(self, data_):
        self._terrainData = data_
        verts, faces = generateTerrainVerts(self._terrainData, translateX=self._terrainStartX)
        del self._terrainMeshData
        self._terrainMeshData = ode.TriMeshData()
        self._terrainMeshData.build(verts, faces)
        del self._floor
        self._floor = ode.GeomTriMesh(self._terrainMeshData, self._space)
    
    def getState(self):
        """ get the next self._num_points points"""
        pos = self._ragdoll.getPosition()
        start = ((pos[0]-(self._terrainStartX) )/self._terrainScale)+1
        if (start+self._num_points+1 > (len(self._terrainData))):
            print ("State not big enough ", len(self._terrainData))
            if (self._validating):
                self.generateValidationTerrain(0)
            else:
                self.generateTerrain()
        start = ((pos[0]-(self._terrainStartX) )/self._terrainScale)+1
        assert start+self._num_points+1 < (len(self._terrainData)), "Ball is exceeding terrain length %r after %r actions" % (start+self._num_points+1, self._state_num)
        # print ("Terrain Data: ", self._terrainData)
        if pos[0] < 0: #something bad happened...
            state=np.zeros((self._num_points+1))
        else:
            state = copy.deepcopy(self._terrainData[start+1:start+self._num_points+2])
            # print ("Start: ", start, " State Data: ", state)
            state[len(state)-1] = fabs(float(floor(start)*self._terrainScale)-pos[0])
            # print ("Dist to next point: ", state[len(state)-1])
        return state
    

if __name__ == '__main__':
    import json
    settings={}
    # game = BallGame2D(settings)
    if (len(sys.argv)) > 1:
        _settings=json.load(open(sys.argv[1]))
        print (_settings)
        game = BallGame2D(_settings)
    else:
        settings['render']=True
        game = BallGame2D(settings)
    game.init()
    for j in range(5):
        # game.generateEnvironmentSample()
        game.generateValidationEnvironmentSample(j)
        game.initEpoch()
        i=0
        # while not game.endOfEpoch():
        for i in range(50):
            print ("Starting new epoch")
            # state = game.getState()
            
            # action = model.predict(state)
            action = [3.73,5.0]
            state = game.getState()
            # print ("state length: " + str(len(state)))
            # print (state)
            
        
            reward = game.actContinuous(action)
            
            print ("Reward: " + str(reward) + " on action: " + str(i))
            print ("Number of geoms in space: ", game._space.getNumGeoms())
            i=i+1
            game._lasttime = time.time()
            
    game.finish()
