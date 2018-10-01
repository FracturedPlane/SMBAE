#!/usr/bin/env python
""" generated source for module Simbicon """
# 
#  *   This is an implementation of the planar character animation system presented in "SIMBICON: Simple Biped Locomotion Control"
#  *   by Kangkang Yin, Kevin Loken and Michiel van de Panne. The purpose of this applet is to provide a simple demo to the aforementioned
#  *   system.
#  *
#  
import java.awt.event

import java.awt.image.BufferedImage

import java.awt.Graphics

import java.awt.image.BufferedImage

import java.awt.image.ImageObserver

import javax.swing.JPanel

import javax.swing.Timer

import java.awt.Color

import java.awt.FlowLayout

import java.awt.BorderLayout

import java.awt.KeyEventDispatcher

import java.awt.DefaultKeyboardFocusManager

class Simbicon(java, applet, Applet, MouseListener, MouseMotionListener, KeyListener):
    """ generated source for class Simbicon """
    bip7 = Bip7()
    gnd = Ground()
    Dt = 0.00005
    DtDisp = 0.0054
    timeEllapsed = 0

    # we'll use this buffered image to reduce flickering
    tempBuffer = BufferedImage()
    timer = Timer()

    # and the controller
    con = Controller()
    Md = float()
    Mdd = float()
    DesVel = 0

    # if this variable is set to true, the simulation will be running, otherwise it won't
    simFlag = False
    simButton = javax.swing.JButton()
    reset = javax.swing.JButton()
    panel = javax.swing.JPanel()
    speedSlider = javax.swing.JSlider()
    label = javax.swing.JLabel()
    shouldPanY = False

    def init(self):
        """ generated source for method init """
        setSize(500, 500)
        addMouseListener(self)
        addMouseMotionListener(self)
        # initialize the biped to a valid state:
        state = [0.463, 0.98, 0.898, -0.229, 0.051, 0.276, -0.221, -1.430, -0.217, 0.086, 0.298, -3.268, -0.601, 3.167, 0.360, 0.697, 0.241, 3.532]
        self.bip7.setState(state)
        delay = 1
        # milliseconds
        taskPerformer = ActionListener()
        self.timer = Timer(delay, taskPerformer)
        self.timer.start()
        self.tempBuffer = BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB)
        initComponents()
        self.con = Controller()
        self.con.addWalkingController()
        self.con.addRunningController()
        self.con.addCrouchWalkController()
        self.addKeyListener(self)
        self.requestFocus()

    def boundRange(self, value, min, max):
        """ generated source for method boundRange """
        if value < min:
            value = min
        if value > max:
            value = max
        return value

    # ////////////////////////////////////////////////////////
    #   PROC: wPDtorq()
    #   DOES: computes requires torque to move a joint wrt world frame
    # ////////////////////////////////////////////////////////
    def wPDtorq(self, torq, joint, dposn, kp, kd, world):
        """ generated source for method wPDtorq """
        joint_posn = self.bip7.State[4 + joint * 2]
        joint_vel = self.bip7.State[4 + joint * 2 + 1]
        if world:
            #  control wrt world frame? (virtual)
            joint_posn += self.bip7.State[4]
            #  add body tilt
            joint_vel += self.bip7.State[5]
            #  add body angular velocity
        torq[joint] = kp * (dposn - joint_posn) - kd * joint_vel

    # ////////////////////////////////////////////////////////
    #  PROC:  jointLimit()
    #  DOES:  enforces joint limits
    # ////////////////////////////////////////////////////////
    def jointLimit(self, torq, joint):
        """ generated source for method jointLimit """
        kpL = 800
        kdL = 80
        minAngle = self.con.jointLimit[0][joint]
        maxAngle = self.con.jointLimit[1][joint]
        currAngle = self.bip7.State[4 + joint * 2]
        currOmega = self.bip7.State[4 + joint * 2 + 1]
        if currAngle < minAngle:
            torq = kpL * (minAngle - currAngle) - kdL * currOmega
        elif currAngle > maxAngle:
            torq = kpL * (maxAngle - currAngle) - kdL * currOmega
        return torq

    def bip7WalkFsm(self, torq):
        """ generated source for method bip7WalkFsm """
        torsoIndex = 0
        rhipIndex = 1
        rkneeIndex = 2
        lhipIndex = 3
        lkneeIndex = 4
        rankleIndex = 5
        lankleIndex = 6
        worldFrame = [False, True, False, True, False, False, False]
        self.con.stateTime += self.Dt
        s = self.con.state[self.con.fsmState]
        computeMdMdd()
        n = 0
        while n < 7:
            target = self.boundRange(target, self.con.targetLimit[0][n], self.con.targetLimit[1][n])
            self.wPDtorq(torq, n, target, self.con.kp[n], self.con.kd[n], worldFrame[n])
            n += 1
        self.con.advance(self.bip7)

    def bip7Control(self, torq):
        """ generated source for method bip7Control """
        body = 0
        stanceHip = int()
        swingHip = int()
        fallAngle = 60
        n = 0
        while n < 7:
            torq[n] = 0
            n += 1
        if not self.bip7.lostControl:
            self.bip7WalkFsm(torq)
        if self.con.state[self.con.fsmState].leftStance:
            stanceHip = 3
            swingHip = 1
        else:
            stanceHip = 1
            swingHip = 3
        if not self.con.state[self.con.fsmState].poseStance:
            torq[stanceHip] = -torq[body] - torq[swingHip]
        torq[0] = 0
        n = 1
        while n < 7:
            torq[n] = self.boundRange(torq[n], self.con.torqLimit[0][n], self.con.torqLimit[1][n])
            self.jointLimit(torq[n], n)
            n += 1

    def computeMdMdd(self):
        """ generated source for method computeMdMdd """
        stanceFootX = self.bip7.getStanceFootXPos(self.con)
        self.Mdd = self.bip7.State[1] - self.DesVel
        self.Md = self.bip7.State[0] - stanceFootX

    def initComponents(self):
        """ generated source for method initComponents """
        self.simButton = javax.swing.JButton()
        self.reset = javax.swing.JButton()
        self.panel = javax.swing.JPanel()
        self.label = javax.swing.JLabel()
        self.label.setText("Speed: ")
        self.speedSlider = javax.swing.JSlider()
        self.speedSlider.setMaximum(100)
        self.speedSlider.setMinimum(0)
        self.speedSlider.setToolTipText("Adjust the speed of the simulation by adjusting this slider.")
        setLayout(BorderLayout())
        self.panel.setLayout(FlowLayout())
        add(self.panel, BorderLayout.NORTH)
        self.panel.add(self.label)
        self.panel.add(self.speedSlider)
        self.panel.add(self.simButton)
        self.panel.add(self.reset)
        self.speedSlider.addMouseListener(java.awt.event.MouseAdapter())
        self.simButton.setText("  Start  ")
        self.reset.setText("Reset")
        self.simButton.addActionListener(java.awt.event.ActionListener())
        self.reset.addActionListener(java.awt.event.ActionListener())

    def resetSimulation(self):
        """ generated source for method resetSimulation """
        self.bip7.resetBiped()
        self.con.stateTime = 0
        self.con.fsmState = 0
        self.con.currentGroupNumber = 0
        self.con.desiredGroupNumber = 0
        repaint()

    def runLoop(self):
        """ generated source for method runLoop """
        if self.simFlag == False:
            return
        self.timer.stop()
        i = 0
        while i < 200:
            self.bip7.computeGroundForces(self.gnd)
            self.bip7Control(self.bip7.t)
            self.bip7.runSimulationStep(self.Dt)
            self.timeEllapsed += self.Dt
            if self.timeEllapsed > self.DtDisp:
                self.update(self.getGraphics())
                self.timeEllapsed = 0
            i += 1
        self.timer.start()

    def update(self, g):
        """ generated source for method update """
        if g == None:
            return
        g2 = self.tempBuffer.getGraphics()
        g2.setColor(Color(255, 255, 255))
        g2.fillRect(0, 0, getSize().width - 1, getSize().height - 1)
        m = Matrix3x3.getTranslationMatrix(0, -300)
        m = m.multiplyBy(Matrix3x3.getScalingMatrix(float(100)))
        panX = self.bip7.State[0]
        panY = self.bip7.State[2]
        if self.shouldPanY == False:
            panY = 0
        m = m.multiplyBy(Matrix3x3.getTranslationMatrix(-panX + 1.5, -panY + 0.5))
        self.bip7.drawBiped(g2, m)
        self.gnd.draw(g2, m)
        g.drawImage(self.tempBuffer, 0, self.panel.getHeight(), self)
        self.panel.repaint()

    def paint(self, g):
        """ generated source for method paint """
        self.update(g)
        self.panel.repaint()

    def keyReleased(self, e):
        """ generated source for method keyReleased """

    def keyPressed(self, e):
        """ generated source for method keyPressed """
        if e.getKeyCode() == e.VK_LEFT:
            self.bip7.PushTime = 0.2
            self.bip7.PushForce = -60
        if e.getKeyCode() == e.VK_RIGHT:
            self.bip7.PushTime = 0.2
            self.bip7.PushForce = 60
        if e.getKeyChar() == 'r' or e.getKeyChar() == 'R':
            self.con.desiredGroupNumber = 1
        if e.getKeyChar() == 'w' or e.getKeyChar() == 'W':
            self.con.desiredGroupNumber = 0
        if e.getKeyChar() == 'c' or e.getKeyChar() == 'C':
            self.con.desiredGroupNumber = 2
        if e.getKeyChar() == '1':
            self.gnd.getFlatGround()
            self.resetSimulation()
        if e.getKeyChar() == '2':
            self.gnd.getComplexTerrain()
            self.resetSimulation()

    def keyTyped(self, e):
        """ generated source for method keyTyped """

    def mouseDragged(self, e):
        """ generated source for method mouseDragged """

    def mouseMoved(self, e):
        """ generated source for method mouseMoved """

    def mousePressed(self, e):
        """ generated source for method mousePressed """
        self.requestFocus()

    def mouseReleased(self, e):
        """ generated source for method mouseReleased """

    def mouseEntered(self, e):
        """ generated source for method mouseEntered """

    def mouseExited(self, e):
        """ generated source for method mouseExited """

    def mouseClicked(self, e):
        """ generated source for method mouseClicked """

    def destroy(self):
        """ generated source for method destroy """
        removeMouseListener(self)
        removeMouseMotionListener(self)

    def getAppletInfo(self):
        """ generated source for method getAppletInfo """
        return "Title: Simbicon\n" + "Author: Stelian Coros, Michiel van de Panne."

