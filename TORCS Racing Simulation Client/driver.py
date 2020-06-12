'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import copy
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# loaded_model_acceleration = pickle.load(open('acceleration_model.sav', 'rb'))
# loaded_model_steer = pickle.load(open('steer_model.sav', 'rb'))
# loaded_model_gear = pickle.load(open('gear_model.sav', 'rb'))

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 200
        self.prev_rpm = None
        self.data_acceleration = None
        self.data_steer = None
        self.data_gear = None
        self.real_data = None
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        # val = copy.deepcopy(msg)
        # print 'msg1 type',type(val),'\n'
        # print '\nmsg1',val,'\n'
        
        
        data_list = []
        
        speed = self.state.getSpeedX()
        if speed is not None:
            data_list.append(speed)
        #print 'speed: ',speed
        
        AngleToTrackAxis = self.state.getAngle()
        if AngleToTrackAxis is not None:
            data_list.append(AngleToTrackAxis)
        ##print 'AngleToTrackAxis: ',AngleToTrackAxis

        
        TrackEdgeSensors = self.state.getTrack()
        if TrackEdgeSensors is not None:
            for item in TrackEdgeSensors:
                if item is not None:
                    data_list.append(item)
        TrackEdgeSensors = np.asarray(TrackEdgeSensors)
        #print 'TrackEdgeSensors: ',TrackEdgeSensors
        
        FocusSensors = self.state.getFocusD()
        if FocusSensors is not None:
            for item in FocusSensors:
                if item is not None:
                    data_list.append(item)
        FocusSensors = np.asarray(FocusSensors)
        #print 'FocusSensors: ',FocusSensors
        
        Gear = self.state.getGear()
        if Gear is not None:
            data_list.append(Gear)
        #print 'Gear: ',Gear
        
        OpponentSensors = self.state.getOpponents()
        if OpponentSensors is not None:
            for item in OpponentSensors:
                if item is not None:
                    data_list.append(item)
        OpponentSensors = np.asarray(OpponentSensors)
        #print 'OpponentSensors: ',OpponentSensors
        
        RacePosition = self.state.getRacePos()
        if RacePosition is not None:
            data_list.append(RacePosition)
        #print 'RacePosition: ',RacePosition
        
        LateralSpeed = self.state.getSpeedY()
        if LateralSpeed is not None:
            data_list.append(LateralSpeed)
       # print 'LateralSpeed: ',LateralSpeed
        
        LapTime = self.state.getCurLapTime()
        if LapTime is not None:
            data_list.append(LapTime)
        #print 'LapTime: ',LapTime
        
        Damage = self.state.getDamage()
        if Damage is not None:
            data_list.append(Damage)
        #print 'Damage: ',Damage
        
        DistanceFromStartLine = self.state.getDistFromStart()
        if DistanceFromStartLine is not None:
            data_list.append(DistanceFromStartLine)
        #print 'DistanceFromStartLine: ',DistanceFromStartLine
        
        DistanceRaced = self.state.getDistRaced()
        if DistanceRaced is not None:
            data_list.append(DistanceRaced)
        #print 'DistanceRaced: ',DistanceRaced
        
        FuelLevel = self.state.getFuel()
        if FuelLevel is not None:
            data_list.append(FuelLevel)
       # print 'FuelLevel: ',FuelLevel
        
        LastLapTime = self.state.getLastLapTime()
        if LastLapTime is not None:
            data_list.append(LastLapTime)
        #print 'LastLapTime: ',LastLapTime
        
        RPM = self.state.getRpm()
        if RPM is not None:
            data_list.append(RPM)
       # print 'RPM: ',RPM
        
        TrackPosition = self.state.getTrackPos()
        if TrackPosition is not None:
            data_list.append(TrackPosition)
        #print 'TrackPosition: ',TrackPosition
        
        WheelSpinVelocity = self.state.getWheelSpinVel()
        if WheelSpinVelocity is not None:
            for item in WheelSpinVelocity:
                if item is not None:
                    data_list.append(item)
        WheelSpinVelocity = np.asarray(WheelSpinVelocity)
       # print 'WheelSpinVelocity: ',WheelSpinVelocity
        
        Z = self.state.getZ()
        if Z is not None:
            data_list.append(Z)
        #print 'Z: ',Z
        
        data = np.asarray(data_list)
        if data.size != 0:
            data = data.reshape(-1, 1)
            data = data.reshape(1, -1)
            # print 'data type: ',type(data)
            # print 'data shape: ',data.shape
           # print 'data: ',data
            self.real_data = copy.deepcopy(data) 
            

        
        loaded_model_acceleration = pickle.load(open('acceleration_model.sav', 'rb'))
        if data.size != 0:
              self.data_acceleration = loaded_model_acceleration.predict(data)
              #print 'The predicted acceleration is: ',loaded_model_acceleration.predict(data),'\n'
          
        loaded_model_steer = pickle.load(open('steer_model.sav', 'rb'))
        if data.size != 0:
              self.data_steer = loaded_model_steer.predict(data)
              #print 'The predicted steering is: ',loaded_model_steer.predict(data),'\n'
             
        loaded_model_gear = pickle.load(open('gear_model.sav', 'rb'))
        if data.size != 0:
              self.data_gear = loaded_model_gear.predict(data)
              #print 'The predicted gear is: ',loaded_model_gear.predict(data),'\n'
        
        #print'*'*20
        
        self.state.setFromMsg(msg)
        
        self.steer()
        
        self.gear()
        
        self.speed()
        
        return self.control.toMsg()
    
    def steer(self):
        # angle = self.state.angle
        # dist = self.state.trackPos
        
        if self.data_steer is not None:
            steer = self.data_steer
            TrackPosition = self.state.getTrackPos()
            if TrackPosition <= -.9:
                steer[0] = .13
            elif TrackPosition >= .9:
                steer[0] = -.13
            #steer = self.data_steer
            self.control.setSteer(steer[0])
        #elf.control.setSteer((angle - dist*0.5)/self.steer_lock)
            #self.control.setSteer(steer)
    
    def gear(self):
        # rpm = self.state.getRpm()
        # gear = self.state.getGear()
        
        # if self.prev_rpm == None:
        #     up = True
        # else:
        #     if (self.prev_rpm - rpm) < 0:
        #         up = True
        #     else:
        #         up = False
        
        # if up and rpm > 7000:
        #     gear += 1
        
        # if not up and rpm < 3000:
        #     gear -= 1
        
        if self.data_gear is not None:
            
            self.data_gear = self.data_gear + 0.5
            if int(self.data_gear) >= 3:
                self.control.setGear(3)
            else:
                self.control.setGear(int(self.data_gear))
        else:
            self.control.setGear(1)
        # if gear is None:
        #     gear = 1
        # if gear < 0:
        #     gear = 0
        
            
    
    def speed(self):
        speed = self.state.getSpeedX()
        # accel = self.control.getAccel()
        
        # if speed < self.max_speed:
        #     accel += 0.1
        #     if accel > 1:
        #         accel = 1.0
        # else:
        #     accel -= 0.1
        #     if accel < 0:
        #         accel = 0.0
        if self.data_acceleration is not None:
            accel =self.data_acceleration
            if accel < 0:
                accel = 0.1
            if accel > 1:
                accel = 1
            if speed >= self.max_speed:
                accel = 0.3
            self.control.setAccel(accel)
        else:
            self.control.setAccel(1)
        # if accel is None:
        #     accel = 1
        # if accel < 0:
        #     accel = 1
        # elif accel > 1:
        #     accel = 1
            
    
    # def getDataAcceleration(self)
    #     return self.data_acceleration
        
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
        