# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:36:05 2018
@author: Fabien Furfaro
"""
import pylab as plt
import os, cv2

import ctypes as c, numpy as np, binascii,serial, time

class ScheduleValv():
    def __init__(self, *arg):
        try : 
            conditions = arg[0]; number = arg[1]; t_fin = arg[2]; self.medium = arg[3]; self.path = arg[4]
            self.ch = {}
            for i in range(len(conditions)):
                # conditions : medium, t_start, width of pulse, freq
                t = np.linspace(0,t_fin,number)
                if conditions[i][2] != 0 :
                    signal = (t > conditions[i][0])*(((t - conditions[i][0]) % conditions[i][2]) < conditions[i][1]) #frequency pulses
                else :
                    if conditions[i][1] != 0 :
                        signal = (t > conditions[i][0])*(t < conditions[i][0] + conditions[i][1]) #one pulse
                    else :
                        signal = (t > conditions[i][0]) #step
                self.ch[str(i+1)] = signal
        except : 
            self.path = ""
            self.ch = {'1' : [0,0,0,0,0,1,1,1,1,1], '2' : [1,1,1,1,1,0,0,0,0,0]} # if no argument --> testing
            self.medium = ['i3','i4'] #0 = NA, 1 = AFK
            print('no arguments : conditions, number and t_fin')
        
        try : SN = ['ELVQS801','ELVQS97J']; self.valv = (Valve(SN[0]),Valve(SN[1]));print('Valve loaded')
        except : print('no valve!')
        
        self.medium = ['i3','i4'] #0 = NA, 1 = AFK
        # indice of valve (name,sn_indice,value) --> cablage
        self.valve_i = {'i1' : [0,8],'i2' : [0,9],'i3' : [0,10],'i4' : [0,11],'i5' : [0,12],'i6' : [0,13],
                        'i7' : [0,14],'i8' : [0,15],'i9' : [0,16],'i10' : [0,17],'i11' : [0,18],'i12' : [0,19],
                        'i13' : [0,20],'i14' : [0,21],'i15' : [0,22],'i16' : [0,23],'flush' : [1,0],
                        'MxOut' : [0,2],'BeadsIn' : [0,1],'Purge' : [0,0],'Pmp1' : [0,3],'Pmp2' : [0,4],
                        'Pmp3' : [0,5],'CellsIn' : [1,3],'3b' : [1,8],'3a' : [1,9],'2b' : [1,10],'2a' : [1,11],
                        '1c' : [1,12],'1b' : [1,14],'1a' : [1,13],'0c' : [1,15],'0a' : [1,16],'0b-flush' : [1,17],
                        'ChIn' : [1,2],'ChOut' : [1,1],'Wast1' : [0,7],'Wast2' : [0,6]}
        self.plan = []

    def applyS(self, n):
        
        data = open(self.path, "a")
        self.plan = []
        for i in range(len(self.ch)):
            self.plan += [[self.ch[str(i+1)][n], i+1]]

        sep = np.asarray(self.plan)[:,0].argsort()
        self.plan = np.asarray(self.plan)[sep]
        self.purgeStep('flush');self.closeStep('flush')
        data.write('\n' + str(time.time()) + '_' + 'flush')
        
        a0 = [i[1] for i in self.plan if i[0] == 0]
        self.feedChamber(a0,self.medium[0]) #no activation medium
        data.write('\n' + str(time.time()) + '_' + 'feedChamberBy_' + self.medium[0] + '_' + str(self.plan))
        self.purgeStep('flush');self.closeStep('flush')
        data.write('\n' + str(time.time()) + '_' + 'flush')
        
        a1 = [i[1] for i in self.plan if i[0] == 1]
        self.feedChamber(a1,self.medium[1]) #activation medium
        data.write('\n' + str(time.time()) + '_' + 'feedChamberBy_' + self.medium[1] + '_' + str(self.plan))
        self.purgeStep('flush');self.closeStep('flush')
        data.write('\n' + str(time.time()) + '_' + 'flush')
        
        data.close()
    
    def setValve(self,statenameList):
        for stn in statenameList :
            state = int(stn[0])
            name = stn[1]
            p = self.valve_i[name]
            self.valv[p[0]].SetBit(self.valv[p[0]].listPin[p[1]],state)    
    
    def purgeStep(self,inVALV):
        self.setValve([[1,inVALV]])
        time.sleep(0.5)
        self.setValve([[1,'MxOut'],[1,'Purge']])
        time.sleep(1)
        self.setValve([[0,'Purge'],[1,'Pmp1'],[1,'Pmp2'],[1,'Pmp3']])
        time.sleep(0.5)
        self.setValve([[0,'0a'],[1,'0b-flush'],[0,'0c'],[1,'1a'],[1,'1b'],[1,'1c'],[1,'2a'],[1,'2b'],[1,'3a'],[1,'3b']])
        time.sleep(0.5)
        self.setValve([[1,'Wast1'],[1,'Wast2']])
        time.sleep(3)
    
    def closeStep(self,inVALV):
        self.setValve([[0,inVALV],[0,'MxOut'],[0,'Pmp1'],[0,'Pmp2'],[0,'Pmp3'],[0,'0b-flush'],[0,'0c'],[0,'1a'],[0,'1b'],[0,'1c'],[0,'2a'],[0,'2b'],[0,'3a'],[0,'3b']])
        time.sleep(0.5)
    
    def feedOneChamber(self,ch):
        bitsV = np.asarray(['0a','0b-flush','0c','1a','1b','1c','2a','2b','3a','3b','4a','4b','5a','5b'])
        mb = self.muxBits(ch)
        m = np.concatenate((mb[:-4,None],bitsV[:-4,None]), axis = 1).tolist()
        self.setValve(m)
        
    def feedAllChamber(self,listCh,I):
        self.purgeStep(I)
        self.setValve([[1,'ChIn'],[1,'ChOut']])
        for l in listCh :
            self.feedOneChamber(l)
            time.sleep(2.5)
        self.setValve([[0,'ChIn'],[0,'ChOut']])
        self.closeStep(I)
    
    def muxBits(self, chamber):
        ch = chamber
        if ch <= 48 : ch = 49 - ch
        ch = ch - 1
        bits = ['0a','0b','0c','1a','1b','1c','2a','2b','3a','3b','4a','4b','5a','5b'] # valve indices
        div = 48
        #arbre binaire
        for i in np.arange(13,6,-2):
            v = ch <= (div-1); bits[i] = not(v); bits[i-1] = v
            ch = ch % div; div = int(div/2)
        #arbre 3 branches n°1
        def switch1(x):
            return { 0 : [1,0,0], 1 : [1,0,0], 2 : [0,1,0],
                    3 : [0,1,0], 4 : [0,0,1], 5 : [0,0,1]}.get(x, [1,1,1])
        bits[3:6] = switch1(ch)
        ch = ch % 2
        #arbre 3 branches n°2
        def switch0(x): return {0 : [1,0,0], 1 : [0,0,1]}.get(x, [1,0,1])

        bits[:3] = switch0(ch)
        return np.asarray(bits).astype('int')
    
    def close(self):
        self.valv[0].close()
        self.valv[1].close()

os.chdir(r"C:\MicroManager\1.4.16")
import MMCorePy

"""
---FT_STATUS (DWORD) : 
FT_OK = 0
FT_INVALID_HANDLE = 1
FT_DEVICE_NOT_FOUND = 2
FT_DEVICE_NOT_OPENED = 3
"""
class Valve():
    
    def __init__(self, SerialNumber):
        self._lib = c.cdll.LoadLibrary("ftd2xx64.dll") #copy this dll on Micromanager folder
        
        #Open by sn :
        FT_OPEN_BY_SERIAL_NUMBER = 1
        dw_flags = c.c_ulong(FT_OPEN_BY_SERIAL_NUMBER)
        self.ftHandle = c.c_ulong()
        status = self._lib.FT_OpenEx(SerialNumber, dw_flags, c.byref(self.ftHandle)); print('Open status :' + str(status))
        
        #Setup :
        self._lib.FT_ResetDevice(self.ftHandle)
        self._lib.FT_SetBaudRate(self.ftHandle, c.c_ulong(921600))
        self._lib.FT_SetDataCharacteristics(self.ftHandle, c.c_uint(8), c.c_uint(0), c.c_uint(0))
        self._lib.FT_SetFlowControl(self.ftHandle, c.c_uint(0), c.c_uint(0), c.c_uint(0))
        self._lib.FT_Purge(self.ftHandle, c.c_uint(1))
        self._lib.FT_Purge(self.ftHandle, c.c_uint(2))
        self._lib.FT_SetDtr(self.ftHandle)
        self._lib.FT_SetRts(self.ftHandle)
        
        null = binascii.unhexlify('00')
        lpBuffer = '!A'+ null + '!B' + null + '!C' + null
        num_bytes = len(lpBuffer)
        self.lpdwBytesWritten = c.c_ulong()
        status = self._lib.FT_Write(self.ftHandle, lpBuffer, num_bytes, c.byref(self.lpdwBytesWritten)) ; print('Setup status :' + str(status))
        
        self.listPin = ['\x00','\x01','\x02','\x03','\x04','\x05',
                        '\x06','\x07','\x08','\x09','\x0a','\x0b',
                        '\x0c','\x0d','\x0e','\x0f','\x10','\x11',
                        '\x12','\x13','\x14','\x15','\x16', '\x17']  
        
    def SetBit(self, pin, state):
        #Set_bits :
        ON, OFF = 'H', 'L'
        if (state == 1) : STATE = ON
        else : STATE = OFF
        null = binascii.unhexlify('00')
        #Pin_index = binascii.unhexlify(str(pin).zfill(2))
        lpBuffer = STATE + pin + null # 3 bits
        num_bytes = len(lpBuffer);
        self.lpdwBytesWritten = c.c_ulong()
        self._lib.FT_Write(self.ftHandle, lpBuffer, num_bytes, c.byref(self.lpdwBytesWritten))

    def purgeValve(self):
        for i in self.listPin :
            self.SetBit(i,1); time.sleep(0.2); self.SetBit(i,0)

    def close(self):
        self._lib.FT_Close(self.ftHandle)

class LStep():
    
    def __init__(self, COM):
        COM = 'COM5'
        self.ser = serial.Serial(COM, 9600, timeout=0, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_TWO)
        self.activeJoystick = binascii.unhexlify(str('55076A0D55500D55430D'))
        self.desactiveJoystick = binascii.unhexlify(str('6A0D'))
        self.mode = True #if stop without close
        self.readX = binascii.unhexlify(str('55430D'))
        self.readY = binascii.unhexlify(str('55440D'))
        self.c = []
        for i in ['00','01','07','0D']:
            self.c += binascii.unhexlify(i)

    def activateJoystick(self):
        self.ser.read_all() #purge buffer
        self.ser.write(self.activeJoystick)
        self.ser.read_all() #purge buffer
        self.mode = True
    
    def desactivateJoystick(self):
        self.ser.read_all() #purge buffer
        self.ser.write(self.desactiveJoystick)
        self.ser.read_all() #purge buffer
        self.mode = False
    
    def readXYposition(self):
        self.ser.read_all() #purge buffer
        self.ser.write(self.readX)
        time.sleep(0.1)
        x = self.ser.read_all()
        self.ser.write(self.readY)
        time.sleep(0.1)
        y = self.ser.read_all()
        return [x,y]
    
    def movetoRelativePosXY(self,X,Y):
        if self.mode == True : self.desactivateJoystick(); self.mode = True
        X,Y = str(X), str(Y)
        self.ser.read_all() #purge buffer
        # r = relative; v = absolute
        cmd = 'U'+ self.c[0] + X + self.c[-1] +'U'+ self.c[1] + Y + self.c[-1] +'U'+ self.c[-2] +'r'+ self.c[-1] +'UP'+ self.c[-1]
        self.ser.write(cmd)
        time.sleep(1)
        if self.mode == True : self.activateJoystick()
    
    def closeCommunication(self):
        self.desactivateJoystick()
        self.ser.close()
        
class XCite():
    def __init__(self, COM):
        COM = 'COM4'
        self.ser = serial.Serial(COM, 9600, timeout=0, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
    
    def setIntensity(self,I):
        I = str(I); self.ser.read_all()
        if I == '100' : self.ser.write(binascii.unhexlify(str('69340D')))
        elif I == '50' : self.ser.write(binascii.unhexlify(str('69330D')))
        elif I == '25' : self.ser.write(binascii.unhexlify(str('69320D')))
        elif I == '12' : self.ser.write(binascii.unhexlify(str('69310D')))
        elif I == '0' : self.ser.write(binascii.unhexlify(str('69300D')))
        else : print('xcite Invalid value')

    def openShutter(self):
        self.ser.read_all()
        self.ser.write(binascii.unhexlify(str('6D6D0D')))
    
    def closeShutter(self):
        self.ser.read_all()
        self.ser.write(binascii.unhexlify(str('7A7A0D')))
    
    def closeCommunication(self):
        self.setIntensity(0)
        self.closeShutter()
        self.ser.close()

class OlympusIX81():
    def __init__(self, COM):
        COM = 'COM1'
        self.ser = serial.Serial(COM, 19200, timeout=0, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_TWO)
        init_ix81 = '314C4F4720494E0D0A'+'324C4F4720494E0D0A' # 1LOG IN + 2LOG IN
        self.ser.write(binascii.unhexlify(init_ix81))
        self.ser.read_all() #flush buffer if exist
        manualFocus = '324A4F47204F4E0D0A'+'314A4F4753454C2046480D0A' # 2JOG ON + 1JOSEL FH
        self.ser.write(binascii.unhexlify(manualFocus))
        self.ser.read_all() #flush buffer if exist
        self.z = None

    def filterCube(self,filterName):
        self.ser.read_all()
        mu = [binascii.unhexlify('314D553F0D0A'+'314D5520'), binascii.unhexlify('0D0A')] #([1MU?\r\n 1MU],[\r\n])
        if filterName == 'RFP' : self.ser.write(mu[0]+binascii.unhexlify('32')+mu[1])
        elif filterName == 'CFP' : self.ser.write(mu[0]+binascii.unhexlify('34')+mu[1])
        elif filterName == 'YFP' : self.ser.write(mu[0]+binascii.unhexlify('31')+mu[1])
        elif filterName == 'GFP' : self.ser.write(mu[0]+binascii.unhexlify('33')+mu[1])
        elif filterName == 'Position-6' : self.ser.write(mu[0]+binascii.unhexlify('36')+mu[1])
        else : print('FilterCube Invalid value')
        time.sleep(1)
    
    def setI(self,I):
        # à coder !!
        self.ser.read_all()
        mV = int(I*10)
        self.ser.write('1LMP '+str(mV).zfill(3))
    
    def LampState(self,State):
        # à coder !!
        self.ser.read_all()
        if State == True :
            self.ser.write('1LMPSW ON')
        elif State == False :
            self.ser.write('1LMPSW OFF')
    
    def getZ(self):
        self.ser.read_all()
        self.ser.write(binascii.unhexlify('32504F533F0D0A')) # 2POS?
        time.sleep(0.1) #sometime it's long !
        self.z = self.ser.read_all()[5:-2]
        return self.z
    
    def setZ(self,Znew):
        Zold = self.getZ()
        DELTA = int(Znew) - int(Zold)
        mov = [binascii.unhexlify('324D4F56'), binascii.unhexlify('0D0A')] # 2MOV'
        if DELTA > 0 : self.ser.write(mov[0] + binascii.unhexlify('204E2C') + str(DELTA) + mov[1]) #N
        elif DELTA < 0 : self.ser.write(mov[0] + binascii.unhexlify('20462C') + str(abs(DELTA)) + mov[1]) #F
        #ne marche pas encore pour descendre !
        #do nothing if DELTA == 0
        time.sleep(1)
    
    def closeCommunication(self):
        self.ser.close()
        #Oubli instruction ENDLOG !

class Sherlock():
    def __init__(self):
        
        try : COM = 'COM1'; self.olympus = OlympusIX81(COM); print('Olympus loaded')
        except : print('no config olympus!')
            
        try : 
            self.mmc = MMCorePy.CMMCore() #micromanager
            self.mmc.loadDevice('Camera','AndorSDK3','Andor sCMOS Camera')
            self.mmc.initializeDevice('Camera')
            self.mmc.setCameraDevice('Camera')
            
            self.cam = self.mmc.getCameraDevice()
            A = self.mmc.getAllowedPropertyValues(self.cam,'Sensitivity/DynamicRange')
            self.mmc.setProperty(self.cam,'Sensitivity/DynamicRange',A[2])  #camera en 16bits
            print('Camera with micromanager loaded')
        except : print('no Cam!')
            
        try : COM = 'COM4'; self.xcite = XCite(COM); print('Xcite loaded')
        except : print('no Fluo!')
        
        try : COM = 'COM5'; self.lstep = LStep(COM); print('LStep loaded')
        except : print('no Lstep!')
    
    def snapPLOT(self, *arg):
        self.mmc.snapImage()
        img = self.mmc.getImage()
        plt.imshow(img)
        return img
    
    def firstPOS(self,number):
        Pos = []
        for n in range(number):
            self.liveCV(50)
            xy = self.lstep.readXYposition()
            z = self.olympus.getZ()
            Pos += [xy + [z]]; print(len(Pos),Pos[-1])
        return Pos
    
    def ajustPOS(self,posList):
        newPos = []
        for p in posList :
            self.lstep.movetoRelativePosXY(p[0],p[1])
            self.olympus.setZ(p[2])
            self.liveCV(100)
            xy = self.lstep.readXYposition()
            z = self.olympus.getZ()
            newPos += [xy + [z]]; print(len(newPos),newPos[-1])
        return newPos
    
    def liveCV(self,exposure):
        exposure = str(exposure)
        self.lstep.activateJoystick()
        self.mmc.setProperty(self.cam,'Exposure', exposure)
        self.mmc.startContinuousSequenceAcquisition(1)
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 800,800)
        while(True):
            frame = 50*self.mmc.getLastImage()
            cv2.imshow('frame',cv2.flip(cv2.rotate(frame,0),1))
            if cv2.waitKey(25) & 0xFF == ord('w'):
                break  # Touche w pour quitter la boucle!
            if cv2.waitKey(25) & 0xFF == ord('o'):
                self.xcite.openShutter()
            if cv2.waitKey(25) & 0xFF == ord('c'):
                self.xcite.closeShutter()
            if cv2.waitKey(25) & 0xFF == ord('r'):
                self.olympus.filterCube('RFP')
            if cv2.waitKey(25) & 0xFF == ord('t'):
                self.olympus.filterCube('Position-6')
            if cv2.waitKey(25) & 0xFF == ord('i'):
                print('max : ', np.max(frame))
                print('min : ', np.max(frame))
        cv2.destroyAllWindows()
        self.mmc.stopSequenceAcquisition()  
        
    def setXYZpos(self,x):
        self.lstep.movetoRelativePosXY(x[0],x[1])
        self.olympus.setZ(x[2])
    
    def close(self):
        self.lstep.closeCommunication()
        self.xcite.closeCommunication()
        self.olympus.closeCommunication()

#TEST :
#mm_s = MM_Sherlock()
#import threading
#t = threading.Thread(target=mm_s.LiveCV())
"""
m = Sherlock()
sv = ScheduleValv()
for i in range(10000000):
    for v in sv.valv :
        v.purgeValve()

"""
