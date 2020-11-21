'''
 objects.py

 This script contains several functions that implement the object selection according to different object definitions.
 The functions are called with (jagged)arrays as imputs and return a boolean mask.

 The functions are stored in a dictionary and saved in a '.coffea' file.

 Usage:
 >> python objects.py

'''


import os, sys
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)
import awkward
import uproot, uproot_methods
import numpy as np
from coffea.arrays import Initialize
from coffea import hist, lookup_tools
from coffea.util import save

outdir  = basepath+'coffeaFiles/'
outname = 'objects'

def isTightMuonPOG(pt,eta,dxy,dz,iso,tight_id, tightCharge, year):
    #dxy and dz cuts are baked on tight_id; tight isolation is 0.15
    mask = ~(pt==np.nan)#just a complicated way to initialize a jagged array with the needed shape to True
    mask = (pt>10)&(abs(eta)<2.5)&(tight_id)&(tightCharge)&(iso<0.15)
    return mask

def isTightElectronPOG(pt,eta,dxy,dz,tight_id,tightCharge,year):
    mask = ~(pt==np.nan)#just a complicated way to initialize a jagged array with the needed shape to True
    mask = ((pt>10)&(abs(eta)<2.5)&(tight_id==4)&(tightCharge)) # Trigger: HLT_Ele27_WPTight_Gsf_v
    return mask
   
def isClean(obj_A, obj_B, drmin=0.4):
   ABpairs = obj_A.cross(obj_B, nested=True)
   ABgoodPairs = (ABpairs.i0.delta_r(ABpairs.i1) > drmin).all()
   return ABgoodPairs

def isTightJet(pt, eta, jet_id, neHEF, neEmEF, chHEF, chEmEF, nConstituents, jetPtCut=30.0):
    mask = (pt>jetPtCut) & (abs(eta)<2.4)# & ((jet_id&6)==6)
    loose = (pt>0)#(neHEF<0.99)&(neEmEF<0.99)&(chEmEF<0.99)&(nConstituents>1)
    tight = (neHEF<0.9)&(neEmEF<0.9)&(chHEF>0.0)
    jetMETcorrection = (pt>0)#(neEmEF + chEmEF < 0.9)
    mask = mask & loose & tight & jetMETcorrection
    return mask

def isCleanJet(jets, electrons, muons, taus, drmin=0.4):
  ''' Returns mask to select clean jets '''
  epairs = jets.cross(electrons, nested=True)
  mpairs = jets.cross(muons, nested=True)
  tpairs = jets.cross(taus, nested=True)
  egoodPairs = (epairs.i0.delta_r(epairs.i1) > drmin).all()
  mgoodPairs = (mpairs.i0.delta_r(mpairs.i1) > drmin).all()
  tgoodPairs = (tpairs.i0.delta_r(tpairs.i1) > drmin).all()
  return (egoodPairs) & (mgoodPairs)# & (tgoodPairs)
  
def isPresMuon(dxy, dz, sip3D, looseId):
  mask = (abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(looseId)
  return mask
  
def isTightMuon(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, mediumPrompt, tightCharge, looseId, minpt=10.0):
  mask = (pt>minpt)&(abs(eta)<2.5)&(abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(looseId)#&(miniIso<0.25)#&(mvaTTH>0.90)&(tightCharge==2)&(mediumPrompt)
  return mask

def isPresElec(pt, eta, dxy, dz, miniIso, sip3D, lostHits):
  mask = (pt>minpt)&(abs(eta)<2.5)&(abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(lostHits<=1)#&(eInvMinusPInv>-0.04)&(maskhoe)&(miniIso<0.25)
  return mask
 
def isTightElec(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, elecMVA, lostHits, convVeto, tightCharge, sieie, hoe, eInvMinusPInv, minpt=15.0):
  maskPOGMVA = ((pt<10)&(abs(eta)<0.8)&(elecMVA>-0.13))|((pt<10)&(abs(eta)>0.8)&(abs(eta)<1.44)&(elecMVA>-0.32))|((pt<10)&(abs(eta)>1.44)&(elecMVA>-0.08))|\
               ((pt>10)&(abs(eta)<0.8)&(elecMVA>-0.86))|((pt>10)&(abs(eta)>0.8)&(abs(eta)<1.44)&(elecMVA>-0.81))|((pt>10)&(abs(eta)>1.44)&(elecMVA>-0.72))
  maskSieie  = ((abs(eta)<1.479)&(sieie<0.011))|((abs(eta)>1.479)&(sieie<0.030))
  maskhoe    = ((abs(eta)<1.479)&(hoe<0.10))|((abs(eta)>1.479)&(hoe<0.07))
  mask = (pt>minpt)&(abs(eta)<2.5)&(abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(lostHits<=1)&\
         (convVeto)&(maskSieie)#&(maskPOGMVA)&(eInvMinusPInv>-0.04)&(maskhoe)&(miniIso<0.25)#&(mvaTTH>0.90)&(tightCharge==2)
  return mask 
 
def isCleanElec(e, mu, drmin=0.05):
  empairs = e.cross(mu, nested=True)
  emgoodPairs = (empairs.i0.delta_r(empairs.i1) > drmin).all()
  return emgoodPairs
 
def isPresTau(pt, eta, minpt=25.0):
  mask = (pt>minpt)&(abs(eta)<2.4)
  return mask

ids = {}
ids['isTightMuonPOG'] = isTightMuonPOG
ids['isTightElectronPOG'] = isTightElectronPOG
ids['isClean'] = isClean
ids['isPresMuon'] = isPresMuon
ids['isTightMuon'] = isTightMuon
ids['isPresElec'] = isPresElec
ids['isTightElec'] = isTightElec
ids['isCleanElec'] = isCleanElec
ids['isPresTau']  = isPresTau
ids['isTightJet'] = isTightJet
ids['isCleanJet']   = isCleanJet

if not os.path.isdir(outdir): os.system('mkdir -p ' + outdir)
save(ids, outdir+outname+'.coffea')
