'''
 selection.py

 This script contains several functions that implement the some event selection. 
 The functinos defined here can be used to define a selection, signal/control region, etc.
 The functions are called with (jagged)arrays as imputs plus some custom paramenters and return a boolean mask.

 The functions are stored in a dictionary and saved in a '.coffea' file.

 Usage:
 >> python selection.py

'''


import os, sys
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)
import uproot, uproot_methods
import numpy as np
from coffea.arrays import Initialize
from coffea import hist, lookup_tools
from coffea.util import save

outdir  = basepath+'coffeaFiles/'
outname = 'selection'
seldic = {}

def passNJets(nJets, lim=2):
  return nJets >= lim

def passMETcut(met, metCut=40):
  return met >= metCut

# Datasets:
# SingleElec, SingleMuon
# DoubleElec, DoubleMuon, MuonEG
# Overlap removal at trigger level... singlelep, doublelep, triplelep

triggers = {
  'SingleMuonTriggers' : ['IsoMu24', 'IsoMu27'],
  'SingleElecTriggers' : ['Ele32_WPTight_Gsf_L1DoubleEG', 'Ele35_WPTight_Gsf'],
  'DoubleMuonTrig' : ['Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8'],
  'DoubleElecTrig' : ['Ele23_Ele12_CaloIdL_TrackIdL_IsoVL', 'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'],
  'MuonEGTrig' : ['Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL', 'Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ'],
  'TripleElecTrig' : ['Ele16_Ele12_Ele8_CaloIdL_TrackIdL'],
  'TripleMuonTrig' : ['TripleMu_12_10_5'],
  'DoubleMuonElecTrig' : ['DiMu9_Ele9_CaloIdL_TrackIdL_DZ'],
  'DoubleElecMuonTrig' : ['Mu8_DiEle12_CaloIdL_TrackIdL'],
}

triggersForFinalState = {}
triggersNotForFinalState = {}


def changeHLT():
  triggers['DoubleMuonTrig'] = ['Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8']
  triggers['MuonEGTrig'] = ['Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ']
  triggers['SingleElecTriggers'] = ['Ele32_WPTight_Gsf_L1DoubleEG', 'Ele35_WPTight_Gsf']

def updateStates():
  triggersForFinalState['ee'] = {
      'MC': triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
      'SingleElectron'     : triggers['SingleElecTriggers'],
      'DoubleEG'     : triggers['DoubleElecTrig'],
  }
  triggersForFinalState['em'] = {
      'MC': triggers['SingleElecTriggers']+triggers['SingleMuonTriggers']+triggers['MuonEGTrig'],
      'SingleElectron'     : triggers['SingleElecTriggers'],
      'MuonEG'     : triggers['MuonEGTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  }
  triggersForFinalState['mm'] = {
      'MC': triggers['SingleMuonTriggers']+triggers['DoubleMuonTrig'],
      'DoubleMuon' : triggers['DoubleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  }
  triggersForFinalState['eee'] = {
      'MC': triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'SingleElectron' : triggers['SingleElecTriggers'],
      'DoubleEG' : triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
  }
  triggersForFinalState['mmm'] = {
      'MC': triggers['SingleMuonTriggers']+triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
      'DoubleMuon' : triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  }
  triggersForFinalState['eem'] = {
      'MC': triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'SingleElectron' : triggers['SingleElecTriggers'],
      'DoubleEG' : triggers['DoubleElecTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  }
  triggersForFinalState['mme'] = {
      'MC': triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleMuonTrig']+triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'SingleElectron' : triggers['SingleElecTriggers'],
      'DoubleMuon' : triggers['DoubleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  }
  triggersForFinalState['eeee'] = {
      'MC': triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'SingleElectron' : triggers['SingleElecTriggers'],
      'DoubleEG' : triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
  }
  triggersForFinalState['mmmm'] = {
      'MC': triggers['SingleMuonTriggers']+triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
      'DoubleMuon' : triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  }
  triggersForFinalState['eeem'] = {
      'MC': triggers['TripleElecTrig']+triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'SingleElectron' : triggers['SingleElecTriggers'],
      'DoubleEG' : triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  }
  triggersForFinalState['eemm'] = {
      'MC': triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['DoubleMuonTrig']+triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig'],
      'SingleElectron' : triggers['SingleElecTriggers'],
      'DoubleEG' : triggers['DoubleElecTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
      'DoubleMuon' : triggers['DoubleMuonTrig'],
  }
  triggersForFinalState['mmme'] = {
      'MC': triggers['TripleMuonTrig']+triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleMuonTrig']+triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'SingleElectron' : triggers['SingleElecTriggers'],
      'DoubleMuon' : triggers['TripleMuonTrig']+triggers['DoubleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],   
  }

  triggersNotForFinalState['ee'] = {
      'SingleElectron' : triggers['DoubleElecTrig'],
      'DoubleEG' : [],
  }
  triggersNotForFinalState['em'] = {
      'MuonEG'     : [],
      'SingleElectron' : triggers['MuonEGTrig'],
      'SingleMuon' : triggers['MuonEGTrig'],
  }
  triggersNotForFinalState['mm'] = {
      'DoubleMuon' : [],
      'SingleMuon' : triggers['DoubleMuonTrig'],
  }
  triggersNotForFinalState['eee'] = { 
      'SingleElectron' : triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'DoubleEG' : [],
  }
  triggersNotForFinalState['mmm'] = {
      'DoubleMuon' : [],
      'SingleMuon' : triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
  }
  triggersNotForFinalState['eem'] = {
      'MuonEG' : [], 
      'SingleElectron' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['SingleMuonTriggers']+triggers['DoubleElecTrig'],
      'DoubleEG' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['SingleMuonTriggers'],
      'SingleMuon' :  triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
  }
  triggersNotForFinalState['mme'] = {
      'MuonEG' : [],
      'SingleElectron' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleMuonTriggers']+triggers['DoubleElecTrig'],
      'DoubleMuon' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers'],
      'SingleMuon' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleMuonTrig'],
  }
  triggersNotForFinalState['eeee'] = {
      'SingleElectron' : triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'DoubleEG' : [],
  }
  triggersNotForFinalState['mmmm'] = {
      'DoubleMuon' : [],
      'SingleMuon' : triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],   
  }
  triggersNotForFinalState['eeem'] = {
      'MuonEG' : [], 
      'SingleElectron' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['SingleMuonTriggers']+triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'DoubleEG' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['SingleMuonTriggers'],
      'SingleMuon' :  triggers['TripleElecTrig']+triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],   
  }
  triggersNotForFinalState['eemm'] = {
      'MuonEG' : [], 
      'SingleElectron' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleMuonTriggers']+triggers['DoubleMuonTrig']+triggers['DoubleElecTrig'],
      'SingleMuon' :  triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['DoubleMuonTrig'], 
      'DoubleEG' :  triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleMuonTriggers']+triggers['DoubleMuonTrig'],
      'DoubleMuon' :  triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
  }
  triggersNotForFinalState['mmme'] = {
      'MuonEG' : [],
      'SingleElectron' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'DoubleMuon' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers'],
      'SingleMuon' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],   
  }


def PassTrigger(df, cat, isData=False, dataName=''):
  tpass = np.zeros_like(df.MET.pt, dtype=np.bool)
  df = df.HLT
  if not isData:
    updateStates()
    paths = triggersForFinalState[cat]['MC']
    for path in paths: tpass = tpass | df[path]
  else:
    if min(df['run']) < 299368:
      changeHLT()
    updateStates()
    passTriggers    = triggersForFinalState[cat][dataName] if dataName in triggersForFinalState[cat].keys() else []
    notPassTriggers = triggersNotForFinalState[cat][dataName] if dataName in triggersNotForFinalState[cat].keys() else []
    for path in passTriggers: 
      tpass = tpass | df[path]
    for path in notPassTriggers: 
      tpass = (tpass)&(df[path]==0)
  return tpass


seldic['passNJets' ] = passNJets
seldic['passMETcut'] = passMETcut
seldic['passTrigger'] = PassTrigger

if not os.path.isdir(outdir): os.system('mkdir -p ' + outdir)
save(seldic, outdir+outname+'.coffea')
