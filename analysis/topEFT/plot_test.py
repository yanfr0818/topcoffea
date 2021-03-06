#Arg settings: -v <var> -c <channel> -l <cut> -t <title>
#For example:
#  python plot_test.py -c em
#  python plot_test.py -v met -c ch4l -l 4j2b
#  python plot_test.py -v m3l -l 2jets

from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import gzip
import pickle
import json
import os
import uproot
import matplotlib.pyplot as plt
import numpy as np
from coffea import hist, processor 
from coffea.hist import plot
import os, sys
import optparse
import re

basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)

path = 'histos/plotsTopEFT.pkl.gz'

processDic = {
  #'Nonprompt' : 'TTTo2L2Nu, tW_noFullHad, tbarW_noFullHad, WJetsToLNu_MLM, WWTo2L2Nu',
  #'DY' : 'DYJetsToLL_M_10to50_MLM, DYJetsToLL_M_50_a',
  #'Other': 'WWW,WZG,WWZ,WZZ,ZZZ,tttt,ttWW,ttWZ,ttZH,ttZZ,ttHH,tZq,TTG',
  #'WZ' : 'WZTo2L2Q,WZTo3LNu',
  #'ZZ' : 'ZZTo2L2Nu,ZZTo2L2Q,ZZTo4L',
  #'ttW': 'TTWJetsToLNu',
  #'ttZ': 'TTZToLL_M_1to10,TTZToLLNuNu_M_10_a',
  'ttH' : 'ttHnobb',#tHq',
  #'data' : 'EGamma, SingleMuon, DoubleMuon',
}
bkglist = ['ttH']# 'Other', 'DY',  'ttH', 'WZ', 'ZZ', 'ttZ', 'ttW']
allbkg  = ['ttH']#['tt', 'tW', 'WW', 'ttG', 'WW', 'WJets', 'Other', 'DY',  'ttH', 'WZ', 'ZZ', 'ttZ', 'ttW']

colordic ={
  'Other' : '#808080',
  'DY' : '#fbff00',
  'WZ' : '#ffa200',
  'ZZ' : '#8fff00',
  'ttW': '#00a278',
  'ttZ': '#6603ab',
  'VVV' : '#c688b4',
  'tttt' : '#0b23f0',
  'Nonprompt' : '#0b23f0',
  'ttVV' : '#888db5',
  'tHq' : '#5b0003',
  'ttH' : '#f00b0b',
  'tZq' : '#00065b',
  'tt' : '#0b23f0',
  'tW' : '#888db5',
  'ttG' : '#5b0003',
  'WW' : '#f00b0b',
  'WJets' : '#00065b',
}

preset = {
  'ch4l'      : ['eeemSSonZ', 'eeemSSoffZ', 'mmmeSSonZ', 'mmmeSSoffZ', 'eemmSSonZ', 'eemmSSoff', 'eeeeSSonZ', 'eeeeSSoffZ', 'mmmmSSonZ', 'mmmmSSoffZ'],
  'ch3l'      : ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ'],
  'ch3lSSonZ' : ['eemSSonZ', 'mmeSSonZ', 'eeeSSonZ', 'mmmSSonZ'],
  'ch2lss'    : ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS'],
}
preset['ch3lp']      = [x+'_p' for x in preset['ch3l']]
preset['ch3lm']      = [x+'_m' for x in preset['ch3l']]
preset['ch3lSSonZp'] = [x+'_p' for x in preset['ch3lSSonZ']]
preset['ch3lSSonZm'] = [x+'_m' for x in preset['ch3lSSonZ']]
preset['ch2lssp']    = [x+'_p' for x in preset['ch2lss']]
preset['ch2lssm']    = [x+'_m' for x in preset['ch2lss']]


usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-v', '--variable',  dest='variable',  help='variable',   default='counts',    type='string')
parser.add_option('-c', '--channel',   dest='channel',   help='channels',   default='ch3l',      type='string')
parser.add_option('-l', '--level',     dest='level',     help='cut',        default='base',      type='string')
parser.add_option('-t', '--title',     dest='title',     help='title',      default='3 leptons', type='string')
(opt, args) = parser.parse_args()

for keys in preset:
    if opt.channel == keys:
        opt.channel = preset[keys]
else                           : channel = opt.channel
level = opt.level


categories = {
 'channel' : channel,
 'cut'     : level    #['base', '2jets', '4jets', '4j1b', '4j2b'],
 #'Zcat' : ['onZ', 'offZ'],
 #'lepCat' : ['3l'],
}

colors = [colordic[k] for k in bkglist]

from plotter.plotter import plotter

def Draw(var, categories, label=''):
  print(categories['channel'])
  plt = plotter(path, prDic=processDic, bkgList=bkglist)
  plt.plotData = True
  plt.SetColors(colors)
  plt.SetCategories(categories)
  plt.SetRegion(label)
  #plt.SetLumi(1./1000)
  plt.Stack(var, xtit='', ytit='')
  plt.GetYields()

Draw(opt.variable, categories, opt.title)
