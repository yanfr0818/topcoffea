#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import numpy as np
import awkward
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea.arrays import Initialize
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, objects, selection, corrections, functions):
        self._samples = samples
        self._objects = objects
        self._selection = selection
        self._corrections = corrections
        self._functions = functions

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
        'dummy'   : hist.Hist("Dummy" , hist.Cat("sample", "sample"), hist.Bin("dummy", "Number of events", 1, 0, 1)),
        'counts'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("counts", "Counts", 1, 0, 2)),
        'invmass' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut","cut"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 200)),
        'njets'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("njets",  "Jet multiplicitu ", 10, 0, 10)),
        'nbtags'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("nbtags", "btag multiplicitu ", 5, 0, 5)),
        'met'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("met",    "MET (GeV)", 40, 0, 400)),
        'm3l'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m3l",    "$m_{3\ell}$ (GeV) ", 20, 0, 200)),
        #'m4l'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m4l",    "$m_{4\ell}$ (GeV) ", 20, 0, 200)),
        'wleppt'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("wleppt", "$p_{T}^{lepW}$ (GeV) ", 20, 0, 200)),
        'e0pt'    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("e0pt",   "Leading elec $p_{T}$ (GeV)", 30, 0, 300)),
        'm0pt'    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m0pt",   "Leading muon $p_{T}$ (GeV)", 30, 0, 300)),
        'j0pt'    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0pt",   "Leading jet  $p_{T}$ (GeV)", 20, 0, 400)),
        'e0eta'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("e0eta",  "Leading elec $\eta$", 20, -2.5, 2.5)),
        'm0eta'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m0eta",  "Leading muon $\eta$", 20, -2.5, 2.5)),
        'j0eta'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0eta",  "Leading jet  $\eta$", 20, -2.5, 2.5)),
        'ht'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("ht",     "H$_{T}$ (GeV)", 40, 0, 800)),
        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):
        # Dataset parameters
        dataset = events.metadata['dataset']
        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights' ]
        isData = self._samples[dataset]['isData']
        datasets = ['DoubleMuon', 'DoubleEG', 'MuonEG', 'SingleMuon', 'SingleElectron']
        for d in datasets: 
          if d in dataset: dataset = dataset.split('_')[0]

        ### Recover objects, selection, functions and others...
        # Objects
        isTightMuon     = self._objects['isTightMuonPOG']
        isTightElectron = self._objects['isTightElectronPOG']
        isGoodJet       = self._objects['isGoodJet']
        isClean         = self._objects['isClean']
        isMuonMVA       = self._objects['isMuonMVA'] #isMuonMVA(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, mediumPrompt, tightCharge, jetDeepB=0, minpt=15)
        isElecMVA       = self._objects['isElecMVA'] #isElecMVA(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, elecMVA, lostHits, convVeto, tightCharge, jetDeepB=0, minpt=15)
        isTauMVA        = self._objects['isTauMVA']
        
        # Corrections
        GetMuonIsoSF    = self._corrections['getMuonIso']
        GetMuonIDSF     = self._corrections['getMuonID' ]

        # Selection
        passNJets   = self._selection['passNJets']
        passMETcut  = self._selection['passMETcut']
        passTrigger = self._selection['passTrigger']

        # Functions
        pow2            = self._functions['pow2']
        IsClosestToZ    = self._functions['IsClosestToZ']
        GetGoodTriplets = self._functions['GetGoodTriplets']

        # Initialize objects
        met = events.MET
        e   = events.Electron
        mu  = events.Muon
        tau = events.Tau
        j   = events.Jet

        
        e['isGood'] = isElecMVA(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, e.mvaTTH, e.mvaFall17V2Iso, e.lostHits, e.convVeto, e.tightCharge,
                                e.sieie, e.hoe, e.eInvMinusPInv, minpt=15)
        leading_e = e[e.pt.argmax()]
        leading_e = leading_e[leading_e.isGood.astype(np.bool)]
                
        mu['isGood'] = isMuonMVA(mu.pt, mu.eta, mu.dxy, mu.dz, mu.pfRelIso03_all, mu.sip3d, mu.mvaTTH, mu.mediumPromptId, mu.tightCharge, minpt=10)
        leading_mu = mu[mu.pt.argmax()]
        leading_mu = leading_mu[leading_mu.isGood.astype(np.bool)]
        
        #tau['isGood'] = (tau.pt>25.0)&(abs(tau.eta)<2.4)
        tau['isGood'] = isTauMVA(tau.pt, tau.eta, minpt=25)
        
        e  =  e[e .isGood.astype(np.bool)]
        mu = mu[mu.isGood.astype(np.bool)]
        tau= tau[tau.isGood.astype(np.bool)]
        nElec = e .counts
        nMuon = mu.counts

        twoLeps   = (nElec+nMuon) == 2
        threeLeps = (nElec+nMuon) == 3
        fourLeps  = (nElec+nMuon) == 4
        twoElec   = (nElec == 2)
        twoMuon   = (nMuon == 2)
        e0 = e[e.pt.argmax()]
        m0 = mu[mu.pt.argmax()]


        j['isgood']  = isGoodJet(j.pt_nom, j.eta, j.jetId, j.neHEF, j.neEmEF, j.chHEF, j.chEmEF, j.nConstituents) #j.pt_nom is the skimmed version
        j['isclean'] = isClean(j, e, mu, tau)
        goodJets = j[(j.isclean)&(j.isgood)]
        njets = goodJets.counts
        ht = goodJets.pt.sum()
        j0 = goodJets[goodJets.pt.argmax()]
        nbtags = goodJets[goodJets.btagDeepB > 0.4941].counts

        ##################################################################
        ### 2 same-sign leptons
        ##################################################################

        # emu
        singe = e [(nElec==1)&(nMuon==1)&(e .pt>-1)]
        singm = mu[(nElec==1)&(nMuon==1)&(mu.pt>-1)]
        em = singe.cross(singm)
        emSSmask = (em.i0.charge*em.i1.charge>0)
        emSS = em[emSSmask]
        nemSS = len(emSS.flatten())
        emSSSign = (np.sign(em.i0.charge+em.i1.charge)>0)
        
        # ee and mumu
        # pt>-1 to preserve jagged dimensions
        ee = e [(nElec==2)&(nMuon==0)&(e.pt>-1)]
        mm = mu[(nElec==0)&(nMuon==2)&(mu.pt>-1)]

        eepairs = ee.distincts()
        eeSSmask = (eepairs.i0.charge*eepairs.i1.charge>0)
        eeonZmask  = (np.abs((eepairs.i0+eepairs.i1).mass-91.2)<15)
        eeoffZmask = (eeonZmask==0)
        eeSSSign = (np.sign(eepairs.i0.charge+eepairs.i1.charge)>0)

        mmpairs = mm.distincts()
        mmSSmask = (mmpairs.i0.charge*mmpairs.i1.charge>0)
        mmonZmask  = (np.abs((mmpairs.i0+mmpairs.i1).mass-91.2)<15)
        mmoffZmask = (mmonZmask==0)
        mmSSSign = (np.sign(mmpairs.i0.charge+mmpairs.i1.charge)>0)

        eeSSonZ  = eepairs[eeSSmask &  eeonZmask]
        eeSSoffZ = eepairs[eeSSmask & eeoffZmask]
        mmSSonZ  = mmpairs[mmSSmask &  mmonZmask]
        mmSSoffZ = mmpairs[mmSSmask & mmoffZmask]
        neeSS = len(eeSSonZ.flatten()) + len(eeSSoffZ.flatten())
        nmmSS = len(mmSSonZ.flatten()) + len(mmSSoffZ.flatten())

        #print('Same-sign events [ee, emu, mumu] = [%i, %i, %i]'%(neeSS, nemSS, nmmSS))

        # Cuts
        eeSSmask   = (eeSSmask[eeSSmask].counts>0)
        mmSSmask   = (mmSSmask[mmSSmask].counts>0)
        eeonZmask  = (eeonZmask[eeonZmask].counts>0)
        eeoffZmask = (eeoffZmask[eeoffZmask].counts>0)
        mmonZmask  = (mmonZmask[mmonZmask].counts>0)
        mmoffZmask = (mmoffZmask[mmoffZmask].counts>0)
        emSSmask   = (emSSmask[emSSmask].counts>0)
        eeSSSign   = (eeSSSign[eeSSSign].counts>0)
        mmSSSign   = (mmSSSign[mmSSSign].counts>0)        
        emSSSign   = (emSSSign[emSSSign].counts>0)        

        
        ##################################################################
        ### 3 leptons
        ##################################################################

        # eem
        muon_eem = mu[(nElec==2)&(nMuon==1)&(mu.pt>-1)]
        elec_eem =  e[(nElec==2)&(nMuon==1)&( e.pt>-1)]
        ee_eem   = elec_eem.distincts()
        ee_eemZmask     = (ee_eem.i0.charge*ee_eem.i1.charge<1)&(np.abs((ee_eem.i0+ee_eem.i1).mass-91.2)<15)
        ee_eemOffZmask  = (ee_eemZmask==0)#(ee_eem.i0.charge*ee_eem.i1.charge<1)&(np.abs((ee_eem.i0+ee_eem.i1).mass-91)>15)
        ee_eemZmask     = (ee_eemZmask[ee_eemZmask].counts>0)
        ee_eemOffZmask  = (ee_eemOffZmask[ee_eemOffZmask].counts>0)

        eepair_eem      = (ee_eem.i0+ee_eem.i1)
        trilep_eem      = eepair_eem.cross(muon_eem)
        trilep_eem      = (trilep_eem.i0+trilep_eem.i1)
        
        group_eem= ee_eem.cross(muon_eem)
        eemSign  = (np.sign(group_eem.i0.charge+group_eem.i1.charge+group_eem.i2.charge)>0)
        eemSign  = (eemSign[eemSign].counts>0)

        # mme
        muon_mme = mu[(nElec==1)&(nMuon==2)&(mu.pt>-1)]
        elec_mme =  e[(nElec==1)&(nMuon==2)&( e.pt>-1)]
        mm_mme   = muon_mme.distincts() 
        mm_mmeZmask     = (mm_mme.i0.charge*mm_mme.i1.charge<1)&(np.abs((mm_mme.i0+mm_mme.i1).mass-91.2)<15)
        mm_mmeOffZmask  = (mm_mmeZmask==0)#(mm_mme.i0.charge*mm_mme.i1.charge<1)&(np.abs((mm_mme.i0+mm_mme.i1).mass-91)>15)
        mm_mmeZmask     = (mm_mmeZmask[mm_mmeZmask].counts>0)
        mm_mmeOffZmask  = (mm_mmeOffZmask[mm_mmeOffZmask].counts>0)

        mmpair_mme     = (mm_mme.i0+mm_mme.i1)
        trilep_mme     = mmpair_mme.cross(elec_mme)
        trilep_mme     = (trilep_mme.i0+trilep_mme.i1)
        
        group_mme= mm_mme.cross(elec_mme)
        mmeSign  = (np.sign(group_mme.i0.charge+group_mme.i1.charge+group_mme.i2.charge)>0)
        mmeSign  = (mmeSign[mmeSign].counts>0)
        
        mZ_mme  = mmpair_mme.mass
        mZ_eem  = eepair_eem.mass
        m3l_eem = trilep_eem.mass
        m3l_mme = trilep_mme.mass


        ### eee and mmm
        eee =   e[(nElec==3)&(nMuon==0)&( e.pt>-1)] 
        mmm =  mu[(nElec==0)&(nMuon==3)&(mu.pt>-1)] 
        ee_pairs = eee.argchoose(2)
        mm_pairs = mmm.argchoose(2)

        # Select pairs that are SFOS.
        eeSFOS_pairs = ee_pairs[(np.abs(eee[ee_pairs.i0].pdgId) == np.abs(eee[ee_pairs.i1].pdgId)) & (eee[ee_pairs.i0].charge != eee[ee_pairs.i1].charge)]
        mmSFOS_pairs = mm_pairs[(np.abs(mmm[mm_pairs.i0].pdgId) == np.abs(mmm[mm_pairs.i1].pdgId)) & (mmm[mm_pairs.i0].charge != mmm[mm_pairs.i1].charge)]
        # Find the pair with mass closest to Z.
        eeOSSFmask = eeSFOS_pairs[np.abs((eee[eeSFOS_pairs.i0] + eee[eeSFOS_pairs.i1]).mass - 91.2).argmin()]
        onZmask_ee = np.abs((eee[eeOSSFmask.i0] + eee[eeOSSFmask.i1]).mass - 91.2) < 15
        mmOSSFmask = mmSFOS_pairs[np.abs((mmm[mmSFOS_pairs.i0] + mmm[mmSFOS_pairs.i1]).mass - 91.2).argmin()]
        onZmask_mm = np.abs((mmm[mmOSSFmask.i0] + mmm[mmOSSFmask.i1]).mass - 91.2) < 15
        offZmask_ee = np.abs((eee[eeOSSFmask.i0] + eee[eeOSSFmask.i1]).mass - 91.2) > 15
        offZmask_mm = np.abs((mmm[mmOSSFmask.i0] + mmm[mmOSSFmask.i1]).mass - 91.2) > 15

        # Create masks
        eeeOnZmask  = onZmask_ee[onZmask_ee].counts>0
        eeeOffZmask = offZmask_ee[offZmask_ee].counts>0
        mmmOnZmask  = onZmask_mm[onZmask_mm].counts>0
        mmmOffZmask = offZmask_mm[offZmask_mm].counts>0
        
        eee_trilep = eee.choose(3)
        eeeSign  = (np.sign(eee_trilep.i0.charge+eee_trilep.i1.charge+eee_trilep.i2.charge)>0)
        eeeSign  = (eeeSign[eeeSign].counts>0)	
        mmm_trilep = mmm.choose(3)
        mmmSign  = (np.sign(mmm_trilep.i0.charge+mmm_trilep.i1.charge+mmm_trilep.i2.charge)>0)
        mmmSign  = (mmmSign[mmmSign].counts>0)
        
        # Leptons from Z
        eZ0= eee[eeOSSFmask.i0]
        eZ1= eee[eeOSSFmask.i1]
        mZ0= mmm[mmOSSFmask.i0]
        mZ1= mmm[mmOSSFmask.i1]

        # Leptons from W
        eW = eee[~eeOSSFmask.i0 | ~eeOSSFmask.i1]
        mW = mmm[~mmOSSFmask.i0 | ~mmOSSFmask.i1]

        eZ = eee[eeOSSFmask.i0] + eee[eeOSSFmask.i1]
        triElec = eZ + eW
        mZ = mmm[mmOSSFmask.i0] + mmm[mmOSSFmask.i1]
        triMuon = mZ + mW

        mZ_eee  = eZ.mass
        m3l_eee = triElec.mass
        mZ_mmm  = mZ.mass
        m3l_mmm = triMuon.mass
        
        
        ##################################################################
        ### 4 leptons
        ##################################################################

        # eeem
        muon_eeem = mu[(nElec==3)&(nMuon==1)&(mu.pt>-1)]
        elec_eeem =  e[(nElec==3)&(nMuon==1)&( e.pt>-1)]
        ee_eeem   = elec_eeem.distincts()
        ee_eeemZmask     = (ee_eeem.i0.charge*ee_eeem.i1.charge<1)&(np.abs((ee_eeem.i0+ee_eeem.i1).mass-91.2)<15)
        ee_eeemOffZmask  = (ee_eeemZmask==0)#(ee_eeem.i0.charge*ee_eeem.i1.charge<1)&(np.abs((ee_eeem.i0+ee_eeem.i1).mass-91)>15)
        ee_eeemZmask     = (ee_eeemZmask[ee_eeemZmask].counts>0)
        ee_eeemOffZmask  = (ee_eeemOffZmask[ee_eeemOffZmask].counts>0)

        #eepair_eeem     = (ee_eeem.i0+ee_eeem.i1)
        #trilep_eeem     = eepair_eeem.cross(muon_eeem)
        #trilep_eeem     = (trilep_eeem.i0+trilep_eeem.i1) 

        # mmme
        muon_mmme = mu[(nElec==1)&(nMuon==3)&(mu.pt>-1)]
        elec_mmme =  e[(nElec==1)&(nMuon==3)&( e.pt>-1)]
        mm_mmme   = muon_mmme.distincts()
        mm_mmmeZmask     = (mm_mmme.i0.charge*mm_mmme.i1.charge<1)&(np.abs((mm_mmme.i0+mm_mmme.i1).mass-91.2)<15)
        mm_mmmeOffZmask  = (mm_mmmeZmask==0)#(mm_mmme.i0.charge*mm_mmme.i1.charge<1)&(np.abs((mm_mmme.i0+mm_mmme.i1).mass-91)>15)
        mm_mmmeZmask     = (mm_mmmeZmask[mm_mmmeZmask].counts>0)
        mm_mmmeOffZmask  = (mm_mmmeOffZmask[mm_mmmeOffZmask].counts>0)

        #mmpair_mmme     = (mm_mmme.i0+mm_mmme.i1)
        #trilep_mmme     = mmpair_mmme.cross(elec_mmme)
        #trilep_mmme     = (trilep_mmme.i0+trilep_mmme.i1)
        #mZ_mmme  = mmpair_mmme.mass
        #mZ_eeem  = eepair_eeem.mass
        #m3l_eeem = trilep_eeem.mass
        #m3l_mmme = trilep_mmme.mass
        
         # eemm
        muon_eemm = mu[(nElec==2)&(nMuon==2)&(mu.pt>-1)]
        elec_eemm =  e[(nElec==2)&(nMuon==2)&( e.pt>-1)]
        ee_eemm   = elec_eemm.distincts()
        mm_eemm   = muon_eemm.distincts()
        ee_eemmZmask  = (ee_eemm.i0.charge*ee_eemm.i1.charge<1)&(np.abs((ee_eemm.i0+ee_eemm.i1).mass-91.2)<15)
        mm_eemmZmask  = (mm_eemm.i0.charge*mm_eemm.i1.charge<1)&(np.abs((mm_eemm.i0+mm_eemm.i1).mass-91.2)<15)
        eemmOnZmask   = (ee_eemmZmask|mm_eemmZmask)
        eemmOffZmask  = (eemmOnZmask==0)
        eemmOnZmask   = (eemmOnZmask[eemmOnZmask].counts>0)
        eemmOffZmask  = (eemmOffZmask[eemmOffZmask].counts>0)
        
        ### eeee and mmmm
        eeee =   e[(nElec==4)&(nMuon==0)&( e.pt>-1)] 
        mmmm =  mu[(nElec==0)&(nMuon==4)&(mu.pt>-1)] 
        # Create pairs
        eeee_groups = eeee.distincts()
        mmmm_groups = mmmm.distincts()
        # Calculate the invariant mass of the pairs
        invMass_eeee = ((eeee_groups.i0+eeee_groups.i1).mass)
        invMass_mmmm = ((mmmm_groups.i0+mmmm_groups.i1).mass)
        # OS pairs
        isOSeeee = ((eeee_groups.i0.charge != eeee_groups.i1.charge))
        isOSmmmm = ((mmmm_groups.i0.charge != mmmm_groups.i1.charge))
        # Get the ones with a mass closest to the Z mass (and in a range of  thr)
        clos_eeee = IsClosestToZ(invMass_eeee, thr=15)
        clos_mmmm = IsClosestToZ(invMass_mmmm, thr=15)
        # Finally, the mask for eeee/mmmm with/without OS onZ pair
        eeeeOnZmask  = (clos_eeee)&(isOSeeee)
        eeeeOffZmask = (eeeeOnZmask==0)
        mmmmOnZmask  = (clos_mmmm)&(isOSmmmm)
        mmmmOffZmask = (mmmmOnZmask==0)
        eeeeOnZmask  = (eeeeOnZmask[eeeeOnZmask].counts>0)
        eeeeOffZmask = (eeeeOffZmask[eeeeOffZmask].counts>0)
        mmmmOnZmask  = (mmmmOnZmask[mmmmOnZmask].counts>0)
        mmmmOffZmask = (mmmmOffZmask[mmmmOffZmask].counts>0)
        
        # Get Z and W invariant masses
        #goodPairs_eeee = eeee_groups[(clos_eeee)&(isOSeeee)]
        #print(len(eeee_groups[clos_eeee.counts>0]))
        #print(len(eeee_groups[isOSeeee.counts>0]))
        #print(len(goodPairs_eeee.i0[goodPairs_eeee.counts>0]))
        #eZ0   = goodPairs_eeee.i0[goodPairs_eeee.counts>0].regular()#[(goodPairs_eee.counts>0)].regular()
        #eZ1   = goodPairs_eeee.i1[goodPairs_eeee.counts>0].regular()#[(goodPairs_eee.counts>0)].regular()
        #goodPairs_mmmm = mmmm_groups[(clos_mmmm)&(isOSmmmm)]
        #mZ0   = goodPairs_mmmm.i0[goodPairs_mmmm.counts>0].regular()#[(goodPairs_eee.counts>0)].regular()
        #mZ1   = goodPairs_mmmm.i1[goodPairs_mmmm.counts>0].regular()#[(goodPairs_eee.counts>0)].regular()
        
        #eeee_reg = eeee[(eeeeOnZmask)].regular()
        #eW = np.append(eeee_reg, eZ0, axis=1)
        #eW = np.append(eW, eZ1,axis=1)
        #eWmask = np.apply_along_axis(lambda a : [list(a).count(x)==1 for x in a], 1, eW)
        #eW = eW[eWmask]
        #mmmm_reg = mmmm[(mmmmOnZmask)].regular()
        #mW = np.append(mmmm_reg, mZ0, axis=1)
        #mW = np.append(mW, mZ1,axis=1)
        #mWmask = np.apply_along_axis(lambda a : [list(a).count(x)==1 for x in a], 1, mW)
        #mW = mW[mWmask]
        
        #eZ      = [x+y for x,y in zip(eZ0, eZ1)]
        #quadElec = [x+y for x,y in zip(eZ, eW)]
        #mZ_eeee  = [t[0].mass for t in eZ]
        #m4l_eeee = [t[0].mass for t in quadElec]
        #mZ      = [x+y for x,y in zip(mZ0, mZ1)]
        #quadMuon = [x+y for x,y in zip(mZ, mW)]
        #mZ_mmmm  = [t[0].mass for t in mZ]
        #m4l_mmmm = [t[0].mass for t in quadMuon]

        # Triggers
        #passTrigger = lambda events, n, m, o : np.ones_like(events['MET_pt'], dtype=np.bool) # XXX
        trig_eeSS = passTrigger(events,'ee',isData,dataset)
        trig_mmSS = passTrigger(events,'mm',isData,dataset)
        trig_emSS = passTrigger(events,'em',isData,dataset)
        trig_eee  = passTrigger(events,'eee',isData,dataset)
        trig_mmm  = passTrigger(events,'mmm',isData,dataset)
        trig_eem  = passTrigger(events,'eem',isData,dataset)
        trig_mme  = passTrigger(events,'mme',isData,dataset)
        trig_eeee = passTrigger(events,'eeee',isData,dataset)
        trig_mmmm = passTrigger(events,'mmmm',isData,dataset)
        trig_eeem = passTrigger(events,'eeem',isData,dataset)
        trig_eemm = passTrigger(events,'eemm',isData,dataset)
        trig_mmme = passTrigger(events,'mmme',isData,dataset)

        # MET filters

        # Weights
        genw = np.ones_like(events['MET_pt']) if isData else events['genWeight']
        weights = processor.Weights(events.size)
        weights.add('norm',genw if isData else (xsec/sow)*genw)

        # P/NP information tracker
        passTracker = {
            'event': events['event'],
            'nElec': nElec, 'nMuon': nMuon, 'njets': njets, 'nbtags': nbtags,
            'trig': {
                'eeSS': trig_eeSS, 'mmSS': trig_mmSS, 'emSS': trig_emSS,
                'eee' : trig_eee,  'mmm' : trig_mmm,  'eem' : trig_eem,  'mme' :  trig_mme,
                'eeee': trig_eeee, 'mmmm': trig_mmmm, 'eeem': trig_eeem, 'eemm': trig_eemm, 'mmme': trig_mmme,
            }
        }
        with open('passTracker','a+') as f:
            for keys in passTracker:
                if keys == 'trig':
                    for tkeys in passTracker[keys]:
                        f.write('{}:'.format('trig_'+tkeys))
                        for i in range(len(passTracker[keys][tkeys])):
                            f.write('{},'.format(str(passTracker[keys][tkeys][i])))
                        f.write('\n')
                else:
                    f.write('{}:'.format(keys))
                    for i in range(len(passTracker[keys])):
                        f.write('{},'.format(str(passTracker[keys][i])))
                    f.write('\n')
            
        # Selections and cuts
        selections = processor.PackedSelection()
        channels2LSS = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
        selections.add('eeSSonZ',  (eeonZmask)&(eeSSmask)&(trig_eeSS))
        selections.add('eeSSoffZ', (eeoffZmask)&(eeSSmask)&(trig_eeSS))
        selections.add('mmSSonZ',  (mmonZmask)&(mmSSmask)&(trig_mmSS))
        selections.add('mmSSoffZ', (mmoffZmask)&(mmSSmask)&(trig_mmSS))
        selections.add('emSS',     (emSSmask)&(trig_emSS))

        channels2LSS += ['eeSSonZ_p', 'eeSSoffZ_p', 'mmSSonZ_p', 'mmSSoffZ_p', 'emSS_p']
        selections.add('eeSSonZ_p',  (eeonZmask)&(eeSSmask)&(trig_eeSS)&(eeSSSign))
        selections.add('eeSSoffZ_p', (eeoffZmask)&(eeSSmask)&(trig_eeSS)&(eeSSSign))
        selections.add('mmSSonZ_p',  (mmonZmask)&(mmSSmask)&(trig_mmSS)&(mmSSSign))
        selections.add('mmSSoffZ_p', (mmoffZmask)&(mmSSmask)&(trig_mmSS)&(mmSSSign))
        selections.add('emSS_p',     (emSSmask)&(trig_emSS)&(emSSSign))

        channels2LSS += ['eeSSonZ_m', 'eeSSoffZ_m', 'mmSSonZ_m', 'mmSSoffZ_m', 'emSS_m']
        selections.add('eeSSonZ_m',  (eeonZmask)&(eeSSmask)&(trig_eeSS)&(eeSSSign==0))
        selections.add('eeSSoffZ_m', (eeoffZmask)&(eeSSmask)&(trig_eeSS)&(eeSSSign==0))
        selections.add('mmSSonZ_m',  (mmonZmask)&(mmSSmask)&(trig_mmSS)&(mmSSSign==0))
        selections.add('mmSSoffZ_m', (mmoffZmask)&(mmSSmask)&(trig_mmSS)&(mmSSSign==0))
        selections.add('emSS_m',     (emSSmask)&(trig_emSS)&(emSSSign==0))

        
        channels3L = ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ']
        selections.add('eemSSonZ',   (ee_eemZmask)&(trig_eem))
        selections.add('eemSSoffZ',  (ee_eemOffZmask)&(trig_eem))
        selections.add('mmeSSonZ',   (mm_mmeZmask)&(trig_mme))
        selections.add('mmeSSoffZ',  (mm_mmeOffZmask)&(trig_mme))

        channels3L += ['eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ']
        selections.add('eeeSSonZ',   (eeeOnZmask)&(trig_eee))
        selections.add('eeeSSoffZ',  (eeeOffZmask)&(trig_eee))
        selections.add('mmmSSonZ',   (mmmOnZmask)&(trig_mmm))
        selections.add('mmmSSoffZ',  (mmmOffZmask)&(trig_mmm))
        
        channels3L += ['eemSSonZ_p', 'eemSSoffZ_p', 'mmeSSonZ_p', 'mmeSSoffZ_p']
        selections.add('eemSSonZ_p',   (ee_eemZmask)&(trig_eem)&(eemSign))
        selections.add('eemSSoffZ_p',  (ee_eemOffZmask)&(trig_eem)&(eemSign))
        selections.add('mmeSSonZ_p',   (mm_mmeZmask)&(trig_mme)&(mmeSign))
        selections.add('mmeSSoffZ_p',  (mm_mmeOffZmask)&(trig_mme)&(mmeSign))

        channels3L += ['eeeSSonZ_p', 'eeeSSoffZ_p', 'mmmSSonZ_p', 'mmmSSoffZ_p']
        selections.add('eeeSSonZ_p',   (eeeOnZmask)&(trig_eee)&(eeeSign))
        selections.add('eeeSSoffZ_p',  (eeeOffZmask)&(trig_eee)&(eeeSign))
        selections.add('mmmSSonZ_p',   (mmmOnZmask)&(trig_mmm)&(mmmSign))
        selections.add('mmmSSoffZ_p',  (mmmOffZmask)&(trig_mmm)&(mmmSign))
        
        channels3L += ['eemSSonZ_m', 'eemSSoffZ_m', 'mmeSSonZ_m', 'mmeSSoffZ_m']
        selections.add('eemSSonZ_m',   (ee_eemZmask)&(trig_eem)&(eemSign==0))
        selections.add('eemSSoffZ_m',  (ee_eemOffZmask)&(trig_eem)&(eemSign==0))
        selections.add('mmeSSonZ_m',   (mm_mmeZmask)&(trig_mme)&(mmeSign==0))
        selections.add('mmeSSoffZ_m',  (mm_mmeOffZmask)&(trig_mme)&(mmeSign==0))

        channels3L += ['eeeSSonZ_m', 'eeeSSoffZ_m', 'mmmSSonZ_m', 'mmmSSoffZ_m']
        selections.add('eeeSSonZ_m',   (eeeOnZmask)&(trig_eee)&(eeeSign==0))
        selections.add('eeeSSoffZ_m',  (eeeOffZmask)&(trig_eee)&(eeeSign==0))
        selections.add('mmmSSonZ_m',   (mmmOnZmask)&(trig_mmm)&(mmmSign==0))
        selections.add('mmmSSoffZ_m',  (mmmOffZmask)&(trig_mmm)&(mmmSign==0))
        
        channels4L = ['eeemSSonZ', 'eeemSSoffZ', 'mmmeSSonZ', 'mmmeSSoffZ', 'eemmSSonZ', 'eemmSSoffZ']
        selections.add('eeemSSonZ',   (ee_eeemZmask)&(trig_eeem))
        selections.add('eeemSSoffZ',  (ee_eeemOffZmask)&(trig_eeem))
        selections.add('mmmeSSonZ',   (mm_mmmeZmask)&(trig_mmme))
        selections.add('mmmeSSoffZ',  (mm_mmmeOffZmask)&(trig_mmme))
        selections.add('eemmSSonZ',   (eemmOnZmask)&(trig_eemm))
        selections.add('eemmSSoffZ',  (eemmOffZmask)&(trig_eemm))
        
        channels4L += ['eeeeSSonZ', 'eeeeSSoffZ', 'mmmmSSonZ', 'mmmmSSoffZ']
        selections.add('eeeeSSonZ',   (eeeeOnZmask)&(trig_eeee))
        selections.add('eeeeSSoffZ',  (eeeeOffZmask)&(trig_eeee))
        selections.add('mmmmSSonZ',   (mmmmOnZmask)&(trig_mmmm))
        selections.add('mmmmSSoffZ',  (mmmmOffZmask)&(trig_mmmm))

        levels = ['base', '2jets', '2j1b', '2j2b', '3jets', '3j1b', '3j2b', '4jets', '4j1b', '4j2b', '5j1b', '5j2b', '6j1b', '6j2b']
        selections.add('base', (nElec+nMuon>=2))
        selections.add('2jets',(njets>=2))
        selections.add('2j1b',(njets>=2)&(nbtags>=1))
        selections.add('2j2b',(njets>=2)&(nbtags>=2))
        selections.add('3jets',(njets>=3))
        selections.add('3j1b',(njets>=3)&(nbtags>=1))
        selections.add('3j2b',(njets>=3)&(nbtags>=2))
        selections.add('4jets',(njets>=4))
        selections.add('4j1b',(njets>=4)&(nbtags>=1))
        selections.add('4j2b',(njets>=4)&(nbtags>=2))
        selections.add('5j1b',(njets>=5)&(nbtags>=1))
        selections.add('5j2b',(njets>=5)&(nbtags>=2))
        selections.add('6j1b',(njets>=6)&(nbtags>=1))
        selections.add('6j2b',(njets>=6)&(nbtags>=2))

        # Variables
        invMass_eeSSonZ  = ( eeSSonZ.i0+ eeSSonZ.i1).mass
        invMass_eeSSoffZ = (eeSSoffZ.i0+eeSSoffZ.i1).mass
        invMass_mmSSonZ  = ( mmSSonZ.i0+ mmSSonZ.i1).mass
        invMass_mmSSoffZ = (mmSSoffZ.i0+mmSSoffZ.i1).mass
        invMass_emSS     = (emSS.i0+emSS.i1).mass

        varnames = {}
        varnames['met'] = met.pt
        varnames['ht'] = ht
        varnames['njets'] = njets
        varnames['nbtags'] = nbtags
        varnames['invmass'] = {
          'eeSSonZ'   : invMass_eeSSonZ,
          'eeSSoffZ'  : invMass_eeSSoffZ,
          'mmSSonZ'   : invMass_mmSSonZ,
          'mmSSoffZ'  : invMass_mmSSoffZ,
          'emSS'      : invMass_emSS,
          'eeSSonZ_p' : invMass_eeSSonZ,
          'eeSSoffZ_p': invMass_eeSSoffZ,
          'mmSSonZ_p' : invMass_mmSSonZ,
          'mmSSoffZ_p': invMass_mmSSoffZ,
          'emSS_p'    : invMass_emSS,
          'eeSSonZ_m' : invMass_eeSSonZ,
          'eeSSoffZ_m': invMass_eeSSoffZ,
          'mmSSonZ_m' : invMass_mmSSonZ,
          'mmSSoffZ_m': invMass_mmSSoffZ,
          'emSS_m'    : invMass_emSS,
          'eemSSonZ'  : mZ_eem,
          'eemSSoffZ' : mZ_eem,
          'mmeSSonZ'  : mZ_mme,
          'mmeSSoffZ' : mZ_mme,
          'eemSSonZ_p'  : mZ_eem,
          'eemSSoffZ_p' : mZ_eem,
          'mmeSSonZ_p'  : mZ_mme,
          'mmeSSoffZ_p' : mZ_mme,
          'eemSSonZ_m'  : mZ_eem,
          'eemSSoffZ_m' : mZ_eem,
          'mmeSSonZ_m'  : mZ_mme,
          'mmeSSoffZ_m' : mZ_mme,
          #'eeeSSonZ'  : mZ_eee,
          #'eeeSSoffZ' : mZ_eee,
          #'mmmSSonZ'  : mZ_mmm,
          #'mmmSSoffZ' : mZ_mmm,
          #'eeeeSSonZ'  : mZ_eeee,
          #'eeeeSSoffZ' : mZ_eeee,
          #'mmmmSSonZ'  : mZ_mmmm,
          #'mmmmSSoffZ' : mZ_mmmm,
        }
        varnames['m3l'] = {
          'eemSSonZ'  : m3l_eem,
          'eemSSoffZ' : m3l_eem,
          'mmeSSonZ'  : m3l_mme,
          'mmeSSoffZ' : m3l_mme,
          'eemSSonZ_p'  : m3l_eem,
          'eemSSoffZ_p' : m3l_eem,
          'mmeSSonZ_p'  : m3l_mme,
          'mmeSSoffZ_p' : m3l_mme,
          'eemSSonZ_m'  : m3l_eem,
          'eemSSoffZ_m' : m3l_eem,
          'mmeSSonZ_m'  : m3l_mme,
          'mmeSSoffZ_m' : m3l_mme,
          #'eeeSSonZ'  : m3l_eee,
          #'eeeSSoffZ' : m3l_eee,
          #'mmmSSonZ'  : m3l_mmm,
          #'mmmSSoffZ' : m3l_mmm,
        }
        #varnames['m4l'] = {
          #'eeeSSonZ'  : m4l_eeee,
          #'eeeSSoffZ' : m4l_eeee,
          #'mmmSSonZ'  : m4l_mmmm,
          #'mmmSSoffZ' : m4l_mmmm,
        #}
        varnames['e0pt' ] = e0.pt
        varnames['e0eta'] = e0.eta
        varnames['m0pt' ] = m0.pt
        varnames['m0eta'] = m0.eta
        varnames['j0pt' ] = j0.pt
        varnames['j0eta'] = j0.eta
        varnames['counts'] = np.ones_like(events.MET.pt, dtype=np.int)

        # Fill Histos
        hout = self.accumulator.identity()
        hout['dummy'].fill(sample=dataset, dummy=1, weight=events.size)
        
        for var, v in varnames.items():
         for ch in channels2LSS+channels3L:
          for lev in levels:
            weight = weights.weight()
            cuts = [ch] + [lev]
            cut = selections.all(*cuts)
            weights_flat = weight[cut].flatten()
            weights_flat = np.ones_like(weights_flat, dtype=np.int)
            if var == 'invmass':
              if   ch in ['eeeSSoffZ', 'mmmSSoffZ', 'eeeSSoffZ_p', 'mmmSSoffZ_p', 'eeeSSoffZ_m', 'mmmSSoffZ_m']: continue
              elif ch in ['eeeSSonZ' , 'mmmSSonZ', 'eeeSSonZ_p', 'mmmSSonZ_p', 'eeeSSonZ_m', 'mmmSSonZ_m']: continue #values = v[ch]
              else: values = v[ch][cut].flatten()
              hout['invmass'].fill(sample=dataset, channel=ch, cut=lev, invmass=values, weight=weights_flat)
            elif var == 'm3l': 
              if ch in channels2LSS: continue
              if ch in ['eeeSSoffZ', 'mmmSSoffZ', 'eeeSSonZ' , 'mmmSSonZ', 'eeeSSoffZ_p', 'mmmSSoffZ_p', 'eeeSSonZ_p' , 'mmmSSonZ_p', 'eeeSSoffZ_m', 'mmmSSoffZ_m', 'eeeSSonZ_m' , 'mmmSSonZ_m']: continue
              values = v[ch][cut].flatten()
              hout['m3l'].fill(sample=dataset, channel=ch, cut=lev, m3l=values, weight=weights_flat)
            else:
              values = v[cut].flatten()
              if   var == 'ht'    : hout[var].fill(ht=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'met'   : hout[var].fill(met=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'njets' : hout[var].fill(njets=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'nbtags': hout[var].fill(nbtags=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'counts': hout[var].fill(counts=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'e0pt'  : 
                if ch in ['mmSSonZ', 'mmSSoffZ', 'mmSSonZ_p', 'mmSSoffZ_p', 'mmSSonZ_m', 'mmSSoffZ_m', 'mmmSSoffZ', 'mmmSSonZ', 'mmmSSoffZ_p', 'mmmSSonZ_p', 'mmmSSoffZ_m', 'mmmSSonZ_m']: continue
                hout[var].fill(e0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'm0pt'  : 
                if ch in ['eeSSonZ', 'eeSSoffZ', 'eeSSonZ_p', 'eeSSoffZ_p', 'eeSSonZ_m', 'eeSSoffZ_m', 'eeeSSoffZ', 'eeeSSonZ', 'eeeSSoffZ_p', 'eeeSSonZ_p', 'eeeSSoffZ_m', 'eeeSSonZ_m']: continue
                hout[var].fill(m0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'e0eta' : 
                if ch in ['mmSSonZ', 'mmSSoffZ', 'mmSSonZ_p', 'mmSSoffZ_p', 'mmSSonZ_m', 'mmSSoffZ_m', 'mmmSSoffZ', 'mmmSSonZ', 'mmmSSoffZ_p', 'mmmSSonZ_p', 'mmmSSoffZ_m', 'mmmSSonZ_m']: continue
                hout[var].fill(e0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'm0eta' :
                if ch in ['eeSSonZ', 'eeSSoffZ', 'eeSSonZ_p', 'eeSSoffZ_p', 'eeSSonZ_m', 'eeSSoffZ_m', 'eeeSSoffZ', 'eeeSSonZ', 'eeeSSoffZ_p', 'eeeSSonZ_p', 'eeeSSoffZ_m', 'eeeSSonZ_m']: continue
                hout[var].fill(m0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'j0pt'  : 
                if lev == 'base': continue
                hout[var].fill(j0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'j0eta' : 
                if lev == 'base': continue
                hout[var].fill(j0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)                
                
         for ch in channels4L:
          for lev in levels:
            weight = weights.weight()
            cuts = [ch] + [lev]
            cut = selections.all(*cuts)
            weights_flat = weight[cut].flatten()
            weights_flat = np.ones_like(weights_flat, dtype=np.int)
            if  var == 'invmass': continue
            elif var == 'm3l'   : continue
            values = v[cut].flatten()
            if   var == 'ht'    : hout[var].fill(ht=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
            elif var == 'met'   : hout[var].fill(met=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
            elif var == 'njets' : hout[var].fill(njets=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
            elif var == 'nbtags': hout[var].fill(nbtags=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
            elif var == 'counts': hout[var].fill(counts=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)

        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    objects     = load(outpath+'objects.coffea')
    selection   = load(outpath+'selection.coffea')
    corrections = load(outpath+'corrections.coffea')
    functions   = load(outpath+'functions.coffea')

    
    topprocessor = AnalysisProcessor(samples, objects, selection, corrections, functions)
    save(topprocessor, outpath+'topeft.coffea')
