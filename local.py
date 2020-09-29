
# from external
import numpy as np
import healpy as hp
import sys
import configparser
import pickle

# from cmblensplus/wrap/
import curvedsky
import basic

# from cmblensplus/utils/
import constants
import misctools
import binning as bn


# fixed values
Tcmb = 2.726e6

# Define directory
def data_directory(root='/global/homes/t/toshiyan/Work/Ongoing/lens_cutsky/'):
    
    direct = {}
    direct['root'] = root
    direct['input'] = '/project/projectdirs/act/data/actsims_data/signal_v0.4/'
    direct['local'] = root + 'data/'
    direct['cmb'] = direct['local'] + 'cmb/'

    return direct


# Define analysis parameters
class analysis_setup():

    def __init__(self,snmin=0,snmax=0,fltr='none',lmin=1,lmax=4096,clmin=100,olmin=1,olmax=2048,bn=30,nside=2048,wtype='com15',ascale=1.,sigma=10.):

        #//// load config file ////#
        conf = misctools.load_config('CMB')

        # rlz
        self.snmin = conf.getint('snmin',snmin)
        self.snmax = conf.getint('snmax',snmax)
        self.rlz   = np.linspace(self.snmin,self.snmax,self.snmax-self.snmin+1,dtype=np.int)
        if self.snmin == 0:
            self.snum  = self.snmax - self.snmin
        else:
            self.snum  = self.snmax - self.snmin + 1

        # multipole range of observed CMB alms
        self.lmin   = conf.getint('lmin',lmin)
        self.lmax   = conf.getint('lmax',lmax)
        
        # filtering multipole below clmin in addition to lx, ly before map->alm
        self.clmin  = conf.getint('clmin',clmin)

        # multipoles of output CMB spectrum
        self.olmin  = conf.getint('olmin',olmin)
        self.olmax  = conf.getint('olmax',olmax)
        self.bn     = conf.getint('bn',bn)
        self.binspc = conf.get('binspc','')

        # cmb map
        self.fltr   = conf.get('fltr',fltr)

        # fullsky map
        self.nside  = conf.getint('nside',nside) #Nside for fullsky cmb map
        self.npix   = 12*self.nside**2

        # window
        self.wtype  = conf.get('wtype',wtype)
        self.ascale = conf.getfloat('ascale',ascale)

        # noise
        self.sigma  = conf.getfloat('sigma',sigma)
        

    def filename(self):

        #//// root directories ////#
        d = data_directory()
        d_map = d['cmb'] + 'map/'
        d_alm = d['cmb'] + 'alm/'
        d_aps = d['cmb'] + 'aps/'
        d_msk = d['cmb'] + 'mask/'

        #//// basic tags ////#
        # for alm
        apotag = 'a'+str(self.ascale)+'deg'
        self.stag = '_'.join( [ self.wtype , apotag , self.fltr , 'lc'+str(self.clmin) ] )
        self.ntag = '_'.join( [ self.wtype , apotag , self.fltr , 'lc'+str(self.clmin) ] )

        # output multipole range
        self.otag = '_oL'+str(self.olmin)+'-'+str(self.olmax)+'_b'+str(self.bn)

        #//// index ////#
        self.ids = [str(i).zfill(5) for i in range(1000)]
        ids = self.ids
        
        #//// base best-fit cls ////#
        # aps of Planck 2015 best fit cosmology
        self.fucl = d['local'] + 'input/cosmo2017_10K_acc3_scalCls.dat'
        self.flcl = d['local'] + 'input/cosmo2017_10K_acc3_lensedCls.dat'

        #//// input alms ////#
        # lensed CMB
        self.fcmb = [ d['input'] + 'fullskyLensedUnabberatedCMB_alm_set00_'+x+'.fits' for x in ids ]
        # input klm realizations
        self.fiklm = [ d['input'] + 'fullskyPhi_alm_'+x+'.fits' for x in ids ]

        #//// Partial sky CMB maps from actsims ////#
        self.fmap = { s: [d_map+s+'_'+x+'.fits' for x in ids] for s in ['s','n'] }

        #//// Derived data filenames ////#
        # cmb signal/noise alms
        self.falm, self.fscl, self.fcls = {}, {}, {}
        for s in ['s','n','c']:
            if s in ['n']: tag = self.ntag
            if s in ['s','c']: tag = self.stag
            self.falm[s] = { m: [d_alm+'/'+s+'_'+m+'_'+tag+'_'+x+'.pkl' for x in ids] for m in ['T','E','B'] }

            # cmb aps
            self.fscl[s] = d_aps+'aps_sim_1d_'+tag+'_'+s+'.dat'
            self.fcls[s] = [d_aps+'/rlz/cl_'+tag+'_'+s+'_'+x+'.dat' for x in ids]

        # suppression
        self.fsup = d_aps + 'supfac_'+self.stag+'.dat'

        # custom mask
        self.amask = d_msk + self.wtype+'_'+apotag+'.fits'


    def array(self):

        #multipole
        self.l  = np.linspace(0,self.lmax,self.lmax+1)
        self.kL = self.l*(self.l+1)*.5

        #binned multipole
        self.bp, self.bc = basic.aps.binning(self.bn,[self.olmin,self.olmax],self.binspc)

        #theoretical cl
        self.ucl = np.zeros((5,self.lmax+1)) # TT, EE, TE, pp, Tp
        self.ucl[:,2:] = np.loadtxt(self.fucl,unpack=True,usecols=(1,2,3,4,5))[:,:self.lmax-1] 
        self.ucl[:3,:] *= 2.*np.pi / (self.l**2+self.l+1e-30) / Tcmb**2
        self.ucl[3,:] *= 1. / (self.l+1e-30)**4 / Tcmb**2

        self.lcl = np.zeros((4,self.lmax+1)) # TT, EE, BB, TE
        self.lcl[:,2:] = np.loadtxt(self.flcl,unpack=True,usecols=(1,2,3,4))[:,:self.lmax-1] 
        self.lcl *= 2.*np.pi / (self.l**2+self.l+1e-30) / Tcmb**2
        
        self.cpp = self.ucl[3,:]
        self.ckk = self.ucl[3,:] * (self.l**2+self.l)**2/4.


#----------------
# initial setup
#----------------

def init_analysis_params(**kwargs):
    # setup parameters, filenames, and arrays
    aobj = analysis_setup(**kwargs)
    analysis_setup.filename(aobj)
    analysis_setup.array(aobj)
    return aobj
      


