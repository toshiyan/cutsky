
import numpy as np
import healpy as hp
import tqdm
import pickle

import curvedsky as cs

import misctools

import local


def interface(run=[],kwargs={},kwargs_ov={}):
    
    aobj = local.init_analysis_params(**kwargs)
    
    if 'map' in run:
        generate_map(aobj,**kwargs_ov)
        
    if 'alm' in run:
        map2alm(aobj,**kwargs_ov)
        
    if 'aps' in run:
        alm2aps(aobj,**kwargs_ov)
    


def generate_map(aobj,overwrite=False,verbose=True):

    # take lensed CMB alms and convert to map
    for rlz in tqdm.tqdm(aobj.rlz,desc='signal map'):
        
        if misctools.check_path(aobj.fmap['s'][rlz],overwrite=overwrite,verbose=verbose): continue
            
        iTlm = np.complex128( hp.fitsfunc.read_alm( aobj.fcmb[rlz], hdu = (1) ) ) / local.Tcmb
        ilmax = hp.sphtfunc.Alm.getlmax(len(iTlm))
        iTlm = cs.utils.lm_healpy2healpix(iTlm, ilmax)[:aobj.lmax+1,:aobj.lmax+1]
        Tmap = cs.utils.hp_alm2map(aobj.nside,aobj.lmax,aobj.lmax,iTlm)
        hp.fitsfunc.write_map(aobj.fmap['s'][rlz],Tmap,overwrite=True)

    # generate random noise alm and covert to map
    nl = np.ones(aobj.lmax+1) * (aobj.sigma*np.pi/10800./local.Tcmb)**2
    
    for rlz in tqdm.tqdm(aobj.rlz,desc='noise map'):
        
        if misctools.check_path(aobj.fmap['n'][rlz],overwrite=overwrite,verbose=verbose): continue

        nlm = cs.utils.gauss1alm(aobj.lmax,nl)
        nmap = cs.utils.hp_alm2map(aobj.nside,aobj.lmax,aobj.lmax,nlm)
        hp.fitsfunc.write_map(aobj.fmap['n'][rlz],Tmap,overwrite=True)
        

        
def map2alm(aobj,overwrite=False,verbose=True):
    
    W = hp.fitsfunc.read_map(aobj.amask,verbose=False)
    
    for rlz in tqdm.tqdm(aobj.rlz,desc='map2alm'):

        if misctools.check_path(aobj.falm['c']['T'][rlz],overwrite=overwrite,verbose=verbose): continue
        
        smap = hp.fitsfunc.read_map(aobj.fmap['s'][rlz],verbose=False)
        nmap = hp.fitsfunc.read_map(aobj.fmap['n'][rlz],verbose=False)
        wslm = cs.utils.hp_map2alm(aobj.nside,aobj.lmax,aobj.lmax,W*smap)
        wnlm = cs.utils.hp_map2alm(aobj.nside,aobj.lmax,aobj.lmax,W*nmap)
        
        pickle.dump((wslm+wnlm),open(aobj.falm['c']['T'][rlz],"wb"),protocol=pickle.HIGHEST_PROTOCOL)    
        pickle.dump((wnlm),open(aobj.falm['n']['T'][rlz],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def alm2aps(aobj,overwrite=False,verbose=True):
    
    cl = {s: np.zeros((len(aobj.rlz),6,aobj.lmax+1)) for s in ['c','n','s']}
    
    print('load W2')
    W = hp.fitsfunc.read_map(aobj.amask,verbose=False)
    wn = tools_cmb.get_wfactor(W)


    for ii, rlz in enumerate(tqdm.tqdm(aobj.rlz,desc='alm2aps')):
        
        if misctools.check_path(aobj.fcls['c'][rlz],overwrite=overwrite,verbose=verbose): continue
        
        alm = {}
        for s in ['c','n']:
            alm[s] = pickle.load(open(aobj.falm[s]['T'][rlz],"rb"))
            cl[s][ii,0,:] = cs.utils.alm2cl(lmax,alm[s])/w2
        cl['s'][ii,0,:] = cs.utils.alm2cl(lmax,alm['c']-alm['n'])/w2
    
        # save cl for each rlz
        np.savetxt(aobj.fcls['c'][rlz],np.concatenate((aobj.l[None,:],cl['c'][ii,:,:])).T)
        np.savetxt(aobj.fcls['s'][rlz],np.concatenate((aobj.l[None,:],cl['s'][ii,:,:])).T)
        np.savetxt(aobj.fcls['n'][rlz],np.concatenate((aobj.l[None,:],cl['n'][ii,:,:])).T)

    for s in ['c','n','s']:
        if misctools.check_path(aobj.fscl[s],overwrite=overwrite,verbose=verbose): continue
        np.savetxt(aobj.fscl[s],np.concatenate((aobj.l[None,:],np.mean(cl[s],axis=0),np.std(cl[s],axis=0))).T)


def get_wfactor(W):
    
    wn = np.ones(5)
    for n in range(1,5):
        wn[n] = np.average(W**n)
    
    M = W.copy()
    M[M!=0] = 1.
    wn[0] = np.average(M)
    return wn
        
