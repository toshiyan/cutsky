#!/usr/bin/env python
# coding: utf-8

import numpy as np
import local
import tools_cmb

# global fixed parameters
kwargs = {'snmin':0,'snmax':100,'ascale':3.0,'wtype':'com15'}
kwargs_ov = {'overwrite':True,'verbose':False}

kwargs_qrec0 = {'n0max':50,'mfmax':100,'rlmin':500,'qlist':['TT'],'bhe':['src']}
kwargs_qrec1 = {'n0max':50,'mfmax':100,'rlmin':500,'qlist':['TT']}

run_cmb = ['map','alm','aps']


tools_cmb.interface(run_cmb,kwargs=kwargs,kwargs_ov=kwargs_ov)

for kwargs_qrec in [kwargs_qrec0,kwargs_qrec1]:
    aobj = local.init_analysis_params(**kwargs)
    tools_lens.interface(aobj,run=['norm','qrec','n0','rdn0','mean','aps'],kwargs_ov=kwargs_ov,kwargs_qrec=kwargs_qrec)

