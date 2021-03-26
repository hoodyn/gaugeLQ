# Integrating out the "Pati-Salam" gauge leptoquark
# to SMEFT.

import numpy as np
from math import pi as Pi
from math import log as Log


# According to hep-ph/980647 and my SM/g3withTresholds.nb
def alpha_s(mu):
	Lambda6 = 0.092 # GeV
	return 4*Pi*(	1/(7*Log(mu**2/Lambda6**2)) 
	  -(26*Log(Log(mu**2/Lambda6**2)))/(343*Log(mu**2/Lambda6**2)**2)  )

class FlavorRange:
	full = [(lbar, l, qbar, q) 
	  for lbar in range(3) for l in range(3) 
	  for qbar in range(3) for q in range(3) ]
	upDiag = [*[
	  (l,l,qbar,q) 
	  for l in range(3)
	  for q in range(3) for qbar in range(q+1)
	  ], *[
	  (lbar, l, qbar, q)
	  for l in range(1,3) for lbar in range(l)
	  for q in range(3) for qbar in range(3)
	  ]]
	upDiag.sort()
	sample = [(lbar, l, 0, 0) 
	  for l in range(3) for lbar in range(3)]

def toString(flavorMultiIndex):
	return "".join([str(i+1) for i in flavorMultiIndex])

#Returns float if complex z is actually real
def realify(z):
	if z.imag == 0.:
		return z.real
	else:
		return z


def ULR2WC(UL, UR, mU1=10000, IgnoreGaugeCoupling=False):
	"""Function returning the WilsonCoefficients in wcxf.Warsaw basis,
	given the numerical form of the quark-lepton mixing matrices UL and UR.
	"""
	if IgnoreGaugeCoupling:
		commonFactor = 1/(mU1**2)
	else:
		commonFactor = 2*Pi*alpha_s(mU1)/(mU1**2)
	_UL = np.array(UL)
	_UR = np.array(UR)
	WC_ed   = WC_dict('ed',  _UR, _UR, -1. *commonFactor,FlavorRange.upDiag)
	WC_ledq = WC_dict('ledq',_UR, _UL, +2. *commonFactor,FlavorRange.full)
	WC_lq1  = WC_dict('lq1', _UL, _UL, -0.5*commonFactor,FlavorRange.upDiag)
	WC_lq3  = WC_dict('lq3', _UL, _UL, -0.5*commonFactor,FlavorRange.upDiag)
	
	WCs = {**WC_ed,**WC_ledq,**WC_lq1,**WC_lq3}
	return WCs

def WC_dict(WCname='WC', U1=None, U2=None, commonFactor=1., flavRange=None):
	if U1 is None:
		U1 = np.identity(3)
	if U2 is None:
		U2 = np.identity(3)
	if flavRange is None:
		flavRange = FlavorRange.sample
	WC_dict = {
	  WCname+'_'+toString((lb,l,qb,q)):
	  realify(commonFactor * U1[q,lb].conjugate() * U2[qb,l])
	  for (lb,l,qb,q) in flavRange
	  }
	return WC_dict

