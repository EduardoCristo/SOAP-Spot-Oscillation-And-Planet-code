![SOAP](https://github.com/iastro-pt/SOAPprivate/workflows/SOAP/badge.svg?branch=main)

[![read SOAP](https://img.shields.io/badge/paper-SOAP-7d93c7.svg?style=flat)](https://arxiv.org/abs/1206.5493)
[![read SOAPT](https://img.shields.io/badge/paper-SOAP_T-7d93c7.svg?style=flat)](https://arxiv.org/abs/1211.1311)
[![read SOAP2](https://img.shields.io/badge/paper-SOAP_2-7d93c7.svg?style=flat)](https://arxiv.org/abs/1409.3594)
[![read SOAP3 rings](https://img.shields.io/badge/paper-SOAP_3%20rings-7d93c7.svg?style=flat)](https://arxiv.org/abs/1709.06443)
[![read SOAP3 DR](https://img.shields.io/badge/paper-SOAP_3%20DR-7d93c7.svg?style=flat)](https://arxiv.org/abs/2002.08227)


SOAP estimates the effects of active regions (spots or plages), planets, and
rings on radial-velocity (RV) and photometric measurements. The code calculates
the photometric variations induced by active regions (flux at 5293 Angstrom) as
well as the RV, bisector span (BIS), and full width at half maximum (FWHM) as
defined with the cross-correlation technique and optimized to reproduce HARPS
observations.


### INSTALLATION

SOAP is written Python.
To install run the following commands

```bash
git clone https://github.com/EduardoCristo/SOAP-Spot-Oscillation-And-Planet-code SOAP
cd SOAP

pip install -e .
```

then you can use SOAP from Python with 

```
>>> import SOAP
```


