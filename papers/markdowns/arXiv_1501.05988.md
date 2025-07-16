# arXiv:1501.05988

**Paper ID:** a68e6b5ba4ec75e63ea8c2cb03e06094

**PDF Path:** apl/Superconductors/arXiv:1501.05988.pdf

**Processing Status:** complete

**Captions Added:** 6

**Generated:** 2025-06-24T13:44:27.013559

---

# Tapering studies for Terawatt level X-ray FELs with a superconducting undulator

C. Emma,<sup>1</sup> J. Wu,<sup>2</sup> P. Emma,<sup>2</sup> Z. Huang,<sup>2</sup> and C. Pellegrini1, 2

<sup>1</sup>University of California, Los Angeles, California 90095, USA

<sup>2</sup>Stanford Linear Accelerator Center, Menlo Park, California 94025, USA

(Dated: August 18, 2024)

We study the tapering optimization scheme for a short period, less than two cm, superconducting undulator, and show that it can generate 4 keV X-ray pulses with peak power in excess of 1 terawatt, using LCLS electron beam parameters. We study the effect of undulator module length relative to the FEL gain length for continous and step-wise taper profiles. For the optimal section length of 1.5m we study the evolution of the FEL process for two different superconducting technologies NbTi and Nb3Sn. We discuss the major factors limiting the maximum output power, particle detrapping around the saturation location and time dependent detrapping due to generation and amplification of sideband modes.

PACS numbers: 41.60.Cr, 41.60.Ap

#### I. INTRODUCTION

Much recent scientific effort has been devoted to studying the possibility of using a tapered undulator [\[1\]](#page-3-0) to achieve terawatt level hard x-ray pulses in the next generation of Free Electron Lasers (FELs) [\[2\]](#page-3-1) [\[3\]](#page-3-2) [\[4\]](#page-3-3) [\[5\]](#page-3-4) [\[6\]](#page-3-5). The motivation for pursuing this goal comes primarily from the bioimaging community where a terawatt power coherent x-ray source would open the doors to single molecule imaging [\[7\]](#page-3-6) [\[8\]](#page-3-7) [\[9\]](#page-3-8). Recent numerical work [\[10\]](#page-4-0) [\[11\]](#page-4-1) shows that a self seeded hard X-ray FEL with LCLS-II like parameters in a 200-m permanent magnet undulator has the capability of reaching TW level pulses with the longitudinal and transverse coherence necessary for coherent X-ray imaging [\[12\]](#page-4-2). Spatial constraints, an increase in tunability and a longer machine lifetime have since led towards considering using superconducting technology for the undulator design rather than permanent magnets [\[13\]](#page-4-3).

In this study we present the results of tapering optimization for the cases of two different superconducting technologies, NbTi and Nb3Sn and assess the possibility of achieving TW levels of power in a 140 m undulator with periodic break sections. We study the impact of changing the undulator section length on the performance for both NbTi and Nb3Sn. We then present GEN-ESIS simulation results for the NbTi case with the optimal choice of section length. Finally we discuss the problem of particle detrapping as a major source of performance degradation in tapered FELs, and propose some ideas to improve on current tapering designs by increasing the trapping and the extraction efficiency in the tapered section of the undulator.

The physical system studied is a 140 m undulator composed of a 20 m SASE section followed by a a 120 m tapered undulator with sections 1 to 3 m in length. The SASE and tapered sections are separated by a self seeding chicane which delivers a 5 MW monochromatic seed at a photon energy of 4keV. We assume the undulator modules to be separated in all cases by 0.5 m break sections where we install the focusing quarupoles in a FODO configuration. The system parameters for both NbTi and

| Parameter Name                                                                                                    | NbTi                                              | Nb3Sn                                            |
|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|--------------------------------------------------|
| Electron Beam:                                                                                                    |                                                   |                                                  |
| Beam Energy E0<br>Energy Spread σE<br>Peak Current Ipk<br>Normalized Emittances x,n/y,n<br>Average β function hβi | 7.2 GeV<br>1.5 MeV<br>4 kA<br>0.4/0.4 µ m<br>12 m | 6.8 GeV<br>1.5 MeV<br>4 kA<br>0.4/0.4µ m<br>12 m |
| Undulator:                                                                                                        |                                                   |                                                  |
| Undulator Period λw<br>Undulator Parameter (RMS) aw<br>Magnetic Gap g<br>Integrated Quad Field Bq                 | 20 mm<br>2.263<br>7.2 mm<br>4.5 T                 | 18 mm<br>2.263<br>7.2 mm<br>4.5 T                |
| Radiation:                                                                                                        |                                                   |                                                  |
| Photon Energy Eγ<br>Peak radiation power input Pseed<br>Seed laser size σr<br>Rayleigh Range ZR                   | 4keV<br>5 MW<br>31 µ m<br>10 m                    | 4keV<br>5MW<br>31 µ m<br>10 m                    |
| FEL:                                                                                                              |                                                   |                                                  |
| FEL parameter ρ<br>3−D<br>FEL 3-D gain length L<br>g<br>Fresnel Parameter Fd                                      | 7.67 ×10−4<br>1.25 m<br>8                         | 7.43 ×10−4<br>1.16 m<br>8.6                      |

Nb3Sn are described in table 1 where the beam energy is chosen in each case to generate 4keV photons.

The tapering optimization method used is the one described in Ref. [\[11\]](#page-4-1) with a transversely parabolic and longitudinally uniform electron beam distribution. The advantages of using a transversely parabolic beam as opposed to a Gaussian in a tapered X-FEL are described in detail in Ref. [\[10\]](#page-4-0). The quadrupole gradient is kept constant for simplicity and is set to achieve an average β function of βav =12 m throughout the undulator. The effect of varying the electron beam size by changing the

![](_page_1_Figure_0.jpeg)

**Caption:** Figure 1 compares the power evolution for different undulator section lengths (Lsec) ranging from 1 to 3 meters, each separated by periodic 0.5 m breaks, using Nb3Sn parameters post self-seeding chicane (z=0). The graph provides insights into how shorter sections facilitate lower transverse beam envelope oscillations and better emulate a continuous magnetic field profile, leading to improved FEL performance. It reveals that 1-2 m sections slightly improve output power between continuous and step tapering, whereas 3 m sections show reduced power from 1.3 TW to 0.5 TW. Consequently, a 1.5 m section length is deemed optimal, balancing interaction length and minimizing beam size oscillations.


FIG. 1: Power evolution comparison for different undulator section lengths Lsec separated by periodic 0.5 m breaks. The simulations are time independent and use the Nb3Sn parameters listed in table 1 starting after the self seeding chicane (z=0). The transverse beam size is normalized to the input beam size σe,<sup>0</sup> and the undulator parameter is normalized to the initial value aw0.

quadrupole focusing strength will further increase the output power as discussed in Ref. [\[11\]](#page-4-1) however this option will not be examined in this work.

# II. SIMULATION RESULTS

#### A. Section length study

In this section we investigate the effect of the undulator section length on the tapered FEL performance. Since the ideal tapering profile is a smoothly varying function of z we expect that a longer section length limits the performance since shorter sections better approximate a continuous magnetic field profile. Furthermore the transverse beam envelope oscillations increase for larger section lengths ∆β <sup>2</sup>/β<sup>2</sup> av = (βavL)/(β 2 av − L 2 ) further degrading the FEL performance as shown in Fig. 1 for the continuous tapering case. We find that for the same average beta function βav as the section length becomes larger than the gain length the maximum output power is significantly reduced as illustrated in Fig. 1. Undulator sections of 1-2 m show a marginal improvement in the output power when going from the continuous to the step taper which can be attributed to the additional phase slippage of the electrons with respect to the ponderomotive wave in the step-wise case vs the continuous case. On the other hand in the 3 m case the maximum power decreases from 1.3 TW to 0.5 TW. This is a result of the more pronounced oscillations in the beam β function and the phase mismatch between undulator sections as the discontinuity in magnetic field value is larger for longer sections. The 1 m undulator sections are thus optimal for minimizing oscillations in the beam size and approximating a smooth taper profile however the filling factor for such short sections limits the interaction length and

![](_page_1_Figure_7.jpeg)

**Caption:** Figure 2 presents the time-independent GENESIS simulation results detailing the optimized taper profile and evolution of FEL radiation power for NbTi and Nb3Sn superconducting undulators. The left graph demonstrates the taper profile, while the right graph shows a comparison of power evolution for NbTi, Nb3Sn, and a permanent magnet undulator. For NbTi and Nb3Sn, a significant increase in output power over a permanent magnet design is observed, reflecting extraction efficiencies of 6.08% and 5.89% respectively. The simulations underscore the potential of superconducting technologies to achieve high output powers over the saturation power factor, thereby promoting their application in advanced FEL systems.


FIG. 2: Time independent GENESIS simulation results for the optimized taper profile and evolution of the FEL radiation power for NbTi and Nb3Sn superconducting undulators. We also plot for comparison the case of a permanent maget undulator of 2.6 cm period and undulator parameter aw<sup>0</sup> = 1.77 as discussed in the text.

overall extraction efficiency. Thus we determine that to obtain a reasonable filling factor and large output power a 1.5 m section length is the optimal choice for the undulator design.

### B. FEL evolution with 1.5 m undulator sections

For our optimal choice of section length time, independent tapering optimizations produce the magnetic field profiles displayed in Fig. 2. In both cases the optimal functional form of the tapering law is very close to quadratic aw(z) ∼ aw<sup>0</sup> - 1 − c(z − z0) 2 with varying taper strengths c and a start location around z<sup>0</sup> ∼ 10 m after the self-seeding chicane. The quadratic scaling can be understood from the 1-D theory of tapered FELs [\[1\]](#page-3-0). The change in energy for a resonant electron is determined by a z dependent poderomotive gradient:

$$\frac{d\gamma\_r^2}{dz} = -\frac{eE(z)a\_w(z)}{mc^2}\sin\Psi\_R\tag{1}$$

where for simplicity we assume the resonant phase Ψ<sup>R</sup> to be constant, E(z) is the magnitude of the growing radiation field and aw(z) is the RMS magnitude of the tapered magnetic field. The FEL will radiate at a wavelength λ<sup>s</sup> if the following z dependent resonance condition is satisfied:

$$
\gamma\_r^2(z) = \frac{\lambda\_w}{2\lambda\_s} \left( 1 + a\_w(z)^2 \right) \tag{2}
$$

If we assume that the radiative process in the tapered section of the undulator is mostly due to coherent emission, with the electric field evolving according to E(z) ∝ z, to satisfy simultaneously Eq. 1-2 the magnetic field aw(z) must decrease quadratically with z. With the optimal taper profiles for NbTi and Nb3Sn we obtain extraction efficiencies of 6.08% and 5.89% respectively, and output powers over a factor of 75 larger than the saturation power in both cases (see Fig. 2). Applying the same tapering optimization for a Permanent Magnet

![](_page_2_Figure_1.jpeg)

**Caption:** Figure 3 demonstrates the real part of the electron beam refractive index (left) and the on-axis resonant phase (right) as derived from time-independent GENESIS simulations for NbTi superconductors. These graphs show how the refractive index evolves with distance in the undulator, and how the resonant phase affects particle trapping under the given electron beam parameters. They underscore the complex interplay of guiding effects and phase relationships in optimizing the FEL process, informing adjustments in system design to maximize efficiency and output.


FIG. 3: Real part of the electron beam refractive index (left) and on axis resonant phase (right) for NbTi superconducting undulator parameters from time independent GENESIS simulation.

Undulator (PMU) with a 26 mm period, beam energy of 6.7 GeV and an RMS undulator parameter aw0=1.77 we report a ∼ 60 % reduction in output power compared to the superconducting case with P PMU rad = 1.1 TW at the undulator exit. As mentioned previously this is one of the motivations for moving from permanent magnets to superconducting technology in the design of Terawatt level X-FELs such as LCLS-II. We also note that for the superconducting case a reduction in the average β function will further increase the efficiency while this cannot be done in the permanent magnet case.

#### C. Limits on optimization due to parasitic effects

The FEL process is dominated by the effects of refractive guiding and particle detrapping in the tapered section of the undulator. In Fig. 3 we show the evolution of the real part of the refractive index [\[14\]](#page-4-4) (in the resonant particle approximation) and the resonant phase given by the expressions:

$$\Re(n-1) = \frac{\omega\_{p0}^2}{\omega\_s^2} \frac{r\_{b,0}^2}{r\_b^2} \frac{a\_w}{2|a\_s|} [JJ] \left\langle \frac{\cos \Psi\_R}{\gamma\_R} \right\rangle \qquad (3)$$

$$
\sin \Psi\_R(z) = \chi \frac{|a\_w'(z)|}{E(z)} \tag{4}
$$

where χ = (2∗me∗c <sup>2</sup>/e)(λw/2λs)(1/ √ 2[JJ]) is a constant independent of z and the other symbols have their usual meaning. From these expressions we see that if we want to maintain strong optical guiding, the bunching must be preserved during the tapered section of the undulator. However as the radiation field grows the effect of optical guiding will decrease as 1/|as| inducing a self-limiting mechanism on the growth of the field [\[15\]](#page-4-5). Furthermore, in order to provide the largest ponderomotive gradient to induce the greatest energy loss in the electrons, it is desirable to increase the resonant phase throughout the tapered section. Again here we find a selflimiting mechanism which limits how much one should increase ΨR, since a larger Ψ<sup>R</sup> results in a smaller ponderomotive bucket creating a tradeoff between the ponderomotive potential strength and the number of trapped

![](_page_2_Figure_9.jpeg)

**Caption:** Figure 4 illustrates the radiation phase (on-axis) and trapping fraction from time-independent GENESIS simulations of the NbTi superconducting undulator. The graph on the left shows the radiation phase becoming more constant as the phase velocity of light approaches c beyond 40 meters of the undulator, indicating reduced gain. The graph on the right tracks the trapping fraction, showing a significant detrapment of particles as the resonant phase increases from zero to ψ<sup>r</sup> = 20° from z=10-50 m, trapping of approximately 20% of the beam occurs due to changes in the ponderomotive bucket. The figure highlights the need for mechanisms such as phase shifters to mitigate particle detrapping and sustain FEL performance.


FIG. 4: Radiation Phase (on axis) and trapping fraction from time independent GENESIS simulations of NbTi. The area after 40 m is when the gain is so low that the phase velocity of light is almost c thus the radiation phase is constant

electrons [\[1\]](#page-3-0). From Fig. 3 and 4 we see that as the resonant phase increases from zero to ψ<sup>r</sup> = 20<sup>o</sup> from z=10-50 m, the phase shift in the ponderomotive bucket causes ∼ 20% of the particles to detrap. This can be mitigated by introducing phase shifters in the break sections between 10-50 m and is an effect which will be examined quantitatively in future studies.

# D. Time dependent effects. Sideband growth and sideband induced detrapping

As has been pointed out in previous work [\[16\]](#page-4-6) [\[11\]](#page-4-1) [\[10\]](#page-4-0), the performance of tapered hard X-ray FELs is also limited by particle detrapping due to time dependent effects. Shot-noise fluctuations in the electron beam current induce modulations in the radiation beam longitudinal profile. This can cause particle detrapping from the top and bottom of the ponderomotive bucket as the modulated radiation field slips through the beam and the particles experience fluctuations in the bucket height. Furthermore, the resonant interaction between the FEL primary wave k<sup>0</sup> and the electron synchrotron motion drives a sideband instability which amplifies parasitic modes at wavenumbers k<sup>0</sup> ± Ω<sup>s</sup> where Ω<sup>s</sup> is the synchrotron wavenumber:

$$
\Omega\_s^2(z) = \frac{eE(z)a\_w(z)}{m\_e c^2 k\_w} \tag{5}
$$

The growth of the sidebands not only increases the bandwidth of the FEL signal but also induces further particle detrapping [\[17\]](#page-4-7) [\[18\]](#page-4-8).

Time dependent effects are analysed by simulating the beam parameters in table I for a longitudinally uniform bunch of 37.5 fs. The simulation results are shown in Fig. 5 for a longer undulator of 160 m. The simulation was done for a longer undulator to emphasize the peak power saturation in the final sections of the tapered undulator. From Fig. 5 we can see that there is a difference in peak output power (averaged over time) compared with the time independent results, namely 1.78 TW in the time independent case compared to 1.55 TW after 120 m time dependent. The drop can be attributed to the growth of

![](_page_3_Figure_0.jpeg)

**Caption:** Figure 5 shows the time-dependent GENESIS simulation results for the NbTi case parameters with a 37.5 fs long electron beam having a flattop longitudinal profile. The left graph depicts the growth of the bunching factor along the undulator length, which significantly varies at different sections. The right graph shows the relative standard deviation of radiated power over input standard deviation as a function of undulator length, indicating power saturations. This figure demonstrates the limitations imposed by time-dependent effects in reaching peak output power, with a drop observed from 1.78 TW in the time-independent case to 1.55 TW after 120 m in the time-dependent analysis. Such effects are attributed to the growth of FEL synchrotron sidebands, which are highlighted here as a critical factor affecting the efficiency and fidelity of the FEL output in long undulator configurations.


FIG. 5: Time dependent GENESIS simulation results for the NbTi case parameters in Table 1. The beam is 37.5 fs long with a flattop longitudinal profile.

![](_page_3_Figure_2.jpeg)

**Caption:** Figure 6 shows the spectrum of FEL synchrotron sidebands, highlighting their growth at various positions within the undulator. Sidebands intensify in regions where the synchrotron frequency changes less rapidly, especially in the latter sections of the undulator, causing detrimental impacts on the coherence and spectral purity of the FEL output. This visualization aids in understanding the effects of synchrotron motion on particle detrapping and informs strategies like varying the synchrotron frequency to mitigate these impacts, contributing to more robust FEL operation.


FIG. 6: Growth of the FEL synchrotron sidebands at different locations in the undulator. The growth of the sidebands intensified in the later section of the undulator where the synchrotron frequency is changing less rapidly (see Fig. 5).

the sidebands shown in a plot of the spectrum at various z locations (see Fig. 6). This growth can be mitigated by changing the synchrotron frequency along the tapered section of the undulator [\[19\]](#page-4-9) [\[20\]](#page-4-10). In the final section of the undulator (z >100m) the variation in the synchrotron frequency is small due to the saturation of the electric field. It might be possible to compensate for this effect by introducing a cubic or quartic term to the taper profile. This option will also be explored in the future.

# III. CONCLUSION

We have presented a numerical study of tapering optimization for a self seeded hard X-ray FEL with superconducting undulator parameters. This work has demonstrated numerically that peak power levels over 1 TW can be achieved in a 120 m undulator with break sections and an optimized taper profile for two separate superconducting technologies: NbTi and Nb3Sn. We demonstrated a 60% increase in output power for the superconducting technology when compared with a permanent magnet design. We have also outlined the effects which limit the growth of the power in the tapered undulator and have found that particle detrapping is the main obstacle to achieving larger extraction efficiencies. The two causes of particle detrapping we have discussed are due to an increase of the resonant phase after the exponential gain regime and sideband induced detrapping due to time dependent effects. To mitigate these two effects we propose to study applying phase shifters at the location of initial saturation and introducing a faster term in the taper profile to reduce the sideband growth by changing the synchrotron frequency faster in the final sections of the undulator.

# ACKNOWLEDGMENT

The authors would like to thank Dr. G. Marcus and J. Duris for useful discussions. This work was supported by U.S. D.O.E. under Grant No. DE-SC0009983.

- <span id="page-3-0"></span>[1] N. M. Kroll, P. L. Morton, and M. Rosenbluth, "Freeelectron lasers with variable parameter wigglers," Quantum Electronics, IEEE Journal of, vol. 17, pp. 1436–1468, August 1981.
- <span id="page-3-1"></span>[2] W. Fawley et al., "Toward tw-level, hard x-ray pulses at lcls," in FEL 2011 Conference proceedings, Shanghai, China, pp. 160 – 163, 2011.
- <span id="page-3-2"></span>[3] X. Huang et al., "Optimization of a terawatt free electron laser. particle accelerator. proceedings, 3rd international particle accelerator conference, ipac 2012, new orleans, usa, may 2-25, 2012," 2012.
- <span id="page-3-3"></span>[4] A. Mak, F. Curbis, and S. Werin, "Methods for the optimization of a tapered free electron laser," in IPAC 2014 Conference proceedings, Dresden, Germany, pp. 2909 – 2911, 2014.
- <span id="page-3-4"></span>[5] S. Serkez et al., "Proposal to generate 10 tw level

femtosecond x-ray pulses from a baseline undulator in conventional sase regime at the european xfel," in FEL 2014 Conference proceedings, Basel, Switzerland, MOP057,2014.

- <span id="page-3-5"></span>[6] E. A. Schneidmiller and M. Yurkov, "Optimization of a high efficiency fel amplifier," in FEL 2014 Conference proceedings, Basel, Switzerland, MOP065,2014.
- <span id="page-3-6"></span>[7] R. Neutze et al., "Potential for biomolecular imaging with femtosecond x-ray pulses," Nature, vol. 406, pp. 752–757, 2000.
- <span id="page-3-7"></span>[8] H. Chapman et al., "Femtosecond diffractive imaging with a soft x-ray free electron laser," Nature Physics, vol. 2, pp. 839–843, 2006.
- <span id="page-3-8"></span>[9] K. Gaffney and H. N. Chapman, "Imaging atomic structure and dynamics with ultrafast x-ray scattering," Science, vol. 316, pp. 1444–1448, 2007.
- <span id="page-4-0"></span>[10] C. Emma, J. Wu, K. Fang, S. Chen, S. Serkez, and C. Pellegrini, "Terawatt x-ray free-electron-laser optimization by transverse electron distribution shaping," Phys. Rev. ST Accel. Beams, vol. 17, p. 110701, Nov 2014.
- <span id="page-4-1"></span>[11] Y. Jiao, J. Wu, Y. Cai, A. W. Chao, W. M. Fawley, J. Frisch, Z. Huang, H.-D. Nuhn, C. Pellegrini, and S. Reiche, "Modeling and multidimensional optimization of a tapered free electron laser," Phys. Rev. ST Accel. Beams, vol. 15, p. 050704, May 2012.
- <span id="page-4-2"></span>[12] C. Emma et al., "Radiation properties of tapered hard x-ray free electron lasers," in FEL 2014 Conference proceedings, Basel, Switzerland, MOC03,2014.
- <span id="page-4-3"></span>[13] P. Emma et al., "A plan for the development of superconducting undulator prototypes for lcls-ii and future fels," in FEL 2014 Conference proceedings, Basel, Switzerland, THA03,2014.
- <span id="page-4-4"></span>[14] D. Prosnitz, A. Szoke, and V. K. Neil, "High-gain, freeelectron laser amplifiers: Design considerations and simulation," Phys. Rev. A, vol. 24, p. 1436, Sep 1981.
- <span id="page-4-5"></span>[15] W. Fawley, "'optical guiding' limits on extraction efficiencies of single pass, tapered wiggler amplifiers," vol. A375,

pp. 550–562, 1996.

- <span id="page-4-6"></span>[16] J. Duris, A. Murokh, and P. Musumeci To be published 2014.
- <span id="page-4-7"></span>[17] R. C. Davidson and J. S. Wurtele, "Singleparticle analysis of the freeelectron laser sideband instability for primary electromagnetic wave with constant phase and slowly varying phase," Physics of Fluids (1958-1988), vol. 30, no. 2, 1987.
- <span id="page-4-8"></span>[18] T. B. Yang and R. C. Davidson, "Influence of the trappedelectron distribution on the sideband instability in a helical wiggler freeelectron laser," Physics of Fluids B: Plasma Physics, vol. 2, no. 10, 1990.
- <span id="page-4-9"></span>[19] R. P. Pilla and A. Bhattacharjee, "Elimination of the sideband instability in variableparameter freeelectron lasers and inverse freeelectron lasers," Physics of Plasmas, vol. 1, no. 2, 1994.
- <span id="page-4-10"></span>[20] W. M. Sharp and S. S. Yu, "Two-dimensional Vlasov treatment of free-electron laser sidebands," Physics of Fluids B, vol. 2, pp. 581–605, Mar. 1990.