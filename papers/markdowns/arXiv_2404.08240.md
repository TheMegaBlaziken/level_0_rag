# arXiv:2404.08240

**Paper ID:** 213714e2f5c2cadcea29b9f003c06dc3

**PDF Path:** apl/Superconductors/arXiv:2404.08240.pdf

**Processing Status:** complete

**Captions Added:** 9

**Generated:** 2025-06-24T13:44:27.999277

---

# BEAM DYNAMICS STUDY OF THE HIGH-POWER ELECTRON BEAM IRRADIATOR USING NIOBIUM-TIN SUPERCONDUCTING CAVITY <sup>∗</sup>

O. Tanaka, Y. Honda, M. Yamamoto, T. Yamada, H. Sakai, High Energy Accelerator Research Organization, KEK 305-0801 Oho, Tsukuba, Ibaraki, Japan olga@post.kek.jp

# ABSTRACT

A compact accelerator design for irradiation purposes is being proposed at KEK. This design targets an energy of 10 MeV and a current of 50 mA. Current design includes a 100 kV thermionic DC electron gun with an RF grid, 1-cell normal-conducting buncher cavity, and Nb3Sn superconducting cavities to accelerate the beam to the final energy of 10 MeV. The goal of the present beam dynamics study is the beam loss suppression (to the ppm level), since it results in a thermal load on the cavity. Then the beam performance at the accelerator exit should be confirmed. The main issue was to transport the beam without loss, since the initial electron energy (100 keV) is low, and the beam parameters are intricately correlated. In addition, the space charge effect is considerable. For this reason, simultaneous optimization of multiple parameters was necessary. Here we report optimization results and their effect on the design of the machine.

*K*eywords Beam injection, extraction and transport · Beam dynamics calculations · Electron beam

# 1 Introduction

At the Compact Energy Recovery Linac (cERL) at KEK [\[1\]](#page-6-0), an irradiation beam line was constructed and successfully commissioned [\[2\]](#page-6-1). Following the recent trend in accelerator science to design a compact high-current irradiation-type accelerators [\[3\]](#page-6-2), [\[4\]](#page-6-3) and based on the results of irradiation experiments at the cERL [\[5\]](#page-6-4), we aim to develop a 50 mA electron beam irradiation facility working at the total beam energy of 10 MeV.

High-current beam acceleration can be achieved by using a superconducting cavities (SCs). In recent years, niobium tin (Nb3Sn) accelerating cavities have attracted attention as next-generation accelerating cavities to replace niobium ones [\[6\]](#page-6-5). A higher transition temperature (18.3 K) of Nb3Sn makes it possible to operate the beam using only a simple small refrigerator instead of the conventional large He one. That allows to design a compact facility. The conceptual design of the proposed machine with special emphasize on the Nb3Sn SCs implementation is given in [\[7\]](#page-6-6).

The overall design of the machine assumes the electron beam to be of 10 MeV and 50 mA. Fig[.1](#page-1-0) shows a schematic of the entire system. It includes: an injector part with an electron gun of specific design, superconducting cryomodule, and an irradiation part, so that the machine can irradiate a large current beam. In order to evaluate the acceleration acceptance of an electron beam irradiation facility, we performed a three-dimensional particle tracking including the space-charge effect with the acceleration cavity system consisted of a buncher, two 1-cell SCs, and five 2-cell SCs. We found solutions to mitigate the beam loss applying a multi objective optimization. In the present study the beam dynamics issues associated with the irradiator design are discussed in details.

# 2 System overview

The electron gun is a 100 kV thermionic DC gun with an RF grid that produces a repetitive longitudinally packed pulsed electron beam at 650 MHz [\[8\]](#page-6-7). After that, we assumed to compress the pulsed beam in the beam direction in a

<sup>∗</sup>*Citation*: Authors. Title. Pages.... DOI:000000/11111.

![](_page_1_Figure_0.jpeg)

**Caption:** Figure 1: Conceptual design diagram of the 10 MeV, 50 mA accelerator using Nb3Sn SRF cavities displays the critical infrastructure from electron gun to irradiation components. The conceptual overview demonstrates how these elements work together in achieving a high-power, compact accelerator. The significance lies in enhancing current transport without beam loss through the use of advanced Nb3Sn technology and compact design principles .


<span id="page-1-0"></span>Figure 1: Conceptual design of 10 MeV, 50 mA accelerator using Nb3Sn SRF cavities.

1-cell normal-conducting buncher cavity, and then accelerate it to 10 MeV using two 1-cell SCs (β = 0.8) and five 2-cell SCs. Note, low-β resonators are just cavities that accelerate efficiently particles with velocity β < 1 [\[9\]](#page-7-0). The RF frequency of all SCs is 1.3 GHz. The buncher cavity operates at 650 MHz. Two solenoids were installed before and after the buncher to provide transverse focusing. In the irradiation section there are two quadrupoles placed 1 m downstream the cryomodule to adjust the beam size. Then the beam is bent downward by the dipole and reaches the extraction window (See Fig[.2\)](#page-1-1).

The design of the electron gun system and the injection part was based on that of cERL [\[1\]](#page-6-0). A thermionic gun was selected for the electron source because it can stably supply a large current of 10 mA or more and an acceleration voltage of 100 kV [\[10\]](#page-7-1), as shown in Fig[.2.](#page-1-1) A collimator was introduced in the injector design to limit the formation of a vacuum pressure step between the electron gun and the SCs. Other reasons were to control beam size and to suppress beam loss in and downstream of the SCs. At the end of the injector section, there is a profile monitor to observe the beam's shape and a Faraday cup to measure the amount of beam charge. An exhaust system is appropriately arranged to maintain ultra-high vacuum in this section.

![](_page_1_Figure_4.jpeg)

**Caption:** Figure 2: This schematic represents the layout of key accelerator components, including the electron gun, normal-conducting buncher cavity, superconducting cavities, solenoids, and extraction systems. Each component's position and function within the compact 10 MeV system is illustrated, designed for optimal transport and acceleration of electron beams. This layout supports system integration by providing insight into the beam path and component interdependencies for achieving high efficiency with minimal beam loss in a compact setup .


<span id="page-1-1"></span>Figure 2: Layout of accelerator components.

Originally we used a 1.3 GHz buncher cavity identical to those at cERL. The beam velocity-modulated in the buncher cavity compresses the bunch length to about 30 ps at the first SC position (Z 2.4 m). This corresponds to a phase width of about ±14 degrees for an acceleration frequency of 1.3 GHz. Bunch compression is seen at 1.3 GHz with RF phase nonlinear components and can also be a source of loss. It is possible to reduce these losses with a collimator, but it is difficult to compress the 100 ps bunch length with 1.3 GHz cERL buncher. Therefore, we redesign and apply the buncher cavity based on 650 MHz to increase the RF phase range and effectively perform bunch compression.

The beam energy is accelerated up to 10 MeV by approximately 2 MV per cavity with five 2-cell cavities (Fig[.3\)](#page-2-0), which is feasible with Nb3Sn. When accelerating in the 2-cell SC cavity after passing through the buncher, the acceleration does not occur immediately. This is because the energy of the beam (100 keV) is in the non-relativistic region, and the speed of the beam is not close to the speed of light. The 2-cell cavity does not accelerate properly due to the mismatch of the cavity cells, resulting in loss of energy. In order to eliminate cavity mismatch, each cell of the 2-cell cavities is made independent and the beam is slightly pre-accelerated by two 1-cell (β = 0.8) cavities. As the graphs of the minimum speed of transportation (see Fig[.4\)](#page-2-1) read, if the minimum velocity is less than zero, the beam will decelerate, and it will be impossible to use it as an accelerator. If the amplitude of the cavity is increased too much, it will enter a region where the beam cannot be accelerated properly. The region that satisfies the condition that the energy finally rises without deceleration is wider for the β = 0.8 cavity.

![](_page_2_Figure_0.jpeg)

**Caption:** Figure 3: This illustration provides a schematic layout of the injector and superconducting cryomodule section of the accelerator, highlighting the key components like the electron gun, buncher, solenoids, and five 2-cell superconducting cavities. The design supports the acceleration of a high-current beam through strategic component placement and involves significant technical considerations ensuring optimal thermal and conductivity properties required for efficient electron acceleration .


<span id="page-2-0"></span>Figure 3: Layout of the injector and the superconducting cryomodule.

![](_page_2_Figure_2.jpeg)

**Caption:** Figure 4: This figure illustrates the minimum speed of transportation within the superconducting cavities for two different cavity configurations: β = 1 (left panel) and β = 0.8 (right panel). The plots use the beam energy in units of the kinetic velocity factor (γβ) against longitudinal position (m). The graphical representation underscores how cavity design affects the beam transport efficiency in non-relativistic conditions. It highlights the broader range of acceptable energy conditions when using β = 0.8 cavities, which is crucial for minimizing energy mismatch losses and improving the overall acceleration process .


<span id="page-2-1"></span>Figure 4: Minimum speed of transportation for β = 1 cavity (left) and β = 0.8 cavity (right).

# 3 Beam dynamics

#### 3.1 Simulation setup

The motivation of the simulation study was the following. Assuming an initial beam, optimization was performed by a simulation including the space charge effect of a multi-particle beam, so that the beam parameters at the exit of the accelerator were confirmed.

To study the beam dynamics in the compact Nb3Sn accelerator, the injector part and the superconducting cryomodule were introduced into simulation. The electron gun distribution was included in this calculation, and the beam transport calculation was performed with the layout shown in Fig[.3.](#page-2-0) We started the calculation by giving the particle distribution of the 100 keV beam at the cathode of the electron source. For the electron gun, solenoids, buncher and 2-cell cavities simulation the 1D field distributions of those from cERL were used. For the 1-cell cavity, the gap length of the original cavity was reduced (β = 0.8), and the 1D field distribution was obtained. General Particle Tracer [\[11\]](#page-7-2) was used for the calculation. For the initial beam parameters please refer to Table [1.](#page-2-2)

<span id="page-2-2"></span>Table 1: Initial parameters of electron beam

| Parameter           | Value                    |
|---------------------|--------------------------|
| Bunch charge        | 77 pC (50 mA at 650 MHz) |
| Number of particles | 106<br>in tracking,      |
|                     | 2000 in optimization     |
| Electron gun energy | 100 keV                  |
| Cathode size        | 5.73 mm (uniform)        |
| Bunch length        | 100 ps (Gaussian ±4σ)    |

The electron emission surface of the hot cathode is approximately 10 mm, and the mean transverse energy (MTE) is 0.5 eV is used to generate a beam. The beam diameter at the electron gun exit is approximately 8 mm at its maximum. To control the space charge effect solenoid focusing is introduces into the design. Thus, the first solenoid focuses the beam, so that the beam diameter becomes 10 mm or less at the collimator position. Then the beam expanding on passing

through the buncher cavity is focused again by the second solenoid. The beam size reaches the maximum diameter of 28 mm at the second solenoid, which is smaller than the beam pipe diameter of 60 mm. Therefore, it can be inferred that the beam loss at this location can be sufficiently suppressed. Another point is that the essential beam size relaxation at the second solenoid location allows in the following effectively shrink the transverse beam size in the cryomodule.

The bunch length primary estimated by the simulation was about 63 ps, but the experimentally obtained bunch length in Ref.[\[12\]](#page-7-3) is longer than the calculated bunch length. Thus, in a particle tracking an initial bunch length of 100 ps was adopted.

#### 3.2 Optimization strategy

The optimization strategy was: for a given layout of the accelerator, it was necessary to adjust the amplitudes and phases of cavities, and strengths of the solenoids to find the optimum conditions. Namely, to reach the target energy (10 MeV), the energy width of the emitted beam was required to be narrow, so that there was no loss during transportation. For low velocity beams the parameters are intricately correlated. In addition, the influence of the space charge effect is also large. For this reason, simultaneous optimization of multiple parameters was necessary.

Optimization targets were the following: (1) to minimize final bunch length; (2) to minimize maximum beam size during transportation; (3) minimize final energy spread; (4) minimize maximum amplitudes of the cavities; (5) final energy should be reached. Then the distribution of 2,000 particles was set up, and the magnetic field of the solenoids on the transport of the beam, the acceleration voltages and the phases of the buncher and SCs were set as free parameters. The beam energy is accelerated up to 10 MeV by approximately 2 MV per cavity with five 2-cell cavities. The acceleration gradient is below 10 MV/m. Bayesian optimization toolbox [\[13\]](#page-7-4) and solver based on multidimensional Newton-Raphson algorithm [\[14\]](#page-7-5) were used independently for the optimization to reach a reliable model of the accelerator. Their results converged. Then the acceleration voltages and phases of the buncher and SCs were confirmed. Together with it, the solenoids suppressed beam divergence due to the large initial space charge effect. As the final result of the optimizations, it was possible to transport 1,000,000 particles without hitting the 70 mm beam pipe.

#### 3.3 Results and discussion

The final results of the optimization procedure are summarized in Table [2.](#page-3-0) Once all free parameters of the model were confirmed, the particle distribution was tracked through the transportation line. The evolution of the beam parameters is shown in Fig[.5.](#page-4-0) And the beam performance at the exit of the cryomodule is given in Table [3.](#page-6-8)

| Cavity name   | Amplitude<br>MV/cav. | Crest ϕ<br>deg. | ϕ off.<br>deg. |
|---------------|----------------------|-----------------|----------------|
| Buncher       | 0.0024               | -58.8           | -90.0          |
| INJ1 (1-cell) | 0.36                 | -62.8           | 1.0            |
| INJ2 (1-cell) | 0.35                 | -127.9          | 0.0            |
| INJ3 (2-cell) | 1.19                 | 145.4           | -85.9          |
| INJ4 (2-cell) | 1.88                 | -42.2           | 4.2            |
| INJ5 (2-cell) | 1.89                 | -174.5          | 0.0            |
| INJ6 (2-cell) | 1.90                 | 56.4            | 0.0            |
| INJ7 (2-cell) | 1.91                 | 133.6           | 0.0            |
| Solenoid name | Current (A)          |                 |                |
| SL1           | 2.75                 |                 |                |
| SL2           | 1.24                 |                 |                |

<span id="page-3-0"></span>Table 2: Optimized parameters

Let's discuss the major knobs that could be used to control and to tune the beam. To control the bunch length effectively a combination of the buncher phase and the first 2-cell cavity phase offsets are used in the model. Buncher cavity involves so-called "zero-cross bunching" to compress the bunch and to leave its phase space linear. Then the bunching process launched at the buncher is stopped using a velocity bunching technique in the first 2-cell cavity (phase offset added to the crest phase). The bunch length behaviour is given in Fig[.5](#page-4-0) to the left. Target compression achieved (see Table [3\)](#page-6-8).

A key knob to control the transverse beam size is solenoid focusing (Fig[.5](#page-4-0) in the middle). As it was mentioned above, the relaxation of the beam size at the location of the second solenoid (after the buncher cavity) is crucial for the

![](_page_4_Figure_0.jpeg)

**Caption:** Figure 5: Time evolution of the beam parameters through the transportation line is shown in three subplots: the left plot depicts the evolution of the rms bunch length (mm) as the beam progresses from the cathode, illustrating the bunch compression; the middle plot shows the rms transverse beam size (mm) also as a function of distance, demonstrating the beam's focusing and defocusing along the path; the right plot presents the total energy (MeV) and energy spread (%), indicating energy gain and spread as the beam accelerates. These visualizations are significant for understanding the impact of cavity focusing and tuning decisions made with quadrupoles and during main acceleration to maintain beam quality and compression during transport.


<span id="page-4-0"></span>Figure 5: Time evolution of the beam parameters through the transportation line: rms bunch length (left); rms transverse beam size (middle); energy and energy spread (right).

![](_page_4_Figure_2.jpeg)

**Caption:** Figure 6: Graphical depiction of the spatial distribution evolution of the beam along the z-axis through the accelerator components. The x-axis represents the spatial distribution along z, while the y-axis depicts transverse positions. As the beam progresses from initial conditions in the gun through the 2-cell cavities, spatial modulation and focusing are observed due to cavity and solenoid effects. This illustrates real-time particle focusing and shifts induced by space charge and accelerating forces, important for maintaining beam integrity.


![](_page_4_Figure_3.jpeg)

**Caption:** Figure 7: Time evolution of the energy distribution of the beam is shown as overlapping snapshots of particles within longitudinal phase space, taken every 1 ns. The visualization demonstrates convergence in energy and time, significant for monitoring the controlled acceleration and bunch compression during passage through the cryomodule. This figure helps underscore the dynamics of energy spread and distribution shaping across fractions of a second.


Figure 7: Time evolution of the energy distribution of the beam.

minimization of the beam size in the presence of the strong space charge force. The cavity focusing effect turned to make a small impact into overall beam size minimization. Nevertheless, the beam size at the exit of the cryomodule is good (5.41 mm). Additional tuning available through 2 quadrupoles at the irradiation section.

Since the main acceleration occurs in five 2-cell cavities, energy tuning knobs are naturally amplitudes of those cavities. One should keep in mind, that the first 2-cell cavity is responsible for the velocity bunching. Therefore the fine energy tuning should be done through the rest four 2-cell cavities. Let us remind that 1-cell cavities are indispensable for the slight acceleration of the 100 keV beam coming from the electron source. This stage is of a great importance to reach the final energy of 10 MeV with a cryomodule. Another point is on the energy spread behaviour. It increases after accelerating in two 1-cell cavities since those fields' fringes oppose a considerable effect on the beam as reads the energy graph in Fig[.5](#page-4-0) to the right.

Standalone optimization of injector parameters up to the second solenoid (see Fig[.3\)](#page-2-0) leads to the best beam parameters at the second solenoid location. However, with the optimization of beam parameters at the exit of the accelerating

![](_page_5_Figure_0.jpeg)

**Caption:** Figure 8: The figure illustrates the final beam transverse spatial distribution 1 meter downstream from the last cavity. Represented are contour and color density plots showing transverse distribution in both x and y planes, using counts in arbitrary units on the axes. This figure sums up the effects of transverse beam dynamics and provides valuable insights into the beam quality exiting the acceleration sections, a key performance metric for ensuring alignment with accelerator design goals.


Figure 8: Final beam transverse spatial distribution 1m downstream the last cavity.

![](_page_5_Figure_2.jpeg)

**Caption:** Figure 9: This figure displays the final beam longitudinal spatial distribution 1 meter downstream from the last cavity. This visualization consists of a histogram representing counts against longitudinal position (Z) and energy (MeV). The composite plot indicates how energy distribution and longitudinal bunch structure conclude at the last cavity, essential for assessing the beam's longitudinal dynamics and ensuring the satisfactory delivery of energy levels needed for downstream applications.


Figure 9: Final beam longitudinal spatial distribution 1m downstream the last cavity.

section, the same beam quality at the second solenoid location is not possible. The final result of the optimization of the beam parameters at the accelerator exit is achieved by relaxing those parameters at the second solenoid location. In this case, beam tracking occurs without losses.

Figure 6 shows a series of snapshots of the particle distribution in real space, taken every 1 nanosecond (ns). We observe the bunches being transported while converging in both the vertical and horizontal directions. This convergence is achieved by the buncher and solenoid focusing the beam. However, as the space charge effect becomes more prominent, the beam starts to diverge. Following this divergence, the beam converges again due to acceleration. The space charge effect also leads to significant scattering of some beam components, which can contribute to the formation of a beam halo and ultimately result in beam loss. These scattered components are evident in Figure 6.

Figure 7 presents overlapping snapshots of the particle distribution in longitudinal phase space, again captured every 1 ns. Here, we see the bunch being transported while converging in both energy and time.

Figures 8 and 9 depict the final particle distributions in the transverse and longitudinal planes, respectively. It is important to note that no beam loss was observed in the simulation with 1M macroparticles.

Table 3: Beam parameters at the exit of the cryomodule

<span id="page-6-8"></span>

| Parameter                | Value             |
|--------------------------|-------------------|
| Total energy             | 10.09 MeV         |
| Rms bunch length         | 0.29 mm / 0.86 ps |
| Rms transverse beam size | 5.41 mm           |
| Energy spread            | 0.52%             |

# 4 SUMMARY

Assuming an initial beam distribution, we performed three-dimensional beam tracking including the space charge effect. As a result of optimization by two independent methods mentioned above, the result of transportation with no loss was obtained. Based on the above calculation results and on the design of the injection section, Nb3Sn cryomodule, and irradiation section [\[7\]](#page-6-6), we designed the entire irradiator. By using the Nb3Sn cavities, it is possible to accelerate 10 MeV and 50 mA without loss, and a very compact irradiation accelerator can be achieved.

# Acknowledgments

This work is supported by the NEDO project "Development of innovative quantum beam technology for high-efficiency nanocellulose (CNF) production".

# References

- <span id="page-6-0"></span>[1] M. Akemoto et al. Construction and commissioning of the compact energy-recovery linac at kek. *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment*, 877:197–219, 2018.
- <span id="page-6-1"></span>[2] Y. Morikawa et al. New Industrial Application Beamline for the cERL in KEK. In *10th International Particle Accelerator Conference*, page THPMP012, 2019.
- <span id="page-6-2"></span>[3] R. C. Dhuley, I. Gonin, S. Kazakov, T. Khabiboulline, A. Sukhanov, V. Yakovlev, A. Saini, N. Solyak, A. Sauers, J. C. T. Thangaraj, K. Zeller, B. Coriton, and R. Kostin. Design of a 10 mev, 1000 kw average power electron-beam accelerator for wastewater treatment applications. *Phys. Rev. Accel. Beams*, 25:041601, Apr 2022.
- <span id="page-6-3"></span>[4] G. Ciovati, J. Anderson, B. Coriton, J. Guo, F. Hannon, L. Holland, M. LeSher, F. Marhauser, J. Rathke, R. Rimmer, T. Schultheiss, and V. Vylet. Design of a cw, low-energy, high-power superconducting linac for environmental applications. *Phys. Rev. Accel. Beams*, 21:091601, Sep 2018.
- <span id="page-6-4"></span>[5] Hiroshi Sakai et al. Stable Beam Operation in Compact ERL for Medical and Industrial Application at KEK. *JACoW*, SRF2021:THOTEV02, 2022.
- <span id="page-6-5"></span>[6] R. D. Porter et al. Next Generation Nb3Sn SRF Cavities for Linear Accelerators. In *Proc. LINAC'18*, pages 462–465. JACoW Publishing, Geneva, Switzerland, 2018.
- <span id="page-6-6"></span>[7] H. Sakai et al. Conceptual design of the high-power electron beam irradiator using niobium-tin superconducting cavity. In *Proc. IPAC'23*. JACoW Publishing, Geneva, Switzerland, 2023. In press.
- <span id="page-6-7"></span>[8] R.J. Bakker, C.A.J. van der Geer, A.F.G. van der Meer, P.W. van Amersfoort, W.A. Gillespie, and G. Saxon. 1 ghz modulation of a high-current electron gun. *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment*, 307(2):543–552, 1991.
- <span id="page-7-0"></span>[9] A. Facco. Low and Medium Beta Superconducting Cavities. In *9th European Particle Accelerator Conference (EPAC 2004)*, 7 2004.
- <span id="page-7-1"></span>[10] K. Fong and D. Storey. Design of an RF Modulated Thermionic Electron Source at TRIUMF. In *9th International Particle Accelerator Conference*, 6 2018.
- <span id="page-7-2"></span>[11] Pulsar Physics. General particle tracer (gpt) code. <https://www.pulsar.nl/>.
- <span id="page-7-3"></span>[12] M. Stefani. *Commissioning & characterization of magnetized gridded thermionic electron source*. PhD dissertation, Old Dominion University, 2021.
- <span id="page-7-4"></span>[13] The GPyOpt authors. GPyOpt: A bayesian optimization framework in python. [http://github.com/](http://github.com/SheffieldML/GPyOpt) [SheffieldML/GPyOpt](http://github.com/SheffieldML/GPyOpt), 2016.
- <span id="page-7-5"></span>[14] Amparo Gil, Javier Segura, and Nico Temme. *Numerical Methods for Special Functions*. 01 2007.