# arXiv:2110.01296

**Paper ID:** 92e01e6a5019c659475fe51df1cead10

**PDF Path:** apl/Superconductors/arXiv:2110.01296.pdf

**Processing Status:** complete

**Captions Added:** 7

**Generated:** 2025-06-24T13:44:27.554314

---

# Thin Film (High Temperature) Superconducting Radiofrequency Cavities for the Search of Axion Dark Matter

J. Golm, S. Arguedas Cuendis, S. Calatroni, C. Cogollos, B. Dobrich, J.D. Gallego, J.M. Garc ¨ ´ıa Barcelo, X. ´ Granados, J. Gutierrez, I.G. Irastorza, T. Koettig, N. Lamas, J. Liberadzka-Porret, C. Malbrunot, W. L. Millar, P. Navarro, C. Pereira Carlos, T. Puig, G. J. Rosaz, M. Siodlaczek, G. Telles and W. Wuensch

*Abstract*—The axion is a hypothetical particle which is a candidate for cold dark matter. Haloscope experiments directly search for these particles in strong magnetic fields with RF cavities as detectors. The Relic Axion Detector Exploratory Setup (RADES) at CERN in particular is searching for axion dark matter in a mass range above 30 µeV. The figure of merit of our detector depends linearly on the quality factor of the cavity and therefore we are researching the possibility of coating our cavities with different superconducting materials to increase the quality factor. Since the experiment operates in strong magnetic fields of 11 T and more, superconductors with high critical magnetic fields are necessary. Suitable materials for this application are for example REBa2Cu3O7−x, Nb3Sn or NbN.

We designed a microwave cavity which resonates at around 9 GHz, with a geometry optimized to facilitate superconducting coating and designed to fit in the bore of available high-field accelerator magnets at CERN. Several prototypes of this cavity were coated with different superconducting materials, employing different coating techniques. These prototypes were characterized in strong magnetic fields at 4.2 K.

*Index Terms*—Superconducting resonators, SRF superconducting radio frequency cavities, Quality factor, 2G HTS Conductors,axion

This project has received funding from the European Union's Horizon 2020 Research and Innovation programme under Grant Agreement No 730871 (ARIES-TNA). BD and JG acknowledge funding through the European Research Council under grant ERC-2018-StG-802836 (AxScale). We also acknowledge funding via the Spanish Agencia Estatal de Investigacion (AEI) and Fondo Europeo de Desarrollo Regional (FEDER) under project PID2019- 108122GB-C33, and the grant FPI BES-2017-079787 (under project FPA-2016-76978-C3-2-P). Furthermore we acknowledge support from SuMaTe RTI2018-095853-B-C21 from MICINN co-financed by the European Regional Development Fund, Center of Excellence award Severo Ochoa CEX2019- 000917-S and CERN under Grant FCCGOV-CC-0208 (KE4947/ATS).

J. Golm is with the European Organization for Nuclear Research (CERN), 1211 Geneva 23, Switzerland and the Friedrich - Schiller - University Jena, 07743 Jena, Germany (email: Jessica.Golm@cern.ch).

S. Arguedas Cuendis, S. Calatroni, B. Dobrich, T. Koettig, J. Liberadzka- ¨ Porret, C. Malbrunot, W. L. Millar, C. Pereira Carlos, G. J. Rosaz, M. Siodlaczek, W. Wuensch are with the European Organization for Nuclear Research (CERN), 1211 Geneva 23, Switzerland.

X. Granados, J. Gutierrez, N. Lamas, T. Puig, G. Telles are with the Institut de Ciencia de Materials de Barcelona, CSIC 08193 Bellaterra, Catalonia, Spain. ` C. Cogollos is with the Instituto de Ciencias del Cosmos, University of Barcelona, 08028 - Barcelona, Spain.

J.D. Gallego is with Yebes Observatory, National Centre for Radioastronomy Technology and Geospace Applications, 19080 - Guadalajara, Spain.

J.M. Garc´ıa Barcelo and P. Navarro are with the Department of Information ´ and Communications Technologies, Technical University of Cartagena, 30203 - Murcia, Spain.

I.G. Irastorza is with CAPA & Departamento de F´ısica Teorica, University de ´ Zaragoza, 50009 - Zaragoza, Spain.

## I. INTRODUCTION

T HE Relic Axion Detector Exploratory Setup (RADES) is an axion haloscope experiment searching for dark matter axions in a strong magnetic fields with high quality factor cavities. It differs from most haloscopes in the fact that thus far it employs dipole magnets and not solenoids. Many experiments successfully use copper cavities in strong magnetic fields to set limits to the axion coupling at low mass ranges. Reference [\[1\]](#page-4-0) and [\[2\]](#page-4-1) and references therein provide a recent review of experimental axion searches and the haloscope technique. The past years superconducting cavities were explored by experiments like QUAX [\[3\]](#page-4-2), [\[4\]](#page-4-3) and CAPP [\[5\]](#page-4-4), [\[6\]](#page-4-5) in order to reach a higher sensitivity. An axion with mass m<sup>A</sup> converts into a photon due to the inverse Primakoff effect. If the converted photon's energy matches the resonance frequency of the cavity, the output power is augmented depending on the axion's coupling strength to photons. For a given axion-photon coupling the figure of merit F of the experiment is

<span id="page-0-0"></span>
$$F \sim g\_{a\gamma}^2 m\_A^2 B^4 V^2 T\_{sys}^{-2} G^4 Q,\qquad(l)$$

where gaγ is the axion coupling to two photons, B the external magnetic field (assumed constant over the cavity volume), V is the cavity volume, Tsys is the detection noise temperature, and G is the geometric form factor of the cavity mode.

The figure of merit of the experiment increases by the power of four with the strength of the magnetic field. Therefore we aim at magnets with fields as high as possible. For the current run we had a 2-m long 11 T dipole magnet in single coil configuration available, for details see [\[7\]](#page-4-6). This sets the requirements for the coatings: we needed a type II superconductor with a critical magnetic field Bc<sup>2</sup> well above 11 T at 4.2 K. The materials should also possess a RF surface resistance R<sup>s</sup> lower than copper at our operating conditions. Experimental results and theoretical predictions have been described in literature, see for example [\[8\]](#page-4-7), [\[9\]](#page-4-8) and references therein for Nb3Sn or high temperature superconductors like REBa2Cu3O7−<sup>x</sup> (RE = Y, Gd, Eu) (REBCO). Both these materials were applied to our cavities. One cavity was sputtercoated with Nb3Sn and REBCO tapes were applied to the second cavity, where the hastelloy substrate was stripped off, such that the REBCO layer is exposed to the RF fields.

## II. CAVITY DESIGN

The previous RADES design consists of rectangular subcavities joined by irises and was specially designed for the axion search in long dipole magnets [\[11\]](#page-4-9). To facilitate the coating of the cavities the design had to be adapted: the irises have been removed and the corners rounded. The cavity is designed to have a large volume (given the frequency), high geometric and quality factor, resonate at about 9 GHz while fitting in the common bores of dipole magnets that were at the time available for RADES experiments. The optimization of the cavity geometry was done in CST and the final design is shown in Fig. [1.](#page-1-0) The cavity is cut along the electrical field lines shown in Fig. [1](#page-1-0) (a). The cut at this position allows us to tune the cavity at a later stage. No RF currents flow across the gap, and the cavity halves can be separated up to 2 mm without degrading sizeably the quality factor [\[12\]](#page-4-10). In Fig. [1](#page-1-0) (b) the direction of the surface currents is illustrated. The superconducting tape was attached to the walls in the flow direction of the currents. In this way the currents do not have to cross the tape boundaries which may result in an increase of the surface resistance across this boundary.

For Nb3Sn cavity halves the entire inner surface was coated with the superconductor, but the layer in the rounded corners may not be optimally coated because of the angle to the sputtering source. However, as shown in Fig. [1](#page-1-0) (b) there is also not much current flowing around this corners. So even if the coating does not reach its optimum in this corners this will only lead to a small degradation in Q0.

For the cavity coated with an HTS tape it was even decided to omit these rounded corners as shown in orange in Fig. [1](#page-1-0) (a). Instead the area was chosen to be copper coated. The simulations showed that for HTS with a surface resistance five times better than copper, the degradation in Q<sup>0</sup> resulting from using copper for the corners is only about 8 %.

The superconducting layer is not expected to shield significantly the magnetic field, due to its low thickness, as demonstrated in [\[10\]](#page-4-11).

## III. COATINGS

## *A. Cu coating*

A reference cavity of the same geometry as the superconducting cavities was coated with copper of two different versions - 'shiny' and 'matt'. The matt copper coating was applied by pulsed galvanic plating and has a high purity (Residual-resistance ratio (RRR) of 680±100 [\[13\]](#page-4-12)) but the surface roughness is very high and it appears matt. The shiny coating was applied by DC galvanic plating with an organic brightener. The surface roughness is much smaller for this coating method but RRR is only around 38±2 [\[13\]](#page-4-12). Fig. [2](#page-1-1) shows photographs of a RADES cavity half with the different copper coatings applied. Since we are operating at frequencies around 9 GHz, the quality factor of the cavities at low temperatures is limited by the anomalous skin effect. Due to this effect the surface roughness matters more than the purity of the material. For that reason, we measured a much higher quality factor for the shiny copper cavity at 4.2 K of about 40,000 (R<sup>s</sup> = 6.6 mΩ) than for the matt one of about

<span id="page-1-0"></span>![](_page_1_Figure_8.jpeg)

**Caption:** Figure 8 displays the design of a RADES cavity modeled in CST Studio Suite®. Depiction (a) outlines the cavity halves design while (b) maps the surface currents associated with the axion mode. These diagrams are fundamental in illustrating how the geometric design facilitates both improved superconducting coating application and superior performance in detecting axion-converted photons at CERN's 9 GHz RADES experiments.


Fig. 1: Cavity design modeled in CST Studio Suite® showing (a) cavity halves design and (b) surface currents of the axion mode.

30,000 (R<sup>s</sup> = 8.8 mΩ). The shiny copper plating was retained for our reference cavity, and was applied also to the HTS cavity before soldering the tapes.

<span id="page-1-1"></span>![](_page_1_Figure_11.jpeg)

**Caption:** Figure 11 provides photographs of a RADES cavity half. Image (a) shows the cavity with a 'matt' copper coating and (b) displays it with a 'shiny' copper coating, highlighting the differences in surface finishes which impact the quality factor at low temperatures due to the anomalous skin effect. These visualizations are crucial in understanding the role of surface roughness versus material purity on the performance of superconducting cavities.


Fig. 2: Photographs of a RADES cavity half coated with matt (a) and shiny (b) copper.

## *B. Nb*3*Sn coating*

The cavity halves were coated separately in two subsequent runs maintaining the same parameters. For each cavity half, a Ta intermediate layer was grown onto the substrates (electropolished stainless steel 316LN) to avoid diffusion of the substrate within the Nb3Sn layer, coated immediately after. This two-step process was made possible by the dualmagnetron configuration of our coating set up, one with a Ta and one with a Nb–Sn stoichiometric target. A custom Joule heater was built in-house to keep the sub-cavities at 750 °C throughout the entire coating and annealing processes. Photographs of (a) the coating setup with both magnetrons and

<span id="page-2-1"></span>TABLE I: Coating parameters for the Ta and Nb3Sn layers.

|          | t     | p      | Pw  | T    | Main pulse |            | Positive Pulse |            |
|----------|-------|--------|-----|------|------------|------------|----------------|------------|
|          | (min) | (mbar) | (W) | (°C) | t<br>(µs)  | ν<br>(kHz) | t<br>(µs)      | ∆t<br>(µs) |
| Ta       | 40    | 1×10−3 | 350 | 750  | 50         | 1          | 200            | 4          |
| Nb3Sn 75 |       | 7×10−4 | 350 | 750  | -          | -          | -              | -          |

<span id="page-2-0"></span>the sample holder in the centre, with the homemade heater attached on top, and (b) a close up of a cavity half in coating position (previous to coating) are shown in Fig. [3.](#page-2-0)

![](_page_2_Picture_3.jpeg)

**Caption:** Picture 3 captures the experimental setup used for coating a cavity half. The image displays the setup equipped with magnetrons and a central sample holder with an attached homemade heater, important for maintaining the required temperature conditions during the Ta and Nb3Sn coating processes. The setup is essential for achieving the desired superconducting properties of the cavity components utilized in high-field magnetic applications.


![](_page_2_Picture_4.jpeg)

**Caption:** Picture 4 showcases a close-up of a cavity half in its position for the coating process prior to the application of coatings. This image is critical for illustrating how the substrate is prepared and positioned to receive Ta and Nb3Sn coatings in a manner ensuring optimal surface characteristics and adherence, contributing to the development of high-performance superconducting cavities.


(a)

Fig. 3: Photographs of the (a) coating setup and (b) a cavity half in coating position.

The Ta layer was deposited by High-Power Impulse Magnetron Sputtering with a potential reversal for ion acceleration (HiPIMS + Positive Pulse) for 40 min at 1 × 10<sup>−</sup><sup>3</sup> mbar, resulting in ≈ 0.9 µm thick layers. After this step, each cavity half was left at the same temperature and pressure conditions (750 °C, 1 × 10<sup>−</sup><sup>3</sup> mbar of Kr) for a 45 min anneal to promote the formation of Ta-α phase. The Nb3Sn layer, ≈ 2 µm thick, was sputtered resorting to Direct Current Magnetron Sputtering (DCMS) for 75 min at a Kr pressure of 7 × 10<sup>−</sup><sup>4</sup> mbar. The final step for each of the cavity halves was a longer second anneal lasting around 24 h in order to promote the formation of the Nb3Sn superconducting phase and cure the potential defects [\[14\]](#page-4-13).

The coating parameters (duration t, pressure p, temperature T and power P w) used for the deposition of each layer are summarized in Table [I.](#page-2-1) The HiPIMS Main Pulse (duration t and frequency ν) and applied Positive Pulse (duration t and delay ∆t) input values used for the Ta layer are also presented.

## *C. REBCO tape*

Each cavity half was coated with 5 segments of 12 mmwide coated conductor provided by the company THEVA. The segments were placed in a way that the REBCO layer was oriented towards the surface of the cavity half (see Fig. [4](#page-2-2) (a)). To homogeneously heat up the system, we used specially designed flexible heaters that could adapt to the curved geometry of the cavity halves. The coated conductors were soldered onto the surface of the cavity halves using Sn60Pb38Cu<sup>2</sup> due to its low-temperature melting point. A standard procedure has been applied for such low temperature solders that enables short heating times of less than 5 minutes at T < 260 ºC. During the development of the soldering technique, samples of 12 x 50 mm<sup>2</sup> where characterized by means of a scanning Hall probe microscope (SHPM) before and after soldering. The soldering technique guarantees no degradation to the REBCO material within the experimental error of our SHPM technique, which we estimate to be under 10 %. During the soldering process, the heaters and the tapes were kept into position by applying some small mechanical pressure onto the system. Once the system cooled down to room temperature, the tapes were peeled off by pulling out the hastelloy substrate. In that way the buffer layer delaminates from the REBCO surface, (see Fig. [4](#page-2-2) (b)).

<span id="page-2-2"></span>![](_page_2_Figure_11.jpeg)

**Caption:** Figure 11 consists of two images: (a) depicts the schematic cross-sectional representation of the copper layers applied over a RADES subcavity, revealing the structural arrangement including Ag, Hastelloy, buffer layers, REBCO, and SnPbCu, with delineated interfaces where delamination might occur; (b) shows a RADES cavity interior post-coating. This detailed view aids in the understanding of the applied REBCO and copper coatings and their roles in superconducting cavity development, which are crucial for creating high-quality factor resonators in axion detection experiments.


Fig. 4: Schematics of the (a) coated conductor segment's position on the RADES cavity and picture of one of the (b) RADES halves after the coating is completed.

## IV. EXPERIMENTAL SET-UP

Before testing the new coatings in magnetic field, the quality factor was measured in the CERN Central Cryogenic Laboratory in liquid helium at 4.2 K. For the Nb3Sn cavity we could measure a Q<sup>0</sup> of about 700,000 (R<sup>s</sup> = 0.4 mΩ) and for the HTS tape about 80,000 (R<sup>s</sup> = 3.3 mΩ), both values being much higher than the Q<sup>0</sup> of 40,000 from our copper reference cavity.

Afterwards the cavity was installed in a 11 T dipole short model at CERN and the characterization of the cavities in high magnetic field was done in parallel to an axion search. One cavity port is weakly coupled (and terminated for axion search) and for the other port we aim for a critical coupling. Since the power of a signal that may be generated by photons converted from axions is very small, a low noise amplifier is connected to the critically coupled port. After amplification the signal is detected by a custom-made data acquisition system

## from TTI [\[15\]](#page-4-14).

In order to be able to measure the quality factor in this configuration a switch was necessary to bypass the low noise amplifier. The signal from the critical coupled port goes from the switch to one port of a vector network analyser while the weakly coupled port goes directly to the device.

The magnet bore was filled with liquid helium which has a dielectric constant of 1.049343 at 4.2 K [\[16\]](#page-4-15). The insertion into a dielectric de-tunes our cavity and results in a resonance frequency of about 8.8 GHz during data-taking. Fig. [5](#page-3-0) shows the experimental set-up. Two cavities were installed at the same time in the magnet bore of 54 mm diameter. The bottom cavity was the Nb3Sn coated cavity and the top one the HTStape cavity. A Hall sensor was attached onto the bottom cavity in order to align the cavity with the magnetic field.

<span id="page-3-0"></span>![](_page_3_Figure_3.jpeg)

**Caption:** Figure 3 provides a schematic of the experimental setup for the quality factor measurements and axion data collection in a high magnetic field, leveraging a handheld FieldFox microwave analyzer and a specialized DAQ system. The diagram illustrates positions and interconnections of signal amplifiers, switch, liquid helium environment, and HTS/Nb3Sn cavities along with the magnetic orientation. This setup is critical for understanding the meticulous design required to measure cavity performance under active experimental conditions, which are vital for RADES-type experiments aimed at axion detection.


Fig. 5: Schematics of the experimental set-up for the quality factor measurements and axion data taking in a 11 T magnetic field at CERN.

## V. RESULTS

The quality factor for each cavity was calculated from the Sparameter measurements with a vector network analyser using the 3dB method to determine Q<sup>l</sup> :

$$\mathbf{Q}\_l = \frac{f\_0}{\Delta f\_{3dB}},\tag{2}$$

where f<sup>0</sup> is the frequency of the maximum amplitude and ∆f3dB the bandwidth at - 3 dB. The coupling of the strongly coupled port was determined using the reflection parameters. During the measurement in liquid helium we observed a fast frequency drift which interfered with the Q measurement and suspect pressure fluctuations to be responsible for this shift. For each measurement we recorded a frequency range of 2 MHz measuring 10,001 points in this range. The frequency sweep took about 6 seconds and the drift in the frequency within this time is reflected by the error bars of the Q values in Fig. [6.](#page-3-1) The magnetic field was ramped up in 1 T steps with a speed of 10 A/s (about 1 kA/T). Afterwards the field was kept constant for 10 minutes and the quality factor of both cavities was measured. The results for the Q<sup>0</sup> of both cavities are shown in Fig. [6.](#page-3-1) The quality factor of the HTS tape cavity remained almost constant between 60,000 (R<sup>s</sup> = 4.4 mΩ)and 80,000 (R<sup>s</sup> = 3.3 mΩ)up to 11.6 T, while Nb3Sn decreased considerably and performed worse than our copper reference cavity above 3 T. Investigations about this behaviour are ongoing. The HTS cavity outperformed the copper cavity by 50% in quality factor and increased the sensitivity of our axion data-taking since the figure of merit scales linear with Q, see equation [\(1\)](#page-0-0). These physics results will be the object of a separate publication. From results presented in [\[8\]](#page-4-7) we expected an improvement of the quality factor by a factor of 5. After the initial characterization of the quality factor of the cavity our studies have shown that the decrease in quality factor has its origin in the 9 mm curvature radius of the cavity. Currently, we are working on the coating of a second haloscope in which we are implementing an upgraded procedure and we expect to obtain a significant improvement in the quality factor.

<span id="page-3-1"></span>![](_page_3_Figure_10.jpeg)

**Caption:** Figure 10 illustrates the quality factor (Q0) measurements for different cavities at a resonance frequency around 8.8 GHz and a temperature of 4.5 K submerged in liquid helium, under varying magnetic fields. The plot presents data for Nb3Sn cavity, REBCO tape cavity, and a copper reference cavity, emphasizing the differences in performance against the magnetic field. Blue triangles, red circles, and an orange diamond signify measurements from Nb3Sn, REBCO tape, and Cu reference cavities respectively. Key results indicate that the REBCO cavity maintains a relatively constant Q0 between 60,000 to 80,000 up to 11.6 T, outperforming the copper cavity, whereas the Nb3Sn cavity shows a significant decrease in Q0 above 3 T, necessitating further investigation. These findings are significant for developing high Q-factor superconducting cavities suitable for axion dark matter detection experiments under high magnetic fields such as those required in RADES setups at CERN.


Fig. 6: Results of quality factor measurements with the cavity immersed in liquid helium. The resonance frequency of the cavities is f = (8.8 ±0.1 ) GHz at 4.5 K.

## ACKNOWLEDGMENT

We are indebted to the CERN technical teams, particularly Gerard Willering, Franco Julio Mangiarotti, Jerome Feuvrier, Marta Bajko, Patrick Viret, Guillaume Pichon, Arnaud Devred, Stephan Russenschuck and Andrzej Siemko, as well as the CERN cryolab team. We also thank the RADES team and Kristof Schmieden for helpful discussions and Giuseppe Ruoso for the loan of equipment used in the measurement.

## REFERENCES

- <span id="page-4-0"></span>[1] I. G. Irastorza and J. Redondo, "New experimental approaches in the search for axion-like particles," *Prog. Part. Nucl. Phys.*, 102, 89-159, 2018.
- <span id="page-4-1"></span>[2] Y. K.Semertzidis and S. Youn, "Axion Dark Matter: How to see it?", *arXiv: hep-ph*, 2021.
- <span id="page-4-2"></span>[3] R. Barbieri et al., "Searching for galactic axions through magnetized media: The QUAX proposal," *Phys. Dark Universe,* vol. 15, pp. 135–141, 2017.
- <span id="page-4-3"></span>[4] D. Di Gioacchino et al., "Microwave Losses in a DC Magnetic Field in Superconducting Cavities for Axion Studies," *IEEE Trans. on Appl. Supercond.*, vol. 29, no. 5, pp. 1-5, 2019
- <span id="page-4-4"></span>[5] D. Ahn et al., "Superconducting cavity in a high magnetic field," *arXiv: Applied Physics*, 2020.
- <span id="page-4-5"></span>[6] D. Ahn et al., "First prototype of a biaxially textured YBa 2Cu 3 O 7 − x microwave cavity in a high magnetic field for dark matter axion search," *arXiv: Instrumentation and Detectors*, [\[arXiv:2103.14515](http://arxiv.org/abs/2103.14515) [physics.insdet]], 2021.
- <span id="page-4-6"></span>[7] F. Savary et al., "Design, Assembly and Test of MBHSP101, a 2-m long 11 T Nb3Sn Single Aperture Dipole Model at CERN," *IEEE Trans. on Appl. Supercond.*, vol. 25, no. 3, Jun. 2015.
- <span id="page-4-7"></span>[8] A. Romanov et al., "High frequency response of thick REBCO coated conductors in the framework of the FCC study" *Sci. Rep.*, (2020) 10:12325
- <span id="page-4-8"></span>[9] A. Alimenti et al., "High frequency response of thick REBCO coated conductors in the framework of the FCC study" *IEEE Trans. Appl. Supercond.*, Vol. 29, No. 5, (2019) 3500104
- <span id="page-4-11"></span>[10] G. Telles et al., "Field Quality and Surface Resistance Studies of a Superconducting REBa2Cu3O7-x - Cu Hybrid Coating for the FCC Beam Screen" *These Proceedings*
- <span id="page-4-9"></span>[11] A. A. Melc ´ on et al, "Axion Searches with Microwave Filters: the ´ RADES project," *JCAP*, 05, 040, 2018.
- <span id="page-4-10"></span>[12] S. Arguedas Cuendis *et al.* "The 3 Cavity Prototypes of RADES: An Axion Detector Using Microwave Filters at CAST," *Springer Proc. Phys.* 245,45-51, 2020.
- <span id="page-4-12"></span>[13] L. Lain Amador*et al.*, "Electrodeposition of copper applied to the manufacture of seamless superconducting rf cavities," textitPhysical Review Accelerators and Beams, 24, 082002, 2021.
- <span id="page-4-13"></span>[14] E. A. Ilyina *et al.*, "Development of sputtered Nb3Sn films on copper substrates for superconducting radiofrequency applications," *Superconductor Science and Technology*, 3, 32 035002, 2019.
- <span id="page-4-14"></span>[15] <https://www.ttinorte.es/>
- <span id="page-4-15"></span>[16] Donnelly, Russell J. and Barenghi, Carlo F. " The Observed Properties of Liquid Helium at the Saturated Vapor Pressure," *Journal of Physical and Chemical Reference Data*, vol. 27, num. 6, 1217-1274, 1998.