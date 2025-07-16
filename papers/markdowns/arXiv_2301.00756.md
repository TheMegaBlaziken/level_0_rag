# arXiv:2301.00756

**Paper ID:** ef91fcabcd08d5332894b4d2e84b7edb

**PDF Path:** apl/Superconductors/arXiv:2301.00756.pdf

**Processing Status:** complete

**Captions Added:** 8

**Generated:** 2025-06-24T13:44:27.697305

---

# Thermal annealing of sputtered Nb3Sn and V3Si thin films for superconducting radio-frequency cavities

Katrina Howard<sup>1</sup>,<sup>2</sup> , Zeming Sun<sup>1</sup> , and Matthias U. Liepe<sup>1</sup>

<sup>1</sup> Cornell Laboratory for Accelerator-Based Sciences and Education, Cornell University, Ithaca, NY 14853, USA

<sup>2</sup> Department of Physics, University of Chicago, Chicago, IL 60637, USA

E-mail: khoward99@uchicago.edu (K.H.); zs253@cornell.edu (Z.S.); mul2@cornell.edu (M.U.L.)

December 2022

Abstract. Nb3Sn and V3Si thin films are promising candidates as thin films for the next generation of superconducting radio-frequency (SRF) cavities. However, sputtered films often suffer from stoichiometry and strain issues during deposition and post annealing. In this study, we explore the structural and chemical effects of thermal annealing, both in-situ and post-sputtering, on DC-sputtered Nb3Sn and V3Si films of varying thickness on Nb or Cu substrates, extending from our initial studies [\[1\]](#page-12-0). Through annealing at 950 °C, we successfully enabled recrystallization of 100 nm thin Nb3Sn films on Nb substrate with stoichiometric and strain-free grains. For 2 µm thick films, we observed the removal of strain and a slight increase in grain size with increasing temperature. Annealing enabled a phase transformation from unstable to stable structure on V3Si films, while we observed significant Sn loss in 2 µm thick Nb3Sn films after high temperature anneals. We observed similar Sn and Si loss on films atop Cu substrates during annealing, likely due to Cu-Sn and Cu-Si phase generation and subsequent Sn and Si evaporation. These results encourage us to refine our process to obtain high-quality sputtered films for SRF use.

Keywords: thermal annealing, A15 superconductors, sputtering, thin film, SRF Submitted to: Superconductor Science and Technology

#### 1. Introduction

Nb3Sn and V3Si thin films are of increasing interest to the superconducting radiofrequency community owing to the quest of achieving high accelerating gradient and efficiency. As niobium-based superconducting radio-frequency (SRF) cavities are reaching the theoretical limits [\[2\]](#page-12-1), alternative materials are of great interest to continue the quest of increasing quality factors, accelerating gradients, and efficiency [\[3\]](#page-12-2). A15 superconductors Nb3Sn and V3Si are promising candidates for this role, used as thin films inside either Nb or Cu cavities [\[3,](#page-12-2) [4\]](#page-12-3). Both candidates have relatively high critical temperatures (Tc,Nb3Sn = 18.3 K and Tc,V3Si = 17.1 K), and Nb3Sn is predicted to yield a superheating field of ∼ 440 mT that doubles the Nb limit of ∼ 240 mT [\[3,](#page-12-2) [5,](#page-12-4) [6,](#page-12-5) [7,](#page-12-6) [8\]](#page-12-7). These properties could allow cavity operation at an elevated temperature of ∼ 4 K and the potential for increased accelerating gradients [\[9\]](#page-12-8). This higher operating temperature allows for reduced cryogenic costs and simpler infrastructure for particle accelerators and their applications [\[3\]](#page-12-2). Due to their brittle nature and low thermal conductivity, Nb3Sn and V3Si are best suited for use as a thin film inside a host cavity with better thermal conductivity, such as Nb or Cu [\[3,](#page-12-2) [10,](#page-12-9) [11\]](#page-12-10).

Nb3Sn thin films have been achieved through vapor diffusion, sputtering, electroplating, and chemical vapor deposition [\[12,](#page-12-11) [13,](#page-12-12) [14,](#page-12-13) [15,](#page-12-14) [16,](#page-13-0) [17\]](#page-13-1). In the state-of-the-art vapor diffusion, a niobium cavity is placed in a hightemperature vacuum furnace, and then tin or tin chloride sources are vaporized and allowed to diffuse into the niobium surface for alloying [\[3,](#page-12-2) [9,](#page-12-8) [13,](#page-12-12) [18,](#page-13-2) [19,](#page-13-3) [20\]](#page-13-4). In contrast, sputtering utilizes high-energy plasma to directly eject target materials onto a substrate at low temperatures [\[4,](#page-12-3) [6,](#page-12-5) [12,](#page-12-11) [21,](#page-13-5) [22\]](#page-13-6). Alternatively, Nb3Sn films are fabricated via electroplating in aqueous solutions working at near-room temperatures and atmospheric pressure followed by heat treatment [\[14,](#page-12-13) [15,](#page-12-14) [16,](#page-13-0) [23\]](#page-13-7), or via chemical vapor deposition that takes advantage of reactions between volatile precursors [\[24\]](#page-13-8). Nb3Sn has been successfully vapor-diffused inside cavities, where a single-cell reached gradients of 24 MV/m, while Nb3Sn 9- and 5-cells reached 15 MV/m, both with Q0's on the order of 10<sup>10</sup> at operating temperature 4.4 K [\[19,](#page-13-3) [20\]](#page-13-4). In cavity tests, maximum surface fields of 120 mT (pulsed operation) and 80 – 100 mT (CW) have been achieved, showing that Nb3Sn cavities can be operated reliably in a flux-free metastable state above the lower critical field of this material (around 40 mT) [\[7,](#page-12-6) [25\]](#page-13-9).

In the sputtering process, the film properties are tailored by controlling the Ar/Kr plasma pressure, substrate temperature, sputtering voltage, sputtering current, rate of deposition, and post-sputtering anneal temperature/duration. In literature [\[4,](#page-12-3) [6,](#page-12-5) [9,](#page-12-8) [12,](#page-12-11) [26\]](#page-13-10), sputtered Nb3Sn films have been demonstrated on Nb and Cu surfaces by using a stoichiometric Nb3Sn target, by co-sputtering with Nb and Sn targets, or through annealing a sputtered Nb/Sn multilayer. A stoichiometric target allows for a design where only a single target is used [\[4,](#page-12-3) [6,](#page-12-5) [26,](#page-13-10) [27\]](#page-13-11). Co-sputtering involves the use of separate Nb and Sn targets that are sputtering at the same time, allowing for tuning of the power applied to each target [\[21,](#page-13-5) [28\]](#page-13-12). Multilayer sputtering also uses separate Nb and Sn targets but alternates the use of each target to create many ultrathin layers of each material [\[11,](#page-12-10) [12,](#page-12-11) [22\]](#page-13-6). Tc's above 17.8 K have been observed for single-target and multilayer sputtering [\[6,](#page-12-5) [11,](#page-12-10) [12\]](#page-12-11).

V3Si films have been attempted by thermal diffusion, magnetron sputtering, and highpower impulse magnetron sputtering (HiP- IMS) [\[10,](#page-12-9) [11,](#page-12-10) [27,](#page-13-11) [28\]](#page-13-12). In thermal diffusion, a vanadium layer on a silicon-on-insulator substrate is annealed at high temperature to produce V3Si [\[28\]](#page-13-12). In the HiPIMS method, power is applied as a set of discrete high-energy pulses at a low-duty cycle, which can be used to ion bombard the substrate, recrystallizing films at a low temperature and allowing more control of the stoichiometry; this method of depositing V3Si films on Cu substrates produced T<sup>c</sup> up to 10 K [\[10\]](#page-12-9). CERN's magnetron sputtered V3Si films on a silver buffer layer upon a Cu substrate have reached T<sup>c</sup> of 11.2K [\[27\]](#page-13-11).

Thermal annealing of the sputtered films, either in situ or post-deposition, is required to minimize the internal stress induced by the sputtering process and improve the stoichiometry and grain structures, which are important for their critical temperature and cavity RF performance [\[4,](#page-12-3) [6,](#page-12-5) [12\]](#page-12-11). However, during annealing of sputtered Nb3Sn or Nb/Sn multilayers, the films suffer from issues such as Sn loss, Cu incorporation into the film from Cu substrates, high strain, and interface issues at the substrate-film boundary [\[4,](#page-12-3) [6,](#page-12-5) [12\]](#page-12-11). Sn loss is a critical issue because of the dependence of T<sup>c</sup> on Sn concentration [\[3\]](#page-12-2). While annealing is frequently performed on Nb3Sn films, these high temperatures have led to Sn loss in the furnace and Nb-rich films with reduced T<sup>c</sup> [\[6,](#page-12-5) [12\]](#page-12-11), which motivates us to mechanistically understand the phase transformation associated with annealing. Cu incorporation can occur during annealing, which lowers the T<sup>c</sup> [\[4\]](#page-12-3). This issue can be addressed by using a barrier layer such as tantalum to reduce the interdiffusion [\[27\]](#page-13-11). The interface between Nb3Sn and Cu also suffers from strain because of their different thermal expansion coefficients and lattice mismatch, which can cause cracking in the film [\[4\]](#page-12-3). Cracking can release high initial strain in the lattice, but does not relieve microstrain and increases surface roughness while decreasing the uniformity of the film [\[4,](#page-12-3) [29\]](#page-13-13). Currently, no sputtered Nb3Sn cavity test has been reported. Moreover, V3Si is much less studied than Nb3Sn, and there has been no RF test to date [\[10,](#page-12-9) [11,](#page-12-10) [27\]](#page-13-11).

One goal of this work is to optimize the sputtering capability of these alternative SRF materials at Cornell and compare our results with existing efforts in the SRF field. Most importantly, we aim to systematically investigate the effect of thermal annealing on the sputtered Nb3Sn and V3Si thin films in order to better understand these observed issues and design an optimal process for SRF use. By understanding the impacts of deposition and annealing parameters, our goal is to find the root of the issues in stoichiometry and strain of thin films. With such knowledge, we hope to provide insights for the development of sputtered Nb3Sn and V3Si cavities. In this study, we investigate Nb3Sn and V3Si films of different thicknesses on both Nb and Cu substrates to optimize the best conditions that minimize strain while producing required stoichiometry and superconducting properties.

## 2. Methods

Nb3Sn and V3Si thin films were deposited using a DC-sputtering system at the Cornell Center for Materials Research. A high vacuum of 10<sup>−</sup><sup>6</sup> torr base pressure was achieved using a cryo-pumped system. All depositions were performed at 5 mTorr Ar pressure. A rotating stage was used, when possible, to ensure uniformity during deposition.

As summarized in Table [1,](#page-3-0) the sputtering parameters varied were the film material (Nb3Sn vs. V3Si), substrate material (Nb

| Film  |    | Substrate Substrate<br>holder | Temperature<br>(°C) | Voltage<br>(V) | Current<br>(A) | Nominal<br>thickness |
|-------|----|-------------------------------|---------------------|----------------|----------------|----------------------|
| Nb3Sn | Nb | Rotating                      | 25                  | 596            | 0.15           | 100 nm               |
| Nb3Sn | Nb | Rotating                      | ><br>25             | 589            | 0.26           | 2<br>µm              |
| Nb3Sn | Cu | Heated                        | 550                 | 466            | 0.214          | 300 nm               |
| V3Si  | Nb | Rotating                      | ><br>25             | 811            | 0.196          | 2<br>µm              |
| V3Si  | Cu | Heated                        | 550                 | 819            | 0.222          | 300 nm               |

<span id="page-3-0"></span>Table 1. Sputtering parameters for Nb3Sn and V3Si film deposition.

vs. Cu), deposition temperature (room temperature vs. 550 °C in situ heating), and film thickness (100 nm, 300 nm, and 2 µm). Bulk Nb3Sn and V3Si targets were used, and they were purchased from ACI alloy, Inc. The impurity concentrations as received were 0.01 at.%.

Nb and Cu squared substrates of 1 cm<sup>2</sup> area were used in order to provide insights for applications in Nb and Cu substrate cavities. Before deposition, Nb substrates were electropolished, and Cu substrates were chemically polished to ensure a smooth surface.

The Nb3Sn and V3Si films were designed to have thicknesses of 100 nm and 2 µm on Nb substrates and 300 nm on Cu substrates. The deposition rate was 2.5 ˚A/s for all samples except for the V3Si film on Cu substrate which was 1.8 ˚A/s (as there was difficulty lighting the plasma). The deposition temperature for the thick 2 µm samples is subject to error because the temperature is uncontrolled upon the rotating stage and increased through the 133-minute deposition. Subsequently, a 550 ℃ heating stage was applied to investigate the effect of in situ heating during deposition.

After the sputtering process, films were annealed in a series of elevated temperatures at 600 ℃, 700 ℃, 800 ℃, and 950 ℃, each for 6 hours, in a Lindberg high-vacuum (3 × 10<sup>−</sup><sup>7</sup> Torr) furnace. The heating rate was 10 ℃ per minute, and the annealing was followed by furnace cooling.

Structural and chemical analyses were conducted between anneals to characterize the films. These analysis methods included scanning electron microscope (SEM) to observe the grain structure and size, energy dispersive X-ray (EDS) and X-ray photoelectron (XPS) spectroscopies to determine the atomic composition, and X-ray diffraction (XRD) to gain insight into the crystal structure of the film and calculate the strain. In this analysis, the key features are the quality of the film surfaces (smoothness, uniformity, grain shape/size), the stoichiometry of the films, and the existence and strain of Nb3Sn and V3Si diffraction planes. Note that EDS results were calibrated with regard to the electron penetration depth in each material and the film thickness.

Finally, on the 100 nm thin Nb3Sn film that yields the best performance, we verified its critical temperature using a quantum design physical property measurement system (PPMS) and quantified the surface roughness using atomic force microscopy (AFM).

#### 3. Results and Discussion

In this section, we first analyze the recrystallization behavior observed in the 100 nm thin Nb3Sn films annealed and discuss the superconducting, composition, and surface properties of these films. Next, we show the composition and strain evolutions as a function of annealing temperature in the 2 µm thick Nb3Sn and V3Si films (on Nb) and attempt to understand the Sn loss and strain relief mechanisms. Finally, we show the ternary phase transformation upon annealing in the 300 nm thick Nb3Sn and V3Si films that were deposited on Cu substrates. Representative surface morphologies of samples upon deposition and after 700 °C and 950 °C anneals are shown in figure S1.

### 3.1. Thin Nb3Sn film: Recrystallization

3.1.1. Recrystallization behavior Figure [1](#page-4-0) shows the evolution of surface morphology with increasing temperature for the 100 nm thin Nb3Sn film on Nb substrate. We observed evident grain recrystallization at 950 ℃ anneals. The grain size increased from a few nanometers as deposited (figure [1a](#page-4-0)) to approximately 300 nm after annealing (figure [1d](#page-4-0)). Recrystallization occurs through the release of strain energy during annealing and the subsequent migration of grain boundaries [\[30\]](#page-13-14).

<span id="page-4-0"></span>Here, we discuss the driving force and boundary mobility for thermodynamic considerations of this recrystallization annealing. The stored energy per unit volume (Es) at a strain level () of 0.2, the maximum strain measured from our sputtered films, is 2.7 × 10<sup>9</sup> J/m<sup>3</sup> , based on a 1-dimensional elastic assumption, E<sup>s</sup> = 1/2 E 2 , where E is Young's modulus and the value for Nb3Sn at 300 K is 13.7 × 10<sup>11</sup> dyn/cm<sup>2</sup> [\[31\]](#page-13-15). Indeed, this value

![](_page_4_Picture_6.jpeg)

**Caption:** Figure 1 provides an SEM comparative analysis of 100 nm Nb3Sn films on Nb substrates at various annealing temperatures: (a) as-deposited, (b) 600 °C, (c) 800 °C, and (d) 950 °C. The images reveal the grain recrystallization dynamics initiated by thermal annealing, with significant grain size increase at 950 °C. These morphological changes reflect crucial shifts in film microstructure, induced by the release of strain energy, vital for achieving optimal superconducting properties in Nb3Sn films .


is dramatically larger than the typical lightly- Figure 1. Surface SEM images for 100 nm thin Nb3Sn films on Nb substrates: (a) as-deposited (a), and (b-d) after annealing: (b) 600 ℃, (c) 800 ℃, and (d) 950 ℃.

![](_page_5_Figure_1.jpeg)

**Caption:** Figure 2 illustrates X-ray diffraction (XRD) patterns of a 100 nm thin Nb3Sn film recorded at various annealing temperatures: (a) as-deposited, (b) 600 °C, (c) 700 °C, (d) 800 °C, and (e) 950 °C. These XRD profiles showcase the persistence of specific Nb3Sn diffraction planes across temperature changes and suggest a stable phase, confirming the film's structural resilience and ability to maintain crystallographic integrity under thermal stress. The data elucidates the crystallization dynamics pivotal for developing stable, high-performance superconducting films .


<span id="page-5-0"></span>Figure 2. XRD patterns taken from the 100 nm thin Nb3Sn film as a function of annealing temperature: (a) as-deposited, (b) 600 ℃, (c) 700 ℃, (d) 800 ℃, (e) 950 ℃. Observed Nb3Sn diffraction planes are labeled at the top.

deformed energy of 10<sup>5</sup> J/m<sup>3</sup> for driving recrystallization in metals [\[32\]](#page-13-16). This suggests a sufficient driving force from the sputtering-induced strain within the film to enable the recrystallization annealing.

Our X-ray diffraction data (figure [2\)](#page-5-0) shows the Nb3Sn phase is consistent in terms of grain orientation at all annealing temperatures including 950 ℃ recrystallizations. We find Nb3Sn peaks near the known powder diffraction peaks at 2θ = 33.6°, 37.7°, 41.5°, 62.8°, 65.6°, 70.6°, and 82.9°[\[34\]](#page-13-17). Due to the large penetration depth of the X-ray probe, strong Nb substrate diffractions are seen at 2θ = 38.4°, 53.3°, and 69.3°. A complete list of known peak locations is shown in table S1. As the annealing temperature was increased to 950 ℃, the grain orientations of Nb3Sn remained while the growth of grain size was significant.

We assume this recrystallization follows a boundary migration mechanism and evaluate the boundary mobility by the Arrhenius law, d = A × exp (- E<sup>a</sup> / RT), where d is the equilibrium grain size, A is the pre-exponential factor, E<sup>a</sup> is the activation energy, and T is annealing temperature. The apparent values of the pre-exponential factor and activation energy were determined to be 2.59 × 10<sup>5</sup> and 63 ± 2 kJ/mol, respectively, by Schelb [\[33\]](#page-13-18). At an annealing temperature of 950 ℃, the maximum attainable grain size is in the range of 434 – 643 nm. The observed ∼ 300 nm grain size in our work is reasonable considering the influence from annealing time (6 h in our work versus up to 200 h in Schelb's work).

In summary, recrystallization anneal above 800 ℃ is effective in relieving the built-in strain from sputtering and thus forming stoichiometric Nb3Sn along with grain coarsening.

3.1.2. Film properties Superconducting properties, atomic composition, and surface roughness were investigated on the 100 nm Nb3Sn sample after the 950 ℃ annealing for 6 hours. As shown in figure [3,](#page-6-0) the critical temperature is determined to be 17.5 K, while the Nb/Sn stoichiometry is 3/1 after sputtering away the surface oxides. Similarly, Sayeed et al. [\[6\]](#page-12-5) reported T<sup>c</sup> values of 17.68 – 17.83 K for 350 nm Nb3Sn sputtered films that were annealed at 800 ℃ for 24 hours and 1000 ℃ for 1 hour with low Sn loss. They observed significant degradation of T<sup>c</sup> down to 10.95 K as a consequence of the dramatic Sn loss down to 4% after annealing for 24 h at 1000 ℃. In contrast, we did not observe the Sn loss in the 100 nm thin films after annealing. We infer recrystallization plays a major role in retaining the Sn ratio as well as maintaining T<sup>c</sup> ∼ 17.5 K in the 100 nm thin films, with the relatively short annealing time helping to retain the film properties. However, our 2 µm thick films as detailed in Section 3.2 showed similar Sn loss behavior at increasing annealing temperatures as compared to the 350 nm thick films in Sayeed's work [\[6\]](#page-12-5), which indicates the importance of a recrystallization process to obtain stoichiometric Nb3Sn films with T<sup>c</sup> 17.5 K. Additionally, the atomic force microscopy (AFM) result is

![](_page_6_Figure_4.jpeg)

**Caption:** Figure 3 illustrates the physical characteristics of a 100 nm Nb3Sn thin film on an Nb substrate post-950 ℃ annealing. Panel (a) details the resistive transition revealing the critical temperature of 17.5 K, critical for its superconductivity performance. Panel (b) shows the XPS spectrum post surface sputtering, presenting the Nb and Sn atomic composition, indicating successful stoichiometry maintenance post-surface oxidation removal. Panel (c) exhibits an AFM image reflecting a low surface roughness, crucial for minimizing scattering losses in superconducting radiofrequency applications. Collectively, these analyses confirm the film's phase purity and surface quality necessary for high-performance superconducting applications .


<span id="page-6-0"></span>Figure 3. Film properties for the 100 nm thin Nb3Sn film on a Nb substrate after 950 ℃ annealing. (a) Resistive transition and critical temperature, (b) XPS spectrum showing the atomic composition after sputtering away the surface 20 nm layer, and (c) AFM image showing low surface roughness.

shown in figure 3c. The film shows low surface roughness, with an average roughness of 18.3 nm, RMS roughness of 25.3 nm, and a maximum height difference of 600 nm. In contrast, Nb substrates used have an average roughness of ∼ 70 nm.

# 3.2. Thick Nb3Sn and V3Si films: relation of strain and composition change

Thermal annealing was performed on 2 µm thick Nb3Sn and V3Si films on the Nb substrate. The initial thickness of the films greatly altered the annealing behaviors as compared to results from 100 nm thin films.

3.2.1. 2 µm thick Nb3Sn films The Nb3Sn grains nucleated in a triangular shape and remained in that shape at all annealing temperatures studied (figure [4a](#page-7-0) and [4b](#page-7-0)). We speculate that these small triangular-shaped grains with 100 – 200 nm in size were induced by the high built-in stress and subsequent plastic deformation during deposition. Inplane stress is typical in physical vapor sputtering, and small grains with angular shapes are favored under the stress [\[35\]](#page-13-19). This argument is supported by the in situ stress versus grain size relationship in Leib's work [\[36\]](#page-13-20).

Upon annealing, as shown in figure [5a](#page-8-0), the 2 µm thick Nb3Sn films experienced significant Sn loss from the as-deposited ∼ 24% down to 21% after the initial anneal at 600 ℃ and further down to nearly 2% after the 950 ℃ anneal. Conversely, Nb3Sn phases were barely observed in the X-ray diffraction until Nb3Sn peaks appeared at 800 ℃ and 950 ℃ anneals. The strain () for a given plane was calculated by = (a<sup>T</sup> - a0) / a0, where a<sup>T</sup> is the measured lattice constant from Nb3Sn plane diffraction and a<sup>0</sup> is the lattice parameter from database [\[34\]](#page-13-17). (Internal strains calculated for all samples are summarized in Table S2.) The relative strain (∆), shown in figure [6a](#page-9-0), was obtained by normalizing strain to the hightemperature anneal limit where we observe negligible strains.

<span id="page-7-0"></span>Here, we analyze the effect of film Figure 4. Surface SEM images for 2 <sup>µ</sup>m thick Nb3Sn

![](_page_7_Picture_6.jpeg)

**Caption:** Figure 4 provides SEM morphologies of 2 µm thick Nb3Sn and V3Si films on Nb substrates under various states: (a, c) as-deposited and (b, d) post-950 °C annealing. This comparison highlights high built-in film strain and its influence on grain size and boundary density variations, reflecting the mechanical repercussions of thermal treatments. The image series is instrumental in visualizing the stress-induced phase transformations in these superconducting films .


(a, b) and V3Si (c, d) films on Nb substrates: (a, c) as-deposited and (b, d) after 950 ℃ annealing.

thickness on the strain. The internal strain () in a biaxial thin film system where inplane stresses are equal (δ = δ<sup>11</sup> = δ22) can be described by a linear relationship as = (2 S<sup>1</sup> + 1/2 S<sup>2</sup> sin<sup>2</sup>φ) δ, where φ is the angle from the film normal to the diffraction plane normal, and S<sup>1</sup> and S<sup>2</sup> are the Xray elastic constants that are determined, in an elastic isotropic scenario, by Young's modulus (E) and Poisson's ratio (υ) and given by – υ / E and (1 + υ) / E, respectively [\[36\]](#page-13-20). The built-in stress increases with film thickness (or deposition time at a fixed deposition rate) in a polycrystalline film system when the deposition goes beyond the initial instantaneous stress stage (< 10 nm thickness) [\[35\]](#page-13-19). This positive correlation, although slightly affected by the growthinterrupt stress relaxation effect and the heating effect, suggests high strain in the 2 µm thick film; however, the high in-plane stress during thicker film sputtering results in plastic deformation as indicated by the observation of small grain sizes and high density of boundaries (figure [4a](#page-7-0)).

Furthermore, we cannot fully explain the Sn loss in the course of annealing the sputtered Nb3Sn films, i.e., the decrease of Sn/(Nb+Sn) atomic ratios with the increasing annealing temperature (figure [5a](#page-8-0)). Note that this Sn loss behavior is repeatedly observed in previous Nb3Sn sputtering work [\[6,](#page-12-5) [12\]](#page-12-11). At the annealing temperatures studied, pure Sn phases are not expected due to their low vaporization temperatures (e.g., 800 ℃ at 10<sup>−</sup><sup>6</sup> Torr), so we primarily consider Nb-Sn alloy phases in the film. The as-deposited film showed a 23 – 25% Sn atomic ratio which suggests minimal Sn-rich phases (Nb6Sn<sup>5</sup> and NbSn2) based on the Nb-Sn phase diagram [\[3\]](#page-12-2); these Sn-rich phases were also not observed in the X-ray diffraction. Without Sn or Sn-

![](_page_8_Figure_3.jpeg)

**Caption:** Figure 5 presents the compositional dynamics of Nb3Sn and V3Si films under annealing. Panel (a) shows Sn/[Sn+Nb] or Si/[Si+V] atomic ratios as functions of annealing temperature, illustrating Sn and Si loss tendencies in thick Nb3Sn and V3Si films deposited on different substrates. The annealing-induced atomic concentration changes are critical for understanding the thermal stability and phase transformations within these films. Panel (b) gives an example of an EDS spectrum highlighting compositional data acquisition in V3Si films, underlining the utility of EDS in compositional analysis. This figure is pivotal in elucidating the metallurgical evolution of superconducting films under thermal stress .


<span id="page-8-0"></span>Figure 5. (a) Sn/[Sn+Nb] or Si/[Si+V] ratios in the Nb3Sn and V3Si films, respectively, as a function of annealing temperature for the 2 µm thick films sputtered on Nb substrates and the 300 nm thick films on Cu substrates (discussed in Section 3.3). Note that the high Si ratios for Cu substrate samples are due to exclusion of Cu signals for calculation. As-deposited Sn/Si composition on 2 µm thick films are 23 – 25 %. (b) Example of the EDS spectrum taken on the 2 µm thick V3Si film for generating the composition dataset.

rich phases, merely Nb3Sn is expected in this study as indicated by the 23 – 25% Sn ratio, but XRD did not show any detectable Nb3Sn diffractions upon deposition; the crystalline Nb3Sn phase has an extremely high (> 2100 ℃) phase transformation temperature, making it unlikely to explain the Sn loss. We, therefore, suspect the generation of
![](0__page_9_Figure_1.jpeg)

**Caption:** Figure 6 depicts the relative strain and phase stability of Nb3Sn thin films across different thermal treatments. Panel (a) presents the relative strain normalized to the high-temperature anneal limit as a function of annealing temperature for 100 nm and 2 µm thick Nb3Sn films on Nb substrates, and a 300 nm Nb3Sn film on Cu substrates, highlighting the thermal strain response. Panel (b) shows the temperature-dependent strain diagram derived from stable and unstable (220) diffraction peak shifts for V3Si films, emphasizing the mechanical stability differences in films of varying thicknesses and substrate types. Lastly, panel (c) provides an example XRD pattern from a 2 µm V3Si film, demonstrating the coexistence of stable and unstable diffraction peaks crucial for understanding the film's structural integrity. This figure underscores the significance of strain management in optimizing superconducting film performance during high-temperature treatments .


<span id="page-9-0"></span>Figure 6. (a) Relative strain that is normalized to the high-temperature anneal limit, as a function of annealing temperature for the 100 nm and 2 µm Nb3Sn films on Nb substrates together with the 300 nm Nb3Sn film on Cu substrates. (b) Temperature-dependent strain diagram calculated from stable (s) and unstable (u) (220) diffraction peak shifting for 2 µm and 300 nm V3Si films on the Nb and Cu substrates, respectively. (c) Example of the XRD patterns taken on the 2 µm thick V3Si film showing the stable and unstable (220) diffraction peaks for generating the strain diagram.

amorphous Nb3Sn phases in the film. Such amorphous phases were reported when using non-equilibrium processing techniques [\[38,](#page-13-0) [39\]](#page-13-1). This could cause the loss of Sn alloys via the generation of α-Nb and also explain the appearance of Nb3Sn diffraction for anneals above 800 ℃, which likely corresponds to the crystallization temperature. This requires further investigation.

3.2.2. 2 µm thick V3Si films Different from thick Nb3Sn films, the as-deposited 2 µm thick V3Si film (figure [6b](#page-9-0)) exhibits a high strain of 15%, which supports the positive relationship between strain and film thickness in an elastic scenario for thick (> 10 nm) polycrystalline films. The initial film shows a near-stoichiometric value of Si (∼ 23%) shown in figure [5b](#page-8-0).

Upon annealing, the V3Si film shows a constant Si concentration for all temperatures; see figure [5a](#page-8-0). In contrast, the strain within the film is significantly relieved together with a transition from the unstable V3Si structure to the stable structure between 800 ℃ and 950 ℃ (figure [6b](#page-9-0)). The structural transformation is observed through the shifting of the (220) and (222) diffraction peaks in figure [6c](#page-9-0). These behaviors demonstrate that thermal annealing contributes to strain reduction and structural stabilization in a thick sputtered film. However, as shown in figure [4d](#page-7-0), large cracks begin to appear on the film after the first anneal at 600 ℃, coinciding with a shift toward a more angular grain shape with increasing temperature. The high strain induced by the sputtering deposition is responsible for the cracks although thermal relaxation has reduced a significant amount of lattice strain.

# 3.3. Nb3Sn and V3Si films on Cu substrates: ternary alloy systems

By studying the temperature-atomic percentage phase diagrams of Nb-Sn [\[3\]](#page-12-0) and V-Si [\[45\]](#page-14-0), as well as the three-element composition phase diagrams of Cu-Nb-Sn [\[40,](#page-13-2) [41,](#page-13-3) [42\]](#page-13-4) and Cu-V-Si [\[43,](#page-14-1) [44\]](#page-14-2), we can gain insight into the phase transformations our films undergo during the annealing process.

As shown in figures [7a](#page-10-0) and [7c](#page-10-0), 300 nm thick Nb3Sn and V3Si films were sputtered on Cu substrates using a 550 ℃ in situ heating stage. Upon annealing, both films undergo dramatic grain structure changes due to the generation of Cu-Sn or Cu-Si phases. Nb3Sn grains start with rounded grains collecting in finger-like formations as deposited on a Cu surface (figure [7a](#page-10-0)) and they remelt into small angular grains collecting in regions of differing densities after 950 ℃ anneal (figure [7b](#page-10-0)). In contrast, V3Si grains begin with a finger-like pattern after deposition (figure [7c](#page-10-0)) and end with small angular grains and large artifacts scattered across the surface after 950 ℃ anneal (figure [7d](#page-10-0)). Overall, there is a trend of grain angularization and pattern restructuring with increasing temperature.

<span id="page-10-0"></span>The ternary phase transformation that includes Cu-alloy in the films primarily determines the film properties. As shown in figure [5a](#page-8-0), the 300 nm Nb3Sn films on Cu substrates suffer from the Sn loss similar to 2 µm thick films on Nb substrates, but the mechanism is different. According to the Nb-Sn-Cu phase diagram [\[40,](#page-13-2) [41,](#page-13-3) [42\]](#page-13-4), Cu-Sn and Nb-Sn phases generate at low temperatures (e.g., 675 ℃ [\[40\]](#page-13-2)) and these Cu-Sn transform into liquid at 800 ℃ under atmospheric pressure [\[41\]](#page-13-3), and at high temperatures (e.g., 1000 ℃ [\[42\]](#page-13-4)), only Nb3Sn and Cu exist. In

![](0__page_10_Picture_5.jpeg)

**Caption:** Figure 7 compares SEM images of 300 nm Nb3Sn and V3Si films on Cu substrates, illustrating surface morphological transformations from as-deposited to after 950 °C annealing. Both films exhibit significant grain structure changes, with Nb3Sn showing angular grain formations and V3Si displaying large surface artifacts post-anneal. These structural evolutions highlight the influence of high-temperature annealing on alloy phase formations and are essential for comprehending phase behavior and optimizing film properties for practical applications in superconducting cavities .


our study, high-vacuum annealing vaporized Figure 7. Surface SEM images for 300 nm Nb3Sn (a, b) and V3Si (c, d) films on Cu substrates: (a, c) as-deposited and (b, d) after 950 ℃ annealing.

the Cu-Sn phases leading to a continuous loss of Sn with increasing temperature. Also, we observed low-intensity Nb3Sn diffraction at all temperatures whereas convoluted XRD peaks that are possibly from Cu-Sn and other Nb-Sn phases appeared at low temperatures. These observations match with the existence of Nb3Sn in the phase diagram at high temperatures although the majority of the film was evaporated.

Different from Nb-Sn-Cu, the V-Si-Cu phase diagram [\[43,](#page-14-1) [44\]](#page-14-2) shows Cu-Si and V-Si phases at low temperatures (e.g., 700 ℃ [\[43\]](#page-14-1)), but there is no liquid phase at high temperatures (e.g., 800 ℃ [\[44\]](#page-14-2)). Instead, these phases transform into V-Si and Cu phases. In our study, as shown in figure [5a](#page-8-0), the Si/(Si+V) ratio begins with a high value of 42% due to the presence of Cu-Si phases generated during the 550 ℃ in situ heated deposition; note that Cu signal is evident, but is not included in the calculation. After annealing, the Si/(Si+V) ratio drops to 20% at 950 ℃. This phenomenon strongly supports the disappearance of Cu-Si phases in the phase diagram, and only V-Si phases together with some Cu metallic inclusions are expected in the annealed films. Our diffraction data suggest these V-Si phases include the stable (220) and unstable (222) V3Si structures.

#### 4. Conclusions and Outlook

In this study, we have demonstrated the capability of annealing the sputtered thin films to produce successful Nb3Sn and V3Si surfaces that have the potential for use inside SRF cavities. We observe that annealing is required to release the strain in the film and promote grain growth. For our Nb3Sn samples, the best results are found on the recrystallized 100 nm film, where large grains form at 950 ℃ anneals. These films are also smooth and have minimal surface defects. The 2 µm Nb3Sn films are not able to overcome the built-in stress and plastic deformation during sputtering, and likely form an amorphous Nb-Sn phase that leads to nearly complete Sn loss upon annealing. In contrast, the V3Si samples retain the stoichiometry at high temperatures, along with a transition in the grain shape to become more angular. Most interesting was the behavior of these films with respect to the unstable and stable phases of V3Si. In the 2 µm film, there was a complete transition from unstable to stable at 800 ℃ along with consistent stoichiometry. Because we observe this transition and the proper stoichiometry at high temperatures, we determine these are successful V3Si films.

For the Cu substrate samples, 550 °C in situ heated deposition and the subsequent lowtemperature anneals produce Cu-Si and Cu-Sn phases. These phases transform at high temperatures, extracting high concentrations of Cu inclusions in the film. The Cu impurities and Cu-related phases could adversely affect the SRF performance of Nb3Sn/V3Si films inside Cu cavities. In a future study, we would be interested in the use of an ultrathin buffer layer between the Cu and the superconducting layer to prevent this effect [\[27\]](#page-13-5).

In our results, we observed a similar Sn loss as in previous studies [\[6\]](#page-12-1). We are interested in finding ways to prevent this loss such as minimizing strong undercooling and avoiding disordered Nb-Sn phases or using encapsulation during the annealing process. We would like to obtain the benefits of annealing such as recrystallization and strain removal while avoiding events such as Sn loss and cracking. Because the 100 nm Nb3Sn film was successful, it would be important in a future study to further investigate films of

similar thickness to optimize grain growth and RF performance.

## Data Availability Statement

The data that support the findings of this study are available upon reasonable request from the authors.

## Conflicts of Interest

The authors declare no competing financial interests.

## Acknowledgements

This work was supported by the U.S. National Science Foundation under Award PHY-1549132, the Center for Bright Beams. This work made use of the Cornell Center for Materials Research Shared Facilities which are supported through the NSF MRSEC program (DMR-1719875).

### References

- [1] Howard K, Liepe M and Sun Z 2021 Thermal annealing of sputtered Nb3Sn and V3Si thin films for superconducting RF cavities Proc. SRF'21 (East Lansing, MI, USA) 82-85 doi: 10.18429/JACoW-SRF2021-SUPFDV009
- [2] Grassellino A et al. 2017 Unprecedented quality factors at accelerating gradients up to 45 MVm<sup>−</sup><sup>1</sup> in niobium superconducting resonators via low temperature nitrogen infusion Supercond. Sci. Technol. 30 9 doi: 10.1088/1361- 6668/aa7afe
- <span id="page-12-0"></span>[3] Posen S and Hall D L 2017 Nb3Sn superconducting radiofrequency cavities: fabrication, results, properties, and prospects Supercond. Sci. Technol. 30 033004 doi: 10.1088/1361- 6668/30/3/033004
- [4] Ilyina E A et al. 2019 Development of sputtered Nb3Sn films on copper substrates for superconducting radiofrequency applications Supercond.

Sci. Technol. 32 035002 doi: 10.1088/1361- 6668/aaf61f

- [5] Valles N and Liepe M 2011 The Superheating Field of Niobium: Theory and Experiment Proc. SRF'2011 (Chicago, IL, USA) 293-301 https://accelconf.web.cern.ch/SRF2011/papers/ tuioa05.pdf
- <span id="page-12-1"></span>[6] Sayeed M N et al. 2021 Properties of Nb3Sn films fabricated by magnetron sputtering from a single target Appl. Surf. Sci. 541 148528 doi: 10.1016/j.apsusc.2020.148528
- [7] Keckert S et al. 2019 Critical fields of Nb3Sn prepared for superconducting cavities Supercond. Sci. Technol. 32 075004 doi: 10.1088/1361- 6668/ab119e
- [8] Catelani G and Sethna J P 2008 Temperature dependence of the superheating field for superconductors in the high-κ London limit Phys. Rev. B 78 224509 doi: 10.1103/PhysRevB.78.224509
- [9] Porter R D et al. 2019 Progress in Nb3Sn SRF cavities at Cornell University Proc. NAPAC'19 37-40 doi: 10.18429/JACoW-NAPAC2019- MOYBB3
- [10] Estrin F L et al. 2021 Using HiPIMS for V3Si superconducting thin films presented at 9th Int. Workshop on Thin Films and New Ideas for Pushing the Limits of RF Superconductivity https://indico.jlab.org/event/405/contributions/ 8111/
- [11] Deambrosis S M et al. 2007 The progress on Nb3Sn and V3Si Proc. SRF'2007 (Beijing, China) 392-399 <http://accelconf.web.cern.ch/accelconf/srf2007/> PAPERS/WE203.pdf
- [12] Sayeed M N et al. 2019 Structural and superconducting properties of Nb3Sn films grown by multilayer sequential magnetron sputtering J. Alloys Compd. 800 272–278 doi: 10.1016/j.jallcom.2019.06.017
- [13] Lee J et al. 2019 Atomic-scale analyses of Nb3Sn on Nb prepared by vapor diffusion for superconducting radiofrequency cavity applications: a correlative study Supercond. Sci. Technol. 32 024001 doi: 10.1088/1361-6668/aaf268
- [14] Barzi E et al. 2016 Synthesis of superconducting Nb3Sn coatings on Nb substrates Supercond. Sci. Technol. 29 015009 doi: 10.1088/0953- 2048/29/1/015009
- [15] Sun Z et al. 2021 Toward stoichiometric and low-surface-roughness Nb3Sn thin films via direct electrochemical deposition Proc.

SRF'21 (East Lansing, MI, USA) 710-713 doi: 10.18429/JACoW-SRF2021-WEOTEV03

- [16] Sun Z et al. 2019 Electroplating of Sn film on Nb substrate for generating Nb3Sn thin films and post laser annealing Proc. SRF'19 (Dresden, Germany) 51-54 doi: 10.18429/JACoW-SRF2019-MOP014
- [17] Xiao L et al. 2019 The technical study of Nb3Sn film deposition on copper by HiPIMS Proc. SRF'19 (Dresden, Germany) 846-847 doi: 10.18429/JACoW-SRF2019-THP008
- [18] Becker C et al. 2015 Analysis of Nb3Sn surface layers for superconducting radio frequency cavity applications Appl. Phys. Lett. 106 082602 doi: 10.1063/1.4913617
- [19] Posen S et al. 2021 Advances in Nb3Sn superconducting radiofrequency cavities towards first practical accelerator applications Supercond. Sci. Technol. 34 025007 doi: 10.1088/1361- 6668/abc7f7
- [20] Eremeev G et al. 2020 Nb3Sn multicell cavity coating system at Jefferson Lab Rev. Sci. Instrum. 91 073911 doi: 10.1063/1.5144490
- [21] Sch¨afer N et al. 2020 Kinetically induced lowtemperature synthesis of Nb3Sn thin films J. Appl. Phys. 128 133902 doi: 10.1063/5.0015376
- [22] Ito R et al. 2019 Nb3Sn thin film coating method for superconducting multilayered structure Proc. SRF'19 (Dresden, Germany) 628- 631 doi: 10.18429/JACoW-SRF2019-TUP077
- [23] Lu M et al. 2019 Electrochemical deposition of Nb3Sn on the surface of copper substrates Proc. SRF'19 (Dresden, Germany) 624-627 doi: 10.18429/JACoW-SRF2019-TUP076
- [24] Ge M et al. 2019 CVD coated copper substrate SRF cavity research at Cornell University Proc. SRF'19 (Dresden, Germany) 381-385 doi: 10.18429/JACoW-SRF2019-TUFUB8
- [25] Godeke A et al. 2006 Nb3Sn for radio frequency cavities Lawrence Berkeley National Laboratory https://escholarship.org/uc/item/6d3753q7
- [26] Rosaz G et al. 2015 Development of Nb3Sn coatings by magnetron sputtering for SRF cavities Proc. SRF'15 (Whistler, BC, Canada) 691-694 doi: 10.18429/JACoW-SRF2015-TUPB051
- <span id="page-13-5"></span>[27] Fernandez S 2020 Magnetron sputtered Nb3Sn and V3Si thin films on copper substrates for SRF application presented at Nb3SnSRF'20 https://indico.classe.cornell.edu/event/1806/ contributions/1491/
- [28] Zhang W et al. 2021 Thin film synthesis of

superconductor on insulator A15 vanadium silicide Sci. Rep. 11 2358 doi: 10.1038/s41598- 021-82046-1

- [29] Tan W et al. 2017 Nb3Sn thin film deposition on copper by DC magnetron sputtering Proc. SRF'17 512-515 doi: 10.18429/JACoW-SRF2017-TUPB055
- [30] Doherty R D et al. 1997 Current issues in recrystallization: a review Mater. Sci. Eng. A 238 219-274 doi: 10.1016/S0921-5093(97)00424-3
- [31] Bussiere J F et al. 1980 Young's modulus of polycrystalline Nb3Sn between 4.2 and 300 K J. Appl. Phys. 51 1024 doi: 10.1063/1.327730
- [32] Humphreys F J and Hatherly M Recrystallization and Related Annealing Phenomena, 2nd Edition, Elsevier Ltd., Amsterdam, 2004.
- [33] Schelb W 1981 Electron microscopic examination of multifilamentary bronze-processed Nb3Sn J. Mater. Sci. 16 2575–2582 doi: 10.1007/BF01113599
- <span id="page-13-6"></span>[34] Jain A et al. 2013 The Materials Project: A materials genome approach to accelerating materials innovation APL Mater. 1 011002 doi: 10.1063/1.4812323
- [35] Abadias G et al. 2018 Stress in thin films and coatings: Current status, challenges and prospects J. Vac. Sci. Technol. A 36 020801 doi: 10.1116/1.5011790
- [36] Leib J S Relationships between grain structure and stress in thin Volmer-Weber metallic films Massachusetts Institute of Technology, 2018.
- [37] Vink T J et al. 1993 Stress, strain, and microstructure in thin tungsten films deposited by dc magnetron sputtering J. Appl. Phys. 74 15 doi: 10.1063/1.354842
- <span id="page-13-0"></span>[38] Matsuki K et al. 1988 New amorphous Cu-Nb- (Si, Ge or Sn) alloys prepared by mechanical alloying of elemental powders Mater. Sci. Eng. 97 47-51 doi: 10.1016/0025-5416(88)90010-9
- <span id="page-13-1"></span>[39] Masumoto T et al. 1980 Superconductivity of ductile Nb-Based amorphous alloys Trans. Jpn. Inst. Met. 21 115-122 doi: 10.2320/matertrans1960.21.115
- <span id="page-13-2"></span>[40] Neijmeijer W L and Kolster B H 1987 The ternary system Nb-Sn-Cu at 675 °C Int. J. Mater. Res. 78 730-737 doi: 10.1515/ijmr-1987-781009
- <span id="page-13-3"></span>[41] Pan V M et al. 1980 Phase equilibria and superconducting properties in niobium-tincopper alloys USSR
- <span id="page-13-4"></span>[42] Hopkins R H et al. 1977 Phase relations and diffusion layer formation in the systems Cu-Nb-

Sn and Cu-Nb-Ge Metall. Trans. A 8 91-97 doi: 10.1515/ijmr-1987-781009

- <span id="page-14-1"></span>[43] Reid J S et al. 1992 Thermodynamics of (Cr, Mo, Nb, Ta, V, or W)-Si-Cu ternary systems J. Mater. Res. 7 2424-2428 doi: 10.1557/JMR.1992.2424
- <span id="page-14-2"></span>[44] Savitskii E M et al. 1979 Influence of copper on the structure and superconducting properties of transition metals Inorg. Mater. 15 512-515
- <span id="page-14-0"></span>[45] Okamoto H et al. 2010 Si-V (Silicon-Vanadium) J. Phase Equilib. Diffus. 31 409–410 doi: 10.1007/s11669-010-9733-5

# Appendix

![](0__page_15_Figure_2.jpeg)

**Caption:** Figure S1 depicts an SEM map of multiple samples at various stages: as-deposited, after 700 °C anneal, and post-950 °C anneal. The images vividly capture the grain growth and morphological evolution of films with increasing annealing temperatures, with significant coarsening and recrystallization occurring at higher temperatures. These transformations, observable across different sample conditions, illustrate critical microstructural changes necessary for enhancing superconducting film performance, ultimately contributing to the understanding and optimization of thin film processes for superconductor applications .


Figure S1. SEM map of all samples upon deposition, after 700 °C anneal, and after 950 °C anneal.

|                 | 1 0 -- 0 ) |       |                 |             |       |               |      |       |
|-----------------|------------|-------|-----------------|-------------|-------|---------------|------|-------|
| ID              | 20         | plane | ID              | 20          | plane | ID            | 20   | plane |
| NbSn2           | 31.5       | 131   | V3Cu-cubic      | 43.2        | 220   | Nb3Sn         | 62.8 | 320   |
| Nb3Sn           | 33.6       | 200   | Cu15Si4         | 43.8        | 332   | Nb3Sn         | 65.5 | 321   |
| NbSn2           | 34.2       | 133   | V = Si3         | 44.1        | 411   | V3Si-unstable | 65.7 | 222   |
| Cu15Si4         | 34.6       | 321   | V3Si-unstable   | 45.1        | 211   | V3Si-stable   | 69.2 | 222   |
| V3Si-unstable   | 36.5       | 200   | Cu15Si4         | 45.8        | 422   | Nh            | 69.3 | 211   |
| Nb6Sn5          | 37.2       | 26    | V3Cu-tetragonal | 46.8        | 4     | V-unstable    | 69.6 | 220   |
| NbSn2           | 37.3       | 117   | V 5Si3          | 47.2        | 222   | Nb3Sn         | 70.6 | 400   |
| Nb3Sn           | 37.7       | 210   | V3Si-stable     | 47.4        | 211   | V3Si-unstable | 71.8 | 321   |
| Nb              | 38.3       | 110   | V-unstable      | 47.6        | 200   | V3Si-stable   | 72.5 | 320   |
| V3Si-stable     | 38.3       | 200   | Cu15Si4         | 47.8        | 210   | Nb3Cu         | 71.3 | 422   |
| Nb6Sn5          | 38.5       | 222   | V3Cu-tetragonal | 49.4        | 200   | Cu            | 74   | 220   |
| Nb3Cu           | 39.3       | 220   | Cu3Si           | 49.5        | 4     | V3Si-stable   | 75.7 | 321   |
| V 5Si3          | 39.4       | 321   | Cu              | 50.4        | 200   | V3Si-unstable | 77.6 | 400   |
| NbSn2           | 40.9       | 224   | Cu3Si           | 50.6        | 200   | V-stable      | 78.3 | 211   |
| V-unstable      | 40.9       | 111   | V3Si-unstable   | 52.6        | 220   | V3Cu-cubic    | 79.1 | 422   |
| Nb3Sn           | 41.5       | 211   | Nb              | 53.3        | 200   | Nb3Sn         | 80.5 | 420   |
| V3Cu-tetragonal | 41.7       | 112   | Nb3Sn           | 54.4        | 310   | V3Si-stable   | 82   | 400   |
| V 6Si5          | 42.3       | 321   | V3Si-stable     | ર્ ર ર : 3  | 220   | Nb            | 82.1 | 220   |
| Cu              | 42.3       | 111   | Nb3Cu           | ર્સ રેને તે | 400   | Nb3Sn         | 82.9 | 421   |
| V-stable        | 42.7       | 110   | V3Si-unstable   | 59.4        | 310   | V-unstable    | 84.1 | 311   |
| V 6Si5          | 42.8       | 132   | Nb3Sn           | 60.1        | 222   | Nb3Sn         | 85.3 | 322   |
| V3Si-stable     | 43         | 210   | V-stable        | 62          | 200   | V-unstable    | 88.7 | 222   |
| V 5Si3          | 43         | 420   | V3Si-stable     | 62.5        | 310   | V3Si-unstable | 88.9 | 420   |
| Cu3Si           | 43.2       | 112   | V3Cu-cubic      | 62.7        | 400   | Cu            | 89.8 | 311   |

Table S1. X-ray diffraction (XRD) peaks of Nb3Sn, V3Si (stable vs. unstable), substrates Nb and Cu, and other possibly relevant phases (NbSn2, Nb6Sn5, V5Si3, V6Si5, Nb3Cu, V3Cu Cu15Si4, Cu3Si4, and V (unstable) from reference [\[34\]](#page-13-6). For NbSn2, Nb6Sn5, V5Si3, and V6Si<sup>5</sup> peaks, we only listed the prominent points.

| 2 um V3Si       | peak plane (2theta) |              |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
|-----------------|---------------------|--------------|------------|-----------------------------------------------------|-------------|--------------|---------------------------|----------------------|------------|----------|-------------------------------------------------------------------------|----------|----------|
| temperature (C) | 200s (38.3)         | 210s (43)    |            | 211s (47.4)  220u/s (52.6/55.3)  222u/s (65.7/69.2) |             | 320s (72.5)  | 321s (75.7)               | 400s (82)            |            |          |                                                                         |          |          |
| 25              |                     |              |            | 0.141414                                            | 0.1416167   |              |                           | -0.0091019           |            |          |                                                                         |          |          |
| 600             |                     |              |            | 0.143407                                            | 0.1431387   |              |                           | -0.0091019           |            |          |                                                                         |          |          |
| 800             | -0.00371            | 0.010709926  | 0.002333   | 0.0154283                                           | -0.0056307  |              | 0.0069136                 | -0.0091019           |            |          |                                                                         |          |          |
| 950             | -0.00371            | 0.0129835    | 0.006355   | 0.0119857                                           | -0.0056307  | 0.004912598  | 0.0092106                 | -0.0091019           |            |          |                                                                         |          |          |
|                 |                     |              |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
| 100 nm Nb3Sn    | peak plane (2theta) |              |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
| temperature (C) | 200 (33.6)          | 210 (37.7)   | 211 (41.5) | 320 (62.8)                                          | 321 (65.6)  | 400 (70.6)   | 421 (82.9)                |                      |            |          |                                                                         |          |          |
| 25              | -0.03165            |              |            | 0.007852                                            | -0.01337    |              | 0.002509                  |                      |            |          |                                                                         |          |          |
| 600             | -0.01792            |              |            |                                                     | -0.01337    | 0.000687     | 0.004506                  |                      |            |          |                                                                         |          |          |
| 800             | -0.00378            | -0.00833     | -0.01311   | 0.003501                                            | -0.00941    | 0.00817      | 0.003506                  |                      |            |          |                                                                         |          |          |
| ਰੇਟੋ0           | -0.01232            | -0.00833     | -0.01311   | 0.004946                                            | -0.01073    | 0.006914     | 0.000522                  |                      |            |          |                                                                         |          |          |
| 2 um Nb3Sn      | peak plane (2theta) |              |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
| temperature (C) | 200 (33.6)          | 210 (37.7)   | 211 (41.5) | 310 (54.4)                                          | 320 (62.8)  | 321 (65.6)   | 400 (70.6)                | 421 (82.9)           | 322 (85.3) |          |                                                                         |          |          |
| 25              |                     | -0.008331347 |            | -0.02744                                            |             |              |                           |                      |            |          |                                                                         |          |          |
| 600             |                     |              |            | -0.02263                                            |             |              |                           |                      |            |          |                                                                         |          |          |
| 800             | -0.0066434          | -0.018277662 | -0.0086    | -0.02585                                            | 0.004946    | -0.008075188 | -0.009077                 |                      | -0.11031   |          |                                                                         |          |          |
| ਰੇਟੋ0           | -0.009488           | -0.018277662 | -0.0086    | -0.02101                                            | 0.004946    | -0.010732407 |                           | -0.00787 0.000522439 | -0.11031   |          |                                                                         |          |          |
| 300 nm Nb3Sn    | peak plane (2theta) |              |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
| temperature (C) | 200 (33.6)          | 210 (37.7)   | 310 (54.4) | 320 (62.8)                                          | 420 (80.5)  | 421 (82.9)   | 322 (85.3)                |                      |            |          |                                                                         |          |          |
| 550             |                     | -0.00581     |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
| 600             | -0.00949            | -0.00581     |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
| 800             | -0.0009             | -0.02073     | 0.016175   | 0.010778                                            | 0.009809    | 0.004506     | -0.12049                  |                      |            |          |                                                                         |          |          |
| ਰੇ 20           | -0.00664            | -0.01084     |            | 0.015205                                            | 0.014072    |              | -0.11966                  |                      |            |          |                                                                         |          |          |
| 300 nm V3Si     | peak plane (2theta) |              |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
| temperature (C) | 200u (36.5)         | 200s (38.3)  | 210s (43)  | 211u (45.1)                                         | 211s (47.4) |              | 220s (55.3)   310u (59.4) |                      |            |          | 310s (62.5) 222u (65.7) 320s (72.5) 400u (77.6) 400s (82)   420u (88.9) |          |          |
| 550             |                     |              |            |                                                     |             |              |                           |                      |            |          |                                                                         |          |          |
| 600             |                     |              | -0.0049    |                                                     |             |              |                           |                      |            | -0.00108 |                                                                         |          | 0.151889 |
| 800             | 0.147913            | -0.00121     | -0.0049    | 0.15903                                             | 0.014506    | 0.020642     | 0.171245                  | -0.00709             | 0.157062   | -0.0058  | 0.157251                                                                | -0.00317 | 0.158064 |
| ਰੇ 20           |                     |              | -0.0005    |                                                     |             | 0.020642     |                           |                      | 0.161794   |          |                                                                         | -0.00016 | 0.149852 |

#### Table S2. Strain for all detected peaks compared to known peak locations.