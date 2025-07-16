# arXiv:2305.02467

**Paper ID:** b282448913e4d8d591f54424fdcbc8c6

**PDF Path:** apl/Superconductors/arXiv:2305.02467.pdf

**Processing Status:** complete

**Captions Added:** 14

**Generated:** 2025-06-24T13:44:27.773161

---

# Surface oxides, carbides, and impurities on RF superconducting Nb and Nb3Sn: A comprehensive analysis

Zeming Sun<sup>1</sup>,2,<sup>∗</sup> , Zhaslan Baraissov<sup>3</sup> , Catherine A. Dukes<sup>4</sup> , Darrah K. Dare<sup>5</sup> , Thomas Oseroff<sup>1</sup>,<sup>2</sup> , Michael O. Thompson<sup>6</sup> , David A. Muller<sup>3</sup> , Matthias U. Liepe<sup>1</sup>,<sup>2</sup>

<sup>1</sup> Cornell Laboratory for Accelerator-based Sciences and Education, Cornell University, Ithaca, New York, 14853, United States of America

<sup>2</sup> Department of Physics, Cornell University, Ithaca, New York, 14853, United States of America

<sup>3</sup> School of Applied & Engineering Physics, Cornell University, Ithaca, New York, 14853, United States of America

<sup>4</sup> Laboratory for Astrophysics and Surface Physics, University of Virginia,

Charlottesville, Virginia, 22904, United States of America

<sup>5</sup> Cornell Center for Materials Research, Cornell University, Ithaca, New York, 14853, United States of America

<sup>6</sup> Materials Science and Engineering, Cornell University, Ithaca, New York, 14853, United States of America

E-mail: zs253@cornell.edu

September 2023

Abstract. Surface structures on radio-frequency (RF) superconductors are crucially important in determining their interaction with the RF field. Here we investigate the surface compositions, structural profiles, and valence distributions of oxides, carbides, and impurities on niobium (Nb) and niobium-tin (Nb3Sn) in situ under different processing conditions. We establish the underlying mechanisms of vacuum baking and nitrogen processing in Nb and demonstrate that carbide formation induced during high-temperature baking, regardless of gas environment, determines subsequent oxide formation upon air exposure or low-temperature baking, leading to modifications of the electron population profile. Our findings support the combined contribution of surface oxides and second-phase formation to the outcome of ultra-high vacuum baking (oxygen processing) and nitrogen processing. Also, we observe that vapor-diffused Nb3Sn contains thick metastable oxides, while electrochemically synthesized Nb3Sn only has a thin oxide layer. Our findings reveal fundamental mechanisms of baking and processing Nb and Nb3Sn surface structures for high-performance superconducting RF and quantum applications.

Keywords: niobium, niobium-tin, surface, oxide, carbide, impurity, superconducting radio-frequency

## 1. Introduction

Niobium (Nb) oxides with diverse chemistries, phase structures, atomic-scale layering, and properties are widely used in various applications, including catalysis, electrolytic capacitors, solar cells, photodetectors, carrier selective contacts, photochromic and electrical-switching devices, and critical temperature sensors [\[1\]](#page-22-0). The oxides in the form of native layers on the superconducting Nb surface are critical – functional or detrimental – in a wide range of emerging superconducting radio-frequency/microwave (SRF) technologies, such as SRF resonators [\[2\]](#page-22-1), SRF cavities and photoinjectors for particle accelerator applications [\[3,](#page-22-2) [4\]](#page-22-3), superconducting quantum computing [\[5,](#page-22-4) [6\]](#page-22-5), microwave parametric amplifiers [\[7\]](#page-22-6), and ultra-sensitive detectors and filters (e.g., kinetic inductance detectors [\[8\]](#page-22-7)).

In high-field (MVm<sup>−</sup><sup>1</sup> -scale), high-operation-temperature (2 – 4.2 K) SRF applications, surface oxide alterations and subsurface impurities significantly affect relevant cavity metrics of the industry-standard Nb (critical temperature T<sup>c</sup> =9.2 K) [\[9,](#page-22-8) [10,](#page-22-9) [11,](#page-22-10) [12,](#page-22-11) [13\]](#page-22-12). These foreign surface structures could also impact the next-generation SRF candidate niobium-tin (Nb3Sn, T<sup>c</sup> = 18 K). The RF field only interacts with the near-surface layer of superconductors, i.e., ∼ 40 nm in Nb and ∼ 100 nm in Nb3Sn, whereas the "dirty" native layer is typically several nanometers thick, with impurities appearing as deep as tens of nanometers (as observed in this work).

An ideal SRF surface could consist of either a dielectric capping layer or a normalconducting proximity-coupled layer proposed by Gurevich and Kubo [\[14,](#page-22-13) [15\]](#page-22-14). The normal-conducting design could potentially utilize oxides of the desired thickness, structure, and electrical properties [\[13\]](#page-22-12). One of the objectives of this work is to provide a map of the structures and depths of surface oxides for (i) in situ heated Nb samples under ultra-high vacuum (UHV) and nitrogen (N2) environments, (ii) 13 types of Nb coupon samples processed by acid/polishing, ozone, anodization, and heat treatments covering nearly all the protocols used in SRF preparations, (iii) Nb3Sn coupons made by conventional vapor diffusion [\[16\]](#page-22-15) and the new electrochemical synthesis [\[17\]](#page-22-16), and their in situ vacuum heated samples.

Inspired by the theoretical surface design [\[14,](#page-22-13) [15\]](#page-22-14), Oseroff [\[18,](#page-22-17) [19\]](#page-22-18) invented a thin-Au/Nb SRF surface, where a nanometer-thin normal-conducting gold (Au) layer could be relevant for losses and potentially beneficial if engineered with subnanometer thinness and specific electrical properties. The advantage of thinner surface oxides was also observed in tunneling experiments [\[13\]](#page-22-12). This work is designed to support the construction of SRF surfaces by fundamentally understanding the nature of surface oxides and valence bonding with impurities and indicating electronic structures.

Additionally, in low-field (Vm<sup>−</sup><sup>1</sup> -scale), low-operation-temperature (mK-scale) SRF applications (e.g., quantum qubits [\[20\]](#page-22-19), cavity quantum electrodynamics (QED) [\[20\]](#page-22-19), and SRF cavities used as QED architecture [\[21\]](#page-22-20)), oxide dielectrics create superconductor-insulator Josephson junction QED and enable the realization of logical operation states. However, amorphous oxides, which act as a two-level system, dominate dissipation and decoherence and hinder the preservation of logical states [\[22,](#page-22-21) [23,](#page-22-22) [24\]](#page-22-23). This study in materials chemistry contributes to the investigation of loss mechanisms and the development of new interfaces for these emerging SRF devices.

Numerous studies [\[3,](#page-22-2) [4,](#page-22-3) [25,](#page-22-24) [26,](#page-22-25) [27,](#page-22-26) [28,](#page-22-27) [29,](#page-22-28) [30,](#page-22-29) [31,](#page-22-30) [32,](#page-23-0) [33,](#page-23-1) [34,](#page-23-2) [35\]](#page-23-3) have investigated surface oxides and oxidization processes on Nb. The Nb surface is easily oxidized and presents signs of oxidization even with low oxygen exposures (0.2 – 2 langmuir) [\[33,](#page-23-1) [34\]](#page-23-2). Oxygen diffuses fairly fast in Nb with coefficients of 3 × 10<sup>−</sup><sup>8</sup> cm<sup>2</sup>/s at 800 °C and 10<sup>−</sup><sup>21</sup> cm<sup>2</sup>/s at 50 °C [\[36\]](#page-23-4). In recent years, native oxides on a high-purity Nb surface exposed to ambient pressure at room temperature (RT) are believed to comprise roughly three layers in the following order: 1 – 6 nm pentoxide (Nb2O5) / 0 – 2 nm dioxide (NbO2) / 0.3 – 1 nm monoxide (NbO) / Nb (table [1\)](#page-3-0). In bulk studies, Nb2O<sup>5</sup> exhibits amorphous, hexagonal, orthorhombic, tetragonal, and monoclinic structures at elevated temperatures [\[37,](#page-23-5) [38\]](#page-23-6), with 3.2 – 4.2 eV band gaps being dielectric [\[39\]](#page-23-7). NbO<sup>2</sup> has a distorted-rutile structure with a 0.7 eV band gap being semiconducting (close to dielectric for the Nb SRF cavity operation). NbO has a vacancy-ordered rocksalt structure [\[40\]](#page-23-8) with a 1.4 K T<sup>c</sup> (normal-conducting at the cavity operation temperature). However, surface oxides do not fully follow their bulk properties. For example, surface Nb2O<sup>5</sup> and NbO<sup>2</sup> decompose and disappear above 250 – 300 °C, instead of transforming into another phase as the bulk behaves. Therefore, resolving accurate surface structures is critical to guiding the interpretation of loss mechanisms and the surface design of dielectric or proximity-coupled normal-conducting passivations.

Despite the large volume of literature (table [1\)](#page-3-0)[\[3,](#page-22-2) [25,](#page-22-24) [26,](#page-22-25) [27,](#page-22-26) [28,](#page-22-27) [29,](#page-22-28) [30,](#page-22-29) [31,](#page-22-30) [32,](#page-23-0) [13\]](#page-22-12), significant variations exist across surface Nb oxides in terms of their native structures, depths, and decomposition products. Such discrepancies hinder a unified theory describing the SRF surface. For example, Eremeev [\[9\]](#page-22-8) observed inconsistent SRF responses between repeated treatments, despite in situ modulation of surface oxide removal and re-growth. The probable causes include the unstable and non-uniform nature of surface oxides, the processing history, small variations between treatments (owing to the Nb surface's extremely high sensitivity/reactivity), and variations in instrument resolutions and analysis methods. These unclear results motivate revisiting the Nb surface oxide chemistry.

Important questions remain, further, of how multiple impurities (e.g., oxygen, O; carbon, C; and nitrogen, N) interact with Nb upon heating and affect subsequent oxidization. Impurity processing has recently become a key element of Nb SRF cavity treatments, producing record accelerating gradients and quality factors approaching the theoretical limit [\[11,](#page-22-10) [12\]](#page-22-11). These treatments involve N<sup>2</sup> processing (so-called "N<sup>2</sup> doping"[\[12\]](#page-22-11) and "N<sup>2</sup> infusion"[\[11,](#page-22-10) [41,](#page-23-9) [42\]](#page-23-10)) at 20 – 45 mTorr N<sup>2</sup> pressures at low (120 – 160 °C) [\[11,](#page-22-10) [41,](#page-23-9) [42\]](#page-23-10), medium (300 – 400 °C) [\[43,](#page-23-11) [44\]](#page-23-12), and high (800 – 1000 °C)[\[12\]](#page-22-11) temperatures, or UHV baking similarly at low [\[10\]](#page-22-9) and medium [\[45\]](#page-23-13) temperatures. With claims of new recipes and new associated physics emerging, there is a need to provide an accurate picture of the processed Nb surface to build the foundation used for relevant theoretical modeling and process development. Secondary ion mass spectroscopy, SIMS,

| Oxide structure and depth                                                                                                                                                           | Thermal decomposition of oxides                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Bake-induced<br>carbides                                                                        | Subsurface<br>impurities                                           | Methods*                                              | Ref.                                           |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------|------------------------------------------------|
| 2 – 6 nm Nb2O5−x<br>/ ∼ 1 nm NbO1−y<br>at<br>(6 – 8)<br>× 10−10 Torr                                                                                                                | < 1 nm<br>NbO1−y<br>1850<br>at<br>at<br>°C<br>8 × 10−10 Torr                                                                                                                                                                                                                                                                                                                                                                                                                                              | –                                                                                               | ∼ 50 nm NbO0.02                                                    | Sputter-assisted<br>ARXPS;<br>XPS;<br>AES             | [3, 25, 26,<br>27]                             |
| Nb2O5 at 10−10 Torr                                                                                                                                                                 | < 1 nm NbO at<br>> 325 °C                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | –                                                                                               | –                                                                  | XPS                                                   | [28]                                           |
| 6 nm Nb2O5<br>at 2<br>× 10−10 Torr                                                                                                                                                  | 1.5 nm Nb2O5<br>/ 4.5 nm NbO2<br>at 280<br>°C<br>at 2<br>× 10−10 Torr<br>< 1 nm NbO2<br>/ 1.5 nm NbO at 380<br>°C<br>at 2<br>× 10−10 Torr<br>< 1 nm<br>NbO<br>at<br>at<br>> 500 °C<br>2 × 10−10 Torr                                                                                                                                                                                                                                                                                                      | C monolayers at<br>< 700 °C;<br>formation<br>NbC<br>during cooling                              | –                                                                  | ARXPS                                                 | [29]                                           |
| 3 nm Nb2O5<br>/ 1 nm NbOx<br>or 2 nm<br>Nb2O5 / NbO2<br>/ other oxides at<br>2 × 10−10 Torr                                                                                         | < 1 nm<br>Nb2O<br>at<br>250 –275<br>at<br>°C<br>10−9 Torr                                                                                                                                                                                                                                                                                                                                                                                                                                                 | –                                                                                               | –                                                                  | Synchrotron<br>glancing<br>XPS;<br>incidence<br>ARXPS | [30, 31]                                       |
| 1 nm Nb2O5<br>/ 0.7 nm NbO2<br>/ 0.3 nm<br>NbO at 8<br>× 10−10 Torr                                                                                                                 | 0.8 nm Nb2O5<br>/ 0.7 nm NbO2<br>/ 0.3 nm<br>NbO at 145<br>°C at 8<br>× 10−10 Torr<br>1 nm NbO at 300<br>°C at 8<br>× 10−10 Torr                                                                                                                                                                                                                                                                                                                                                                          | –                                                                                               | ∼ 10 nm O inter<br>stitials at 145<br>°C<br>at UHV                 | XRR; X-ray dif<br>scattering;<br>fuse<br>XPS          | [32]                                           |
| 1 –3 nm Nb2O5<br>/ 0.2 –1 nm NbO2<br>/<br>∼ 1 nm NbO at 8<br>× 10−9 Torr                                                                                                            | 2 nm Nb2O5<br>/ 1 nm NbO2<br>/ ∼ 1 nm<br>NbO at 120<br>°C at 8<br>× 10−9 Torr<br>∼ 1 nm NbO under 120<br>°C 25 mTorr N2<br>"infusion"<br>0.5 nm Nb2O5<br>/ 0.2 nm NbO2<br>/ ∼ 1 nm<br>NbO at 200<br>°C at 8<br>× 10−9 Torr<br>0.2 nm NbO2<br>/ ∼ 1 nm NbO at 250<br>°C<br>at 8<br>× 10−9 Torr<br>0.2 nm NbO2<br>/ ∼ 1 nm NbO at 250<br>°C<br>25 mTorr N2<br>"infusion"<br>0.2 nm NbO2<br>/ ∼ 1 nm NbO / NbxNy<br>under 500<br>°C 25 mTorr N2<br>"infusion"<br>∼ 1 nm NbO at 800<br>°C at 8<br>× 10−8 Torr | –                                                                                               | –                                                                  | XRR; X-ray dif<br>fuse scattering;<br>XPS             | [13]                                           |
| 1 nm Nb2O5<br>+ Nb / 2 nm Nb2O5<br>+ NbO2<br>+ NbO + Nb / 2 nm Nb2O5<br>+ NbO2<br>+<br>NbO + NbOx<br>+ Nb / 1 nm NbO2<br>+ NbO<br>+ NbOx<br>+ Nb / 1 nm NbO + NbOx<br>+ Nb<br>+ NbC | 1 nm Nb2O5<br>+ NbO2<br>+ Nb / 2 nm Nb2O5<br>+ NbO2<br>+ NbO + Nb / 2 nm NbO2<br>+<br>NbO + NbOx+ Nb / 2 nm NbO + NbOx<br>+ Nb + NbC<br>at<br>200°C<br>at<br>(0.7 –<br>2) × 10−10 Torr<br>No higher-order oxides<br>in situ<br>at 500<br>°C<br>+ NbO2<br>+ NbOx<br>+ Nb + NbC /                                                                                                                                                                                                                           | NbC underneath<br>the oxides<br>in situ<br>at 200°C<br>70% – 80% NbC<br>in situ<br>at 500<br>°C | > 7 nm O & C at<br>200°C at UHV<br>> 5 nm O & C at<br>500°C at UHV | Sputter-assisted<br>ARXPS;<br>XPS;<br>STEM/EELS       | Nb<br>EP'ed<br>(this work)                     |
| 1 nm Nb2O5<br>+ NbOx<br>+ Nb + NbC / 1 nm Nb2O5<br>6 nm NbO + NbOx<br>+ Nb + NbC<br>in situ<br>at 120                                                                               | Up to 60% NbC<br>in situ<br>at 120<br>°C,<br>300 °C, 400<br>°C, &<br>800 °C                                                                                                                                                                                                                                                                                                                                                                                                                               | > 40 nm<br>C<br>&<br>> 50 nm<br>O<br>at<br>300<br>120 °C,<br>°C,<br>400 °C, & 800<br>°C         | Sputter-assisted<br>XPS                                            | 500°C pre<br>Nb<br>baked<br>(this work)               |                                                |
| 1 nm Nb2O5<br>+ Nb + NbC                                                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Up to 50% NbC                                                                                   | > 30 nm C & O                                                      | Sputter-assisted<br>XPS                               | 800°C pre<br>Nb<br>baked<br>(this work)        |
| 1 nm Nb2O5<br>+ Nb + NbC / 1 nm Nb2O5<br>NbO2 + NbOx<br>+ Nb + NbC / 1 nm NbO2<br>+ NbOx<br>+ Nb + NbC<br>(embedded<br>ni<br>tride nano-crystals)                                   | +                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Up to 60% NbC                                                                                   | > 40 nm C & O                                                      | Sputter-assisted<br>STEM<br>XPS;<br>/EELS             | 800°C N2-<br>processed<br>(this<br>Nb<br>work) |
| 1 nm Nb2O5<br>+ Nb + NbC                                                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Up to 20% NbC                                                                                   | > 20 nm C & O                                                      | Sputter-assisted<br>STEM<br>XPS;<br>/EELS             | N2-<br>processed<br>EP'ed<br>+<br>(this work)  |

<span id="page-3-0"></span>Table 1. Native surface oxides on Nb, thermal decomposition of oxides, bake-induced carbides, and subsurface impurities.

\*XPS: X-ray photoelectron spectroscopy; ARXPS: angle-resolved XPS; AES: Auger electron spectroscopy; XRR: X-ray reflectivity; STEM/EELS: scanning transmission electron microscopy with electron energy loss spectroscopy; UHV: ultra-high vacuum.

results have been widely reported [\[11,](#page-22-10) [42,](#page-23-10) [46,](#page-23-14) [44,](#page-23-12) [17\]](#page-22-16). Confusion abounds concerning the extremely low nitrogen concentration (∼ 0.01 at.%) in "N2-infused" Nb, in contrast to 1 – 3 orders of magnitude higher oxygen and carbon concentrations (figure S1) [\[46\]](#page-23-14). Similarly, "N2-doped" Nb, which requires post-electropolishing of the µm-thick surface, contains < (0.1 – 1) at.% nitrogen and absolutely higher oxygen and carbon concentrations. Therefore, this work also aims to untangle the mystery of subsurface impurities in Nb in terms of their concentrations, structures, and valence distributions.

Combining the implications of surface oxides and subsurface impurities, we extend the understanding of a desirable SRF surface that implies collective contributions. Previous studies have reported on the thinned oxide layer of N2-processed Nb [\[13,](#page-22-12) [4\]](#page-22-3) and the oxygen dissolution in UHV-baked Nb [\[32\]](#page-23-0). Here, we expand on the efforts by considering multiple impurities and deconvoluting their multiplet structures, including carbides and suboxides. We provide a comprehensive understanding of structural identification and impurity bonding. To advance accuracy and throughput, our considerations span several aspects:

- Sputter-assisted XPS and cross-sectional STEM/EELS are combined to calibrate depth-resolved chemical states;
- Characteristics of XPS chemical states are identified in >500 in situ samples using GENPLOT semi-automatic search algorithms to overcome the ambiguity of binding energy references;
- Empirical electronic structures (valence band electron distribution) are analyzed using XPS valence spectra fingerprints in the impurity region where hybridization is distinguishable [\[47\]](#page-23-15);
- Results are correlated between in situ measurements during continuous heating at elevated temperatures and in situ measurements of individual conditions (no heating history) versus coupon samples;
- Characterization is extended to Nb3Sn, and the different oxide structures on vapordiffused and electrochemically synthesized samples are highlighted. [\[17\]](#page-22-16).

## 2. Experimental Section

## 2.1. Nb and Nb3Sn samples and treatments

High-purity (> 300 residual-resistivity ratio), large-grain (> 50 mm), 3 mm thick Nb plates were used. The Nb samples, unless otherwise specified, were polished following standard SRF protocols: first, buffered chemical polishing (BCP) using 1:1:1 hydrofluoric (HF, 48%), nitric (70%), and phosphoric (85%) acids; second, 100 µm electropolishing (EP) using 1:9 HF (48%) and sulfuric (98%, H2SO4) acids. The average surface roughness (Ra) of the EP'ed Nb was 40 nm.

As summarized in table [2,](#page-5-0) treatments of interest include (i) acid-related treatments such as BCP, EP, and 2% HF soak; (ii) heat-related treatments such as 800 °C UHV (10<sup>−</sup><sup>10</sup> torr) baking, "N<sup>2</sup> doping", and "N<sup>2</sup> infusion"; and (iii) environment-related treatments such as modulated 20%:80% O<sup>2</sup> / N<sup>2</sup> soak, ozone processing (Jelight 144AX),

| Sample #    | SMP1             | SMP1           | SMP1          | SMP1          | SMP2          | SMP3           | SMP3        |
|-------------|------------------|----------------|---------------|---------------|---------------|----------------|-------------|
| Step #      | STP1             | STP2           | STP3          | STP4          | –             | STP1           | STP2        |
| Treatment   | BCP              | HF<br>soak     | BCP<br>2nd    | ex<br>Ozone   | 100 µm EP     | 100<br>µm EP   | RT O2/N2    |
|             |                  | 30 min         | 30 min        | posure 3 d    |               |                | flow > 1 d  |
| Pre         | N/A              | N/A            | N/A           | N/A           | BCP           | No<br>prior    | N/A         |
| treatment   |                  |                |               |               |               | BCP            |             |
| air<br>Post | > 6 mo           | < 3 d          | < 3 d         | < 3 d         | < 3 d         | < 3 d          | < 3 d       |
| exposure    |                  |                |               |               |               |                |             |
| Sample #    | SMP4             | SMP5           | SMP6          | SMP6          | SMP7          | SMP8           | SMP9        |
| Step #      | –                | –              | STP1          | STP2          | –             | –              | –           |
| Treatment   | 800<br>5 h<br>°C | "2/800"        | 100 µm EP     | "2/800"       | 100 µm EP     | N2<br>160 °C   | Anodization |
|             | +<br>baking      | N2<br>doping   | RT<br>N2<br>+ | N2<br>doping  | RT<br>N2<br>+ | infusion       |             |
|             | RT O2/N2         | 5<br>+<br>µm   | storage       |               | storage       | 96 h           |             |
|             | flow > 1 d       | light EP       |               |               |               |                |             |
| Pre         | BCP + EP         | BCP + EP       | prior<br>No   | 5 h<br>800 °C | N/A           | BCP + EP       | EP          |
| treatment   |                  | 800<br>+<br>°C | BCP           | baking        |               | 800<br>+<br>°C |             |
|             |                  | 5 h baking     |               |               |               | 5 h baking     |             |
| air<br>Post | > 6 mo           | > 6 mo         | < 3 d         | < 3 d         | < 3 d         | > 6 mo         | < 3 d       |
| exposure    |                  |                |               |               |               |                |             |
|             |                  |                |               |               |               |                |             |

<span id="page-5-0"></span>Table 2. Summary of acid/polishing, ozone, anodization, and heat treatments on Nb coupons.

and exposure under ambient conditions for varying periods. The typical "2/800 N<sup>2</sup> doping" recipe involved exposing the Nb to 45 mTorr N<sup>2</sup> at 800 °C for 2 min, followed by a 5 µm EP removal [\[12,](#page-22-11) [42\]](#page-23-10). Except for ozone exposure and anodization, these treatments adhered to the typical procedures employed in Nb cavity preparation [\[46\]](#page-23-14). The reactive ozone treatment was designed to modify surface oxides, while anodization is a typical treatment of Nb substrates utilized prior to the Sn-vapor diffusion process to produce Nb3Sn.

Approximately 2 µm thick Nb3Sn films were produced using two methods: (i) Sn-vapor diffusion [\[16\]](#page-22-15) and (ii) a recently developed process that involves combining electrochemical Sn deposits with thermal conversion to a smoother Nb3Sn [\[17\]](#page-22-16). The average surface roughnesses (Ra) for these two methods were 300 nm and 40 nm, respectively.

After undergoing treatments, the samples were stored in a nitrogen-flowed glovebox and transported in cleanroom plastic bags before any characterization.

## 2.2. XPS and STEM/EELS characterizations

Quantitative determination of chemical states, elemental compositions, and valence distributions was primarily achieved through XPS depth profiling, supported by crosssectional STEM/EELS.

Three XPS systems were employed for this study, namely a PHI Versaprobe III, a Surface Science Instruments SSX-100 ESCA Spectrometer, and a Thermo Fisher Nexsa G2, with binding energy calibrations cross-referenced using EP'ed Nb. Comparing the monoatomic argon and cluster ion sputtering results from Nexsa (figure S2(a)), we found mono-sputtering provided effective surface removal and minimal oxide residuals. The non-destructive ARXPS results (figure S2(b)) were consistent with the structural evolution of the oxide layers as resolved by sputter-assisted XPS. Although preferential sputtering, atomic mixing, and the effect of roughness can occur [\[48\]](#page-23-16), sputter-assisted XPS allows probing the impurity region, sitting as deep as tens of nanometers (beyond the < 5 nm capability of ARXPS). Also, understanding chemical bonding (XPS peak position) is the priority over other tasks in this work, so mono-sputter-assisted depth profiling was primarily used.

2.2.1. In situ XPS with heating stage and reaction cells To capture critical changes during UHV and N<sup>2</sup> bakings, as well as after subsequent air exposure, in situ investigations were performed in three sets using the PHI Versaprobe XPS instrument with a heating stage. First, EP'ed Nb was repeatedly heated at 200 °C, 500 °C, 120 °C, 300 °C, 400 °C, and 800 °C at (0.7 – 2) × 10<sup>−</sup><sup>10</sup> Torr UHV for 30 min, with intentional ambient air exposure for 10 – 24 h between measurements to imitate SRF cavity treatments. Depth profiles were collected in situ during heating at the same temperature as the 30 min pre-heating, except for 800 °C processing. For this, an adjacent UHV reaction chamber (8 × 10<sup>−</sup><sup>9</sup> Torr) was required. The sample processed at 800 °C was transferred directly into the analysis chamber, without exposure to the atmosphere, under 8 × 10<sup>−</sup><sup>10</sup> Torr and measured at 500 °C, the highest temperature available in the XPS analysis chamber. Additionally, the 500 °C data were collected several times to confirm the observation of high carbon concentrations induced, which increased the number of sample heating cycles. Second, fresh-EP'ed Nb was processed to resemble the 800 °C "N<sup>2</sup> doping" procedures, which involved subjecting the Nb to 800 °C N<sup>2</sup> processing at 1 mTorr for 90 min (the same amount of N<sup>2</sup> exposure as "N<sup>2</sup> doping"), followed by re-exposure to air and a final 5 µm EP, with depth profiling conducted between each procedure. Third, a reference 800 °C UHV baked sample was examined immediately after baking and after air exposure, which provided comparisons with "N<sup>2</sup> doping" and avoided effects induced by the heating history. The samples were tightly mounted on the heating stage, and the stage temperature was calibrated and showed negligible variation (typically < +/− 1 degree).

High-resolution spectra of Nb 3d, Nb 3p, O 1s, C 1s, and valence photoelectrons were probed using a 100 µm monochromatic Al k-alpha X-ray (1486.6 eV) beam. After optimization (figure S3), the scan parameters were set to 45° emission angle, 26 eV pass energy, 50 ms/step, and up to 60 sweeps with the dual-neutralization on, resulting in a 0.3 –0.6 eV FWHM resolution for Nb subpeaks. For depth profiles, a 3 keV Ar<sup>+</sup> beam was rastered over 2 × 2 mm<sup>2</sup> area with Zalar rotation. The sputtering rate of 1.6 ˚A/s was determined using a SiO<sup>2</sup> standard and compared with the cross-sectional STEM result. Depth profiles were collected with 6 s intervals at the surface region and 12 – 60 s intervals toward the bulk.

Vapor-diffused Nb3Sn was in situ measured at RT, 200 °C, and 500 °C under

continuous heating, without exposure to air, using the same parameters as for Nb measurements.

2.2.2. Coupon inspections To obtain depth profiles of oxides on the coupons, we conducted survey scans of representative Nb samples listed in table [2,](#page-5-0) as well as two types of Nb3Sn samples, using the SSX-100 XPS instrument. Monochromatic Al kalpha X-ray (1486.6 eV) photoelectrons were collected under a 10<sup>−</sup><sup>9</sup> Torr vacuum from an 800 µm analysis spot with a 55° emission angle. The scan parameters were set to 150 eV pass energy, 1 eV step size, and 100 s/step. For the depth profile, a 4 kV Ar<sup>+</sup> beam with a spot size of ∼ 5 mm was rastered over a 2 × 4 mm<sup>2</sup> area.

We used an FEI/Thermo Fisher Titan Themis STEM with beam energy up to 300 kV and resolution down to 0.08 nm to image the surface cross-sections of EP'ed and "N<sup>2</sup> doped" Nb and vapor-diffused Nb3Sn. The equipped EELS was utilized to analyze the chemical states as a function of depth at local regions. The cross-sectional specimens were prepared using a Thermo Fisher Helios G4 UX focused ion beam.

## 3. Results and Discussion

## 3.1. Native oxides on Nb at RT

The cross-sectional STEM image (figure [1\(](#page-8-0)a)) reveals an amorphous oxide region of ∼ 7 nm on the surface of EP'ed Nb (measured at RT). Due to the limitations of each technique, structural deconvolution requires the combination and correlation of findings from EELS, ARXPS, and sputter-assisted XPS. The structure consists of Nb2O<sup>5</sup> mixed with metallic Nb at the outermost layer, along with hydroxide and organics adsorbed. The next layer, which is 4 – 5 nm thick, comprises a mixture of Nb2O5, NbO2, NbO, NbO<sup>x</sup> (suboxide), and Nb. The following layer, which is 1 – 2 nm thick, is a mixture of NbO, NbOx, and Nb, with NbO<sup>x</sup> continuing and NbC appearing deeper. The evidence of these identifications is detailed below.

EELS spectra (figure [1\(](#page-8-0)b)), taken at the locations indicated in figure [1\(](#page-8-0)a), reveal two distinct regions with thicknesses of ∼ 5 nm and ∼ 2 nm, respectively. Figure S4 summarizes the EELS fingerprints collected from Nb2O5, NbO2, and NbO in the literature [\[49\]](#page-23-17), showing that the Nb2O<sup>5</sup> and NbO<sup>2</sup> features have close energy states while NbO exhibits distinguishable energies near 535 eV and 548 eV. Based on this, the EELS data suggests that the first 5 nm of the oxide region is mixed with Nb2O5/NbO<sup>2</sup> and NbO, while the second 2 nm is primarily NbO. This local probe substantiates the mixing composition within each layer of surface oxides.

The XPS concentration profile (figure [1\(](#page-8-0)c)) also shows an oxide region of ∼ 7 nm in thickness, followed by deeper oxygen and carbon impurities. Additionally, contamination was observed at the outermost surface of all coupon samples studied, likely due to methanol cleaning. The deconvoluted structure of the hydroxyl, methoxy, carbonyl, and carboxyl dangling bonds is depicted in figure S5.

ARXPS measurements, taken at a 15°take-off angle (figure [1\(](#page-8-0)d)), strongly indicate the presence of metallic Nb mixed within Nb2O5, as previously observed in a 2° grazing angle measurement [\[31\]](#page-22-30). Deeper probing at higher angles between 45° and 90° detected the NbO<sup>2</sup> and NbO signals. This near-grazing measurement supports the observation of metallic Nb within the oxide layers in sputter-assisted XPS.

The sputter-assisted XPS spectra in figure [1\(](#page-8-0)e) clearly show the evolution of structural changes as a function of depth. To deconvolute this complex system with diverging references, we utilized GENPLOT semi-auto searching in vast amounts of data to elucidate the binding energies of all components present on the Nb surface, as summarized in figure [2.](#page-9-0) The identification is further corroborated by concurrent features observed in the valence spectra, such as C 2s (figure [1\(](#page-8-0)f)). We have identified Nb at 202.2 ± 0.1 eV, NbC at 203 ± 0.1 eV, NbO<sup>x</sup> at 203.8 ± 0.1 eV, NbO at 204.5 ± 0.1 eV, NbO<sup>2</sup> at 205.8 ± 0.1 eV, and Nb2O<sup>5</sup> at 207.5 ± 0.1 eV. All subpeaks, except for NbOx, exhibit their characteristic peaks, as exemplified in figures [2\(](#page-9-0)b)–(d); the incorporation of NbO<sup>x</sup> is nevertheless necessary to fit the spectra in 375 samples. With the support of EELS and ARXPS analyses, we can trust the validity of these observations, although sputtering effects may slightly influence the results.

![](_page_8_Figure_2.jpeg)

**Caption:** Shows structural details of native oxides on EP'ed Nb at room temperature captured using cross-sectional STEM, EELS, XPS, and ARXPS. The figure illustrates a layered oxide structure topped with Nb2O5/NbO2 before transitioning to NbO. These multilayer oxides are crucial for understanding and adjusting surface properties in superconducting Nb materials .


Intensity

[arb. unit]

<span id="page-8-0"></span>Figure 1. Native oxides on EP'ed Nb at RT: (a) Cross-sectional STEM image showing surface oxides on Nb. (b) EELS O-K edge spectra at various depths as labeled in (a). (c) Concentration depth profiles of Nb, O, and C collected by XPS at RT (and during in situ heating at 200°C and 500°C). (d) ARXPS spectra collected at take-off angles of 15°, 45°, and 90°. (e) Nb 3d and (f) valence XPS spectra as a function of depth. The intensity units in (b) and (d – f) are arbitrary.
![](0__page_9_Figure_0.jpeg)

**Caption:** This figure shows the XPS binding energy analysis of Nb in various chemical states, including metallic Nb, carbide, suboxide, monoxide, dioxide, and pentoxide. The binding energies are presented as points with colors corresponding to their chemical identification. Panels (b), (c), and (d) display representative XPS spectra at room temperature, after sputtering 3 nm, and at 500 °C, respectively, highlighting the characteristic peaks used for peak fitting. The intensity units are arbitrary, emphasizing the qualitative rather than quantitative analysis. This analysis is crucial for understanding the surface chemistry of Nb, which is pivotal for optimizing its superconducting properties .


<span id="page-9-0"></span>Figure 2. Nb 3d XPS binding energy analysis: (a) Binding energies of Nb in different chemical states (metallic Nb, carbide, suboxide, monoxide, dioxide, and pentoxide) determined using (i) this work, (ii) the NIST reference database [\[50\]](#page-23-0) and (iii) EELS chemical shifts [\[49\]](#page-23-1). (b,c,d) Representative XPS spectra of EP'ed Nb showing characteristic peaks used for peak fitting: (b) at RT without sputtering, (c) at RT after sputtering 3 nm, and (d) at 500 °C without sputtering. In (b – d), the labeled circles are color-coded to correspond with the identifications in (a), and the intensity units are arbitrary.

We emphasize that metallic Nb coexists with amorphous oxides exhibiting a mixing feature, likely involving non-stoichiometry and defective energy states near the Fermi level. Both findings are consistent with the observation of electron populations at the Fermi level in the valence spectra of surface layers, while bulk signals contribute to the Fermi electrons at deeper layers, as shown in figure [1\(](#page-8-0)f). Our observations strongly suggest that the entire ∼ 7 nm surface oxide region is conductive, most likely normal-conducting, and plays a crucial role in determining the surface properties of superconducting Nb. Further identification of sub-nanometer local amorphous structures is helpful to confirm this finding.

## 3.2. Effects of UHV baking temperature on Nb

In situ XPS measurements were taken on Nb at various temperatures, with deliberate air exposure between each measurement. A high concentration of carbon appeared in the sample at 500°C (figure [1\(](#page-8-0)c)), which crucially affected the subsequent oxidization and impurity bonding. Thus, we divide the results into two groups: (i) the effects of UHV baking temperature on EP'ed Nb, with a heating history of RT, 200°C, and 500°C, and (ii) the effects of UHV baking temperature on high-temperature baked Nb (close to the 800°C "outgassing" baking condition), with a history of 500°C, 120°C, 300°C, 400°C, and 800°C. The complete Nb 3d and valence spectra are included in figures S6 and S7, respectively.

3.2.1. Effects of UHV baking temperature on EP'ed Nb Figure [3](#page-10-0) presents a map of the surface oxide and carbide motifs for all baking conditions. When heated at 200°C, the oxides undergo slight reconstruction and thickness reduction, while the layout remains similar to RT, which is consistent with the decomposition threshold at ∼ 250 – 300 °C

![](0__page_10_Figure_0.jpeg)

**Caption:** This figure illustrates structural profiles of UHV-baked Nb, showing the depth-resolved proportion of Nb motifs from XPS peak fitting of Nb 3d spectra across various temperatures: room temperature, 200°C, 500°C, 120°C, 300°C, 400°C, and 800°C. The results reveal significant transformation at 500°C with carbide dominance, important for controlling oxidation and surface chemistry during Nb processing.


<span id="page-10-0"></span>Figure 3. Structural profiles of UHV-baked Nb: Comparative proportion of different Nb motifs resolved by XPS peak fitting of Nb 3d spectra, plotted as a function of depth in nm. The spectra were taken in situ at (a) RT, (b) 200°C, (c) 500°C, (d) 120°C, (e) 300°C, (f) 400°C, and (g) 800°C, in the indicated sequence, with air exposure between measurements. A reference sample was (h) baked directly at 800°C and then (i) exposed to air. The expected fitting residue is between 5% – 10 %.

reported in the literature [\[29,](#page-22-0) [4,](#page-22-1) [32\]](#page-23-2). However, upon heating at 500°C, nearly 80% of the surface structures convert to carbides, with the higher-order oxides disappearing. The binding energy of the main peak in the spectra shifts to 203 eV (figure [2\(](#page-9-0)c)). These carbides significantly alter the subsequent oxidization process and the diffusion of impurities into the bulk upon re-exposure to air.

3.2.2. Effects of UHV baking temperature on high-temperature baked Nb Figures [4\(](#page-11-0)a) and (b) illustrate the changes in carbon and oxygen concentrations with baking temperature after the 500°C pre-baking. A thick impurity region is observed with a carbon thickness of 20 – 40 nm and an oxygen thickness exceeding 40 nm (using a conservative residue signal threshold of 5 at.%). We argue that the influence from chamber gas residues is negligible since nitrogen consistently remains below the detection limit (< 1000 ppm), as measured by residual gas analysis. We also argue that the sputtering effect is minor since the RT data is consistent with the results of STEM imaging. However, we find that the heating history affects the impurity profile. For instance, directly heating the EP'ed sample to 800°C results in a carbon thickness of 20 nm and an oxygen thickness of ∼ 30 nm.

![](0__page_11_Figure_0.jpeg)

**Caption:** This figure presents depth-dependent concentration profiles of carbon and oxygen in UHV-baked Nb samples, analyzing variations at temperatures from 500°C to 800°C, with focus on depths at 10 nm, 20 nm, and 40 nm. The findings highlight the alterations in impurity profiles due to temperature, illustrating effects like oxygen uptake and carbide-induced passivation, which are important for improving SRF cavity performance .


<span id="page-11-0"></span>Figure 4. Composition profiles of UHV-baked Nb: (a) Carbon and (b) oxygen concentrations as a function of depth measured in situ at different temperatures, following a heating history of 500°C, 120°C, 300°C, 400°C, and 800°C. (c) Carbon and (d) oxygen concentrations at selective depths as a function of baking temperature. The starting RT reference for this experiment was an EP'ed sample.

The comparison between the 200°C (figures [3\(](#page-10-0)b) and S6(b)) and 120°C (figures [3\(](#page-10-0)d) and S6(d)) data, both temperatures of which are below the decomposition threshold, implies that the initial surface plays a critical role in oxidation. The 200°C oxides form on an EP'ed surface, whereas the 120°C oxides grow on a surface populated with carbides from a 500°C pre-baking. The presence of carbides essentially minimizes the generation of higher-order oxides. The transition metal (e.g., Nb) favors dissociation into chemisorbed oxygen, and any foreign phases interrupt the self-passivating oxides, which allows for oxygen (and carbon) dissolution into the subsurface [\[51\]](#page-23-3). Considering the low equilibrium O and C solubilities in Nb (<< 1 at.% even at 800°C [\[52,](#page-23-4) [53\]](#page-23-5)), a pertinent question is regarding the oversaturation of oxygen and carbon in the subsurface.

Although we adopt the buried oxide (or precipitate) hypothesis and treat them as carbides and suboxides in this work, there are arguments regarding the accurate picture [\[51\]](#page-23-3). Our other work observed the existence of carbides in the subsurface [\[54\]](#page-23-6), while our valence analyses (below) suggest that these subsurface impurities are not likely to form a specific phase, especially for the suboxide. Nevertheless, our data indicate that the UHV baking mechanism in the SRF field mainly takes advantage of preceding carbide formation during the high-temperature treatment, e.g., 800 °C outgassing. This process surpasses the limiting rate during impurity dissolution, resulting in a mixture of carbides and suboxides, in addition to varying oxide thickness.

Upon heating to 300°C, oxygen accumulates on the surface over carbon in concentration (figures [4\(](#page-11-0)c) and (d)), which matches the onset of native oxide decomposition. However, carbides continue to dominate over suboxides (figure [3\(](#page-10-0)e)). Baking at 400°C yields similar composition and motif proportion profiles compared to baking at 300°C, while carbide formation is intensified at the relatively surface region with a 60% proportion compared to the bulk region with a 20% proportion, as shown in figure [3\(](#page-10-0)f). At 800°C (figure [4\(](#page-11-0)d)), there is a surface depletion of oxygen, with the proportion of pure Nb increasing to above 30% at the 20 nm surface in addition to 50% carbides (figure [3\(](#page-10-0)g)). For reference, the sample that was directly heated to 800°C (figure [3\(](#page-10-0)h)) has 30% – 40% carbides and 60% –70 % pure Nb on the surface with minimal suboxides. However, air exposure of this sample (figure [3\(](#page-10-0)i)) immediately introduces higher numbers of carbides and suboxides within 20 nm, along with a one-nanometer-thin higher-order oxide region, demonstrating the higher-order oxide formation limitation and impurity uptake due to carbide formation.

Additionally, upon inspecting the O 1s spectra (e.g., figure S8), we identify two types of oxygen-related motifs, assigned to O<sup>2</sup><sup>−</sup> and O<sup>−</sup>. As illustrated in figure S9, the change in the relative ratio of O<sup>−</sup> likely correlates with the carbon concentration as a function of depth, indicating a possible interaction between carbon and certain oxygen species.

![](0__page_12_Figure_3.jpeg)

**Caption:** Depicts the depth profiles of carbon, oxygen, and nitrogen in N2-processed Nb, revealing minimal detectable nitrogen despite a significant nitride surface grain presence. This finding emphasizes challenges in N2 doping attempts and highlights the impact of such processing on surface chemistry.


<span id="page-12-0"></span>Figure 5. Composition profiles of N2-processed Nb: Depth profiles of (a) carbon, (b) oxygen, and (c) nitrogen concentrations for different processing conditions: (i) after pre-EP, (ii) during in situ N<sup>2</sup> processing at 800°C ("N<sup>2</sup> doping"), (iii) after air exposure, and (iv) after re-EP (light). A reference sample was subjected to (v) UHV baking at 800°C and then (vi) air exposure. (d) Cross-sectional phase-contrast STEM image of the N2-processed coupon (condition iii), showing surface nitride nano-grains (labeled "3" and "4") and an oxide layer (labeled "0").

## 3.3. Effects of N<sup>2</sup> processing on Nb

Two sets of EP'ed Nb were processed: one at 800°C with 5.4 × 10<sup>6</sup> langmuir N<sup>2</sup> exposure to simulate "N<sup>2</sup> doping", and the other under UHV as a reference. The complete Nb 3d and valence spectra are included in figures S10 and S11, respectively.

Figure [5](#page-12-0) shows that both in situ conditions introduce significant amounts of carbon and oxygen, but nitrogen is below the detection limit in all samples studied (< 1000 ppm), even for the "N<sup>2</sup> doping" condition, despite the observation of two nitride nano-grains during STEM inspection. Structural deconvolution reveals that the N<sup>2</sup> processed sample (figure [6\(](#page-13-0)b)) contains slightly higher levels of carbides and suboxides than the UHV baked sample (figure [6\(](#page-13-0)e)). Upon air exposure (figure [6\(](#page-13-0)c)), the suboxide layer on the N2-processed sample surface substantially increases, along with more higherorder oxides. In contrast, the UHV baking reference (figure [6\(](#page-13-0)f)) only exhibits a 3 nm suboxide layer with proportions exceeding 10% and 1 nm with higher-order oxides. These findings, together with the UHV baking data, prove that the thickness of the suboxide layer, grown on surfaces populated with carbides, is proportional to the ratio of carbides, as illustrated in figure [7.](#page-14-0) Specifically, an initial surface with a higher number of carbides results in a larger amount of suboxide growth. The suboxide layer profile, including its peak concentration and location, determines the type and thickness of higher-order oxides formed on top of the suboxide layer.

To achieve high-performance SRF cavities, a 5 µm (light) EP treatment is commonly employed after N<sup>2</sup> processing to remove any nitrides. We have examined the surface profile following air exposure of the EP'ed, N2-processed surface. Our study shows that this surface (figure [6\(](#page-13-0)d)) contains the lowest quantities of carbides and suboxides,

![](0__page_13_Figure_4.jpeg)

**Caption:** The figure illustrates structural profiles of N2-processed Nb coupons, detailing depth variations post pre-EP, in situ N2 processing, air exposure, and re-EP treatments. This figure emphasizes the growth of suboxide layers and alteration in impurity compositions, crucial for tailoring Nb surface for superconducting applications. The comparison between N2 and UHV treatments provides insight into the effects of nitrogen exposure on surface chemistry .


<span id="page-13-0"></span>Figure 6. Structural profiles of N2-processed Nb: Comparative proportion of different Nb motifs as a function of depth in nm. The spectra were taken (a) after pre-EP, (b) during in situ N<sup>2</sup> processing at 800°C ("N<sup>2</sup> doping"), (c) after air exposure, and (d) after re-EP (light) treatments. A reference sample was subjected to (e) UHV baking at 800°C and then (f) air exposure. The expected fitting residue is between 5% – 10 %.

![](0__page_14_Figure_0.jpeg)

**Caption:** The graph details the correlation between suboxide layer thickness and the surface carbide ratio, derived through various thermo-chemical processes including UHV baking and N2 processing. This correlation is significant in optimizing the surface layers of Nb for enhanced superconducting performance by managing the underlying carbide content .


<span id="page-14-0"></span>Figure 7. Effect of second-phase formation: The correlation between the thickness of suboxide layers and the varying ratio of carbides on the surface where suboxides are formed. The data was obtained through in situ UHV baking (history: 500°C, 120°C, 300°C, 400°C, and 800°C), in situ direct 800°C UHV baking and N<sup>2</sup> processing, and their subsequent air exposure.

alongside the thinnest higher-order oxides. Notably, variations in the oxide profile exist between EP'ed, N2-processed Nb and EP'ed Nb. While the electrochemical mechanism is beyond the scope of this study, our focus centers on valence analyses to demonstrate the modification of electron populations in orbitals resulting from these treatments.

## 3.4. Valence analyses

The valence bonding in carbides and oxides typically involves hybridization, which causes overlapping energy levels in momentum space. This makes it challenging to distinguish the contribution of electronic density of states in each orbital to the overall electronic structure. First-principles calculations [\[55,](#page-23-7) [56\]](#page-23-8) have reported that higherorder Nb oxides exhibit strong hybridization (as observed in figure [1\(](#page-8-0)f)). Conversely, lower-order Nb oxides and carbides show weaker hybridization and more distinguishable orbital occupation [\[55,](#page-23-7) [57\]](#page-23-9).

When the transition metal d-orbitals are partially filled with low numbers of electrons (< 3), these orbitals and the O or C orbitals are roughly separated[\[58,](#page-23-10) [59\]](#page-23-11). XPS studies (figure S12) have revealed that pure Nb displays three characteristic peaks at 0 – 3 eV, with the majority contribution coming from the Nb 4d orbital [\[60,](#page-23-12) [61,](#page-23-13) [62\]](#page-23-14). Exposure to oxygen results in a peak near 6.5 eV, mainly from O 2p [\[33\]](#page-23-15). NbC shows two peaks: one near 4.4 eV, mainly from C 2p, and another near 11 eV from C 2s [\[63,](#page-23-16) [61,](#page-23-13) [57\]](#page-23-9).

Figures [8](#page-15-0) and [9](#page-15-1) demonstrate that we repeatedly and clearly identify these characteristic peaks in over 500 valence spectra, except for those with higher-order oxides. To deconvolute the spectra, we perform peak fitting using 7 Gaussian peaks, as exemplified in figure [8\(](#page-15-0)b). The sub-peak denoted "i" near the Fermi edge may include surface states and a large contribution from Nb 5s (and possibly 5p if hybridized).

![](0__page_15_Figure_0.jpeg)

**Caption:** The figure displays valence spectral analysis in UHV-baked Nb across different temperatures, depicting in situ spectra at 0 nm (panel a-1) and 9 nm (panel a-2). It also shows peak fitting at 1 nm during 800°C baking (panel b-1) and spectra comparison at 55 nm (panel b-2). The binding energy varies with temperature, showing significant spectral shifts related to structural changes, crucial for understanding Nb's surface chemistry under thermal treatment .


Intensity

Y

Axis

[AU]

<span id="page-15-0"></span>Figure 8. Valence spectral analysis in UHV-baked Nb: (a) Comparison of valence spectra taken in situ at different temperatures, following the sample's heating history of RT, 200°C, 500°C, 120°C, 300°C, 400°C, and 800°C, at depths of (a-1) 0 nm and (a-2) 9 nm (or the maximum depth probed). (b-1) Example of valence peak fitting for spectra taken at 1 nm during in situ baking at 800°C, and (b-2) comparison with spectra taken at 55 nm, demonstrating characteristic features. The binding energy scales in (a-2) and (b-2) are equivalent to those in (a-1) and (b-1), respectively.

![](0__page_15_Figure_2.jpeg)

**Caption:** The valence spectral analysis in UHV-baked and N2-processed Nb shows electron population shifts across different processing steps, contributing to understanding electron distribution and its implications on material properties under varied thermal and chemical environments .


<span id="page-15-1"></span>Figure 9. Valence spectral analysis in N2-processed Nb: Comparison of valence spectra taken after pre-EP, during in situ N<sup>2</sup> processing at 800°C ("N<sup>2</sup> doping"), after air exposure, and after re-EP (light): (a) at 0 nm and (b) at 9 nm. A reference sample was baked at 800°C under UHV and then exposed to air. These results are re-plotted as figures [6\(](#page-13-0)e) and (f) to facilitate comparison. The intensity units in (a) and (b) are arbitrary.

Another small sub-peak denoted "j" is likely a hybridization product of Nb-O, as indicated by higher-order oxides. Consequently, these sub-peaks are not included in the quantitative calculations.

![](0__page_16_Figure_0.jpeg)

**Caption:** Illustrates electrons' depth-dependent population in UHV-baked and N2-processed Nb samples, exploring the intrinsic effects of baking temperature and processing methods on Nb, O, and C electrons, offering insight for optimizing superconducting niobium's microstructural features .


<span id="page-16-0"></span>Figure 10. Electron population analysis in UHV-baked and N2-processed Nb: Electron population as a function of depth in nm for Nb (solid line), O (dotted), and C (dashed) orbitals. (a) The in situ UHV baked sample, following a heating history of 500°C (black square), 120°C (red up-triangle), 300°C (blue down-triangle), 400°C (orange star), and 800°C (purple diamond). (b) The in situ 800°C N<sup>2</sup> processed ("N<sup>2</sup> doped") sample (red up-triangle), followed by air exposure (blue down-triangle) and light EP (orange star). A reference sample was directly baked at 800°C under UHV (purple diamond) and then exposed to air (green square).

We calculate the electron population at each bonding orbital through realistic normalization, such as in Nb (NNb), by [\[59,](#page-23-11) [64,](#page-23-17) [65\]](#page-24-0)

$$N\_{\rm Nb} = \frac{A\_{\rm Nb} \,/\, [\delta\_{\rm Nb} \, \lambda\_{\rm Nb} c\_{\rm Nb} (1 - f\_{\rm met.})]}{\sum\_{i} A\_{i} \,/\, (\delta\_{i} \, \lambda\_{i} c\_{i})} \times N\_{\rm t}. \tag{1}$$

In the case of a multicomponent system, this quantification is subject to several assumptions and simplifications. To specifically study the interaction between Nb and impurities, we first subtract the metallic Nb component (fmet.) using the Nb 3d data. Due to the presence of multiple impurities, we estimate the total number of electrons involved in bonding (Nt) and the number of bonds by using a weighted average based on impurity concentrations (ci). Since the homogeneity of the system is uncertain, we utilize matrix factor approximations. For carbon and oxygen, we assume the hybridization of 2s and 2p orbitals. We calculate the Gaussian peak area (Ai) and obtain the values for cross-section area (σi) and inelastic mean free path (λi) from literature [\[66,](#page-24-1) [67\]](#page-24-2). Lastly, molecular orbital theory is not involved. By following these methods, we can effectively compare samples processed by different heat treatments (although the absolute values should be refined). For instance, in figure [10,](#page-16-0) the electron populations observed at different depths after the 800°C UHV baking, regardless of the heating history, exhibit striking similarities, indicating the intrinsic effect on bonding of this heat treatment.

Figure [10\(](#page-16-0)a) demonstrates the versatile capability of suboxides to arrange the number of electrons in the presence of carbides. The 120°C low-temperature baking reveals the effect of air exposure on the carbide-populated surface, showing that oxygen captures more electrons and creates more ionic character, especially within the first 10 nm. In figure [10\(](#page-16-0)b), the air exposure of the N2-processed sample and its EP'ed sample exhibit similar behavior, with a more significant impact that results in nearly no electrons around the Nb orbital within the first 5 nm. The topic of ionic bonding is relevant to theories for high T<sup>c</sup> oxides and carbides [\[68\]](#page-24-3), but caution should be taken when interpreting it in the Nb system, where BCS theories apply.

Moreover, mid-temperature baking at 300°C generates a uniformly distributed region of high-electron-affinity oxygen, coinciding with the decomposition of higherorder oxides on the surface. However, this effect gradually diminishes as the baking temperature increases to 400°C and 800 °C, yielding a slope of oxygen's electron population as a function depth. In contrast, the N2-processed samples at 800°C, as well as those UHV-baked at 120°C and 300°C, induce more ionic character at the surface region or exhibit a uniform profile of high ionic character throughout the sample. Overall, the carbide-facilitated oxygen diffusion and surface higher-order oxide decomposition affect the electron distribution in the suboxides mixed within the superconducting Nb.

## 3.5. Nb coupon studies

Qualitative analyses are conducted on Nb coupons listed in table [2](#page-5-0) using Nb 3p and O 1s data collected with the SSX-100 XPS instrument. The concentration values obtained from high-intensity survey scans are reliable, while the Nb 3p spectra, which contain well-separated spin orbitals and better calibration in the SSX-100 XPS instrument, are

![](0__page_17_Figure_4.jpeg)

**Caption:** Oxygen concentration as a function of depth is analyzed on Nb coupons with differing treatments measured using the SSX-100 XPS. This includes EP'ed, BCP'ed, HF soaked, 800°C baked, and N2-processed samples. The data reveal treatment-specific oxidation characteristics, underpinning the necessity of precise surface manipulation for superconducting applications. The EP-treated samples show substantial high-order oxides, significant for enhancing cavity surface characteristics .


Figure 11. Coupon composition analysis: Oxygen concentration as a function of depth on Nb coupons measured with the SSX-100 XPS, compared with thicker lines representing in situ data taken with the PHI XPS. (a) EP'ed (light blue), (b) BCP'ed (dark green), (c) HF soaked (light green), (d) 800°C baked (purple), (e) "N<sup>2</sup> doped" and exposed to air (red), (f) "N<sup>2</sup> doped" and EP'ed (orange), (g) "N<sup>2</sup> infused" (pink), (h) modulated 20% O<sup>2</sup> / 80% N<sup>2</sup> flow for > 1 day (grey), (i) ozone treated for 3 days (black), and (j) electrochemically anodized (dark blue). The dashed lines indicate samples exposed to air for > 6 months.
used for comparison.

The oxygen concentration profiles in figure [11](#page-17-0) demonstrate that the Nb surface is highly sensitive to various acid-related and heat treatments, as well as environmental factors. The EP'ed samples (light blue) have been used as a calibration reference between different samples. We find a significantly higher quantity of higher-order oxides with greater depth in the EP'ed and N2-processed samples, regardless of whether they are N<sup>2</sup> "doped" (red) or "infused" (pink), which is consistent with our in situ investigations. In contrast, the BCP'ed (green), 800°C UHV baked (purple), and "N<sup>2</sup> doped" + EP'ed (orange) samples exhibit a thinner higher-order oxide layer on the surface.

After 6 months of exposure to ambient air, most samples develop a typical oxide stack that matches a reference oxide model derived from ozone treatment (black). However, the "N2-doped" + EP'ed sample (orange) does not show an increase in higherorder oxide thickness after the same exposure period. This suggests that passivating oxides are generated under this unique condition, and the EP mechanism for this type of sample requires further investigation.

Furthermore, figure [12](#page-18-0) presents the oxide structural differences imposed by

![](1__page_18_Figure_4.jpeg)

**Caption:** Depth-dependent XPS 3p3/2 spectra of Nb coupons showing varied responses based on acid/polishing and oxidizer exposure treatments. Subfigures (a) through (c) depict conditions of BCP’ed, EP’ed, and HF soaked samples stored in air for less than three days. Subfigures (d) through (f) show samples after prolonged exposure to oxidizers, illustrating distinct spectral characteristics for BCP’ed, EP'ed with O2/N2 flow, and ozone-treated samples. This depth analysis underscores the effect of environmental exposure on oxidation profiles, valuable for tailoring surface treatments in Nb superconducting applications .


<span id="page-18-0"></span>Figure 12. Coupon spectral analysis: Depth-dependent XPS 3p3/<sup>2</sup> spectra on Nb coupons measured at depths of 0 – 6.5 nm and 39 nm. Acid/polishing treatments: (a) BCP'ed, (b) EP'ed, and (c) HF soaked. These samples were stored in air for < 3 days before measurement. Effects of oxidizer exposure: (d) BCP'ed sample after > 6 months in air, (e) EP'ed sample exposed to a modulated 20% O<sup>2</sup> / 80% N<sup>2</sup> flow for > 1 day, (f) EP'ed sample exposed to ozone for 3 days. The binding energies of different oxide structures in the Nb 3p spectra are indicated in figure S13. The intensity units are arbitrary.

acid/polishing and oxidizer exposure factors, showing similar observations.

# 3.6. Native oxides on Nb3Sn at RT

Y

Axis

Characterizing surface oxides on Nb3Sn is crucial due to the current challenges faced by Nb3Sn cavities, which are hindered by unclear quench mechanisms [\[17,](#page-22-0) [16\]](#page-22-1). The theory surrounding how surface oxides influence Nb3Sn cavities is evolving [\[18,](#page-22-2) [19\]](#page-22-3). Here we report our findings on oxides in both vapor-diffused and electrochemically synthesized Nb3Sn, the only two types capable of producing high-performance SRF cavities to date. Our observations of distinct oxide structures and thicknesses provide atomic models for the development of emerging theories.

Figure [13\(](#page-19-0)a) displays a cross-sectional STEM image of vapor-diffused Nb3Sn, revealing an amorphous oxide region with a thickness exceeding 9 nm. EELS O-K edge spectra (figure [13\(](#page-19-0)b)), while slightly affected by the Sn-M edge, exhibit nearly identical features that indicate a mixed structure throughout the oxide region. XPS Nb 3d and 3p spectra in figure [13\(](#page-19-0)c) demonstrate a similar oxide evolution as a function of depth compared to Nb, but with an almost doubled thickness. Notably, these Nb oxides are

![](1__page_19_Figure_4.jpeg)

**Caption:** This figure depicts native oxides on Nb3Sn at room temperature, shown through cross-sectional STEM images and EELS O-K edge spectra. The XPS spectra for Nb 3d and Sn 3d reflect the evolution of surface oxides, crucial for understanding mechanisms that affect Nb3Sn superconducting properties and SRF cavity performance. The mixed oxide composition provides a basis for investigating quench mechanisms in Nb3Sn cavities .


<span id="page-19-0"></span>Figure 13. Native oxides on Nb3Sn at RT: (a) Cross-sectional STEM image showing surface oxides on Nb3Sn produced by vapor diffusion. (b) EELS O-K edge spectra, collected at various depths as indicated in (a). (c) Nb 3d and 3p spectra and (d) Sn 3d spectra as a function of depth, measured with the PHI XPS on vapor-diffused Nb3Sn. (Data measured with the SSX-100 XPS is included in figure S14). (e) Sn 3d and (f) Nb 3p spectra as a function of depth measured with the SSX-100 on electrochemical Nb3Sn. The intensity units in (b – f) are arbitrary.

![](1__page_20_Figure_0.jpeg)

**Caption:** Profiles concentration-depth of Nb, Sn, O, and C in vapor-diffused Nb3Sn during in situ UHV heating at 200°C and 500°C. This dataset aids in defining oxide characteristics responsible for affecting superconducting performance in Nb3Sn cavities, as these conditions are relevant for maintaining or improving SRF cavity efficiency .


<span id="page-20-0"></span>Figure 14. UHV baking in Nb3Sn oxides: (a) Concentration depth profiles of Nb, Sn, O, and C, collected by the PHI XPS on vapor-diffused Nb3Sn at RT and during in situ heating at 200°C and 500°C under UHV without exposure to air between measurements. (b,c) Nb 3d and Sn 3d spectra as a function of depth during in situ heating at (b) 200°C and (c) 500°C. The intensity units in (b) and (c) are arbitrary.

further mixed with either SnO<sup>2</sup> or SnO, which have close binding energies, as shown in figure [13\(](#page-19-0)d).

Particularly noteworthy are the strikingly different surface oxides present on the electrochemically synthesized Nb3Sn. As shown in figures [13\(](#page-19-0)e) and (f), this type of Nb3Sn only has a surface layer of < 2.6 nm SnO2/SnO, without any higher-order Nb oxides. The existence of suboxides needs further investigation.

# 3.7. Effects of UHV baking temperature on Nb3Sn

Upon in situ heating at 200°C (figure [14\(](#page-20-0)b)), slight reconstruction and decomposition occur in both Nb oxides and Sn oxides mixed on the surface. However, these metastable oxides undergo dramatic decomposition and transformation during in situ 500 °C baking. As shown in figure [14\(](#page-20-0)c), Sn oxides disappear, and Nb oxides begin to incorporate a significant quantity of oxides with all charges, resulting in wide spectra throughout the 8 nm probed. Additionally, up to 20 at.% carbon appears on the surface.

The valence spectra of these baking results in figure S15 indicate the conducting behavior, most likely normal-conducting, of this thick oxide region.

# 4. Conclusions

In summary, we investigated the effects of different baking and processing conditions, including UHV baking, N<sup>2</sup> processing, acid/polishing, and oxidizer exposure treatments, on the structures, compositions, depths, and valence distributions of surface oxides, carbides, and impurities on Nb and Nb3Sn.

Our findings revealed that these treatments significantly alter the type and thickness of Nb surface oxides. We found that the entire surface region (e.g., ∼ 7 nm in EP'ed Nb), containing amorphous oxides and metallic components, is most likely normalconducting, which can impact the superconducting bulk. Using XPS, STEM/EELS, and in situ measurements we accurately established the structural profiles of the oxide regions as a function of depth under various baking and processing conditions.

Moreover, we resolved the underlying mechanism of baking and N2-processing in Nb (which had previously achieved record SRF performance). We discovered that hightemperature baking induces carbide formation, regardless of gas environment. This, in turn, determines the subsequent formation of suboxides and higher-order oxides upon air exposure and low-temperature baking. These findings, combined with previous studies [\[13,](#page-22-4) [14,](#page-22-5) [15,](#page-22-6) [19\]](#page-22-3), strongly support that both surface oxides and second-phase formation collectively contribute to the effects induced by UHV baking (or oxygen processing) and nitrogen processing. Specifically, UHV baking at 500°C and N<sup>2</sup> processing at 800°C, with significant carbide formation, introduce a large number of suboxides and create more ionic character toward the surface after air exposure and 120°C – 300°C baking. Additionally, the decomposition of surface higher-order oxides at 250°C – 300°C also modifies the electron population profile at the surface.

Furthermore, our investigation revealed that vapor-diffused Nb3Sn contains >9 nm thick oxides mixed with Nb oxides and Sn oxides, while electrochemically synthesized Nb3Sn only has < 2.6 nm thin Sn oxides on the surface. The thick metastable oxides on vapor-diffused Nb3Sn may pose critical issues.

Overall, this study provides important insights into the fundamental mechanisms of baking and processing Nb and Nb3Sn surface structures for high-performance SRF and quantum applications.

# 5. Supporting Information

The Supporting Information is submitted.

# 6. Data Availability Statement

Data are available by contacting the corresponding author.

# 7. Conflicts of Interest

The authors declare no competing financial interest.

# 8. Acknowledgement

This work was supported by the U.S. National Science Foundation under Award PHY-1549132, the Center for Bright Beams. Utilization of the PHI Versaprobe III XPS within UVa's Nanoscale Materials Characterization Facility (NMCF) was fundamental to this project; we acknowledge NSF MRI award #1626201 for the acquisition of this instrument. This work also made use of the Cornell Center for Materials Research Shared Facilities which are supported through the NSF MRSEC program (DMR-1719875), and was performed in part at the Cornell NanoScale Facility, an NNCI member supported by NSF Grant NNCI-2025233. Z.S. thanks H. G. Conklin and T. M. Gruber for assisting with sample preparation and Dr. M. Salim for XPS assistance.

# References

- [1] Nico C, Monteiro T and Gra¸ca M P F 2016 Progress in Materials Science 80 1–37
- [2] Zmuidzinas J 2012 Annual Review of Condensed Matter Physics 3 169
- [3] Halbritter J 1987 Applied Physics A 43 1
- [4] Semione G D L, Pandey A D, Tober S, Pfrommer J, Poulain A, Drnec J, Sch¨utz G, Keller T F, Noei H, Vonk V, Foster B and Stierle A 2019 Physical Review Accelerators and Beams 22 103102
- [5] de Leon N P, Itoh K M, Kim D, Mehta K K, Northup T E, Paik H, Palmer B S, Samarth N, Sangtawesin S and Steuerman D W 2021 Science 372 253
- [6] Murray C E 2021 Materials Science & Engineering R 146 100646
- [7] Esposito M, Ranadive A, Planat L and Rocha N 2021 Applied Physics Letters 119 120501
- [8] Doyle S, Mauskopf P, Naylon J, Porch A and Duncombe C 2008 Journal of Low Temperature Physics 151 530
- [9] Eremeev G 2008 Study of the high field Q-slope using thermometry Ph.D. thesis Cornell University
- [10] Romanenko A, Grassellino A, Barkov F and Ozelis J P 2013 Physical Review Accelerators and Beams 16 012001
- [11] Grassellino A et al. 2017 Superconductor Science and Technology 30 094004
- [12] Grassellino A, Romanenko A, Sergatskov D, Melnychuk O, Trenikhina Y, Crawford A, Rowe A, Wong M, Khabiboulline T and Barkov F 2013 Superconductor Science and Technology 26 102001
- <span id="page-22-4"></span>[13] Lechner E M, Oli B D, Makita J, Ciovati G, Gurevich A and Iavarone M 2020 Physical Review Applied 13 044044
- <span id="page-22-5"></span>[14] Kubo T and Gurevich A 2019 Physical Review B 100 064522
- <span id="page-22-6"></span>[15] Gurevich A and Kubo T 2017 Physical Review B 96 184515
- <span id="page-22-1"></span>[16] Porter R 2021 Advancing the maximum accelerating gradient of niobium-3 tin superconducting radiofrequency accelerator cavities: RF measurements, dynamic temperature mapping, and material growth Ph.D. thesis Cornell University
- <span id="page-22-0"></span>[17] Sun Z, Baraissov Z, Porter R D, Shpani L, Shao Y T, Oseroff T, Thompson M O, Muller D A and Liepe M U 2023 Superconductor Science and Technology
- <span id="page-22-2"></span>[18] Oseroff T 2022 Advancing a superconducting sample host cavity and its application for studying proximity-coupled normal layers in strong microwave fields Ph.D. thesis Cornell University
- <span id="page-22-3"></span>[19] Oseroff T, Sun Z and Liepe M 2023 Superconductor Science and Technology
- [20] Blais A, Grimsmo A L, Girvin S and Wallraff A 2021 Reviews of Modern Physics 93 025005
- [21] Romanenko A, Pilipenko R, Zorzetti S, Frolov D, Awida M, Belomestnykh S, Posen S and Grassellino A 2020 Physical Review Applied 13 034032
- [22] Martinis J M et al. 2005 Physical Review Letters 95 210503
- [23] M¨uller C, Cole J H and Lisenfeld J 2019 Reports on Progress in Physics 82 124501
- [24] Romanenko A and Schuster D I 2017 Physical Review Letters 119 264801
- [25] Grundner M and Halbritter J 1980 Journal of Applied Physics 51 397
- [26] Darlinski A and Halbritter J 1987 Surface and Interface Analysis 10 223
- [27] Grundner M and Halbritter J 1984 Surface Science 135 144
- [28] King B R, Patel H C, Gulino D A and Tatarchuk B J 1990 Thin Solid Films 192 351
- [29] Dacc`a A, Gemme G, Mattera L and Parodi R 1998 Applied Surface Science 126 219
- [30] Ma Q and Rosenberg R A 2003 Applied Surface Science 206 209
- [31] Ma Q, Ryan P, Freeland J W and Rosenberg R A 2004 Journal of Applied Physics 96 7675
- [32] Delheusy M, Stierle A, Kasper N, Kurt R P, Vlad A, Dosch H, Antoine C, Resta A, Lundgren E and Andersen J 2008 Applied Physics Letters 92 101911
- [33] Miller J N, l Lindau, Stefan P M, Weissman D L, Shek M L and Spicer W E 1982 Journal of Applied Physics 53 3267
- [34] Wang Y, Wei X, Tian Z, Cao Y, Zhai R, Ushikubo T, Sato K and Zhuang S 1997 Surface Science 372 L285
- [35] Sanz J M and Hofmann S 1983 Journal of the Less-Common Metals 92 317
- [36] Perkins R A and R A Padgett J 1977 Acta Metallurgica 25 1221
- [37] Ko E and Weissman J 1990 Catalysis Today 8 27
- [38] Schafer H, Gruehn R and Schulte F 1966 Angewandte Chemie International Edition 5 27
- [39] Sathasivam S, Williamson B A D, Althabaiti S A, Obaid A Y, Basahel S N, Mokhtar M, Scanlon D O, Carmalt C J and Parkin I P 2017 ACS Applied Materials & Interfaces 9 18031
- [40] Wimmer E, Schwarz K, Podloucky R, Herzig P and Neckel A 1982 Journal of Physics and Chemistry of Solids 43 439
- [41] Umemori K, Kako E, Konomi T, Michizono S, Sakai H, Tamura J and Okada T 2019 Study on nitrogen infusion using KEK new furnace 19th International Conference on RF Superconductivity p 95
- [42] Koufalis P N, Furuta F, Ge M, Gonnella D, Kaufman J J, Liepe M, Maniscalco J T and Porter R D 2016 Low temperature nitrogen baking of a SRF cavity International Linear Accelerator Conference 2016 p 472
- [43] Ito H, Araki H, Takahashi K and Umemori K 2021 Progress of Theoretical and Experimental Physics 2021 071G01
- [44] Yang Z, Hao J, Quan S, Lin L, Wang F, Jiao F and Liu K 2022 Physica C: Superconductivity and its applications 599 1354092
- [45] He F et al. 2021 Superconductor Science and Technology 34 095005
- [46] Koufalis P N 2022 Field and frequency dependence of the surface resistance of superconducting microwave resonators for particle accelerators Ph.D. thesis Cornell University
- [47] Bergmann D and Hinze J 1996 Angewandte Chemie International Edition 35 150
- [48] Hofmann S, Liu Y, Wang J Y and Kovac J 2014 Applied Surface Science 314 942
- [49] Bach D, Schneider R, Gerthsen D, Verbeeck J and Sigle W 2009 Microscopy and Microanalysis 15 505
- [50] NIST X-ray Photoelectron Spectroscopy Database, Version 4.1 (National Institute of Standards and Technology, Gaithersburg, 2012); http://srdata.nist.gov/xps/.
- [51] Over H and Seitsonen A P 2002 Science 297 2003
- [52] Bryant R T 1962 Journal of the Less Common Metals 4 62
- [53] H¨orz G, Lindenmaier K and Klaiss R 1974 Journal of the Less Common Metals 35 97
- [54] Sun Z et al. 2023 Advanced Electronic Materials 9
- [55] Kurmaev E Z, Moewes A, Bureev O G, Nekrasovc I A, Cherkashenkoa V M, Korotina M A and Ederer D L 2002 Journal of Alloys and Compounds 347 213
- [56] Zhang W, Wu W, Wang X, Cheng X, Yan D, Shen C, Peng L, Y W and L B 2013 Surface and Interface Analysis 45 1206
- [57] Edamoto K, Maehama S, Miyazaki E and Kato H 1989 Physical Review B 39 7461
- [58] Greiner M T, Helander M G, Tang W M, Wang Z B, Qiu J and Lu Z H 2012 Nature Materials 11 76
- [59] Grosvenor A P, Wik S D, Cavell R G and Mar A 2005 Inorganic Chemistry 44 8988
- [60] Hochst H, Hufner S and Goldmann A 1976 Solid State Communications 19 899
- [61] Hochst H, Hufner S and Goldmann A 1980 Zeitschrift fur Physik B 37 27
- [62] Hochst H, Steiner P, Reiter G and Hufner S 1981 Zeitschrift fur Physik B Condensed Matter 42 199
- [63] Weaver J H and Schmidt F A 1980 Physics Letters 77A 73
- [64] Seah M P 1980 Surface and Interface Analysis 2 222
- [65] Powell C J and Larson P E 1978 Applications of Surface Science 1 186
- [66] Yeh J J and Lindau I 1985 Atomic Data and Nuclear Data Tables 32 1
- [67] Powell C J and Jablonski A 1999 Journal of Physical and Chemical Reference Data 28 19
- [68] Rosner H, Kitaigorodsky A and Pickett W E 2002 Physical Review Letters 88 127001