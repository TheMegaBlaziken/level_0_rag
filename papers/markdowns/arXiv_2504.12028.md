# arXiv:2504.12028

**Paper ID:** d80337bf01012737bf43d1ba2af78e76

**PDF Path:** apl/Superconductors/arXiv:2504.12028.pdf

**Processing Status:** complete

**Captions Added:** 11

**Generated:** 2025-06-24T13:44:28.070599

---

# Magnetic and Mechanical Analysis of Bi-2212 Rutherford Cable in a Cos-Theta Sub-Scale Dipole Coil

A. D'Agliano, A. V. Zlobin, I. Novitski, D. Turrioni, E. Barzi, *Senior Member, IEEE*, S. Donati and V. Giusti

*Abstract*—The U.S. Magnet Development Program (US-MDP) explores high-field accelerator magnets compatible with operational conditions beyond the limits of Nb3Sn technology. The ongoing R&D High-Temperature Superconductors (HTS) suggests using Bi2Sr2CaCu2O8−<sup>x</sup> (Bi-2212) as superconducting element. Bi-2212 Rutherford cables maintain a high critical current (I<sup>C</sup> ) when exposed to a large external magnetic field. However, Bi-2212 exhibits an oversensitive stress-strain response when subject to large Lorentz forces. This paper reports on the magnetic and mechanical analysis of the Bi-2212 cosine-theta insert being developed at Fermilab for a hybrid magnet composed of two external layers of Nb3Sn and two internal layers of Bi-2212. We performed a FEM analysis of the insert to estimate the HTS stress state in the coil's strands under magnetic and mechanical loads.

*Index Terms*—Finite-element analysis, Bi-2212, Rutherford cable, cosine-theta, superconducting accelerator magnet.

## I. INTRODUCTION

I N the past few years, scientists have gained increasing interest in realizing high-field magnets using HTSconductors. The research team at the Lawrence Berkeley National Laboratory (LBNL) has developed and tested magnet models based on canted-cosine-theta, racetrack, and pancake coils [\[1\]](#page-7-0), [\[2\]](#page-7-1). Fermilab is developing a cosine-theta dipole insert based on a stress-managed structure and a Bi-2212 Rutherford cable [\[3\]](#page-7-2), [\[4\]](#page-7-3).

This work is supported by Fermi Research Alliance, LLC, under contract No. under contract No. DE-AC02-07CH11359 with the U.S. Department of Energy and U.S. Magnet Development Program.

A. D'Agliano is with the Fermi National Accelerator Laboratory, Batavia, IL, 60510, USA, and also with the University of Pisa, Pisa, 56126, Italy (Corresponding author, e-mail: dagliano@fnal.gov).

E. Barzi, I. Novitski, D. Turrioni and A. V. Zlobin are with the Fermi National Accelerator Laboratory, Batavia, IL, 60510, USA.

S. Donati and V. Giusti are with the University of Pisa, Pisa, 56126, Italy.

Manuscript received October 4, 2024; revised January 14, 2025; Accepted for publication February 5, 2025.

Bi-2212 has been chosen since it can carry a large current density (JE) in a high external magnetic field. Bi-2212 could allow breaking the threshold of a 20 T magnetic field inside a dipole bore [\[5\]](#page-7-4), which is one of the main goals of the US Magnet Development Program [\[6\]](#page-7-5). The insert will be placed in the innermost part of a multi-layer hybrid magnet and, thus, exposed to an external magnetic field that could exceed 14.6 T, which is the record value for a cosine-theta dipole bore field achieved at Fermilab during the test of the Nb3Sn Cosine-Theta MDPCT1 in 2020 [\[7\]](#page-7-6). Experimental tests demonstrated that Bi-2212 can still carry a current density above 1000 A/mm<sup>2</sup> at 14.6 T [\[8\]](#page-7-7). This makes Bi-2212 one of the best candidates for significantly improving high-field magnets' performance for accelerator applications.

![](_page_0_Figure_13.jpeg)

**Caption:** Figure 13 provides a composite overview of the HTS cosine-theta dipole insert, showcasing detailed components from mandrels to Bi-2212 strands. The sequence illustrates the transition from the Inconel-718 mandrels to the complete Rutherford cable assembly, emphasizing the structural hierarchy and constituent materials such as epoxy and mullite. This comprehensive view highlights material integration aiding in mechanical stability and functional harmony, crucial for the robust performance of high-field superconducting magnets.


<span id="page-0-0"></span>Fig. 1. Schematic representation of the dipole insert and Bi-2212 Rutherford cable as implemented in the Ansys FEM simulation (heterogeneous model).

Fermilab's team aims to characterize the magnetic and mechanical behavior of a Bi-2212 small-aperture cosinetheta insert (Fig. [1\)](#page-0-0) with a simulation and validate the readiness of magnet design for fabrication and test. We used the Ansys Parametric Design Language (APDL) to perform a 2D FEM analysis. In [\[9\]](#page-7-8) has been reported a study of a previous version of the insert for which a simplified model of the cable, defined as a rectangle of homogeneous material placed inside the mandrel grooves, was adopted. This cable model, referred to as the "homogeneous model" in the following, provides an approximate estimate of the equivalent stress in the coil. To improve the simulation quality, we introduced a more detailed model of the cable to distinguish the relevant parts of the conductor: a) the Ag hexagonal core and outer strand ring, b) the Bi-2212 conductor area, c) the Mullite sleeve as insulation, d) the epoxy in all the gaps (Fig. [1\)](#page-0-0). This model will be referred to as the "heterogeneous model". The heterogeneous model allows to specify the pure isotropic material properties inside the cable cross-section (i.e., mechanical characteristics, stress-strain behavior, current degradation related to applied load) and to obtain a more accurate estimate of the stress and strain distributions in the conductor. This paper describes the most recent design of the Bi-2212 cosine-theta dipole model being developed at Fermilab. It also reports the results of magnetic and mechanical analyses performed in the two configurations of homogeneous and heterogeneous Bi-2212 cable models.

## II. HTS COSINE-THETA BI-2212 COIL DESIGN

## *A. Cross-Section Geometry Evolution*

The insert coil was designed to fit into the 60 mm aperture of the outer dipole [\[10\]](#page-7-9) realized at Fermilab with two layers of Nb3Sn. The mandrel is the insert's main structural part and provides mechanical support to the coil turns during the winding process, the heat treatment, and the coil powering. Table [I](#page-1-0) reports the geometrical and electrical parameters of Rutherford cable and Bi-2212 strands [\[4\]](#page-7-3). The position of the Rutherford cables in the coil transversal and axial cross-sections and the mandrel geometry have been chosen to maximize the magnetic field intensity and the field quality in the bore using ROXIE code [\[11\]](#page-7-10).

<span id="page-1-0"></span>TABLE I BI-2212 RUTHERFORD CABLE AND STRAND PARAMETERS

| Parameter                         | values      | unit  |
|-----------------------------------|-------------|-------|
| Cable ID                          | LBNL - 1110 |       |
| Number of strands                 | 17          |       |
| Bare cable width                  | 7.8         | mm    |
| Bare cable thickness              | 1.44        | mm    |
| Cable transposition pitch         | 58          | mm    |
| Billet ID                         | PMM180207-2 |       |
| Strand diameter before/after reac | 0.80/0.778  | mm    |
| tion                              |             |       |
| Strand architecture               | 55 x 18     |       |
| Strand fill factor                | 23          | %     |
| Strand twist pitch                | 25          | mm    |
| Strand Ic<br>(4.2K, 5T)           | 460-640     | A/mm2 |

The mandrel structure has evolved [\[9\]](#page-7-8) to optimize the magnetic parameters and minimize the bending radius of the turns located at the cosine-theta pole regions. Fig. [2](#page-1-1) shows the original 2-layer 15-turn insert design as described in [\[9\]](#page-7-8) and the final 2-layer 9-turn insert design analyzed in this paper. The original design allowed the cable to be wound from the cylinder's internal surface to improve mechanical robustness. However, the minimum bending radius in the pole region was approximately 3.6 mm, which was too small to guarantee the integrity of the Bi-2212 strands. The final design addressed this issue by reducing the number of turns in both layers and increasing the minimal bending radius to approximately 5.5 mm which ensures the integrity of the Bi-2212 strands. To minimize the risk of damaging the cable and simplify the winding procedure, the turns are inserted in the grooves from the external surface of the barrel, at the slight expense of the coils' radial stiffness.

![](_page_1_Picture_9.jpeg)

**Caption:** Figure 3 offers a schematic comparison between original (left) and final (right) transverse designs of a superconducting insert coil, showcasing differences in coil layer configurations. The evolution features adjustments to enhance field homogeneity and mechanical resilience, optimizing the support structure to suit Bi-2212 strand placement while ensuring integrity during high-stress operations .


Fig. 2. Schematic representation of the insert transverse crosssection. The original design is on the left, and the final design is on the right.

<span id="page-1-1"></span>Fig. [3](#page-1-2) shows the magnetic diagrams in the coil turns calculated with ROXIE at 10 kA. In the simulation, the final insert design was modeled inside an iron yoke structure characterized by a 60 mm inner diameter and a magnetic relative permeability of 1000. The area in the aperture where the relative error between the B<sup>1</sup> field component and the field computed considering all the harmonics [\[12\]](#page-7-11) is below 10−<sup>4</sup> , has an approximate diameter of 5.4 mm.

![](_page_1_Figure_12.jpeg)

**Caption:** Figure 12 displays the magnetic field distribution across the cosine-theta HTS dipole insert as simulated with ROXIE. This illustration shows the uniformity of field intensity along the coil structure at a 10 kA current. The color scale bar indicates magnetic field strength in teslas (T), with distinct color bands highlighting field variations across the coil. Roxie simulations confirm the applicability of this configuration to achieve a consistent field profile, advantageous for minimizing field-induced mechanical stress on the Bi-2212 strands, thus signifying relevance in high-precision and robust superconducting magnet designs for applications requiring stable magnetic fields.


<span id="page-1-2"></span>Fig. 3. Magnetic field intensity for the HTS insert dipole at 10 kA as determined with ROXIE.

## *B. Insert Coil Practice Winding*

Fig. [4](#page-2-0) shows the six mandrel pieces that constitute the mechanical structure of the insert coil. The two inner parts (number 1 and number 2 in the picture) support the inner coil, while the four outer parts (number 3,4,5,6) support the outer coil. The insert internal diameter is 15.5 mm, and the external diameter is 57.6 mm, with a gap of 0.5 mm between the two layers to insert Mullite insulation [\[11\]](#page-7-10). Coil winding starts from the interface between parts 1 and 2 (Fig. [4\)](#page-2-0). The transition from the inner to the outer layer is performed with a layer jump at the pole region. The Rutherford cable lead end enters the mandrel from the same side where the return end exits. In this way, the two ends connect with the power supply splices on the same side of the insert. The winding procedure has been successfully tested with a dummy cable.

![](_page_2_Picture_3.jpeg)

**Caption:** Figure 4 shows a 3D-printed model of a 2-layer mandrel utilized to test winding methodologies and validate procedural soundness for the Bi-2212 insert coil. Each layer's composition and construction are detailed, underscoring the transition from inner to outer via specified component parts numbered to indicate integral role assignment in the winding process. This model serves as a tool for validating mechanical stability against thermal and electromagnetic stresses .


Fig. 4. 3D-printed 2-layer mandrel used to test the winding feasibility and validate the procedure.

## <span id="page-2-0"></span>III. BI-2212 INSERT FEM ANALYSIS

## *A. Magnetic Analysis*

We performed the magnetic analysis of the Bi-2212 insert with the homogeneous and the heterogeneous models. In both cases, we used the ANSYS element type PLANE 53, which models 2-D planar and axisymmetric magnetic fields. The simulation is planar, and proper boundary conditions guarantee compliance with the symmetry condition. The element is defined by 8 nodes and is based on the magnetic vector potential formulation for the computation of the magneto-static analysis. PLANE 53 allows the introduction of a user-defined iron B-H curve. The symmetry constraints are applied on the vertical axis and the outer cylindrical lines of the outer air area to impose that the z-component of the magnetic vector potential is equal to zero. We applied an I<sup>C</sup> load of 10 kA distributed on the Bi-2212 strands in the cable (Fig. [1\)](#page-0-0). The total current load corresponds to a J<sup>E</sup> of approximately 1900 A/mm<sup>2</sup> (at 5 T, 4.2 K). This current density is consistent with the data of the Bi-2212 short samples tested at these specific values of the magnetic field and temperature [\[8\]](#page-7-7).

TABLE II BI-2212 INSERT PARAMETERS AT 4.2 K

<span id="page-2-1"></span>

|                             | HTS Insert |            |  |
|-----------------------------|------------|------------|--|
|                             | Inner coil | Outer coil |  |
| Bore field [T]              | 4.36       |            |  |
| Peak field [T]              | 4.85       | 3.95       |  |
| Current [kA]                | 10.0       |            |  |
| Inductance [mH/m]           | 0.21       |            |  |
| Stored Energy @10 kA [kJ/m] | 10.7       |            |  |
| Fr [MN/m]                   | 0.07       | 0.04       |  |
| Fa [MN/m]                   | -0.06      | -0.09      |  |
| Number of turns             | 3          | 6          |  |

We implemented a detailed geometry of the cable components in the heterogeneous model: a) the Bi-2212 region, b) the silver core and outer ring of the strand, c) the Mullite insulation area, and d) all the gaps filled with epoxy. After importing the ROXIE coordinates of each cable in the ANSYS code, a user-defined APDL macro created all the coil turns at the designed locations. The magnetic simulation of the heterogeneous model shows that with the nine turns of the Bi-2212 cable, a maximum field of 4.85 T can be generated in the coil and 4.36 T in the bore at 4.2 K. The maximum difference between the magnetic field computed with ANSYS in the heterogeneous conductor and ROXIE is of the order of 4.7% (4.63 T at 10 kA, Fig. [3\)](#page-1-2), and it is 2.3% if the ANSYS homogeneous model is considered.

Fig. [5](#page-3-0) shows the distribution of the magnetic field and the Lorentz force as provided by the ANSYS simulation. Table [II](#page-2-1) summarizes the main magnetic parameters of the Bi-2212 insert. The total Lorentz forces are the sum of all the nodal results on the turns of the specified layer and are reported in a cylindrical coordinate system (radial and azimuthal components).

Fig. [6](#page-3-1) shows the magnetic field intensity along specific directions for the homogeneous and the heterogeneous models. The fields computed for the two models are almost identical, except for the positions of the nodes, which define the conductor components in the heterogeneous model. The maximum difference between the two models is 0.18 T along the direction at 25.6 degrees from the x-axis, which intersects the conductors of layers 1 and 2 (Fig. [5\)](#page-3-0). We verified that the two models provide almost identical nodal sums of the Lorentz forces as computed on each coil.

![](_page_3_Figure_1.jpeg)

**Caption:** Figure 1 presents the magnetic field intensity (in the top section) and Lorentz forces (in the bottom section) within the Bi-2212 strands as determined using ANSYS for the heterogeneous model. The color gradient denotes the magnetic field strength, measured in teslas (T), while the superimposed streamline pattern outlines regions of differing field intensity. The plot indicates significant variances in field intensity across the aperture, coils, and iron yoke. The bottom section shows Lorentz forces (in N/m) acting on the strands, demonstrating the forces distributed around the serpentine path of the Bi-2212 conductor. This analysis is pivotal for understanding the mechanical stresses imposed under operational magnetic fields, contributing crucial insight for the improved mechanical performance of superconducting materials in high-field applications .


<span id="page-3-0"></span>Fig. 5. Magnetic field intensity in the aperture, coils, and iron yoke (Top) and Lorentz forces in the Bi-2212 strands (Bottom) for the heterogeneous model as determined with ANSYS.

## *B. Conductor Mechanical Properties*

Fig. [7](#page-3-2) shows the geometrical description of the Bi-2212 sub-scale dipole insert as modeled for the AN-SYS mechanical analysis according to the heterogeneous model, including all the structural and cable components. One of the major challenges of a FEM analysis is the definition of the material properties. To better define the mechanical properties of the Bi-2212 strands, we measured stress-strain curves of heat-treated and nonheat-treated Bi-2212 strand samples with a tensile test Instron machine. Both types of Bi-2212 specimens were 150 mm long with a 0.8 mm diameter. Each strand edge

![](_page_3_Figure_5.jpeg)

**Caption:** Figure 8 depicts magnetic flux intensity profiles (in Telsa) in an HTS Insert as analyzed. It profiles the magnetic field across multiple axes (X, Y, radial axes at angles) using both homogeneous ('hom_') and heterogeneous ('het_') models. This data is vital in calibrating structural elements responding to varying magnetic flux intensities, optimizing the HTS insert design under operational magnetic loads.


<span id="page-3-1"></span>Fig. 6. Magnetic field intensity for the HTS insert dipole as determined along the x-axis, y-axis, the line at 25.6 degrees, and the bisector of the first quadrant. The 'hom\_' and 'het\_' refer to the homogeneous and heterogeneous models, respectively.

was glued with epoxy to a washer to improve the grip with the machine clamps during the traction test.

![](_page_3_Figure_8.jpeg)

**Caption:** Figure 9 showcases the cross-section composition of the HTS insert, detailing material components like iron yoke, mandrel, epoxy insulation, silver, and Bi-2212. This visualization is imperative for understanding the layered architecture contributing to the overall electromagnetic and mechanical behavior of the superconducting structure under study .


<span id="page-3-2"></span>Fig. 7. Geometry of the Bi-2212 sub-scale dipole insert modeled in ANSYS mechanical analysis with the cable described according to the heterogeneous model. The structural components are also highlighted.

The goal of the tensile analysis was to obtain the conductor's Young modulus (E) along its axial direction. The analysis showed that the non-heat-treated and the heattreated specimens have almost identical Young moduli, EnHT = 21.99 ± 5.3% GPa and EHT = 21.26 ± 10.8% GPa, respectively. The two samples showed different yield strength results. To define the material properties

![](_page_4_Figure_1.jpeg)

**Caption:** Figure 5 displays the tensile stress-strain curves for Bi-2212 strand samples, highlighting variations between heat-treated (top) and non-heat-treated (bottom) samples. The stress (σ) is plotted against strain (ε), showing mechanical properties such as yield strength around 140 MPa and a near-linear elasticity limit. The heat-treated samples exhibit improved plasticity properties critical for optimizing the resilience of superconducting cables under operational loading .


<span id="page-4-1"></span>Fig. 8. Stress-strain curves of heat-treated (Top) and non-heat-treated (Bottom) samples of Bi-2212 strand tested with an Instron machine.

of the Bi-2212 elements in the mechanical analysis, we used the measured EHT and the yield strength of the best heat-treated sample of approximately 140 MPa. However, the reacted strands broke unevenly during the plastic transition, suggesting that more data would be necessary to determine a more accurate yield strength of the Bi-2212 strand after the thermal reaction. We needed to use the reacted samples data since, at the beginning of the simulation, the entire coil structure is intended to have already been appropriately heat-treated in a 50 bar oxygen flow and respecting the specific temperature profile [\[8\]](#page-7-7).

We referred to the literature for the mechanical properties of the structural materials at warm and cold temperatures, as reported in Table [III](#page-4-0) [\[3\]](#page-7-2), [\[14\]](#page-7-12)–[\[17\]](#page-7-13). In some cases (Inconel-718, iron, Kapton), the Young modulus variation between room and cold temperature is negligible, and we simply used one single value independent of temperature. This reduced the computational load of the analysis without significantly compromising the simulation accuracy.

## *C. Mechanical Analysis*

The main goal of the mechanical FEM simulation is to compute the stress and strain distributions within the Bi-2212 and silver in the strands and compare

TABLE III MATERIAL PROPERTIES AT 300/4.2 K

<span id="page-4-0"></span>

|                 | Properties value          |                            |                                                 |  |
|-----------------|---------------------------|----------------------------|-------------------------------------------------|--|
|                 | Young<br>Modulus<br>[GPa] | Yield<br>Strength<br>[MPa] | Thermal<br>Contraction<br>α · 10−6<br>[K−1<br>] |  |
| Mullite&CTD101K | 12.9/19.7                 | 790/1360                   | 26                                              |  |
| Bi-2212         | 21.3                      | 140                        | 11.1                                            |  |
| Silver          | 82.5/91                   | 54/80                      | 14.6                                            |  |
| Inconel-718     | 208                       | 1100/1700                  | 8.2                                             |  |
| Iron            | 205                       | 100/300                    | 7                                               |  |
| Kapton          | 2.5                       | 89.6                       | 70                                              |  |
| SS 304          | 199/210                   | 215/439                    | 10.3                                            |  |

the results obtained with the homogeneous and heterogeneous models. The mandrel structure of the twolayer coil doesn't change between the two simulations (Fig. [7\)](#page-3-2). Since the mandrel pieces are free to slide axially during heat treatment, and the grooves' design allows the insulated cable to move, we can neglect the residual stress state due to thermal cool-down after the reaction process. For this reason, the non-linear mechanical analysis follows a load-step approach which consists of three steps: a) pre-stress load, implemented as a 100 µm interference compression at the contact surface between the second layer skin and the iron yoke. That value was chosen to minimize coil displacements due to Lorentz forces. b) thermal cool-down from 300 K to 4.2 K, and c) energization at 10 kA, reaching 4.8 T and generating the computed Lorentz forces. We used the element type PLANE 183, a high-order 2D, 8 node element with quadratic displacement behavior for plasticity implementation (as for Bi-2212 in Fig. [8\)](#page-4-1) and generalized plane strain options needed for mechanical simulation. The magnet symmetry allows to model only one quarter of the structure cross-section with symmetry boundary condition on X and Y axes. The area elements follow a bi-linear perfectly plastic rheological behavior. The homogeneous model is characterized by the Bi-2212 anisotropic material properties reported in [\[17\]](#page-7-13). The heterogeneous model is characterized by the isotropic material properties of Mullite plus epoxy, silver and Bi-2212 (Fig. [7\)](#page-3-2). We defined the conductors, mullite/epoxy and insulation areas with a finer mesh compared to the mandrel and iron yoke areas. We applied CONTACT 172 and TARGET 169 to the material interfaces of mandrelsepoxy, epoxy-ground insulation, and ground insulationskin, with interference only along the contact highlighted by the green dashed arc in Fig. [7.](#page-3-2)

![](_page_5_Figure_2.jpeg)

**Caption:** Figure 2 illustrates the stress history comparison (in MPa) of HTS Dipole components across different models (heterogeneous 'het_' vs homogeneous 'hom_') over three load steps: pre-stress (PS), cool-down (PS-Th), and energization (EN). The stress distribution, noted for components like mandrel, epoxy insulation, and coil, highlights the mechanical impact during the energization phase (at 4.8 T) using ANSYS simulations. The plot emphasizes disparities between models, with stress peaks notably around PS-Th load steps in het_Bi2O3 and het_Epoxy/Coil series. Such data underline the structural integrity and reliability of HTS dipoles under rigorous operational conditions.


<span id="page-5-0"></span>Fig. 9. Maximum equivalent Von Mises stress (MPa) for the insert components and the three load steps in the homogeneous and heterogeneous models. The 'hom\_' and 'het\_' refer to the homogeneous and heterogeneous model, respectively.

Fig. [9](#page-5-0) shows the maximum equivalent Von Mises stress for each component of the insert (mandrel, epoxy insulation, epoxy coil, Bi-2212, silver) at each of the three simulation load steps (pre-stress 'PS', cool-down 'PS-Th', and energization 'EN' at 4.8 T) as computed in the two different simulations. Here are the results for each structural component.

- Mandrel: In the homogeneous (heterogeneous) model the maximum stress at energization (EN) is 419 (427) MPa at the bottom of the fourth groove from the mid-plane of the outer layer. Stress intensity in the mandrel drops at the cool-down load step since Inconel-718 is characterized by a higher thermal contraction factor compared to iron. For this reason, the pre-stress compression is partially relieved. The largest difference between the two models is 70 MPa at the bottom of the third groove from the mid-plane of the internal layer at prestress (PS). This is probably due to the different behavior of the heterogeneous and homogeneous cables inside the structure during compression.
- Epoxy insulation: In the homogeneous (heterogeneous) model the maximum stress at energization (EN) is 243 (256) MPa. The largest difference between the two models is 53 MPa at cool-down (PS-Th). We believe that this difference is due to the presence of fillets inserted between the mandrel and the epoxy insulation only in the heterogeneous model (Fig. [7\)](#page-3-2).

![](_page_5_Figure_7.jpeg)

**Caption:** Figure 7 details maximum equivalent Von Mises stress (in MPa) from analysis through different load conditions for insert components like mandrel, and epoxy insulation at 'EN' (energization) step. It contrasts homogeneous ('hom_') versus heterogeneous ('het_') model results, revealing disparities that guide structural optimizations. The heterogeneous model nuances mechanical performance predictions critical for complex, stress-induced superconductor behavior.


<span id="page-5-1"></span>

Fig. 10. Equivalent Von Mises stress distribution at EN load step (MPa) in the homogeneous (Top) and heterogeneous models (Bottom). The maximum stress is highlighted with a red circle in both of the models.

• Conductor: In the homogeneous model, the maximum stress is 141 MPa at the edge of the cable. If only the area where the Bi-2212 is located in the heterogeneous model is considered, the maximum stress is 50 MPa. In the heterogeneous model, the maximum stress is 143 MPa in the epoxy coil, and 68 MPa in the Bi-2212 (Fig. [10\)](#page-5-1). Silver reaches plasticity and the stress settles at its yield strength of 80 MPa. No other material reaches plasticity during the simulation load steps. In practice, homogeneous cable and the coil epoxy stress histories in the heterogeneous model are very similar.

In Table [IV](#page-6-0) are highlighted the differences between the equivalent Von Mises stress of components that could be directly compared between the two models (Mandrel, Epoxy insul. and Coil epoxy), at each load step.

Table [V](#page-6-1) reports the maximum and minimum values of radial, azimuthal, and axial stress components for

<span id="page-6-0"></span>TABLE IV STRESS DIFFERENCES BETWEEN THE COMPARABLE COMPONENTS IN THE HOMOGENEOUS AND HETEROGENEOUS MODELS [MPA]

|       | Components |         |            |
|-------|------------|---------|------------|
|       | Coil Ins.  | Mandrel | Epoxy Ins. |
| PS    | -14.6      | -70     | 1          |
| PS-Th | 14         | -19     | -63        |
| EN    | -2         | -8      | -13        |

the homogeneous and heterogeneous models at the energization load step (EN). Directional stress, excluding the axial one, spans a ∼140 MPa range for the heterogeneous model, which is wider than the ∼120 MPa range estimated for the homogeneous model. However, the maximum absolute value is lower for the heterogeneous model, where the radial and azimuthal stresses are 72 MPa in compression and 80 MPa in traction, respectively, compared to 118 MPa in compression and 108 MPa in compression for the homogeneous model.

TABLE V DIRECTIONAL STRESS RESULTS IN THE CABLE [MPA]

<span id="page-6-1"></span>

| Stress component (min/max) | Homogeneous | Heterogeneous |
|----------------------------|-------------|---------------|
| Radial                     | -108/9      | -72/70        |
| Azimuthal                  | -118/6      | -63/80        |
| Axial                      | -30/6       | -21/101       |

We also performed a sensitivity analysis to search for possible correlations between stress and the geometrical deformation of the strand in the cable. The Bi-2212 strands can be deformed during the cold-rolling extrusion of the cable from the Turks Head. To reproduce this effect, we developed an APDL macro to modify the cable with a strand shape parameter (SP). If SP = 1, strands have an elliptical shape. If SP = 0, strands are completely packed and compressed inside the cable area (Fig. [11\)](#page-6-2). Table [VI](#page-6-3) reports the maximum equivalent stress in all the different structural components at the energization load step (EN) for four representative SP values (SP = 0.0, 0.3, 0.7, and 1.0). Table [VI](#page-6-3) shows that the configuration with SP = 0.7 is the least conservative one concerning the stress value inside the Bi-2212 since the stress reaches its minimum value. However, the structural materials show moderate sensitivity to the stress as SP varies. Although this is the least conservative configuration, we chose SP = 0.7 since it better matches the actual Bi-2212 strand geometry.

<span id="page-6-3"></span>TABLE VI VON MISES STRESS DEPENDENCE ON THE STRAND SHAPE PARAMETER (MAX/MIN) [MPA]

|                  | Strand Shape Parameter (SP) |         |         |         |
|------------------|-----------------------------|---------|---------|---------|
|                  | 0.0                         | 0.3     | 0.7     | 1.0     |
| Bi-2212          | 96/25                       | 73/26   | 68/26   | 78/25   |
| Silver           | 80/63                       | 80/68   | 80/67   | 80/61   |
| Coil Epoxy       | 138/61                      | 137/59  | 143/60  | 157/62  |
| Epoxy insulation | 250/69                      | 254/69  | 256/69  | 259/69  |
| Mandrel          | 422/0.3                     | 423/0.5 | 427/0.5 | 429/0.2 |

![](_page_6_Figure_10.jpeg)

**Caption:** Figure 6 presents stress distribution maps for varying Strand Shape Parameters (SP), ranging from 0.0 to 1.0, depicted in multiple panels (a-d). These maps demonstrate the mechanical impact of cable compaction on the stress profile across the Bi-2212 strands at the energization step (EN). These findings illustrate the relationship between cable geometry and mechanical stress, aiding in the refinement of strand and winding configurations for optimal performance in superconducting magnet systems .


<span id="page-6-2"></span>Fig. 11. Variation of the cable geometry and stress distribution between four different SP values: (a) 0.0, (b) 0.3, (c) 0.7, (d) 1.0.

## IV. CONCLUSION

In this paper, the up-to-date design of the Bi-2212 insert coil developed at Fermilab was presented. We reported the results of the magnetic and mechanical simulations performed with ROXIE and ANSYS. We developed two alternative models of the Bi-2212 cable. In the "homogeneous model", the cable is represented in the simulation as a simple rectangle of homogeneous material. In the "heterogeneous model", all cable components are modeled at the strand level (Sect. I). The magnetic analysis shows no significant differences in the magnetic field distribution and Lorentz force intensity between the two models (Sect. III.A). To obtain the Bi-2212 cable mechanical properties needed for the mechanical simulation of the heterogeneous model, we performed a series of measurements with a tensile test Instron machine at Fermilab to plot the Bi-2212 strands stressstrain curves (Sect. III.B). Those measurements allow us to set the heat-treated superconductor Young modulus value at 21.3 GPa and the yield stress at 140 MPa in the mechanical analysis. We computed the maximum equivalent Von Mises stress for the three load steps (pre-stress (PS), cool-down (PS-Th), and energization (EN) at 4.8 T) in the insert components for both the homogeneous and the heterogeneous model (Sec. III.C). For the homogeneous model, the maximum stress in the conductor is 141 MPa located on the edge of the cable. That value can be reduced by optimizing precompression load and shim thickness to reduce stresses in the coil instead of minimizing coil displacement after energization. Considering, instead, the area within the homogeneous Rutherford cable where the Bi-2212 strands should be located, the maximum stress is 50 MPa. With the heterogeneous model, the maximum stress in the Bi-2212 areas is 68 MPa, which respects the threshold of 120 MPa [\[17\]](#page-7-13). The maximum stress in the structural components remains within the acceptable value of yield strength set for their material, except for silver and iron yoke where plasticity occurs. The accuracy of the Bi-2212 dipole insert mechanical analysis was improved thanks to the heterogeneous model, which gives more detailed information about stress state distribution and intensity in all the coil components. We also investigated the dependence of the equivalent Von Mises stress in the cable on the strand deformation (SP parameter). The higher sensitivity is in the Bi-2212 conductor area since silver reaches plasticity, and the strand shape variation affects only indirectly the other structural materials. Even if this is the least conservative configuration, we used the SP value to 0.7 when comparing the heterogeneous and homogeneous models, since it represents the most realistic cable geometry. That SP value will be set as standard for the future FEM analysis based on the heterogeneous Rutherford cable model.

## ACKNOWLEDGMENTS

This work was supported by the EU Horizon 2020 Research and Innovation Program under the Marie Sklodowska-Curie Grant Agreement No. 734303, 822185, 858199 and 101003460, and the Horizon Europe Research and Innovation Program under the Marie Sklodowska-Curie Grant Agreement No. 101081478. We want to thank collaborators from CEA Paris-Saclay, Lawrence Berkeley National Laboratory, and CERN for their technical advice and suggestions, which enhanced the quality of the work published in this article.

## REFERENCES

<span id="page-7-0"></span>[1] Shen T., Garcia Fajardo L. "Superconducting Accelerator Magnets Based on High-Temperature Superconducting Bi-2212 Round Wires." Instruments 2020, 4, 17. [https://doi.org/10.3390/](https://doi.org/10.3390/instruments4020017) [instruments4020017](https://doi.org/10.3390/instruments4020017)

- <span id="page-7-1"></span>[2] L. Garcia Fajardo, T. Shen, X. Wang, C. Myers, D. Arbelaez, E. Bosque, L. Brouwer, S. Caspi, L. English, S. Gourlay, A. Hafalia, M. Martchevskii, I. Pong and S. Prestemon, "First demonstration of high current canted-cosine-theta coils with Bi-2212 Rutherford cables", Superconductor Science and Technology, Jan. 2021, IOP Publishing, vol. 34, no. 2, [https://dx.doi.org/10.1088/1361-6668/](https://dx.doi.org/10.1088/1361-6668/abc73d) [abc73d](https://dx.doi.org/10.1088/1361-6668/abc73d)
- <span id="page-7-2"></span>[3] A. V. Zlobin, I. Novitski, E. Barzi and D. Turrioni, "Development of a Small-Aperture Cos-Theta Dipole Insert Coil Based on Bi2212 Rutherford Cable and Stress Management Structure", IEEE Transactions on Applied Superconductivity, vol. 32, no. 6, pp. 1-5, Sept. 2022, Art no. 4003605.
- <span id="page-7-3"></span>[4] E. Barzi, "Conductor Properties and Coil Technology for a Bi2212 Dipole Insert for 20 Tesla Hybrid Accelerator Magnets", Contribution to Snowmass 2021, Oct. 2022, [https://doi.org/10.](https: //doi.org/10.48550/arXiv.2204.01072) [48550/arXiv.2204.01072](https: //doi.org/10.48550/arXiv.2204.01072)
- <span id="page-7-4"></span>[5] A. V. Zlobin and I. Noviski and E. Barzi and P. Ferracin. "20 T Dipole Magnet Based on Hybrid HTS/LTS Cos-Theta Coils with Stress Management", 2023, 2305.06776. [https://arxiv.org/](https://arxiv.org/abs/2305.06776) [abs/2305.06776](https://arxiv.org/abs/2305.06776)
- <span id="page-7-5"></span>[6] U.S. Magnet Development Program Website. ["https://usmdp.lbl.](https://usmdp.lbl.gov/) [gov/"](https://usmdp.lbl.gov/).
- <span id="page-7-6"></span>[7] A. V. Zlobin et al., "Development and First Test of the 15 T Nb3Sn Dipole Demonstrator MDPCT1", in IEEE Transactions on Applied Superconductivity, vol. 30, no. 4, pp. 1-5, June 2020, Art no. 4000805, doi: 10.1109/TASC.2020.2967686.
- <span id="page-7-7"></span>[8] J. Jiang et al., "High-Performance Bi-2212 Round Wires Made With Recent Powders", in IEEE Transactions on Applied Superconductivity, vol. 29, no. 5, pp. 1-5, Aug. 2019, Art no. 6400405, doi: 10.1109/TASC.2019.2895197.
- <span id="page-7-8"></span>[9] A. V. Zlobin, I. Novitski, E. Barzi, and D. Turrioni, "Development of a Bi-2212 Dipole Insert at Fermilab", IEEE Transactions on Applied Superconductivity, vol. 33, pp. 1-5, 2023.
- <span id="page-7-9"></span>[10] I. Novitski, A. V. Zlobin, E. Barzi and D. Turrioni, "Design and Assembly of a Large-Aperture Nb3Sn Cos-Theta Dipole Coil With Stress Management in Dipole Mirror Configuration", in IEEE Transactions on Applied Superconductivity, vol. 33, no. 5, pp. 1-5, Aug. 2023, Art no. 4001405, doi: 10.1109/TASC.2023.3244894.
- <span id="page-7-10"></span>[11] Roxie Home Webpage. ["https://roxie.docs.cern.ch/"](https://roxie.docs.cern.ch/).
- <span id="page-7-11"></span>[12] Stephan Russenschuck, "Field computation for accelerator magnets: Analytical and numerical methods for electromagnetic design and optimization.", 2010, ISBN: 9783527407699.
- [13] K. Couturier, P. Ferracin, E. Todesco, D. Tommasini, W. Scandale, "Elastic modulus measurements of the LHC dipole superconducting coil at 300 K and at 77 K", vol. 613, pp. 377– 382, 2002, AIP Conference Proceedings.
- <span id="page-7-12"></span>[14] Y. Chen, "Mechanical Analysis of the DSB Cross-Section.", SSCL-Preprint-317, May 1993, [https://lss.fnal.gov/archive/other/](https://lss.fnal.gov/archive/other/ssc/sscl-preprint-317.pdf) [ssc/sscl-preprint-317.pdf.](https://lss.fnal.gov/archive/other/ssc/sscl-preprint-317.pdf)
- [15] David R. Smith and F. R. Fickett, "Low-Temperature Properties of Silver", Vol. 100, Number 2, March–April 1995, Journal of Research of the National Institute of Standards and Technology.
- [16] H. Schneider, J. Schreuer, B. Hildmann, "Structure and Properties of Mullite - A Review.", vol. 12, pp. 329-344, December 2008, Journal of the European Ceramic Society.
- <span id="page-7-13"></span>[17] P. Li, Y. Wang, A. Godeke, L. Ye, G. Flanagan and T. Shen, "Thermal-Mechanical Properties of Epoxy-Impregnated Bi-2212/Ag Composite.", in IEEE Transactions on Applied Superconductivity, vol. 25, no. 3, pp. 1-4, June 2015, Art no. 8400904, doi: 10.1109/TASC.2014.2376178.