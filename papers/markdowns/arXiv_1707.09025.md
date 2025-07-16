# arXiv:1707.09025

**Paper ID:** bf8626238a353ecfec24bdcc3b18ee1f

**PDF Path:** apl/Superconductors/arXiv:1707.09025.pdf

**Processing Status:** complete

**Captions Added:** 11

**Generated:** 2025-06-24T13:44:27.165056

---

# **SRF THEORY DEVELOPMENTS FROM THE CENTER FOR BRIGHT BEAMS**<sup>∗</sup>

D. B. Liarte† , T. Arias, D. L. Hall, M. Liepe, J. P. Sethna, N. Sitaraman, Cornell University, Ithaca, NY, United States of America A. Pack, M. K. Transtrum, Brigham Young University, Provo, UT, USA

#### *Abstract*

We present theoretical studies of SRF materials from the Center for Bright Beams. First, we discuss the effects of disorder, inhomogeneities, and materials anisotropy on the maximum parallel surface field that a superconductor can sustain in an SRF cavity, using linear stability in conjunction with Ginzburg-Landau and Eilenberger theory. We connect our disorder mediated vortex nucleation model to current experimental developments of Nb3Sn and other cavity materials. Second, we use time-dependent Ginzburg-Landau simulations to explore the role of inhomogeneities in nucleating vortices, and discuss the effects of trapped magnetic flux on the residual resistance of weakly- pinned Nb3Sn cavities. Third, we present first-principles density-functional theory (DFT) calculations to uncover and characterize the key fundamental materials processes underlying the growth of Nb3Sn. Our calculations give us key information about how, where, and when the observed tin-depleted regions form. Based on this we plan to develop new coating protocols to mitigate the formation of tin depleted regions.

#### **INTRODUCTION**

The fundamental limit to the accelerating *E*-field in an SRF cavity is the ability of the superconductor to resist penetration of the associated magnetic field *H* (or equivalently *B*). SRF cavities are routinely run at peak magnetic fields above the maximum field *H*c<sup>1</sup> sustainable in equilibrium; there is a metastable regime at higher fields due to an energy barrier at the surface [\[2\]](#page-4-0). *H*sh marks the stability threshold of the Meissner state. In Fig. [1](#page-0-0) we show results from linear stability analysis [\[14\]](#page-4-1), valid near *T*c, for *H*sh as a function of the Ginzburg-Landau parameter κ, the ratio λ/ξ of the London penetration depth λ to the coherence length ξ. Niobium has κ <sup>≈</sup> <sup>1</sup>.5, most of the promising new materials have large κ. At lower temperatures, one must move to more sophisticated Eliashberg theories [\[4\]](#page-4-2), for which *H*sh is known analytically for large κ; numerical studies at lower κ are in progress [\[3\]](#page-4-3). Broadly speaking, the results so far for isotropic materials appear similar to those of Ginzburg-Landau.

This manuscript will briefly summarize theoretical work on *H*sh (the threshold of vortex penetration and hence the quench field). First, we discuss the effect of materials anisotropy on *H*sh [\[12\]](#page-4-4). Second, we discuss theoretical estimates of the effect of disorder [\[13\]](#page-4-5), and preliminary unpub-

<span id="page-0-0"></span>![](_page_0_Figure_9.jpeg)

**Caption:** Numerical estimate of the superheating field in Ginzburg-Landau theory across varying κ values. Depicted are the Ginzburg-Landau results (black solid line), large-κ expansion (red dashed line), and small-κ Padé approximation (blue dotted line), providing comprehensive insights into how superheating fields adapt with material property variations.


Figure 1: From ref. [\[14\]](#page-4-1), showing a numerical estimate of *H*sh in Ginzburg-Landau theory over many orders of magnitude of κ (black solid line), along with a large-κ expansion (red dashed line), and a Padé approximation for small κ (blue dotted-dashed line).

lished simulations of the effects of surface roughness and materials inhomogeneity. Third, we discuss key practical implications of theoretically calculated point defect energies, interactions, relaxation times, and mobilities in the promising new cavity material Nb3Sn. Finally, some magnetic flux is trapped in cavities during the cooldown phase, and the response of these flux lines to the oscillating external fields appears to be the dominant source of dissipation in modern cavities. We model potentially important effects of multiple weak-pinning centers on this dissipation due to trapped flux.

# **THE EFFECT OF MATERIALS ANISOTROPY ON THE MAXIMUM FIELD**

Some of the promising new materials are layered, with strongly anisotropic superconducting properties (MgB<sup>2</sup> and the pnictides, for example, but not Nb3Sn or NbN). Fig. [2](#page-1-0) illustrates an anisotropic vortex (magnetized region blue, vortex core red) penetrating into the surface of a superconductor (grey). The anisotropy here is characteristic of MgB<sup>2</sup> at low temperatures, except that the vortex core is expanded by a factor of 30 to make it visible.

Near *T*c, we find in ref. [\[12\]](#page-4-4) that a simple coordinate change and rescaling maps the anisotropic system onto the isotropic case (Fig. [1](#page-0-0) above, as studied in ref. [\[14\]](#page-4-1)). We find, near *T*<sup>c</sup> where Ginzburg-Landau theory is valid, that *<sup>H</sup>*sh is nearly isotropic for large <sup>κ</sup> materials (Fig. [3.](#page-1-1) At lower temperatures, different heuristic estimates of the effects of anisotropy on *H*sh yield conflicting results. Further work at lower temperatures could provide valuable insight into the

<sup>∗</sup> This work was supported by the US National Science Foundation under Award OIA-1549132, the Center for Bright Beams.

<sup>†</sup> dl778@cornell.edu

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

**Caption:** Figure 2 represents the anisotropic nature of vortices in superconducting MgB2. The diagrams show a vortex disk (blue) and its core (red) in the ac plane, highlighting significant anisotropy compared to isotropic models. The core is exaggerated 30 times for visibility. This figure demonstrates the influence of material anisotropy on vortex behavior, crucial for designing and optimizing superconducting materials and SRF cavities.


Figure 2: From ref. [\[13\]](#page-4-5), showing vortex (blue disk) and vortex core (red disk) of zero-temperature MgB<sup>2</sup> in the *ac* plane, with the external magnetic field parallel to the normal of the plane of the figure. We have drawn the core region about 30 times larger with respect to the penetration depth, so that the core becomes discernible.

possible role of controlled surface orientation for cavities grown from these new materials.

<span id="page-1-1"></span>![](_page_1_Figure_3.jpeg)

**Caption:** The phase diagram of anisotropic superconductors illustrates the dependence of superheating field anisotropy on the Ginzburg-Landau parameters. The diagram divides the behavior into Type I and Type II regions, with mixed behavior in between. Key superconducting materials like YBCO, BiSCCO, NbSe2, and MgB2 are plotted, showcasing varying degrees of anisotropy. This aids in categorizing material behavior and selecting suitable materials for high-performance applications.


Figure 3: From ref. [\[12\]](#page-4-4), showing the phase diagram of anisotropic superconductors in terms of mass anisotropy and GL parameters. The shaded blue and orange regions correspond to regions where the superheating field anisotropy can be approximated by γ 1/2 and 1, respectively, within 10% of accuracy. Note that the superheating field of MgB<sup>2</sup> is nearly isotropic near *T* = *T*c.

# **DISORDER-MEDIATED FLUX ENTRY AND MATERIALS ANISOTROPY**

Defect regions and inhomogeneity of superconductor properties can weaken the performance of SRF cavities. In ref. [\[12\]](#page-4-4) we used simple estimates based on Bean and Livingston's energy barrier arguments [\[2\]](#page-4-0), to estimate the effects of disorder in lowering *H*sh by providing flaws that lower the barrier to vortex penetration. Here we use these calculations to shed light about the relationship between tin depleted regions, low critical temperature profiles, defect sizes and quench fields.

Consider an external magnetic field *B*, parallel to the surface of a semi-infinite superconductor occupying the halfspace *<sup>x</sup>* <sup>&</sup>gt; 0. If *<sup>B</sup>* is larger than the lower critical field *<sup>B</sup>*c<sup>1</sup> (and smaller than *B*c2), the vortex lattice phase is thermodynamically favored. However, if the field is not large enough, a newborn vortex line near the superconductor surface will have to surpass an energy barrier to penetrate the superconductor towards the bulk of the material. This instability typically is surmounted by the simultaneous entry of an entire array of vortices, whose interactions lower one another's barriers. Disorder, in contrast, will lead to a localized region allowing one vortex entry at a time. Bean and Livingston provided simple analytical calculations for the energy barrier felt by one vortex line; we extended their calculation to estimate the dirt needed to reduce this barrier to zero at a quench field *<sup>H</sup>*<sup>q</sup> <sup>&</sup>lt; *<sup>H</sup>*sh.

The new materials have larger κ, and in particular smaller vortex core sizes ξ; naively one would expect vortex penetration when flaws of size ξ arise. Are these new materials far more sensitive to dirt than niobium? Reassuringly, Fig. [4](#page-1-2) shows that the low values of the coherence length do not make these new materials substantially more susceptible to disorder-induced vortex penetration [\[13\]](#page-4-5).

<span id="page-1-2"></span>![](_page_1_Figure_10.jpeg)

**Caption:** Reliability of vortex nucleation in superconductors presented through a model of Gaussian random disorder. The comparison between projection models (3D semicircular and 2D pancake vortex nucleation) and experimental data underscore the robustness of these models in predicting vortex behavior. The findings are central to enhancing the understanding of disorder-driven phenomena in superconductors.


Figure 4: From ref. [\[13\]](#page-4-5), showing the reliability of vortex nucleation, in a simple model of Gaussian random disorder, for three candidate superconductors. Solid curves are for a 3D semicircular vortex barrier model; dashed curves are for 2D pancake vortex nucleation in a 2D superconducting layer.

We can use our model to estimate the suppressed superconducting transition temperature *T* min c and the flaw depth *D*<sup>c</sup> needed to allow vortex penetration, as a function of *H*<sup>q</sup> (or, in Tesla, *B*q) (Fig. [5\)](#page-2-0). For Nb3Sn we find a flaw size of *D*<sup>c</sup> ∼ 100nm and *T* min <sup>c</sup> ∼ 12K can allow vortex nucleation and quenches at fields of ∼ 77mT (Fig. [5\)](#page-2-0), consistent with experimental results [\[8\]](#page-4-6).

# **TIME-DEPENDENT GINZBURG-LANDAU SIMULATIONS OF ROUGH SURFACES AND DISORDER**

To quantify the dependence of *H*sh on surface roughness and disorder, we have developed a time-dependent Ginzburg-Landau simulation. Fig. [6](#page-2-1) shows the density <sup>|</sup>ψ<sup>|</sup> <sup>2</sup> of superconducting electrons at a field just above *H*sh(top left), showing the entry of several vortices for a 2D system with an irregular surface. On the bottom left, we show the corresponding

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

**Caption:** Figure 5 illustrates the critical temperature profile necessary for vortex nucleation in Nb3Sn superconducting cavities at approximately 77 mT magnetic fields. Subfigure (a) shows how the critical temperature (Tc) varies with depth (x) in nanometers, while subfigure (b) displays the relationship between the minimal superconducting transition temperature (Tmin c) and flaw depth (Dc) as they change with quench field (B). These findings are pivotal in understanding the conditions under which vortices lead to quenching in these materials, thereby influencing the design of robust superconducting surfaces resistant to quenching.


Figure 5: (a) Critical temperature profile that allow nucleation of vortices in Nb3Sn cavities at a field of ∼ 77mT. (b) Suppressed superconducting transition temperature *T* min c (black), and flaw depth *D*<sup>c</sup> (red), as a function of the quench field.

supercurrent **j**; on the top right we show the magnetic field *H* (perpendicular to the plane of the simulation), and on the bottom right we show the effect of surface roughness on <sup>|</sup>ψ(θ)|<sup>2</sup> around the perimeter. Our initial results quantify how inward-curving regions in the plane perpendicular to the applied field on the perimeter can act as vortex nucleation sites in this geometry. An open question remains what the effect of curvature and surface roughness have when oriented parallel to the applied field.

The effect of roughness in Fig. [6](#page-2-1) is to lower *H*sh by a few percent. By systematically varying the details of the roughness parameters, we can use this tool to identify at what scale roughness will have significant impact on vortex nucleation. SRF cavity roughness can be smoothed to varying degrees. Our TDGL environment can be used to find dangerous regimes or configurations that can have serious consequences for cavity performance.

We can also use this tool as a way to explore vortex dynamics and the effects of pinning sites on trapped residual magnetic flux. Pinning sites originate from inhomogeneities in the material, such as grain boundaries or spatial inhomogeneities in the alloy stoichiometry. By incorporating this information into our TDGL environment we can try to better understand the mechanisms driving residual resistance for typical cavities.

<span id="page-2-1"></span>![](_page_2_Figure_5.jpeg)

**Caption:** This visualization covers the spatial dependence of superconducting electron density, supercurrent, induced magnetic field, and order parameter variations around an irregularly surfaced Nb3Sn cavity. It demonstrates the nuanced interplay of geometric features and electromagnetic properties in determining vortex nucleation sites, a key factor in optimizing cavity design and performance.


Figure 6: Spatial dependence of the density of superconducting electrons (top left), supercurrent (bottom left), and the induced magnetic field (top right). On the bottom right, we show the variation of the order parameter around the perimeter of the superconductor.

#### **DFT CALCULATIONS**

Nb3Sn cavities are created by depositing tin vapor on the surface of a niobium cavity, which reacts with the niobium to form an irregular surface layer of the compound. Of interest are regions of "tin depleted" Nb3Sn, known to have a lower superconducting transition temperature than the surrounding Nb3Sn. These regions may be the nucleation centers responsible for quenches observed well below *H*sh expected for perfect Nb3Sn [\[8\]](#page-4-6).

Density functional theory (DFT) can be used to study layer growth, tin depletion, and other features of Nb3Sn layers at the single-particle level. This information, combined with experimental data and accounting for the effects of grain boundaries and strain, makes it possible to build a multiscale model of layer growth.

<span id="page-2-2"></span>![](_page_2_Picture_10.jpeg)

**Caption:** Illustration of antisite disorder in Nb3Sn layers. Here, it is estimated that approximately 1% of lattice sites have antisite defects, which are likely due to high temperatures during coating processes. This defect is surmised to be the most common form of point defect in Nb3Sn, potentially affecting lattice structure, electron mean free paths, and contributing to collective weak pinning.


Figure 7: Illustration of antisite disorder. We estimate that on the order of 1% of lattice sites are affected by antisite defects "frozen in" from the high coating temperature. This would make them by far the most common point defect in Nb3Sn layers.

Our initial work uses in-house DFT software to calculate defect formation and interaction energies, impurity energies, and energy barriers in Nb3Sn. We have found that antisite disorder (Figure [7\)](#page-2-2), rather than impurities or vacancies, likely sets the electron mean free path in Nb3Sn and may

also be responsible for collective weak pinning. We have also found that under certain conditions during growth, it is energetically favorable for Nb3Sn to form at tin-depleted stoichiometry, while during annealing existing Nb3Sn near the surface or grain boundaries can become tin-depleted by diffusion (Figure [8\)](#page-3-0). Either or both of these tin depletion mechanisms may result in quench nucleation centers; by understanding them we can for the first time make informed modifications to the coating process in an attempt to limit tin depletion and produce better cavities.

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

**Caption:** This figure depicts spatially resolved measurements of tin depletion on a Nb3Sn superconducting layer. The image on the left shows a broad view with tin density levels depicted by a green color gradient, indicating regions of significant tin depletion. The right-hand images offer close-ups of specific areas with around 1-2% tin depletion close to the surface. This figure underscores the impact of tin depletion on superconducting properties, potentially serving as nucleation sites for vortex penetration and influencing the superconducting transition temperature and performance of SRF cavities.


Figure 8: Experimental data showing tin depletion. At left is a tin density map of a layer cross section showing regions of significant (7-8%) tin depletion, in this case mostly deep in the layer relative to the RF penetration depth (dashed line). At right are close ups showing slight (1-2%) tin depletion right at the surface of the layer. Images by Thomas Proslier at Argonne National Lab, received via personal communication with Daniel Hall.

# **DYNAMICS OF TRAPPED VORTICES; POTENTIAL ROLE OF WEAK PINNING**

When the field is high enough for penetration of new vortices, one expects a cascade of vortices leading to a quench. Vortices trapped during the cooling process, while not immediately fatal, do act as sources of residual resistance. Experiments show that the non-BCS surface resistance is proportional to the trapped flux, both for nitrogen-doped Nb cavities [\[6\]](#page-4-7) and for Nb3Sn [\[9\]](#page-4-8). This suggests that trapped vortices may be a dominant contribution to the quality factor of the cavity.

Previous studies of the residual resistance due to a trapped flux line [\[7\]](#page-4-9) focused on the Bardeen-Stephen viscous dissipation [\[1\]](#page-4-10) of a free line pinned a distance below the surface, as the external field drags the line through a otherwise uniform superconducting medium. Experimental measurements in nitrogen-doped Nb cavities showed good agreement to this theory, except that the distance to the pinning center was presumed to change linearly with the mean-free path [\[6\]](#page-4-7) as it changes due to nitrogen doping. Since nitrogen (or other contaminant gases [\[10,](#page-4-11) [11\]](#page-4-12)) should act as weak pinning centers (with many impurities per coherence length cubed), we have been modeling the role of weak pinning in vortex dissipation.

<span id="page-3-1"></span>![](_page_3_Figure_6.jpeg)

**Caption:** Vortex line solutions in Nb3Sn cavities shown over time, highlighting the dynamics under measured parameters. The figure illustrates the classical depinning transition where defects act as a random potential requiring a critical force per unit length to depin. This dynamic is crucial for understanding and mitigating dissipation mechanisms in superconducting devices like Nb3Sn cavities.


Figure 9: From ref. [\[9\]](#page-4-8), showing vortex line solutions at several times, using measured parameters for Nb3Sn.

Line defects pulled through a disordered medium is one of the classical depinning transitions [\[5\]](#page-4-13). The disorder acts as a random potential, and macroscopically there is a threshold force per unit length *f*pin needed to depin the line and start motion (Fig. [9\)](#page-3-1). This depinning transition is preceded by avalanches of all sizes (local regions of vortex motion) and followed by fluctuations on all scales (jerky motion of the vortex line in space and time). For our initial estimates, we have ignored these fluctuations, using a 'mean-field' model where our superconducting vortex line has a threshold supercurrent *j*<sup>d</sup> ∝ *f* 2/3 pin for motion. We presume also that the energy dissipated is *f*pin times the area swept out by the vortex as the external surface field pulls it to and fro (Fig. [10\)](#page-3-2).

<span id="page-3-2"></span>![](_page_3_Figure_9.jpeg)

**Caption:** Illustration of a vortex line subjected to external RF magnetic fields and interacting with various pinning centers. This figure models the balance between applied field magnitude and pinning center concentration, pivotal in understanding how these factors dictate residual resistance within superconducting materials like Nb3Sn.


Figure 10: Illustration of a vortex line (red curve) subject to an external rf magnetic field and the collective action of many pinning centers.

The residual resistance measured in Nb3Sn cavities shows a linear dependence on the peak RF field (Fig. [11,](#page-4-14) [\[9\]](#page-4-8)). The scaling properties of the terms included in the earlier work [\[7\]](#page-4-9) all predict no dependence on the strength of the external oscillating field. Our theory including weak pinning but ignoring the viscous dissipation produces a dissipation that is linear in this external field. Our estimates, however, suggest that our theory should be valid at MHz frequencies, but at the operating GHz frequencies the viscous term must be important for the energy dissipation. Our preliminary

calculations suggest that incorporating both can provide a reasonable explanation of the experimental data, but we still do not obtain quantitative agreement.

<span id="page-4-14"></span>![](_page_4_Figure_1.jpeg)

**Caption:** This graph examines the relationship between sensitivity to trapped magnetic flux and peak surface RF magnetic field. Data is split into two types: external field data (blue markers) and thermal gradient data (red markers). The notable trend observed is a linear increase in sensitivity as the RF magnetic field strength increases, indicating the pivotal role of flux management in optimizing cavity performance.


Figure 11: From ref. [\[9\]](#page-4-8), showing the sensitivity of residual resistance to trapped magnetic flux, as a function of the peak rf field.

#### **CONCLUSION**

The collaboration between scientists inside and outside traditional accelerator physicists made possible by the Center for Bright Beams has been immensely fruitful. This proceedings illustrates the richness of the science at the intersection of accelerator experimentalists working on SRF cavities with condensed-matter physicists with interests in continuum field theories and *ab-initio* electronic structure calculations of materials properties. (One must also note the important contributions of experimental condensed matter physicists in the collaboration.) Current SRF cavities are pushing fundamental limits of superconductors, and are a source of fascinating challenges for theoretical condensedmatter physics. Conversely, we find that theoretical calculations are remarkably fruitful in guiding and interpreting experimental findings.

#### **ACKNOWLEDGMENT**

We thank Alex Gurevich for useful conversations.

#### **REFERENCES**

- <span id="page-4-10"></span>[1] J. Bardeen, and M. J. Stephen, "Theory of the Motion of Vortices in Superconductors" *Phys. Rev.*, vol. 140, p. A1197, 1965.
- <span id="page-4-0"></span>[2] C. P. Bean, J. D. Livingston, "Surface Barrier in Type-II Superconductors" *Phys. Rev. Lett.*, vol. 12, p. 14, 1964.
- <span id="page-4-3"></span>[3] G. Catelani, M. K. Transtrum, and J. P. Sethna, Unpublished.
- <span id="page-4-2"></span>[4] G. Catelani, and J. P. Sethna, "Temperature dependence of the superheating field for superconductors in the high-κ London limit" *Phys. Rev. B*, vol. 78, p. 224509, 2008.
- <span id="page-4-13"></span>[5] D. S. Fisher, "Collective transport in random media: from superconductors to earthquakes" *Phys. Rep.*, vol. 301, p. 113, 1998.
- <span id="page-4-7"></span>[6] D. Gonnella, J. Kaufman, and M. Liepe, "Impact of nitrogen doping of niobium superconducting cavities on the sensitivity of surface resistance to trapped magnetic flux" *J. Appl. Phys.*, vol. 119, p. 073904, 2016.
- <span id="page-4-9"></span>[7] A. Gurevich, and G. Ciovati, "Effect of vortex hotspots on the radio-frequency surface resistance of superconductors" *Phys. Rev. B*, vol. 87, p. 054502, 2013.
- <span id="page-4-6"></span>[8] D.L. Hall, P. Cueva, D. Liarte, M. Liepe, J.T. Maniscalco, D.A. Muller, R.D. Porter, and J.P. Sethna, "Quench Studies in Single-Cell Nb3Sn Cavities Coated Using Vapour Diffusion", in *Proc. 8th Int. Particle Accelerator Conf. (IPAC'17)*, Copenhagen, Denmark, May 2017, paper MOPVA116, pp. 1119–1122, [http://jacow.org/](http://jacow.org/ipac2017/papers/mopva116.pdf) [ipac2017/papers/mopva116.pdf](http://jacow.org/ipac2017/papers/mopva116.pdf), [https://doi.org/](https://doi.org/10.18429/JACoW-IPAC2017-MOPVA116) [10.18429/JACoW-IPAC2017-MOPVA116](https://doi.org/10.18429/JACoW-IPAC2017-MOPVA116), 2017.
- <span id="page-4-8"></span>[9] D.L. Hall, D. Liarte, M. Liepe, and J.P. Sethna, "Impact of Trapped Magnetic Flux and Thermal Gradients on the Performance of Nb3Sn Cavities", in *Proc. 8th Int. Particle Accelerator Conf. (IPAC'17)*, Copenhagen, Denmark, May 2017, paper MOPVA118, pp. 1127–1129, [http://jacow.org/](http://jacow.org/ipac2017/papers/mopva118.pdf) [ipac2017/papers/mopva118.pdf](http://jacow.org/ipac2017/papers/mopva118.pdf), [https://doi.org/](https://doi.org/10.18429/JACoW-IPAC2017-MOPVA118) [10.18429/JACoW-IPAC2017-MOPVA118](https://doi.org/10.18429/JACoW-IPAC2017-MOPVA118), 2017.
- <span id="page-4-11"></span>[10] P. N. Koufalis, D. L. Hall, M. Liepe, and J. T. Maniscalco, "Effects of interstitial carbon and oxygen on niobium superconducting cavities" *arXiv:1612.08291*, 2016.
- <span id="page-4-12"></span>[11] P. N. Koufalis, M. Liepe, and J. T. Maniscalco, "Low temperature doping of niobium cavities: what is really going on?", in *Proc. of the 18th Int. Conf. on RF Superconductivity*, Beijing, China, 2017.
- <span id="page-4-4"></span>[12] D. B. Liarte, M. K. Transtrum, and J. P. Sethna, "Ginzburg-Landau theory of the superheating field anisotropy of layered superconductors" *Phys. Rev. B*, vol. 94, p. 144505, 2016.
- <span id="page-4-5"></span>[13] D. B. Liarte, S. Posen, M. K. Transtrum, G. Catelani, M. Liepe and J. P. Sethna, "Theoretical estimates of maximum fields in superconducting resonant radio frequency cavities: stability theory, disorder, and laminates", *Supercond. Sci. Technol.*, vol. 30, p. 033002, 2017.
- <span id="page-4-1"></span>[14] M. K. Transtrum, G. Catelani and J. P. Sethna, "Superheating field of superconductors within Ginzburg-Landau theory" *Phys. Rev. B.*, vol. 83, p. 094505, 2011.