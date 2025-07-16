# arXiv:2003.03362

**Paper ID:** ea07e21280110097ee04af797535686b

**PDF Path:** apl/Superconductors/arXiv:2003.03362.pdf

**Processing Status:** complete

**Captions Added:** 16

**Generated:** 2025-06-24T13:44:27.353261

---

# Analysis of Magnetic Vortex Dissipation in Sn-Segregated Boundaries in Nb3Sn Superconducting RF Cavities

Jared Carlson,[∗](#page-11-0) Alden Pack, and Mark K. Transtrum[†](#page-11-1)

Department of Physics and Astronomy, Brigham Young University, Provo, Utah 84602, USA

Jaeyel Lee[∗](#page-11-0) and David N. Seidman

Department of Materials Science and Engineering, Northwestern University, Evanston, Illinois 60201, USA

Danilo B. Liarte,[∗](#page-11-0) Nathan Sitaraman,[∗](#page-11-0) Alen Senanian, Michelle Kelley, James P. Sethna, and Tomas Arias Cornell University, Ithaca, New York 14853, USA

Sam Posen

Fermi National Accelerator laboratory, Batavia, Illinois 60510, USA (Dated: November 2, 2021)

We study mechanisms of vortex nucleation in Nb3Sn Superconducting RF (SRF) cavities using a combination of experimental, theoretical, and computational methods. Scanning transmission electron microscopy (STEM) image and energy dispersive spectroscopy (EDS) of some Nb3Sn cavities show Sn segregation at grain boundaries in Nb3Sn with Sn concentration as high as ∼35 at.% and widths ∼3 nm in chemical composition. Using ab initio calculations, we estimate the effect excess tin has on the local superconducting properties of the material. We model Sn segregation as a lowering of the local critical temperature. We then use time-dependent Ginzburg-Landau theory to understand the role of segregation on magnetic vortex nucleation. Our simulations indicate that the grain boundaries act as both nucleation sites for vortex penetration and pinning sites for vortices after nucleation. Depending on the magnitude of the applied field, vortices may remain pinned in the grain boundary or penetrate the grain itself. We estimate the superconducting losses due to vortices filling grain boundaries and compare with observed performance degradation with higher magnetic fields. We estimate that the quality factor may decrease by an order of magnitude (10<sup>10</sup> to 10<sup>9</sup> ) at typical operating fields if 0.03% of the grain boundaries actively nucleate vortices. We additionally estimate the volume that would need to be filled with vortices to match experimental observations of cavity heating.

#### I. INTRODUCTION

Superconducting Radio-Frequency (SRF) cavities are used in accelerators to transfer energy to beams of charged particles. Oscillating electric fields induce magnetic fields parallel to the surface of the cavity. However, for very large induced magnetic fields, the superconducting Meissner state is unstable, so that a fundamental limit to the cavity's performance is the material's ability to expel large magnetic fields. For type-II materials, complete flux expulsion is thermodynamically stable up to a lower-critical field, Hc<sup>1</sup>, and a mixed state characterized by superconducting vortices is stable for fields up to an upper-critical field, Hc<sup>2</sup>. Thus, by limiting the fields on the walls of the SRF cavities, the superconductor can be kept in the flux-free Meissner state, so that surface dissipation is extremely small and quality factors ∼ 10<sup>10</sup> can be achieved. For magnetic fields parallel to the cavity surface, the superconducting Meissner state can be maintained above the stability limit in a metastable state up to a limit (for an ideal surface) of the so-called superheating field Hsh [1](#page-11-2) . Hsh has been studied extensively by the condensed matter community, primarily in the context of Ginzburg-Landau theory at ideal interfaces[2](#page-11-3)[–5](#page-11-4). Because high-power SRF cavities routinely operate in the

metastable regime[6](#page-11-5) , there has been renewed interested by the condensed matter community in the behavior of superconductors in large magnetic fields. Calculations extend results to the semi-classical theory of Eilenberger theory in both the clean[7](#page-11-6) and dirty[8](#page-11-7) limits and Time-Dependent Ginzburg Landau (TDGL) simulations that account for material[9](#page-11-8) and spatial inhomogeneities[10–](#page-11-9)[13](#page-11-10) . In this paper, we explore the role of grain boundaries (GBs) in SRF cavity performance motivated by experimental observations of inhomogeneities in real-world SRF cavities. It is well-established that grain boundaries can strongly affect the superconducting properties of Nb3Sn. They are known to pin vortices and determine critical currents in Nb3Sn wires[14,](#page-11-11)[15](#page-11-12). Compositional variation within Nb3Sn grain-boundaries on the scale of a few nanometers have been observed[16,](#page-11-13)[17](#page-11-14). This study brings together the expertise of many areas of condensed matter and accelerator physics to explore fundamental physics of superconductors in extreme conditions and connect those results to real systems.

Recently there has been significant progress towards the employment of Nb3Sn in SRF cavities as a higher performance alternative to the industry standard Nb for next generation particle accelerator applications[18,](#page-11-15)[19](#page-11-16) . Nb3Sn cavities are prepared with Nb3Sn films ∼2 µm (nearly 20 penetration depths) thick coated on the surface of Nb cavities using the Sn vapor-diffusion process. Nb3Sn is an intermetallic alloy with A15 crystal structure; it is a promising material for next-generation SRF cavities in large part because of its large (predicted) superheating field (∼400 mT). It also has a higher critical temperature (Tc ∼18K), making possible a higher quality factor (Q0, another critical metric of cavity performance) at a given temperature compared to Nb (Tc ∼9K).

In practice, however, real world Nb3Sn cavities quench well below the theoretically predicted value. The maximum accelerating gradient that has been achieved within these cavities is about 24 MV/m, which corresponds to the surface magnetic field of ∼98 mT. These cavities exhibit a high Q<sup>0</sup> of ∼ 10<sup>10</sup> at 4.2 K[18](#page-11-15)? ; however, in some cavities, Q<sup>0</sup> begins to degrade significantly before the limiting quench field is reached, a phenomenon described as "Q-slope"[21](#page-11-17) .

Understanding the mechanism underlying the Q-slope phenomenon is an important question for cavity development. Several mechanisms have been proposed in terms of imperfections in the Nb3Sn coatings[6](#page-11-5)[,22,](#page-11-18)[23](#page-11-19) such as thin grains[24](#page-12-0)[,25](#page-12-1) and Sn-deficient regions[26](#page-12-2). Another potential mechanism that may have detrimental effects on the performance of Nb3Sn cavities is Sn segregation at grain boundaries[27](#page-12-3). In some Nb3Sn coatings, tin concentrations as high as ∼35 at.% have been observed in grain boundaries with the segregated zone extending by as much as ∼3 nm, comparable to the coherence length of Nb3Sn (∼3 nm). Because of the inferior superconducting properties, magnetic flux may penetrate the segregated region, degrade Q0, and lead to premature quench.

In support of this hypothesis, witness samples coated with high-performance (quench at ∼24 MV/m with Q∼10<sup>10</sup> at 4.4 K) Nb3Sn cavities at Fermilab did not show Sn segregation at the grain boundaries in energy dispersive X-ray spectroscopy (EDS) and in scanning transmission electron microscopy (STEM). Similarly, a direct cutout from a high-performance Nb3Sn cavity fabricated at Cornell did not show Sn segregation at grain boundaries within the detection limit of STEM-EDS. In contrast, Nb3Sn cavities, which show Sn segregation at grain boundaries in witness samples coated together with the cavities, displayed negative Q-slope for accelerating fields in the 5-15 MV/m range, see Fermilab cavity 1 and 2, Fig. [1.](#page-1-0) These experimental results, summarized in Fig. [1,](#page-1-0) suggest a potential link between Sn segregation at grain boundaries and cavity performance[27](#page-12-3) .

Experimentally, it is difficult to isolate the effects of Sn segregation at grain boundaries from other imperfections, such as Sn-deficient regions and surface roughness. To overcome these challenges, we use numerical tools to theoretically understand the role of segregation in grain boundaries for SRF cavity performance. We use density functional theory to estimate the effective T<sup>c</sup> of the material within the segregation zone. Next, we use time-dependent Ginzburg-Landau simulations with spatial varying material properties motivated by the ab ini-

![](_page_1_Figure_5.jpeg)

**Caption:** Figure 1 presents a Q0 vs Eacc (accelerating field) plot for several Nb3Sn SRF cavities processed at Fermilab and Cornell. The figure contrasts the behaviors observed between cavities with and without significant Sn segregation at grain boundaries, plotted as different symbols. Sn segregation correlates to noticeable Q-slopes in Fermilab Cavity 1 and 2, showing steep decreases in Q0 starting around 8 MV/m. These results substantiate the relevance of Sn segregation as a critical factor adversely impacting cavity performance.


<span id="page-1-0"></span>FIG. 1. Q vs E curve of various Nb3Sn SRF cavities coated at Fermilab and Cornell. The GBs of a witness sample (Fermilab Cavity 3) and direct cutout (Cornell Cavity 2) of a high-performance cavity are characterized in STEM-EDS and showed no Sn segregation at GBs. On the other hand, witness samples of Fermilab Cavity 1 and 2, which show Q-slope starting at 8 MV/m, were found to have Sn segregation at GBs (reprint from[27](#page-12-3)).

tio DFT calculations. Time-dependent Ginzburg-Landau theory allows us to conduct numerical experiments on a mesoscale that probe the role of grain boundaries and segregated zones for vortex nucleation, pinning, and quenching. Finally, motivated by our TDGL simulations, we estimate power dissipated by vortex nucleation within segregated grain boundaries during an RF cycle and make quantitative comparisons to actual SRF cavities.

This paper is organized as follows: First, we present experimental and first-principles analyses of grainboundary defect segregation in Nb3Sn cavities in section [II.](#page-2-0) We then report on first principles DFT calculations of superconducting properties for segregation zones in section [III](#page-2-1) and time-dependent Ginzburg-Landau simulations of flux penetration in section [IV.](#page-4-0) We estimate the effect on cavity performance in section [V.](#page-6-0) Our numerical experiments isolate the effects of Sn-segregated grain boundaries from potentially confounding mechanisms. Our results indicate that the effects of Sn-segregated grain boundaries alone are consistent with observed behaviors. Specifically, grain boundaries may nucleate and then trap a limited number of vortices, leading to degradation in the cavity's quality factor. We conclude by discussing the implications of these results for cavity development and further theoretical studies in section [VI.](#page-10-0)

## <span id="page-2-0"></span>II. EXPERIMENTAL AND THEORETICAL STUDY OF NB3SN DEFECT SEGREGATION

We have systematically investigated the composition of GBs in Nb3Sn coatings and the origin of Sn-segregation at GBs using experimental and first-principles methods. The high angle annular dark field (HAADF)-STEM image in Figure [2](#page-2-2) displays a Nb3Sn coating on Nb prepared by the Sn vapor diffusion process using coating parameters from the early stage of the development of Nb3Sn films at Fermilab[18](#page-11-15). EDS mapping is performed across a GB in a Nb3Sn coating prepared by the same coating parameters and reveals that Sn is segregated at the GB, Figure [3.](#page-3-0) The maximum concentration of Sn at the GB is ∼33 at.% and the width of the Sn segregated region is ∼3 nm. The Gibbsian interfacial excess of Sn is ∼16 atom/nm<sup>2</sup> . Previous analyses of the structures of Snsegregated GBs in Nb3Sn employing HR-STEM and firstprinciple calculations indicated that most of the segregated Sn exist as Sn-antisite defects near the GBs rather than forming Sn-rich phases such as Nb6Sn<sup>5</sup> or other non-equilibrium phases[27](#page-12-3)[,28](#page-12-4). Another GB from a witness sample of a high-performance cavity, which is prepared by the coating procedure with less Sn and higher furnace temperature than the previous coating to mitigate the Sn-segregation at GBs[27](#page-12-3), is characterized by HR-STEM EDS, Fig. [4.](#page-3-1) It is noted that there is no Sn segregation at the GB within the detection limit of EDS (∼1 at.%). This indicates a possible correlation between the Sn segregation at GBs and cavity performance, and motivates us to investigate the effects of Sn-segregation on GB properties using DFT and Ginzburg-Landau calculations.

To investigate the origin of the Sn-segregation in the Nb3Sn coating, we consider our previous experimental study of Nb3Sn GBs using STEM-EDS and APT analyses[27](#page-12-3). Nine batches of coatings were characterized by randomly selecting 2-4 GBs as representatives for each coating and analyzing them. Our results show that the effect of the GB structure (characterized by five degrees of freedom given by the rotation axis (c), the disorientation angle (θ), and the normal vector to the GB plane (n)) on Sn-segregation is insignificant, which is in contrast with the usual behavior of GB segregation behaviors of alloys in an equilibrium state[29](#page-12-5)[,30](#page-12-6). Even small-angle GB reveal a similar amount of Sn-segregation at GBs to high-angle GBs nearby. In parallel, we performed a detailed DFT study surveying a variety of GB structures in Nb3Sn including both symmetric tilt and twist GBs, identifying a number of low energy structures[31](#page-12-7). We found a rapid decay of local strain while moving away from the GB, and a disruption to the electronic structure that decays to bulk behavior ∼1–1.5 nm from the GB. Importantly, this disruption reduces the Fermi-level density-of-states by more than a factor of 2 in the core region. Associated with this density-of-states reduction, we also found a strong binding of tin on niobium antisite defects to GBs that extends outward to the same distances of ∼1–1.5 nm.

![](_page_2_Picture_3.jpeg)

**Caption:** This image illustrates a high-angle annular dark field scanning transmission electron microscopy (HAADF-STEM) image of a typical Nb3Sn coating of approximately 2 µm thickness on niobium (Nb), deposited via a Sn vapor-diffusion process. The visualization highlights the morphological uniformity and integrity of the coating, which is critical for assessing the performance capabilities of Nb3Sn in SRF cavities. The image plays a crucial role in understanding the microstructural basis affecting Nb3Sn's superconducting properties and provides a baseline for comparing different coating batches and their superconductive performance .


FIG. 2. HAADF-STEM image of a typical ∼2 µm thick Nb3Sn coating on Nb prepared by Sn vapor-diffusion.

<span id="page-2-2"></span>Based on these results, we consider both kinetic and thermodynamic factors that may contribute to Snsegregation at GBs. For the kinetic factor, we considered the Sn-diffusion process along GBs, and it is found that the amount of Sn-segregation at GBs in Nb3Sn is affected by Sn-flux during the coating process: higher Snflux causes a more significant amount of Sn-segregation at GBs. This suggests that the kinetic Sn GB diffusion process plays a significant role in determining the chemical composition of GB. For the thermodynamic factor, the attractive electronic interaction between Sn-antisite defects and GBs may also play an important role, as we find that this interaction has a range similar to the observed radius of Sn-segregation, and depends only weakly on the structure of the GB. The current coating procedures at Fermilab and Cornell successfully mitigates the Sn-segregation at GBs by annealing without the external Sn-source[32](#page-12-8) .

Finally, we observe divots in the Nb3Sn surface at GBs; the HAADF-STEM image in Fig. [5](#page-3-2) displays the geometry of a GB on the top surface. It has ∼80 nm of depth and ∼420 nm of width. The unique compositional and geometrical changes at GBs may provide pathways for flux to penetrate the surface. This observations motivates us to consider surface geometry as an additional factor in our Ginzburg-Landau modeling of vortex penetration.

## <span id="page-2-1"></span>III. THE EFFECT OF TIN-RICH STOICHIOMETRY ON T<sup>c</sup>

Previous works have demonstrated that both defects and strain can reduce the T<sup>c</sup> of Nb3Sn, and that this effect can be attributed to a reduction of the Fermi-level density of states[33,](#page-12-9)[34](#page-12-10). At GBs, both of these effects may play an important role, but because these regions are so small, it is difficult to directly probe their superconducting properties. To determine the effect of Sn-antisite defects, we consider ab initio T<sup>c</sup> values calculated using Eliashberg theory[35](#page-12-11) and Density Functional Theory

![](_page_3_Figure_1.jpeg)

**Caption:** This figure shows a HAADF-STEM image combined with concentration profiles of niobium (Nb) and tin (Sn) across a grain boundary (GB) between two grains in a Nb3Sn coating. The x-axis represents the distance across the grains in nanometers (nm), and the y-axis denotes the concentration in atomic percentage (at.%) for Nb and Sn. There is a distinct segregation of Sn at the grain boundary, reaching a maximum concentration of approximately 33 at.% with a segregation width of around 3 nm. This segregation is significant as it affects the superconducting properties of the material, influencing the performance and efficiency of superconducting radio frequency (SRF) cavities by promoting vortex nucleation, which impairs the cavity quality factor.


<span id="page-3-0"></span>FIG. 3. The HAADF-STEM image and corresponding Nb and Sn concentration profiles across the GB between Grain 1 and Grain 2. Sn is segregated at the GB up to ∼33 at.% Sn and the width of the Sn segregated region is ∼3 nm.

![](_page_3_Figure_3.jpeg)

**Caption:** This graph provides the Nb and Sn concentration profiles at a grain boundary (GB) in a Nb3Sn sample, using bright-field scanning transmission electron microscopy (BF-STEM) data. The plot illustrates distinct Sn enrichment at the GB, highlighting Sn as high as 33 at.% within a narrow zone (~3 nm), which underscores the detrimental impact of such segregation zones on superconducting performance and cavity efficiency by facilitating vortex penetration.


<span id="page-3-1"></span>FIG. 4. BF-STEM and corresponding Nb and Sn concentration profiles across a GB from a witness sample of highperformance Nb3Sn SRF cavity prepared at Fermilab.

(DFT)[33](#page-12-9)[,36](#page-12-12). We present such results obtained using a Wannier-based k-point sampling approach[37](#page-12-13). For the experimentally measured stoichiometry range of the A15 phase, the predicted T<sup>c</sup> values are similar to or modestly higher than experimental values, as described in Table 1. For experimentally inaccessible tin-rich stoichiometry,

<span id="page-3-2"></span>![](_page_3_Picture_6.jpeg)

**Caption:** This high-angle annular dark field (HAADF) scanning transmission electron microscopy (STEM) image displays the cross-sectional view of a Nb3Sn grain boundary on the surface, emphasizing the significant structural changes at these boundaries. This image is integral in visualizing microstructural features that contribute to flux entry pathways, relevant for developing enhanced performance strategies for Nb3Sn-based SRF cavities by addressing the structural integrity at the grain scale.


FIG. 5. HAADF-STEM image of the cross-section of the top surface of a Nb3Sn GB.

![](_page_3_Figure_8.jpeg)

**Caption:** Figure 6 showcases experimental (gray squares) and calculated (black circles) critical temperatures (T<sub>c</sub>) for Nb-Sn superconductors with varying stoichiometries. The data points indicate a pronounced decrease in T<sub>c</sub>, reaching a minimum of about 5 Kelvin at approximately 31.25% Sn content, illustrating the vulnerability of tin-rich regions within grain boundaries to temperature-driven performance reductions. This finding highlights the stoichiometric sensitivity of Nb3Sn, key to designing SRF materials with optimized stability and superconducting efficacy.


<span id="page-3-3"></span>FIG. 6. Experimental T<sup>c</sup> [38](#page-12-14) (grey squares) and calculated T<sup>c</sup> (black circles) for A15 Nb-Sn of different stoichiometries. For some stoichiometries, multiple possible defect configurations were considered. The calculated T<sup>c</sup> reaches a minimum of about 5 Kelvin in the tin-rich regime.

T<sup>c</sup> values fall to a minimum of about 5K at 31.25% Sn stoichiometry (Fig. [6\)](#page-3-3). This is well within the range that has been observed around GBs. As expected, we observe a close correlation between calculated T<sup>c</sup> and Fermi-level density of states for all calculations.

Our atomistic GB calculations show that the presence of tin antisite defects dramatically widens the reduction in the Fermi-level density of states around the GB, from the range of ∼1-1.5 nm described in section [II](#page-2-0) to a range of at least ∼2 nm[31](#page-12-7). By considering the average Fermilevel density-of-states over the volume of a Cooper pair, we estimate the local suppression of T<sup>c</sup> from both a clean GB and a GB containing the observed tin-concentration profiles[27](#page-12-3). Our estimates show clean GBs suppress T<sup>c</sup> by only about 20%, but GBs with representative Sn-antisite concentrations suppress T<sup>c</sup> by close to 70%[31](#page-12-7). Based on this result, we conclude that the combined impact from anti-site defects and strain is significantly larger than the impact from strain alone. This predicted sensitivity of GB properties to defect segregation further motivates our Ginzburg-Landau modeling of GBs with different local superconducting properties.

TABLE I. Calculated vs. Measured T<sup>c</sup> Composition Experimental T<sup>c</sup> (K) Calculated T<sup>c</sup> (K) 18.75% Sn 6 9.2† 20.83% Sn 9.5 11.3 23.44% Sn 16 16.1 25.00% Sn 18 18.2 31.25% Sn n/a 5.3†

† Averaged over multiple configurations.

#### <span id="page-4-0"></span>A. Introduction to Methods

Time-dependent Ginzburg-Landau (TDGL) theory is sophisticated enough to capture vortex dynamics without becoming too algebraically complicated and computationally expensive. We solve the TDGL equations using a finite element method implemented in FEniCS[39,](#page-12-15)[40](#page-12-16) . We follow the finite element formulation described by Gao et. al[9](#page-11-8)[,41](#page-12-17). The equations are

$$\frac{\partial \psi}{\partial t} + i\phi\psi = -\alpha'\psi + |\psi|^2\psi + \left(\frac{-i}{\kappa\_0}\nabla - \mathbf{A}\right)^2\psi$$

$$\mathbf{j} = \nabla \times \nabla \times \mathbf{A}$$

$$=-\frac{1}{u\_0} \left(\frac{\partial \mathbf{A}}{\partial t} + \frac{1}{\kappa\_0} \nabla \phi\right) - \frac{i}{2\kappa\_0} \left(\psi^\* \nabla \psi - \psi \nabla \psi^\*\right) - |\psi|^2 \mathbf{A} \tag{2}$$

$$\left(\frac{i}{\kappa\_0}\nabla\psi + \mathbf{A}\psi\right)\cdot n = 0 \text{ on surface} \tag{3}$$

$$(\nabla \times \mathbf{A}) \times n = \mathbf{H} \times n \text{ on surface} \tag{4}$$

$$-\left(\nabla \phi + \frac{\partial \mathbf{A}}{\partial t}\right) \cdot n = 0 \text{ on surface} \tag{5}$$

These equations depend on the order parameter ψ, the magnetic vector potential A, and the electric potential φ, all of which can vary in space and time. There are also three dimensionless material parameters that are modified from the usual definition order to accommodate material inhomogeneities. We follow the procedure in reference[9](#page-11-8) and define material parameters with respect to reference point, which, in this case, we take to be the bulk Nb3Sn. The constant κ<sup>0</sup> is the Ginzburg-Landau parameter, the ratio of the penetration depth λ<sup>0</sup> and the coherence length ξ<sup>0</sup> in the bulk. Similarly, u<sup>0</sup> is the ratio of the characteristic time scales for variations in the order parameter and current in the bulk. Finally, Sn segregation is modeled through a spatially varying critical temperature quantified in the parameter α <sup>0</sup> = (1 − T /Tc)/(1 − T /Tbulk c ) ∝ 1 − T /Tc.

The Ginzburg-Landau model has some inherent limitations. Importantly, it is quantitatively valid only near Tc, and the dynamics Eqs. [\(1\)](#page-4-1)- [\(2\)](#page-4-2) are appropriate for gapless superconductors. The model captures the relevant physics for describing vortex nucleation and pinning while accommodating inhomogenous material parameters, making it a useful model at level of detail for this study. Here, we use the model as a tool for probing the stability of superconducting configurations in the presence of segregated grain boundaries and varying applied magnetic fields. The limitations of the model mean that the results of these simulations are not quantitativleiy accurate; however, they do give a qualitative insights into a potential explanation for the Q-slope phenomenon. A model for gapped superconductors introduces a second time scale[42](#page-12-18), and future work may incorporate these more

<span id="page-4-3"></span>FIG. 7. The black square interior to the film geometry marks our domain of simulation. It lies perpendicular to the applied magnetic field Ha. We assume there are no variations in the direction of H<sup>a</sup> so that we can simulate a 2D domain.

realistic dynamics.

This formulation reduces the full three-dimensional problem into a two-dimensional problem by assuming symmetry along the z-axis. The magnetic field points along the z-axis, fixing variations in the order parameter and magnetic vector potential to the x-y plane.

For all simulations in this section, we consider the film geometry seen in Figure [7.](#page-4-3) We take a rectangular cross-section lying in the x-y plane. We enforce periodic boundary conditions on the left and right side. On the top and bottom, we enforce Dirichlet boundary conditions for the magnetic field and Neumann boundary conditions for the order parameter.

We model geometric defects by removing an exponential cut out from the top and bottom of the film. The region removed is given by de<sup>−</sup> |x| <sup>w</sup> where d is the depth

<span id="page-4-2"></span><span id="page-4-1"></span>![](_page_4_Figure_16.jpeg)

**Caption:** Figure 16 depicts calculated temperature profiles near a grain boundary for the Nb3Sn/Nb system subjected to external temperatures of 2K (blue curve) and 4K (yellow curve). The temperature rises significantly near the boundary in response to an applied magnetic field, illustrating localized thermal oscillations that can impact the superconducting performance by nearing critical temperature thresholds necessary for quenching. The red dashed lines represent the bath temperatures, and vertical lines indicate relevant distances from the grain boundary. These findings are pivotal for understanding the thermal management requirements for maintaining SRF cavity performance under operational stress.


![](_page_5_Figure_0.jpeg)

**Caption:** Figure 12 illustrates a set of vortex lines in a superconductor under a surface magnetic field, focusing on a grain boundary as the area of weakest superconductivity where vortices nucleate. This depiction shows how these defects serve as nucleation points for magnetic vortices that degrade superconducting performance, emphasizing the need to address these areas to maintain high SRF cavity quality.


<span id="page-5-0"></span>FIG. 8. Projection of α <sup>0</sup> = 1 − T /T<sup>c</sup> onto a mesh, where α <sup>0</sup> = 0 (i.e., T ≈ Tc) in the region |x|/λ ≤ 0.125 and α <sup>0</sup> = 1 elsewhere. The width of this region comes from experimental observations.

of the cut out and w determines the width. The depth and width are chosen to match experimentally observed geometries.

To capture Sn segregation we allow T<sup>c</sup> to vary over the domain through the parameter α 0 . We take T bulk <sup>c</sup> = 19 K and T = 4.4 K, so that T /T<sup>c</sup> = 1 − 0.77α 0 . To mimic the distribution of material inhomogeneities shown in Figure [3,](#page-3-0) we let α <sup>0</sup> ≤ 1 in the GB region |x| ≤ ξ/2 and α <sup>0</sup> = 1 elsewhere. The projection of this onto the mesh is shown in Figure [8.](#page-5-0)

In addition to stoichiometry, previous work suggests that strain effects in the neighborhood of a grain boundary can also effect local superconducting properties. For example, a 1% strain can result in a 10% reduction in the Fermi level density of states[34](#page-12-10). In contrast, an excess of 5% Sn can result is more than a 50% reduction in the density of states at the Fermi level. Here, we focus on the dominant, stoichiometry effect and model a reduction in T<sup>c</sup> that is strongly localized to the grain boundary.

The Ginzburg-Landau parameter for Nb3Sn is κ = λ/ξ ∼26, which is challenging to simulate because of the extreme separation in length scales that require a very refined mesh. However, the relevant physics for vortex nucleation are variations in material parameters on length scales comparable to the superconducting coherence length, ξ. Therefore, we have simulated a moderate type-II material (κ = 4) but scaled the width of the segregated region (i.e., depleted Tc) so that its dimensions relative to ξ are the same as that observed in Figure [3.](#page-3-0) At sufficiently high fields, the line of vortices undergoes a buckling instability when the vortex-vortex repulsion overcomes the pinning strength of the grain boundary. Because our simulations use a smaller value of the penetration depth, we under-estimate the vortex-vortex repulsion and overestimate the critical field at which the chain becomes unstable. However, the pinning force in real-world cavities also depends on details, such as the slope of the α 0 in the transition region, as we discuss below. These details will affect quantitative predictions for real-world cavities; however, we expect these simulations to capture the qualitative picture.

#### B. Vortex Nucleation in Grain Boundaries

To simulate the nucleation of vortices into a grain boundary, we set the value of the magnetic field at the top and bottom boundary such that it is low enough that an array of vortices do not penetrate directly into the bulk, but large enough for vortices to enter into the grain boundary[9](#page-11-8) . As we evolve in time (assuming a constant applied field), two different behaviors are observed depending on the magnitude of the applied field. In the first scenario, vortices enter into the grain boundary at the geometric divot. With increasing field, the spacing between vortices decreases until it reaches critical levels. In the second scenario, vortices first fill the grain boundary, as in the first case, but then begin to penetrate into the grain from the boundary.

Once a vortex has penetrated into the grain boundary, it is pushed from the surface, allowing more vortices to come in after it. Once space is available, another vortex penetrates. This continues until the vortices have filled the grain boundary, i.e. an optimal spacing between the vortices inside grain boundary has been achieved. This is illustrated by the sequence in Figure [9.](#page-6-1) Note that vortices are manifest as regions in which the order parameter is reduced to near zero at their center and exponentially decay radially outward to unity.

If the applied magnetic field is sufficiently high, vortices will also begin to penetrate into the bulk once the grain boundary has been filled. See, for example, the bottom of Figure [9.](#page-6-1) The escape of vortices into the bulk occurs through a buckling transition similar to that observed in anisotropic, layered superconductors[43](#page-12-19). The field at which vortices penetrate from the grain boundary into the grain is dependent on the distribution of α <sup>0</sup> = 1 − T /Tc. The shallower the slope of α 0 the lower the applied field needs to be to nucleate vortices into the grain from the grain boundary. These results are summarized in the phase diagram in Figure [10.](#page-6-2) Comparing with results from section [III,](#page-2-1) for a segregated region with T<sup>c</sup> ∼ 5K in a cavity operating at T = 4.2K (T /T<sup>c</sup> ∼ 1), we observe a non-trivial region of the phase diagram that admits flux trapped at the grain boundary.

The value of the applied field at which the vortices first leave the grain boundary and penetrate the bulk depends on the properties in the transition zone between the segregated and non-segregated region. If the transition form α <sup>0</sup> < 1 (segregated region) to α <sup>0</sup> = 1 (non-segregated region) is very sharp (as the blue solid curve in Figure [11\)](#page-6-3), then vortices will be trapped in the grain boundary for larger fields. However, if the transition is more gradual (such as the orange dashed curve), then it is easier for vortices to escape the boundary and penetrate the bulk. Figures [\[8,](#page-5-0) [9,](#page-6-1) [10\]](#page-6-2) were generated with a very sharp inter-

![](_page_6_Figure_1.jpeg)

**Caption:** Figure 9 details the sequenced penetration of vortices through a grain boundary under increasing magnetic fields. Initial vortex penetration occurs at the geometric defect (top), progress to the grain bulk (middle), and eventually disperses into an equilibrium distribution (bottom). These observations highlight critical transitions between confined and delocalized vortex behaviors, essential for assessing SRF cavity performance under varied operational conditions.


<span id="page-6-1"></span>FIG. 9. Sequence of behaviors for increasingly large applied magnetic fields. An applied magnetic field is set as a boundary condition to the top and bottom of the film, with periodic boundary conditions on the right and left sides. At moderate applied magnetic fields, vortices penetrate only into the grain boundary from the geometric defect (top). At higher fields, a critical vortex spacing is reached (second pane) above which vortices begin to penetrate the bulk from the grain boundary (third pane). Finally, having entered the bulk, vortices disperse and fill the grain with an equilibrium distribution (bottom).

face.

## <span id="page-6-0"></span>V. ESTIMATES OF VORTEX DISSIPATION AT GRAIN BOUNDARIES

Inhomogeneous properties of superconductors have high impact on the performance of SRF cavities, affecting figures of merit such as the residual resistance

![](_page_6_Figure_6.jpeg)

**Caption:** Figure 10 displays a phase diagram from time-dependent Ginzburg-Landau (TDGL) simulations predicting flux penetration within the presence of a grain boundary as a function of the applied magnetic field and the minimum critical temperature at the grain boundary. The diagram partitions regions with no flux penetration (blue), flux pinned to the grain boundary (yellow), and flux extending into the bulk (red) at increasing field strengths. These results underscore the complexity of magnetic flux behavior in segregated Nb3Sn grain boundaries, vital for comprehending cavity efficiency and designing improved SRF cavity solutions.


<span id="page-6-2"></span>FIG. 10. Phase diagram of TDGL predictions for flux penetration in the presence of the grain boundary as a function of the the applied field and minimum critical temperature inside the grain boundary. At small applied fields, no flux penetrates (blue). At intermediate fields, flux penetrates but is constrained to the grain boundary (yellow). At sufficiently high fields, the flux penetrates from the grain boundary in to the bulk material (red).

![](_page_6_Figure_8.jpeg)

**Caption:** This figure demonstrates the behavior of a function T/Tc correlated with x/λ in a superconducting Nb3Sn film. The y-axis is non-dimensionalized as T/Tc minus a constant, illustrating phase transition patterns of the superconducting state as a function of the normalized distance along the vortex penetration length. The depicted sharp transition (solid lines) and gradual transformation (dashed lines) both highlight the effects of boundaries on superconducting properties, emphasizing their implications for vortex dynamics within SRF cavities and the importance of maintaining stable superconducting phases across the span.


<span id="page-6-3"></span>FIG. 11. Profiles of potential α 0 for the transition between the segregated and non-segregated regions. Sharp transitions (such as the blue solid curve) keep the vortices constrained to the boundary for larger applied fields. A more shallow transition (such as the orange dashed curve), however, allow vortices to escape into the grain more easily.

due to trapped magnetic flux[44–](#page-12-20)[46](#page-12-21), and the superheating field[1,](#page-11-2)[2](#page-11-3)[,7,](#page-11-6)[47–](#page-12-22)[49](#page-12-23). Grain boundaries well aligned with the surface magnetic field can become weak spots for the nucleation of superconducting vortices (see Fig. [12\)](#page-7-0), and could be ultimately responsible for the quench of an SRF cavity. Note that our Ginzburg-Landau model did not simulate the full semi-loops of Figure [12.](#page-7-0) In this section, we discuss simple estimates for the power dissipation and heat diffusion due to nucleation of vortices at grain boundaries in Nb3Sn cavities.

We start with an estimate for the power dissipated in an SRF cavity when a grain boundary is filled with n superconducting vortices. We assume of order O(n) vortex
![](0__page_7_Picture_0.jpeg)

**Caption:** This figure illustrates vortex nucleation in a superconductor subjected to an applied surface magnetic field (H), where vortex lines (in red) appear in areas of weakened superconductivity, specifically at grain boundaries illustrated in dark gray. As the magnetic field increases, these vortices nucleate at the boundaries due to weakened superconducting properties in these regions, a phenomenon pivotal for understanding SRF cavity performance degradation. This visual serves to highlight the role of grain boundaries as initial sites for vortex entry, emphasizing the significant effects these boundaries have on the superconductor's overall electromagnetic properties and its superconducting performance.


FIG. 12. Illustrating vortex nucleation (red lines) in a superconductor (light gray region) subject to a surface magnetic field H. Vortex entry starts at regions where superconductivity is weakest (dark gray regions, here representing grain boundaries).

<span id="page-7-0"></span>lines are annihilated once per cycle, their energy is lost, and the power dissipated per vortex line is simply the drop in the energy of the outside magnetic field times the RF frequency. Our estimate gives a rough estimate for the actual power dissipated by vortices at grain boundaries, if the field reaches high enough values for them to enter.

The drop in magnetic energy when a vortex line of length D nucleates into the superconductor is given by:

$$
\Delta E = \left| \int \frac{1}{2\mu} \left( B - \frac{\Phi\_0 D}{V} \right)^2 dV - \int \frac{B^2}{2\mu} dV \right|
$$

$$
\approx \frac{B\Phi\_0 D}{\mu},
\tag{6}
$$

where V is the volume, µ is the permeability of free space, Φ<sup>0</sup> is the fluxoid quantum, and λ is the penetration depth. Note that ∆E is also approximately the work done by the external magnetic field to push a vortex into the bulk of the superconductor: W ≈ f<sup>L</sup> · D · λ = (Φ0B/(µλ))Dλ = ∆E, where f<sup>L</sup> is the Lorentz force per length. Thus, our calculation gives the vortex dissipation at grain boundaries assuming that the vortices do not leave the grain boundary as the external field drops and changes sign, as our simulations indicate, and as one would expect for vortices that enter a distance more than λ, past the surface nucleation barrier. The total energy drop for a grain boundary of linear size D with vortices spaced by λ (see Fig. [12\)](#page-7-0) is ∆EGB ≈ ∆E(D/λ) = BΦ0D<sup>2</sup>/(µλ), and the power dissipated per grain boundary is given by

$$P\_{GB} \equiv f \Delta E\_{GB} = \frac{B \Phi\_0}{\mu} \frac{fD^2}{\lambda},\tag{7}$$

where f is the cavity frequency. For a 1.3GHz Nb3Sn cavity with typical grain size of D ≈ 1µm, we find PGB ≈ 621nW at B = 60mT.

Note that our estimate relies on the assumption that vortices quickly fill the grain boundary before being annihilated during the RF cycle. If the vortex line relaxes slowly, the RF field might vanish and change sign before vortices had time to fill the grain boundary, and our assumption would not be valid. We now show that that is not the case — vortices move at extremely high speeds in the typical environment of Nb3Sn SRF cavities.

Consider the one-dimensional motion of a "train" of N vortices moving through a grain boundary towards the superconductor interior (see Fig. [13\)](#page-8-0). Assuming overdamped dynamics, the equations of motion for each vortex line i of velocity v<sup>i</sup> read:

$$\begin{aligned} \eta\_0 v\_1 &= f\_L - f\_{2,1}, \\ \eta\_0 v\_2 &= f\_{1,2} - f\_{3,2}, \\ &\vdots \\ \eta\_0 v\_{N-1} &= f\_{N-2,N-1} - f\_{N,N-1}, \\ \eta\_0 v\_N &= f\_{N-1,N}, \end{aligned}$$

where f<sup>L</sup> = Φ0Hrf /λ is the Lorentz force per length at the surface, Hrf is the surface magnetic field, η<sup>0</sup> = Φ<sup>o</sup> 2 /(2πξ<sup>2</sup>ρn) is the Bardeen-Stephen viscosity[50](#page-12-0) , ρ<sup>n</sup> is the resistivity of the normal state and fi,j is the repulsion force from vortex i into vortex j. Thus,

$$v\_{BS} \equiv \frac{1}{N} \sum\_{i=1}^{N} v\_i = \frac{f\_L}{N\eta\_0} = \frac{2\pi}{\mu \Phi\_0} \frac{\rho\_n \xi^2}{\lambda} \frac{B\_{rf}}{N} . \tag{8}$$

For Nb3Sn at Brf = 60mT, we find vBS = 2.4µm/ns and vBS = 24µm/ns for ten vortices and one vortex, respectively. The average velocity of the vortex train is high enough for vortices to quickly fill in the grain boundary during the RF cycle, but this numerical value should be taken with a grain of salt. The Bardeen-Stephen formula is not valid at these high speeds, which also exceed the pair-breaking limit of the superconducting condensate[51](#page-12-1) . In principle, one could incorporate pair-breaking mechanisms into our simple estimates by using the nonlinear viscosity calculated by Larkin and Ovchinnikov (LO)[52](#page-12-2): η(v) = η0/[1+ (v/v0) 2 ], where η<sup>0</sup> is the Bardeen-Stephen viscosity and the LO velocity v<sup>0</sup> marks the onset of highly dissipative states for straight vortices. Following the same steps that we have used to calculate vBS, and assuming that each vortex moves with the same vortextrain velocity vLO, we find

$$v\_{LO} = \frac{v\_0}{2\upsilon\_{BS}} \left( 1 - \sqrt{1 - 4\frac{v\_{BS}s^2}{v\_0^2}} \right)$$

$$\approx v\_{BS} \left( 1 + \frac{v\_{BS}}{v\_0^2} \right),\tag{9}$$

for vBS v0, showing that nonlinear corrections lead to an increase of the velocity of the vortex train from the Bardeen-Stephen result. However, note that estimates for Nb3Sn at low temperatures[53](#page-12-3) yield v<sup>0</sup> ≈ 0.1km/s, which is much smaller than our estimates for vBS, so that a more sophisticated analysis is needed to better describe vortex motion at these high-speed regimes. In recent work[54](#page-12-4), Pathirana and Gurevich consider the nonlinear dynamics of curvilinear vortices subject to LO viscous forces to study dissipation and vortex teardown instabilities.

![](0__page_8_Figure_4.jpeg)

**Caption:** Figure 4 schematically illustrates a sequence of vortices (represented as blue disks) translocating through a grain boundary (dark gray) under the influence of surface Lorentz forces caused by an applied RF magnetic field. Each vortex is subject to mutual repulsion from neighboring vortices. This design effectively visualizes the mechanism by which grain boundaries act as weak points in superconductors, suggesting a fundamental process by which grain boundary activations might induce degradation of the quality factor in high-performing SRF cavities under operational conditions.


<span id="page-8-0"></span>FIG. 13. Illustrating a "train" of N vortex lines (blue disks) moving through a grain boundary (dark gray) of a superconductor (light gray). The first vortex is subject to a surface Lorentz force from the RF field and each vortex is repelled by its nearest neighbors.

Grain boundary activation might be associated with the degradation of the quality factor Q of SRF cavities at high fields. We now use our estimates to calculate the number of active grain boundaries needed to deplete Q by a certain amount.

The quality factor is given by Q = GBrf 2 /(2µ <sup>2</sup>P), where P is the dissipated power per unit area and G is a geometry factor. We break up the total surface area s of the cavity into N blocks, so that s = NsGB, where sGB is the average area occupied by one grain boundary. Assume inactive and active blocks dissipate power P<sup>1</sup> and P<sup>1</sup> + PGB, respectively, where by active block we mean a block with a grain boundary filled with vortices. For M active blocks,

$$Q = \frac{G B\_{rf}{}^2}{2\mu^2 (N P\_1 + M P\_{GB}) / (N s\_{GB})}$$

$$= \frac{G B\_{rf}{}^2 s\_{GB}}{2\mu^2 (P\_1 + x P\_{GB})},\tag{10}$$

where x ≡ M/N is the ratio of active grain boundaries. In the absence of active grain boundaries, we assume Q = Q<sup>1</sup> is constant (i.e. P<sup>1</sup> ∼ Brf 2 ), so that

<span id="page-8-1"></span>
$$P\_1 = \frac{G B\_{rf} \,^2 s\_{GB}}{2 \mu^2 Q\_1}.\tag{11}$$

Plugging Eq. [\(11\)](#page-8-1) into Eq. [\(10\)](#page-8-2) and solving for x, we find

$$x = \frac{G s\_{GB} B\_{rf}}{2\mu^2 P\_{GB}} \left(\frac{1}{Q} - \frac{1}{Q\_1}\right). \tag{12}$$

To estimate the percentage of active grain boundaries that is needed for Q to drop by a certain amount, we assume Q is constant and equal to Q<sup>1</sup> = 10<sup>10</sup> for fields up to Brf = 22mT, and then decreases exponentially until it reaches Q = 10<sup>9</sup> at Brf = 66mT. This form for the Qslope profile is shown in yellow in Fig. [14,](#page-8-3) and is chosen to roughly mimic the experimental data shown with red symbols in Figure [1.](#page-1-0) Here we assume that Q degradation is solely due to grain activation; the field of vortex penetration is chosen to roughly match that of the onset of the experimental Q slope (≈ 22mT). Note that Calculations for the exactly solvable limit of Josephson vortices which penetrate along weakly-coupled planar junctions yield a threshold critical current density J<sup>c</sup> ∼ φ0/(λ λ<sup>J</sup> 2 ), where λ<sup>J</sup> is the Josephson penetration depth, which is assumed to be much larger than the penetration depth[55](#page-12-5). Figure [14](#page-8-3) shows a plot of the percentage of active grain boundaries (100 x, blue curve) as a function of Brf corresponding to the artificial Q-slope profile shown in the yellow curve (using Nb3Sn parameters with Q<sup>1</sup> = 10<sup>10</sup> , sGB = 0.5µm<sup>2</sup> and G = 278Ω). Note that about 0.03% of the surface grain boundaries need be filled with vortices for Q to drop from 10<sup>10</sup> to 10<sup>9</sup> for Nb3Sn at about 66mT.

![](0__page_8_Figure_14.jpeg)

**Caption:** Figure 14 plots the estimated percentage of active grain boundaries, expressed as a fraction, that contribute to the degradation of the SRF cavity quality factor (Q) as a function of the applied RF magnetic field strength (Brf) in millitesla (mT). The blue curve demonstrates a growing activation of surface grain boundaries correlating with the decline in Q depicted by the yellow curve, illustrating Q dropping from 10^10 to 10^9 as Brf increases from 22 mT to 66 mT. This model suggests that only a minuscule percentage (~0.03%) of grain boundaries need be activated to effect a significant Q degradation at the sulfur-RF conditions common to Nb3Sn cavities.


<span id="page-8-3"></span>FIG. 14. Estimated percentage of active surface grain boundaries (blue curve) as a function of Brf , corresponding to the quality factor profile displayed in the yellow curve for Nb3Sn with Q<sup>1</sup> = 10<sup>10</sup> .

<span id="page-8-2"></span>We end this section with a simple model calculation of the steady-state thermal heating at a grain boundary. Figure [15](#page-9-0) illustrates our model for thermal diffusion near an active grain boundary. Cross sections of Nb and Nb3Sn layers are illustrated below and above the yellow dashed line in Fig. [15,](#page-9-0) respectively. The Nb surface is in contact with a low-temperature He bath. The Nb3Sn surface is in contact with vacuum, and is subject to a parallel oscillating magnetic field. The grain boundary is represented by a red rectangle at the top center of Figure [15,](#page-9-0) and has linear size D. White arrows depict approximate directions for heat diffusion in our model. First, we assume one-dimensional heat diffusion away from the grain boundary up to a distance r ≈ D. We expect the heat front to attain a semi-spherical shape for distances r ' D. We then assume three-dimensional heat diffusion away from a half-sphere of radius D for distances D ≤ r ≤ R1. For r ≤ R<sup>1</sup> we consider the Nb3Sn thermal conductivity κ = κNb3Sn. At last, we assume three-dimensional heat diffusion away from a half-sphere of radius R<sup>1</sup> for distances R<sup>1</sup> ≤ r ≤ R2, with κ given by the Nb thermal conductivity κNb. Note that our assumptions are stronger when R<sup>2</sup> R<sup>1</sup> (R2/R<sup>1</sup> ∼ 10<sup>3</sup> for typical Nb3Sn/Nb SRF cavities).

![](0__page_9_Picture_1.jpeg)

**Caption:** This schematic model represents heat diffusion near an active grain boundary in a superconducting Nb3Sn/Nb system. The sketch displays a cross-section where red regions indicate hotter zones, and blue indicates cooler areas. The yellow dashed line marks the Nb/Nb3Sn interface with directional heat flux depicted by white arrows. This model highlights how grain boundary-induced thermal gradients can emerge, influenced by disparities in thermal conductivities between Nb3Sn and Nb, and help explain localized heating, critical for SRF cavity performance insights.


FIG. 15. Sketch of our model for heat diffusion near an active grain boundary (red rectangle at the top center). Red and blue tones represent contour regions of high and low temperatures, respectively. The dashed yellow line represent the interface between Nb, whose outer surface is in contact with the He bath (bottom of the figure), and Nb3Sn, whose inner surface is in contact with vacuum and the rf field (top). White arrows represents approximate directions for the heat flux. In our model, heat fronts move along one and three dimensions for distances smaller and greater than the grain size D, respectively. The change in the direction of heat flux at the interface is due to the wide separation of the thermal conductivities of Nb and Nb3Sn.

<span id="page-9-0"></span>The equilibrium temperature profile can then be cast from the stationary solutions of the heat equation in each region:

$$T(r) = \begin{cases} \zeta(r), & \text{for } 0 \le r \le D, \\ c/r + d, & \text{for } D \le r \le R\_1, \\ e/r + g, & \text{for } R\_1 \le r \le R\_2, \end{cases} \tag{13}$$

where c, d, e and g are constants, and ζ(r) is the stationary solution of the one-dimensional heat equation (the Nb3Sn thermal conductivity strongly varies with temperature in this region[56](#page-12-6), which complicates the problem

of finding an analytical solution for ζ(r))[57](#page-12-7). Note that we relax the definition of the coordinate r here, which should be interpreted as a lateral distance away from the grain boundary for 0 ≤ r ≤ D, and a depth coordinate towards the Helium bath for distances r > D.

To calculate ζ(r), we use Fourier's law — Q˙ = −κ dT /dr, where Q˙ is the heat flux. The stationary solution of the heat equation can be found from the solution of dQ˙ /dr = 0, i.e.

<span id="page-9-1"></span>
$$-\kappa\_{\rm Nb3Sn}(T)\frac{dT}{dr} = a,\tag{14}$$

where a is constant.

The thermal conductivity κNb3Sn(T) in the superconducting layer has two important contributions: a phonon contribution and an electronic component (carried by superconducting quasiparticles). The low-temperature phonon thermal conductivity is strongly dependent on the morphology of the crystal[58](#page-12-8); in clean insulating crystals it is dominated by scattering off grain boundaries and sample boundaries, and varies as T 3 . Scattering off impurities can cut off the contribution of high-frequency phonons, or even resonantly cut off certain frequency bands. All of these mechanisms lead to a thermal conductivity that monotonically increases with temperature, so we avoid the complexity by using a constant phonon thermal conductivity k1, giving a lower bound for the conductivity and hence an upper limit to the heating. The electronic portion of the thermal conductivity k<sup>2</sup> in the normal metal at low temperatures is roughly independent of temperature, and is set by the electronic meanfree path. In the superconductor, it decreases exponentially as exp(−∆(0)/kBT), as seen experimentally[56](#page-12-6). Using the BCS relation between the gap and the transition temperature, we therefore use

<span id="page-9-4"></span>
$$\kappa\_{\rm Nb3Sn}(T) = k\_1 + k\_2 \exp\left(-1.76 \, T\_c/T\right). \tag{15}$$

We use the normal electron thermal conductivity k<sup>2</sup> = 2 × e <sup>1</sup>.<sup>76</sup> W/m·K from[56](#page-12-6). Because the electronic contribution is negligible at the operating temperature of the cavity, we set k<sup>1</sup> = 10<sup>−</sup><sup>2</sup>W/m·K as the approximate total thermal conductivity of Nb3Sn at 2K[56](#page-12-6). Both of these constants are dependent upon the preparation of the film, and also could vary from one region of the film to another as the growth conditions or the underlying Nb grain orientations vary.

Integration of Eq. [\(14\)](#page-9-1) results in

<span id="page-9-2"></span>
$$
\Pi(T) = -a \, r + b,\tag{16}
$$

<span id="page-9-3"></span>where b is constant, and

$$\begin{aligned} \Pi\left(T\right) &= k\_1 \, T + k\_2 \, T \, e^{-1.76T\_c/T} \\ &+ 1.76 \, k\_2 \, T\_c \, E\_i \left(-1.76 \frac{T\_c}{T}\right), \end{aligned} \tag{17}$$

with Ei(x) ≡ R <sup>∞</sup> −x [e <sup>−</sup><sup>t</sup>/t]dt denoting the exponential integral function. ζ(r) is then the solution of Eq. [\(16\)](#page-9-2) for T. Note that our simple model assumes that the quasiparticles and the phonons remain at the same effective temperature (the inelastic electronic mean free path is small), and that both remain diffusive (the elastic phonon and electron mean free paths are small). Violating either of these assumptions would likely lower the transport of energy away from the grain boundary, making the heating more dangerous.

We focus our attention on grain-boundary activation and ignore other sources of power dissipation. These sources, in particular the BCS dissipation that exponentially increases with temperature, may have a significant impact on stationary temperature profiles, with possible indicatives of thermal runaway and cavity quenches at low fields. As we discuss at the end of this section, our simple estimates suggest that larger grain boundaries or multiple nearby boundaries could already raise the temperature high enough to quench the cavity. Hence, although the inclusion of other sources would lead to a more accurate description of heat diffusion near active grain boundaries, they will not lead to qualitative changes in our conclusions. Thus, we assume that the heat flux is the power dissipated per grain boundary (PGB) per unit area. We use Fourier's law to determine the coefficients a, c and e in Eqs. [\(13\)](#page-9-3) and [\(16\)](#page-9-2). For 0 ≤ r ≤ D, Q˙ = PGB/D<sup>2</sup> , so that:

$$a = \frac{P\_{GB}}{D^2}.\tag{18}$$

For D ≤ r ≤ R<sup>1</sup> (R<sup>1</sup> ≤ r ≤ R2), Q˙ = PGB/2πr<sup>2</sup> , κ = κNb3Sn (κNb) and dT /dr = −c/r<sup>2</sup> (−e/r<sup>2</sup> ), so that:

$$c = \frac{P\_{GB}}{2\pi\kappa\_{\rm Nb3Sn}}, \quad e = \frac{P\_{GB}}{2\pi\kappa\_{\rm Nb}}.\tag{19}$$

To find g, we use T(R2) = THe, where THe is the temperature of the Helium bath:

$$g = T\_{\rm He} - \frac{P\_{GB}}{2\pi\kappa\_{\rm Nb}R\_2}.\tag{20}$$

To find d and b, we use the continuity of T(r) at r = R<sup>1</sup> and r = D, respectively (we ignore the Kapitza resistance at the interface between Nb3Sn and Nb). Thus,

$$c/R\_1 + d = e/R\_1 + g \quad \Rightarrow \quad d = \frac{e-c}{R\_1} + g,\qquad \text{(21)}$$

and

$$b = aD + \Pi \left(\frac{c}{D} + d\right). \tag{22}$$

Figure [16](#page-10-0) shows temperature profiles (Eq. [\(13\)](#page-9-3)) near an active grain boundary for Nb3Sn/Nb systems at B = 60mT. Note that the temperature of an active grain boundary increases to about 10K near the boundary surface for a Helium temperature of 2K. Although this increase in temperature is not large enough to drive Nb3Sn into the normal state, it certainly has significant impact

![](0__page_10_Figure_12.jpeg)

**Caption:** Figure 16 details the temperature profiles as a function of distance from a grain boundary in an Nb3Sn/Nb system when exposed to external bath temperatures of 2K and 4K. The x-axis represents the distance in meters (m), and the y-axis indicates the temperature in Kelvin (K). The profiles reveal that temperatures near active grain boundaries can rise up to approximately 10K, significantly affecting the superconducting properties by localizing heating effects due to grain boundary activation at fields of B = 60mT. The spatial decay of temperature underscores the heating influence as being constrained primarily to the grain boundaries. This visualization aids in understanding local thermal anomalies that could drive quenching in SRF cavities .


<span id="page-10-0"></span>FIG. 16. Temperature as a function of distance from the grain boundary for the Nb3Sn/Nb system at a bath of 2K (blue) and 4K (yellow). Vertical dashed lines correspond to D = 1µm, R<sup>1</sup> = 2µm and R<sup>2</sup> = 3mm, from left to right. Bottom and top red dashed lines correspond to THe = 2K and 4K, respectively. We have used temperature-dependent κNb3Sn given by Eq. [\(15\)](#page-9-4), κNb = 10W/m·K, and PGB = 621nW at B = 60mT, according to our previous estimates.

on the superconducting properties. Also, note that the temperature decays to nearly THe as r approaches twice the grain size D, suggesting that heating due to grainboundary activation is mostly localized.

A temperature rise of 10K at the grain boundary is over half of the critical temperature of the film, suggesting that larger grain boundaries or multiple nearby boundaries could raise the temperature high enough to quench the cavity. Cavities with tin-rich grain boundaries and more pristine grain boundaries show the same quench fields, suggesting that another mechanism controls the quench fields of existing Nb3Sn cavities. If the excess dissipation in the cavities with tin-rich boundaries is due to vortex penetration (Fig. [14\)](#page-8-3), one would expect rare events with large or multiple grain boundaries would happen, suggesting that our grain-boundary heating estimate is unduly pessimistic. Alternatively, it remains possible that the grain boundaries have high superheating fields, and the excess dissipation has another explanation. In any case, our estimates suggest that vortex entry at grain boundaries should be expected for tin-rich boundaries well below the superheating field for a perfect crystal, and that the subsequent heat release should be important both as a contribution to the overall dissipation and as a quench mechanism for the cavity.

## VI. CONCLUSION

In this work we have presented an interdisciplinary, multi-scale study of vortex nucleation in Sn-segregated grain boundaries and its subsequent effect on SRF performance. Scanning transmission electron microscopy images and energy dispersive spectroscopy show Sn concentration as high as ∼35 at.% and widths ∼3nm in chemical composition in grain boundaries. We used density functional theory to estimate the effective critical temperature for the material in the segregation zone and find that the effective T<sup>c</sup> can be reduced to as low as 5 K for Sn concentrations in excess of ∼30 at.%. Next, we used these calculations as inputs into timedependent Ginzburg-Landau simulations. These simulations demonstrate that grain boundaries can act as nucleation sites for magnetic vortices. The grain boundaries then act as pinning "planes" after nucleation that allow vortices to move vertically along the grain boundary, but prevent them from moving laterally into the bulk. We have seen that for a range of applied fields, vortices may nucleate at but remain constrained to the grain boundary. These vortices will nucleate and annihilate once per RF cycle, and we estimate the superconducting losses of this phenomenon at the scale of SRF cavities. We have shown that as long as vortices do not penetrate the bulk grain, losses are localized near the grain boundary and will not lead to a global quench. However, the annihilation process each cycle will lead to a reduction in the quality factor that increases with larger applied fields, consistent with the experimentally observed Q-slope.

SRF cavities are an important application area that require multi-disciplinary talents to address. This study has leveraged the skills of accelerator physicists, material scientists, and condensed matter theorists with expertise across a range of scales to explore a question fundamental to the advancement of next-generation SRF material, Nb3Sn. This study has presented evidence that segregation zones in grain boundaries play an important role in cavity performance. Understanding the mechanism behind the Q-slope will motivate new manufacturing protocols and help constrain the design space of future cavities.

## VII. ACKNOWLEDGEMENT

We would like to thank Matthias Liepe, and Richard Hennig for helpful conversations. This research is supported by the United States Department of Energy, Offices of High Energy. Fermilab is operated by the Fermi Research Alliance LLC under Contract No. DE-AC02- 07CH11359 with the United States Department of Energy. This work made use of the EPIC, Keck-II, and/or SPID facilities of Northwestern University's NUANCE Center, which received support from the Soft and Hybrid Nanotechnology Experimental (SHyNE) Resource (NSF ECCS-1542205); the MRSEC program (NSF DMR-1121262) at the Materials Research Center; the International Institute for Nanotechnology (IIN); the Keck Foundation; and the State of Illinois, through the IIN. This work was supported by the US National Science Foundation under Award OIA-1549132, the Center for Bright Beams.

- <sup>∗</sup> These authors contributed equally
- † [mktranstrum@byu.edu](mailto:mktranstrum@byu.edu)
- <sup>1</sup> D. B. Liarte, S. Posen, M. K. Transtrum, G. Catelani, M. Liepe, and J. P. Sethna, [Superconductor Science and](http://dx.doi.org/ 10.1088/1361-6668/30/3/033002) Technology 30[, 033002 \(2017\).](http://dx.doi.org/ 10.1088/1361-6668/30/3/033002)
- <sup>2</sup> M. K. Transtrum, G. Catelani, and J. P. Sethna, Physical Review B 83, 094505 (2011).
- <sup>3</sup> S. J. Chapman, SIAM Journal on Applied Mathematics 55, 1233 (1995).
- <sup>4</sup> A. J. Dolgert, S. J. Di Bartolo, and A. T. Dorsey, Physical Review B 53, 5650 (1996).
- <sup>5</sup> L. Kramer, Physical Review 170, 475 (1968).
- <sup>6</sup> S. Posen, N. Valles, and M. Liepe, Physical review letters 115, 047001 (2015).
- <sup>7</sup> G. Catelani and J. P. Sethna, Physical Review B 78, 224509 (2008).
- <sup>8</sup> F. P.-J. Lin and A. Gurevich, Physical Review B 85, 054513 (2012).
- <sup>9</sup> A. R. Pack, J. Carlson, S. Wadsworth, and M. K. Transtrum, arXiv preprint arXiv:1911.02132 (2019).
- <sup>10</sup> A. Y. Aladyshkin, A. Mel'Nikov, I. Shereshevsky, and I. Tokman, Physica C: Superconductivity 361, 67 (2001).
- <sup>11</sup> L. Burlachkov, M. Konczykowski, Y. Yeshurun, and F. Holtzberg, Journal of applied physics 70, 5759 (1991).
- <sup>12</sup> D. Y. Vodolazov, Physical Review B 62, 8691 (2000).
- <sup>13</sup> P. Soininen and N. Kopnin, Physical Review B 49, 12087 (1994).
- <sup>14</sup> R. Scanlan, W. Fietz, and E. Koch, Journal of Applied Physics 46, 2244 (1975).
- <sup>15</sup> A. Godeke, Superconductor Science and Technology 19, R68 (2006).
- <sup>16</sup> M. Suenaga and W. Jansen, Applied physics letters 43, 791 (1983).
- <sup>17</sup> M. Sandim, D. Tytko, A. Kostka, P. Choi, S. Awaji, K. Watanabe, and D. Raabe, Superconductor Science and Technology 26, 055008 (2013).
- <sup>18</sup> S. Posen and D. L. Hall, Superconductor Science and Technology 30, 033004 (2017).
- <sup>19</sup> S. Posen and M. Liepe, Physical Review Special Topics-Accelerators and Beams 17, 112001 (2014).
- <sup>20</sup> S. Posen, J. Lee, O. Melnychuk, Y. Pischalnikov, D. Seidman, D. Sergatskov, and B. Tennis, in 19th Int. Conf. on RF Superconductivity (SRF'19), Dresden, Germany, 30 June-05 July 2019 (JACOW Publishing, Geneva, Switzerland, 2019) pp. 818–822.
- <sup>21</sup> G. M¨uller, H. Piel, J. Pouryamout, P. Boccard, and P. Kneisel, in Proceedings of the Workshop on Thin Film Coating Methods for Superconducting Accelerating Cavities (2000).
- <sup>22</sup> A. Gurevich, Superconductor Science and Technology 30, 034004 (2017).
- <sup>23</sup> D. L. Hall, New Insights into the Limitations on the Efficiency and AchievableGradients in Nb 3 Sn SRF Cavities (Cornell University, 2017).
- <sup>24</sup> J. Lee, S. Posen, Z. Mao, Y. Trenikhina, K. He, D. L. Hall, M. Liepe, and D. N. Seidman, Superconductor Science and Technology 32, 024001 (2018).
- <sup>25</sup> Y. Trenikhina, S. Posen, A. Romanenko, M. Sardela, J.-M. Zuo, D. Hall, and M. Liepe, Superconductor Science and Technology 31, 015004 (2017).
- <sup>26</sup> C. Becker, S. Posen, N. Groll, R. Cook, C. M. Schlep¨utz, D. L. Hall, M. Liepe, M. Pellin, J. Zasadzinski, and T. Proslier, Applied Physics Letters 106, 082602 (2015).
- <sup>27</sup> J. Lee, Z. Mao, K. He, Z. H. Sung, T. Spina, S.-I. Baik, D. L. Hall, M. Liepe, D. N. Seidman, and S. Posen, Acta Materialia 188, 155 (2020).
- <sup>28</sup> S. J. Dillon, M. Tang, W. C. Carter, and M. P. Harmer, [Acta Materialia](http://dx.doi.org/https://doi.org/10.1016/j.actamat.2007.07.029) 55, 6208 (2007).
- <sup>29</sup> D. N. Seidman, Annual Review of Materials Research 32, 235 (2002).
- <sup>30</sup> J. Rittner, D. Udler, and D. Seidman, Interface Science 4, 65 (1997).
- <sup>31</sup> M. M. Kelley, N. S. Sitaraman, and T. A. Arias, arXiv preprint arXiv:2005.12133 (2020).
- <sup>32</sup> B. Kempshall, B. Prenitzer, and L. Giannuzzi, Scripta materialia 47, 447 (2002).
- <sup>33</sup> N. S. Sitaraman, J. Carlson, A. R. Pack, R. D. Porter, M. U. Liepe, M. K. Transtrum, and T. A. Arias, arXiv preprint arXiv:1912.07576 (2019).
- <sup>34</sup> A. Godeke, F. Hellman, H. ten Kate, and M. Mentink, Superconductor Science and Technology 31, 105011 (2018).
- <sup>35</sup> W. L. McMillan, Physical Review 167, 331 (1968).
- <sup>36</sup> R. Sundararaman, K. Letchworth-Weaver, K. A. Schwarz, D. Gunceler, O. Y., and T. A. Arias, SoftwareX 6, 278 (2017).
- <sup>37</sup> N. Marzari and D. Vanderbilt, Physical Review B 56, 12847 (1997).
- <sup>38</sup> H. Devantay, J. L. Jorda, M. Decroux, J. Muller, and R. Fl¨ukiger, Materials Science 16, 2145–2153 (1981).
- <sup>39</sup> M. S. Alnæs, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg, C. Richardson, J. Ring, M. E. Rognes, and G. N. Wells, [Archive of Numerical Software](http://dx.doi.org/10.11588/ans.2015.100.20553) 3 (2015), [10.11588/ans.2015.100.20553.](http://dx.doi.org/10.11588/ans.2015.100.20553)
- <sup>40</sup> A. Logg, K.-A. Mardal, and G. Wells, Automated solution of differential equations by the finite element method: The FEniCS book, Vol. 84 (Springer Science & Business Media, 2012).
- <sup>41</sup> H. Gao and W. Sun, Journal of Computational Physics 294, 329 (2015).
- <sup>42</sup> L. Kramer and R. Watts-Tobin, Physical Review Letters 40, 1041 (1978).
- <sup>43</sup> M. J. Dodgson, Physical Review B 66, 014509 (2002).
- <sup>44</sup> A. Gurevich and G. Ciovati, [Phys. Rev. B](http://dx.doi.org/10.1103/PhysRevB.87.054502) 87, 054502 [\(2013\).](http://dx.doi.org/10.1103/PhysRevB.87.054502)
- <sup>45</sup> D. Hall, D. Liarte, M. Liepe, R. Porter, and J. Sethna, in [Proc. of International Conference on RF Super](http://dx.doi.org/ https://doi.org/10.18429/JACoW-SRF2017-THPB042)[conductivity \(SRF'17\), Lanzhou, China, July 17-21,](http://dx.doi.org/ https://doi.org/10.18429/JACoW-SRF2017-THPB042) [2017](http://dx.doi.org/ https://doi.org/10.18429/JACoW-SRF2017-THPB042) , International Conference on RF Superconductivity No. 18 (JACoW, Geneva, Switzerland, 2018) pp. 844–847, https://doi.org/10.18429/JACoW-SRF2017-THPB042.
- <sup>46</sup> D. B. Liarte, D. Hall, P. N. Koufalis, A. Miyazaki, A. Senanian, M. Liepe, and J. P. Sethna, [Phys. Rev. Applied](http://dx.doi.org/ 10.1103/PhysRevApplied.10.054057) 10, [054057 \(2018\).](http://dx.doi.org/ 10.1103/PhysRevApplied.10.054057)
- <sup>47</sup> L. Kramer, Phys. Rev. 170[, 475 \(1968\).](http://dx.doi.org/10.1103/PhysRev.170.475)
- <sup>48</sup> D. B. Liarte, M. K. Transtrum, and J. P. Sethna, [Phys.](http://dx.doi.org/10.1103/PhysRevB.94.144504) Rev. B 94[, 144504 \(2016\).](http://dx.doi.org/10.1103/PhysRevB.94.144504)
- <sup>49</sup> V. Ngampruetikorn and J. A. Sauls, [Phys. Rev. Research](http://dx.doi.org/10.1103/PhysRevResearch.1.012015) 1[, 012015 \(2019\).](http://dx.doi.org/10.1103/PhysRevResearch.1.012015)
- <span id="page-12-0"></span><sup>50</sup> J. Bardeen and M. J. Stephen, [Phys. Rev.](http://dx.doi.org/10.1103/PhysRev.140.A1197) 140, A1197 [\(1965\).](http://dx.doi.org/10.1103/PhysRev.140.A1197)
- <span id="page-12-1"></span><sup>51</sup> L. Embon, Y. Anahory, Z. L. Jeli´c, E. O. Lachman, ˇ Y. Myasoedov, M. E. Huber, G. P. Mikitik, A. V. Silhanek, M. V. Miloˇsevi´c, A. Gurevich, et al., Nature communications 8, 85 (2017).
- <span id="page-12-2"></span><sup>52</sup> A. Larkin and Y. Ovchinnikov, Sov. Phys. JETP 41, 960 (1975).
- <span id="page-12-3"></span><sup>53</sup> A. Gurevich and G. Ciovati, [Phys. Rev. B](http://dx.doi.org/10.1103/PhysRevB.77.104501) 77, 104501 [\(2008\).](http://dx.doi.org/10.1103/PhysRevB.77.104501)
- <span id="page-12-4"></span><sup>54</sup> W. P. M. R. Pathirana and A. Gurevich, [Phys. Rev. B](http://dx.doi.org/10.1103/PhysRevB.101.064504) 101[, 064504 \(2020\).](http://dx.doi.org/10.1103/PhysRevB.101.064504)
- <span id="page-12-5"></span><sup>55</sup> A. Sheikhzada and A. Gurevich, [Phys. Rev. B](http://dx.doi.org/10.1103/PhysRevB.95.214507) 95, 214507 [\(2017\).](http://dx.doi.org/10.1103/PhysRevB.95.214507)
- <span id="page-12-6"></span><sup>56</sup> G. D. Cody and R. W. Cohen, [Rev. Mod. Phys.](http://dx.doi.org/10.1103/RevModPhys.36.121) 36, 121 [\(1964\).](http://dx.doi.org/10.1103/RevModPhys.36.121)
- <span id="page-12-7"></span><sup>57</sup> Note that the temperature dependence of the thermal conductivity of Nb3Sn should also change the stationary solution of the three-dimensional heat equation for D ≤ r ≤ R1. Here we ignore this effect for the sake of mathematical simplicity, since our qualitative conclusions are not altered if we incorporate the temperature dependence of κ in this region.
- <span id="page-12-8"></span><sup>58</sup> D. G. Cahill and R. O. Pohl, [Annual Re](http://dx.doi.org/10.1146/annurev.pc.39.100188.000521)[view of Physical Chemistry](http://dx.doi.org/10.1146/annurev.pc.39.100188.000521) 39, 93 (1988), [https://doi.org/10.1146/annurev.pc.39.100188.000521.](http://arxiv.org/abs/https://doi.org/10.1146/annurev.pc.39.100188.000521)