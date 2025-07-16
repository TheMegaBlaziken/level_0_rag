# arXiv:2201.10733

**Paper ID:** 52048bf61df497eddc920579fcbb1db3

**PDF Path:** apl/Superconductors/arXiv:2201.10733.pdf

**Processing Status:** complete

**Captions Added:** 21

**Generated:** 2025-06-24T13:44:27.578251

---

# Measurement of high quality factor superconducting cavities in tesla-scale magnetic fields for dark matter searches

S. Posen,<sup>∗</sup> M. Checchin,† O. S. Melnychuk, T. Ring, I. Gonin, and T. Khabiboulline

Fermi National Accelerator Laboratory, Batavia, Illinois 60510, USA

(Dated: November 17, 2022)

In dark matter searches using axion haloscopes, the search sensitivity depends on the quality factors (Q0) of radiofrequency cavities immersed in multi-tesla magnetic fields. Increasing Q<sup>0</sup> would increase the scan rate through the parameter space of interest. Researchers developing superconducting radiofrequency cavities for particle accelerators have developed methods for obtaining extremely high Q<sup>0</sup> ∼ 10<sup>11</sup> in µT-scale magnetic fields. In this paper, we describe efforts to develop high Q cavities made from Nb3Sn films using a technique developed for particle accelerator cavities. Geometry optimization for this application is explored, and two cavities are tested: an existing particle accelerator-style cavity and a geometry developed and fabricated for use in high fields. A quality factor of (5.3 ± 0.3)×10<sup>5</sup> is obtained at 3.9 GHz and 6 T at 4.2 K.

#### I. INTRODUCTION

Superconducting radiofrequency (SRF) cavities have been used in particle accelerator applications for decades, and substantial research efforts have gone into developing techniques to maximize cavity performance, in particular the quality factor Q0—which determines the heat dissipated to the cryogenic systems—and the accelerating gradient Eacc—which determines the energy gain per unit length [1]. Modern SRF cavities can routinely achieve Q<sup>0</sup> on the order of 10<sup>11</sup> and Eacc on the order of 40 MV/m.

Axion haloscopes use radiofrequency cavities to search for dark matter [2]. Cavities are placed in magnetic fields on the order of several tesla. Theoretical models for axions [3–5] predict that, if they exist as dark matter [6–8], there is a chance that passing axion particles will be converted to photons, with frequency related to the axion mass (for a review see [9]). If the frequency of the cavity matches that of photons converted from axions, a small signal may be detectable. By tuning the cavity frequency, different potential axion masses can be evaluated. The signal predicted is extremely small, and there is a large mass range of interest. As a result, researchers are developing ways to achieve high sensitivity within a short sampling time for a given frequency, so that the scan rate can be as high as possible, allowing wide ranges to be scanned within reasonable experimental timeframes.

The scan rate is proportional to B<sup>4</sup> 0V <sup>2</sup>C <sup>2</sup>QeffT −2 n , where B<sup>0</sup> = max|B| is the applied magnetic field strength inside the cavity, V is the cavity volume, C is a geometric form factor related to the RF electric field distribution and its alignment with the applied magnetic field (the axion couples to ERF · B, where ERF is the RF electric field), Qeff is the effective quality factor of the cavity, and T<sup>n</sup> is the system noise temperature. Qeff nominally depends on Q0, Qext, and Qa, where Qext is the external quality factor (which depends on the coupling to the cavity and is typically set close to Q0) and Q<sup>a</sup> is the effective quality factor of the galactic axion field ∼ 10<sup>6</sup> . Ref. [10] shows that in both the limit of Q<sup>L</sup> Q<sup>a</sup> and the limit of Q<sup>L</sup> Qa, Qeff ≈ Q<sup>L</sup> = (Q −1 <sup>0</sup> + Q −1 ext) −1 . Thus, there is a substantial benefit to the scan rate to increasing Q<sup>0</sup> up to Q<sup>a</sup> and beyond.

The dependence on quality factor makes the use of SRF cavities a possible avenue for increasing scan rate. Past experiments have typically used copper cavities, including ADMX and HAYSTAC [11, 12], with typical Q<sup>0</sup> values ∼ 104−10<sup>5</sup> depending on the frequency. Early efforts on SRF cavities have been showing promising results, using superconductors like Nb3Sn, NbTi, and YBCO [13– 15]. Promising results have also been obtained by using dielectrics to screen fields from the walls of copper cavities [16, 17]. In this paper, we present high magnetic field Q<sup>0</sup> measurements of SRF cavities that were fabricated with a vapor diffusion technique to coat niobium cavities with an inner layer of Nb3Sn. This method is typically used to coat cavities for particle accelerator applications that can operate with high Q<sup>0</sup> at higher temperatures than cavities made from the standard material Nb [18]. For the first time, we explore the RF performance of vapor diffusion Nb3Sn cavities under multi-tesla magnetic fields. We present a model for flux dissipation in an SRF cavity in a large magnetic field, including a figure of merit and resulting considerations for designing the cavity geometry. Frequency dependence and misalignments are also considered in the model. The cavity results are compared to the model as well as to other experimental efforts with cavities in multi-tesla fields.

## II. MAGNETIC FLUX DISSIPATION IN RF SUPERCONDUCTOR

SRF cavities provide orders of magnitude higher Q<sup>0</sup> than copper in accelerator applications, but the operating conditions for haloscopes are substantially different. SRF cavities in accelerators are typically made of niobium at operate at ∼ 2 or ∼ 4 K, and in magnetic field environments ∼ 1 µT. Under these conditions, the

<sup>∗</sup> sposen@fnal.gov

<sup>†</sup> now at SLAC National Accelerator Laboratory, Menlo Park, CA

some small amount of trapped flux from the ambient field during cooldown. As a result, the surface resistance Rs—which determines Q<sup>0</sup> by Q<sup>0</sup> = G/Rs, where G is a geometry-dependent factor that is independent of frequency—generally is dominated by the temperaturedependent BCS resistance [19] and trapped flux surface resistance: under the influence of RF currents, flux trapped in the superconductor during cooldown undergoes motion, which generates dissipation [20, 21].

niobium walls are primarily in the Meissner state, with

In haloscopes, cavities are operated at tens or hundreds of mK to reduce the noise temperature, where the BCS resistance is exponentially suppressed. However, the magnetic field is typically several tesla, above the upper critical field of niobium, which would leave it in the normal conducting state. Superconductors with substantially higher critical fields would be required to maintain superconductivity, such as those used in superconducting magnets, like Nb3Sn, NbTi, YBCO, and MgB2. Under these conditions, these superconductors would not be in the Meissner state but in the vortex state, with substantial amounts of flux in the superconductor. This is expected to be the dominant source of surface resistance.

To inform efforts to reduce dissipation and maximize Q0, we present a model to describe qualitatively the Qfactor as a function of the trapped magnetic field. The interaction between vortices is neglected and we assume small values of RF field amplitude—lower than the depinning value, as defined in Ref. [22]—so that the vortex response is linear, and independent of the applied RF field. We then calculate the vortex RF response by solving numerically the vortex motion equation for a vortex frozen in the superconductor (z ≥ 0 domain) and subjected to an RF current oscillating along the x direction:

$$\eta\_0 \dot{u}(t, z) = \varepsilon u''(t, z) - \kappa\_\mathrm{p} u(t, z) + \gamma \cos(\omega t) e^{-z/\lambda}, \quad (1)$$

where u(t, z) is the vortex displacement in the y direction per unit of local surface magnetic field Bs, ˙u(t, z) its first order time derivative and u <sup>00</sup>(t, z) its second order derivative with respect to z. A schematic of the vortex is shown in Fig. 1. The parameters η0, ε, κp, and λ are the flux-flow viscosity, vortex elasticity, pinning constant, and penetration depth, respectively; while γ = φ0/((µ0λ)). The viscosity is defined by the Bardeen-Stephen model [23], and it is equal to η<sup>0</sup> = φ<sup>0</sup> Bc2/ρ, where ρ is the normal conductive resistivity, that for Nb3Sn was experimentally found to be in the range ∼ (5 − 90) µΩ cm, [24, 25]. A detailed description of ε is instead found in Ref. [22]. Initial and boundary conditions are u(0, z) = 0, u 0 (t, 0) = 0, and u(t, ∞) = 0.

We assume an average pinning potential U(y) acting on each single vortex described by a Lorentzian function as defined in Ref. [22]. We determine the depth of U(y) by calculating the value that yields a maximum on the pinning force as a function of the vortex displacement equal to the elementary pinning force per unit of length p—for Nb3Sn filaments prepared via "internal Sn"

and "bronze" processes, p is measured to be in the range ∼ (60 − 100) µN/m [26, 27]. For low B<sup>s</sup> amplitudes the vortex displacement is small and the pinning potential can be approximated by a parabola. The pinning constant is then calculated as κ<sup>p</sup> = −u(t, z) <sup>−</sup><sup>1</sup> dU(y)/dy.

The resistance due to N vortices in the area element Σ is defined as:

$$\begin{split} R\_{\text{fl}} &= 2\mu\_0^2 \frac{N}{\Sigma} \frac{\langle P \rangle}{B\_{\text{s}}^2} \\ &= 2\mu\_0^2 \frac{\langle P \rangle}{B\_{\text{s}}^2} \mathbf{B} \cdot \hat{\mathbf{n}}, \end{split} \tag{2}$$

where the flux trapped in the surface element Σ is defined as φ = Nφ<sup>0</sup> = B · ˆn Σ, and hPi is the average singlevortex dissipated power, that is equal to:

$$\langle P \rangle = \frac{\gamma \omega B\_s^2}{2\pi} \int\_0^{2\pi/\omega} \int\_0^\infty \dot{u}(t, z) \cos(\omega t) e^{-z/\lambda} \, dt \, dz. \quad (3)$$

Hence, the total power dissipated by the vortices trapped in the cavity walls (Pc) is defined as the integral of Rfl over the cavity surface—depending on the local internal product B · ˆn—and the local surface RF magnetic field squared B<sup>2</sup> s :

$$P\_{\rm c} = \frac{1}{2\mu\_0^2} \int R\_{\rm fl} B\_{\rm s}^2 \, dA. \tag{4}$$

The cavity internal Q-factor is then easily calculated as:

$$Q\_0 = \frac{\omega W}{P\_\mathbf{c}},\tag{5}$$

![](_page_1_Figure_17.jpeg)

**Caption:** Schematic diagram depicting the oscillation of a vortex in the yz-plane, driven by an RF current along the x-axis. This representation is part of a theoretical model to calculate the resistance generated by trapped vortex lines within superconducting cavities, important for optimizing cavity design for minimal power dissipation under high magnetic fields【4:9†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 1. Schematics of a vortex oscillating in the yz plane driven by an RF current directed along the x axis.

with stored energy defined as W = κB<sup>2</sup> p , where κ is a geometry-dependent parameter calculated via COMSOL [28] simulations (elliptical geometry: κ = 34.8 W/T<sup>2</sup> , CIGAR geometry: κ = 82.3 W/T<sup>2</sup> ) and B<sup>p</sup> the peak surface magnetic field.

Based on the model just described, we can define the proper figure of merit (FoM) necessary to optimize the cavity geometry (for a fixed frequency) with respect to the Q-factor when immersed in a magnetic field:

$$\text{FoM} = \frac{\int \mathbf{B} \cdot \hat{\mathbf{n}} \, |\mathbf{B}\_{\text{RF}}|^2 \, dA}{B\_0 \int |\mathbf{B}\_{\text{RF}}|^2 \, dV},\tag{6}$$

where BRF is the RF magnetic field in the resonator (N.B. at the cavity surface |BRF| = Bs). The FoM so defined is inversely related the to quality factor Q<sup>0</sup> ∼ 1/FoM, the numerator is proportional to the power dissipated by the resonator due to vortex oscillation (see Eq. 4), while the denominator is proportional to the total stored energy W ∝ R |BRF| <sup>2</sup> dV . The highest Q-factor is obtained for geometries that minimize the FoM of Eq. 6, i.e. geometries that simultaneously minimize vortex dissipation and maximize the total stored energy.

Using Eq. 6, one can propose design choices for the cavity geometry to reduce dissipation. It should be possible to greatly reduce the integral R B· ˆn |BRF| <sup>2</sup> dA by developing a geometry with low surface RF magnetic field B<sup>s</sup> in regions where the applied DC magnetic field B is perpendicular to the surface. In other words, in areas where B · ˆn is high, B<sup>s</sup> should be small, and in areas where B<sup>s</sup> is high, B · ˆn should be small. For example, compared to a pillbox geometry, which may have relatively large RF magnetic field on the endcaps which are perpendicular to the applied field, one would expect a benefit from a geometry with more gently tapering ends.

The frequency dependence of vortex surface resistance is also an important dependence to take into consideration during the design of new resonator geometries. Previous studies showed that both PbIn and NbTa alloys and Nb show a sigmoid-like behavior of the vortex surface resistance as a function of the logarithm of frequency [29, 30], the same behaviour is to be expected for Nb3Sn.

In Fig. 2, the vortex surface resistance as a function of frequency expected for Nb3Sn (using the material parameters described above) is shown. As described in Ref. [30], the plateau in the low frequency regime is governed by pinning, while the plateau at high frequency regime is governed by flux-flow and represents maximum surface resistance value per amount of B0. The depinning frequency—defined as the frequency value at which Rfl is half the flux-flow value—is simulated to be equal to about 12 GHz.

#### III. CAVITY GEOMETRIES

The first cavity geometry chosen to study was that of TESLA [31], shown in Figure 3. This geometry is widely

![](_page_2_Figure_10.jpeg)

**Caption:** Graph showing the dependence of vortex surface resistance on frequency for a TESLA-style superconducting cavity. The resistance calculation assumes a particular alignment of the cavity relative to an external magnetic field, emphasizing critical aspects of resonator design to minimize power losses in high-field applications【4:8†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 2. Vortex surface resistance dependence over frequency. Rfl was calculated assuming B · ˆn = 1.

used in particle accelerators, including for example the European XFEL and LCLS-II [32, 33]. This geometry is optimized for accelerators, not for high magnetic fields, but existing cavities were on hand to conduct first experiments. In the Appendix, it is shown that on a cavity of this shape, substantial flux losses occur in regions where surface currents are nearly perpendicular to the applied field. The accelerating mode would also be the mode relevant for axion searches, the fundamental TM010 mode of the cavity.

While first studies were being set up on a TESLA-style cavity, a new cavity geometry was developed and began fabrication. Considering the the analysis in Section II, the geometry was developed with an aim to achieve low surface RF magnetic field B<sup>s</sup> in regions where the applied DC magnetic field B is perpendicular to the surface. The overall optimization was conducted by assuming an uniform magnetic field oriented along the z direction (that coincides with the cavity symmetry axis).

A cigar-shaped cavity was chosen out of the different geometries evaluated, shown in Figure 4. The relevant mode for axion searches is the TM010 mode, which is the ninth-lowest frequency mode of the cavity, above four degenerate TE11n modes. The geometric form factor C for the TM010 mode is simulated to be 0.499 and the RF volume is 5.05 × 10<sup>−</sup><sup>4</sup> m<sup>3</sup> . Figure 5 shows a comparison of the two geometries, using the numerator of the figure of merit in Eq. 6, illustrating the advantage the cigarshaped geometry compared to the TESLA geometry for high magnetic field applications.

![](_page_3_Figure_1.jpeg)

**Caption:** This figure illustrates the normalized electric (left) and magnetic (right) field intensity of the TM010 mode in the cross-section of a TESLA cavity, commonly used in particle accelerators like those at the European XFEL and LCLS-II. The visualization highlights the field distribution, which is crucial for understanding how the cavity interacts with magnetic fields and influences the quality factor (Q0) in high magnetic field environments, relevant to efficiently searching for axion dark matter【4:13†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 3. Normalized electric (left) and magnetic (right) field intensity of the TM010 mode in a cross section through the middle of a TESLA cavity used in particle accelerators [31].

![](_page_3_Figure_3.jpeg)

**Caption:** Visualization of the normalized electric and magnetic field intensities of the TM010 mode in a cross-section of the newly developed cigar-shaped cavity. This geometry, compared to the traditional TESLA cavity, offers reduced flux dissipation in high magnetic fields, increasing cavity Q-factor significantly for both fundamental studies and practical applications【8:19†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 4. Normalized electric (left) and magnetic (right) field intensity of the TM010 mode in a cross section through the middle of the cigar geometry cavity developed as part of this work.

### IV. EXPERIMENTAL APPARATUS

The high magnetic field test stand available for carrying out the measurements was an Oxford TeslatronTM system at Fermilab [34], which previously has mainly been used to test wires for superconducting magnet application. It has a NbTi solenoid magnet capable of reaching ∼ 6 T with a 147 mm bore and is operated in liquid helium. A special insert was fabricated that allowed for a cavity to be assembled and evacuated in a cleanroom, then inserted into the dewar. The system is shown in

![](_page_3_Figure_7.jpeg)

**Caption:** Comparison of the magnetic field orientation relative to the RF surface currents in TESLA and cigar-shaped cavities, normalized to the same stored energy. The cigar-shaped cavity demonstrates a designed reduction in the RF magnetic field's impact, aiming to increase the quality factor by minimizing power dissipation, a significant advancement for cavities operating under high magnetic fields【4:19†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 5. Comparison of ˆz · ˆn|BRF| 2 from Eq. 6 in the TESLA and cigar-shaped cavities. For the plots that include BRF, the cavities are normalized to the same stored energy. The cigar-shaped cavity was designed to substantially reduce the numerator in Eq. 6.

Figure 6.

In this system, the solenoid current is measured, and converted to peak applied magnetic field on the central axis based on information from the vendor. Cavities are assembled with two antennae attached to feedthroughs. Coaxial cables bring the signal through the helium volume to feedthroughs on the top plate, which in turn are connected to a vector network analyzer. The network analyzer is used to measure the loaded quality factor Q<sup>L</sup> by fitting the resonance curve in transmission through the cavity. A simple signal map is shown in Figure 7.

Figure 8 shows a magnetic field intensity map of the solenoid (map was simulated based on coil position and should be considered approximate) as well as how the cavities are designed to be positioned inside of the system. The solenoid is closed on the bottom, and the TESLA cavity position had to be raised slightly higher than the region of maximum field intensity to fit above the bottom plate with the RF hardware attached. The figure illustrates how the magnetic field lines cross the TESLA cavity nearly perpendicularly in regions with high surface RF magnetic field, while they are nearly parallel to the relevant regions of the cigar-shaped cavity.

The Fermilab particle accelerator research program had already generated Nb3Sn TESLA-style cavities with frequency 1.3 GHz and 3.9 GHz (the 3.9 GHz cavity doesn't exactly match the TESLA geometry, but it is close; for simplicity we refer to it as the TESLA cavity). The 3.9 GHz cavity was chosen because it fit in the bore of the solenoid (∼ 4 GHz is also an interesting frequency

![](_page_4_Picture_1.jpeg)

**Caption:** Photograph showing the TeslatronTM system at Fermilab with its cryostat and solenoid components clearly visible. This system supports the setup and testing of superconducting cavities such as the Nb3Sn TESLA cavity, facilitating experiments aimed at achieving high Q-factors under controlled magnetic field conditions【8:11†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 6. TeslatronTM system at Fermilab including cryostat (background) and solenoid (foreground). The inset shows the cavity insert with the 3.9 GHz Nb3Sn TESLA cavity installed.

![](_page_4_Figure_3.jpeg)

**Caption:** A schematic of the signal map in the experimental setup used to measure the quality factor Q0 of SRF cavities in high magnetic fields. This diagram is essential to understanding the configuration and data flow in the high magnetic field Q0 measurement, which is pivotal for applications in axion dark matter searches【12:8†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 7. Signal map for high magnetic field Q<sup>0</sup> measurement.

for future axion studies, e.g., [35]). For proper comparison, 3.9 GHz was also chosen for the frequency at which to fabricate the cigar-shaped cavity.

The previous performance of the 3.9 GHz TESLA-style cavity is shown in Figure 9. The performance was measured at 4.4 K after cooldown in a < 1 µT magnetic field. The Nb3Sn cavity started as a niobium cavity, which was electropolished, then Nb3Sn vapor diffusion coated, as described in [18], then rinsed with high pressure ultrapure water, and assembled for measurement in a class 10 cleanroom.

To prepare for the high magnetic field measurement, the TESLA cavity was disassembled from its previous assembly, high pressure rinsed, then assembled in a clean-

![](_page_4_Figure_8.jpeg)

**Caption:** Calculated magnetic field intensity within the Dewar, depicting field directions and intensities at a solenoid center field of 6 T. The illustration includes the positioning outlines for TESLA and cigar-shaped cavities, enabling the analysis of how the field orientation affects cavity performance due to alignment with the resonator's surface RF magnetic field【8:15†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 8. Calculated magnetic field intensity inside the dewar when the center of the solenoid is at 6 T. R = 0 corresponds to the axis of symmetry. The dotted blue line shows the boundary of the region inside the solenoid. The red arrows indicate the magnetic field direction and relative intensity. The white solid line is the outline of the TESLA cavity positioned as expected in the system, and the black outline similarly is for the cigar-shaped cavity.

room to the insert shown in Figure 6.

The cigar-shaped cavity was milled from a pair of solid blocks of reactor-grade niobium. NbTi flanges were then welded to its ports. The cavity was then electropolished using a typical niobium SRF cavity electropolishing process [36], with a specially shaped aluminum anode and corresponding fixturing, shown in Figure 10. The cavity was coated with Nb3Sn, then rinsed with high pressure ultrapure water, and assembled for measurement, as shown in Figure 11.

#### V. RESULTS

The main results of this work are shown in Figure 12. Results are given in terms of Q<sup>L</sup> vs applied magnetic field B0. The applied magnetic field is the peak DC field that would be expected to be measured on the central

![](_page_5_Figure_0.jpeg)

**Caption:** Performance data of a Nb3Sn TESLA cavity, illustrating Q0 as a function of Eacc at a test temperature of 4.4 K. The plot reveals the cavity's operational characteristics after a cooldown in magnetic fields below 1 µT, critical for assessing the cavity's viability in high-field environments intended for dark matter axion searches【12:11†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 9. Previous performance of Nb3Sn TESLA cavity at 4.4 K, after cooldown in a < 1 µT magnetic field.

![](_page_5_Picture_2.jpeg)

**Caption:** This image showcases the electropolishing setup for a cigar-shaped cavity, wherein an aluminum anode is lowered to match the cavity's geometry. Electropolishing is a critical step in preparing the cavity surface for Nb3Sn coating, which in turn affects the cavity's superconducting properties and enhances its potential use in high magnetic field applications for dark matter detection【4:3†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 10. Side view of the electropolishing setup for the cigarshaped cavity. The aluminum anode is lowered from above; it was fabricated with geometry to match the cavity.

axis of the solenoid. All cavities were cooled down with near-zero current in the solenoid, then the current was increased in steps. The Q<sup>L</sup> was measured by the network analyzer. At low B0, Q<sup>L</sup> is expected to be dominated by the external quality factors Qext of the coupling ports (Q −1 <sup>L</sup> = Q −1 <sup>0</sup> + Q −1 ext1 + Q −1 ext2) and by trapped flux from during the cooldown of the cavity (from Earth's magnetic field and from residual magnetization in the dewar). At high B0, Q<sup>L</sup> is expected to be dominated by Q<sup>0</sup> of the cavity. Qext values were chosen to be at least an order of magnitude above the expected range of Q<sup>0</sup> so that it wouldn't contribute significantly to the overall Q<sup>L</sup> at high fields, but still low enough that the power coupled between the antennae and the cavity would provide a strong signal. For the TESLA cavity the Qext values were measured to be (3.4 ± 0.5) × 10<sup>8</sup> and (2.3 ± 0.3) × 10<sup>8</sup> . For the cigar-shaped cavities the Qext values were measured to be (8.2 ± 4.1) × 10<sup>8</sup> and (7.4 ± 3.7) × 10<sup>8</sup> (there was higher uncertainty in measuring Qext values at room temperature for the cigar cavity due to other modes close to the TM010 modes).

For these Nb3Sn-coated Nb cavities, the niobium bulk dominated the behavior at low fields. The cavity acted as

![](_page_5_Picture_6.jpeg)

**Caption:** Photograph of the cigar-shaped SRF cavity post-Nb3Sn coating and its assembly for testing. This cavity design aims to enhance performance at high magnetic fields by reducing flux dissipation, showcasing enhancements in superconductor technology that could be leveraged for particle accelerators and axion dark matter detection【8:1†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 11. Top: Cigar-shaped cavity after Nb3Sn coating. Bottom: Cigar-shaped Nb3Sn cavity assembled to the test insert.

![](_page_5_Figure_8.jpeg)

**Caption:** Plot showing the loaded quality factor as a function of applied magnetic field for both TESLA and cigar-shaped Nb3Sn-coated cavities. The inset zooms in on the low-field region. Results depict how field penetration affects superconducting performance, where the cigar-shaped cavity maintains a significantly higher quality factor at 6 T, offering promising results for axion dark matter searches【4:3†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 12. Loaded quality factor vs applied magnetic field data for the TESLA and cigar-shaped Nb3Sn cavities. The inset provides a zoom of the low field region. For the cigar-shaped Nb3Sn cavity, measurements were made both for increasing and for decreasing field.

a Meissner shield to the applied field until around 0.1 T, at which point it appears that flux begins to penetrate the niobium walls. This is approximately half of the nominal field at which flux penetration is expected for niobium, with the difference likely due to demagnetization effects. After flux penetrates, the Q<sup>0</sup> decreases steadily with increasing field, then begins to start decreasing more slowly towards higher applied fields, particularly above ∼ 3 T. By the time the cavities reach 6 T, the TESLA cavity Q<sup>0</sup> is (4.3±0.2)×10<sup>4</sup> , and the cigar-shaped Nb3Sn cavity Q<sup>0</sup> is (5.3±0.3)×10<sup>5</sup> (this assumes that Q<sup>0</sup> dominates Q<sup>L</sup> based on the comparatively high Qext values at B<sup>0</sup> = 6 T).

At the maximum applied field of 6 T, the incident power from the network analyzer was lowered to evaluate if would alter the measurement. The power was adjusted from 10 dBm to 0 dBm and then to -10 dBm.
![](0__page_6_Figure_1.jpeg)

**Caption:** The relationship between frequency shift and applied magnetic field for TESLA and cigar-shaped Nb3Sn cavities. The data underline the minimal deformation of the cigar-shaped cavity at higher fields due to its stiffer design, supporting its viability in maintaining frequency stability essential for high-precision dark matter searches【8:5†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 13. Frequency vs applied magnetic field data for the TESLA and cigar-shaped Nb3Sn cavities.

For both cavities, no change in Q<sup>0</sup> was observed within uncertainty for these 3 power levels.

For the cigar shaped cavity, after reaching the peak field of 6 T, the field was decreased in steps to check for hysteresis. The increasing curve was followed until the field was decreased to ∼ 0.5 T. At low field values, the Q<sup>L</sup> was smaller than for the increasing curve, suggesting additional trapped flux was present compared to after cooldown.

The frequency change vs applied magnetic field data is shown in Figure 13. The cavities were not tuned exactly to 3.9 GHz – the zero-field frequency f<sup>0</sup> for the TESLA Nb3Sn cavity was 3870 MHz. f<sup>0</sup> for the cigar-shaped Nb3Sn cavity was 3891 MHz. The frequency of the cigarshaped cavity changed less at higher fields. This may be due to the stiffer cavity resulting in smaller cavity deformation from magnetic forces.

# VI. COMPARISON OF RESULTS TO THEORETICAL MODEL

In Fig. 14(a), the model data from Section II is compared to the experimental data for the two geometries assuming p = 8 µN/m and ρ = 60 µΩ cm. In Fig. 14(b) and (c), the dissipation pattern expected on the cavity surface for the two geometries is shown. The simulations were performed assuming the actual DC magnetic field distribution in the dewar. The pinning force assumed in the simulations is lower than the range reported in literature [26, 27]; however, since the material considered in this work is produced by direct solid-vapour reaction of Sn on Nb (no Cu or bronze are involved in the reaction), it has high purity and large grains [18], and one can reasonably expect lower pinning force than what previously measured in previous works [26, 27].

While the model matches the experimental data reasonably well, some discrepancy between the two can be observed. As introduced previously, this model neglects the interaction between vortices, which is not negligible

![](0__page_6_Figure_9.jpeg)

**Caption:** This figure presents simulated internal quality factor (Q0) as a function of the maximum magnetic field generated by a solenoid (a). Power dissipation per unit area on two different cavity geometries is shown in (b) and (c), highlighting the impact of cavity design on performance in high magnetic environments. Such analysis is crucial for improving efficiency in systems aiming to detect axion dark matter【4:7†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 14. Internal quality factor dependence as a function of the maximum magnetic field generated by the solenoid is shown in figure (a). In figures (b) and (c), the distribution of power dissipation per unit of area on the cavity surface is shown for the two geometries under consideration (normalized to maximum dissipation).

at high fields where vortices organize in an Abrikosov's lattice [37]. In such a condition, the deformation of the vortex lattice subjected to the RF current should be taken into consideration [38], but this goes beyond the scope of this qualitative model. Because of this approximation, the model overestimates the dissipation at high fields. The more rigid nature of the lattice compared to a single vortex implies less displacement per RF semiperiod, traducing to a lower vortex velocity and hence lower power dissipation. Nevertheless, this simple model can provide important information to properly design resonators for operation in high-magnetic field values.

To further examine possible sources of discrepancy, we explore the effect of misalignment of the two resonators with respect to the Dewar magnetic center. To do this, we introduce the parameters δ, ∆x, and ∆z that represent the tilt angle about the y-axis with respect to the z axis, and displacements along the x-axis and z-axis of the Dewar, respectively. The dot product is defined as:

$$\begin{aligned} \mathbf{B} \cdot \hat{\mathbf{n}} &= B\_{\mathbf{x}}(x', y', z') \left( n\_{\mathbf{x}} \cos \delta + n\_{\mathbf{z}} \sin \delta \right) \\ &+ B\_{\mathbf{y}}(x', y', z') \, n\_{\mathbf{y}} \\ &+ B\_{\mathbf{z}}(x', y', z') \left( -n\_{\mathbf{x}} \sin \delta + n\_{\mathbf{z}} \cos \delta \right), \end{aligned} \tag{7}$$

where the coordinates x 0 , y 0 , and z 0 represents points on the cavity surface after rotation and translation, and are defined as:

$$
\begin{pmatrix} x' \\ y' \\ z' \end{pmatrix} = \begin{pmatrix} x\cos\delta + z\sin\delta + \Delta x \\ y \\ -x\sin\delta + z\cos\delta + \Delta z \end{pmatrix} . \tag{8}
$$

In Fig. 15, the results of the analysis are shown. The plots in Fig. 15(a), (d), and (g) show the variation of Q<sup>0</sup> with respect to the perfect alignment condition times the maximum magnetic field in the solenoid (B0) as a function of δ, ∆x, and ∆z. The power areal density plots in Fig. 15(b), (c), (e), (f), (h), and (i), show instead

![](0__page_7_Figure_0.jpeg)

**Caption:** Graphs displaying the variation in the quality factor Q0 with respect to misalignment parameters δ, Δx, and Δz. Adjacent figures depict the power dissipation pattern on the cavity surface at maximum misalignment. This highlights the sensitivity of cigar-shaped resonators to misalignment, underscoring the importance of precision in cavity alignment for high-field applications【4:2†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. 15. Variation of the quality factor times B<sup>0</sup> as a function of the misalignment parameters δ, ∆x, and ∆x are shown in figures (a), (d), and (g). The distribution of power dissipation per unit of area on the cavity surface for the two geometries under consideration calculated at the respective maximum misalignment value are shown in figures (b), (c)—δ = 15 deg, (e), (f)—∆x = 1.5 cm, and (h), (i)—∆z = 3 cm (all normalized to maximum dissipation).

the trapped flux dissipation pattern assuming the maximum misalignment simulated—δ = 15 deg for (b) and (c), ∆x = 1.5 cm for (e) and (f), and ∆z = 3 cm for (h) and (i). Clearly, the CIGAR resonator is more sensitive to misalignment compared to the elliptical resonator, even though its shape minimizes vortex dissipation when properly aligned to the solenoid magnetic center.

TABLE I. Comparison of the results from this work to several previous studies of superconducting cavities in high magnetic fields for axion research.

| Source          | Material     | f (GHz) B0 |      | (T) T (K) Q0 |                   |
|-----------------|--------------|------------|------|--------------|-------------------|
| This work Nb3Sn |              | 3.9        | 6.0  | 4.2          | (5.3 ± 0.3) × 105 |
| [13]            | NbTi/Cu 9.08 |            | 5    | 4.2          | 2.95 × 105        |
| [14]            | Nb3Sn        | 9          | 8    | 4.2          | 6 × 103           |
| [14]            | REBCO        | 9          | 11.6 | 4.2          | 7 × 104           |
| [15]            | YBCO         | 6.93       | 8.0  | 4.2          | 3.2 × 105         |

## VII. DISCUSSION

The highest Q<sup>0</sup> reported in this study at 6 T is (5.3 ± 0.3)×10<sup>5</sup> for the Nb3Sn cigar shaped cavity. The surface resistance for a copper cavity at the same temperature, magnetic field, and frequency can be calculated based on [39] to be 0.0029 Ω. Using this surface resistance and the same geometry factor as the cigar cavity would yield a Q<sup>0</sup> of 1.6 × 10<sup>5</sup> , a factor of 3.4 lower than the Nb3Sn cigar shaped cavity. This suggests that the use of Nb3Sn coatings could be a promising direction for improving axion search scan rate near this frequency. It may also improve at lower temperatures.

The results presented here are compared to several previous studies of superconducting cavities in high magnetic fields in Table I. It should be noted in these comparisons that differences in frequency, cavity geometry, and magnetic field value could substantially impact the Q<sup>0</sup> value. The previous materials studied included conventional superconductors like NbTi as well as high temperature superconductors like REBCO (RE stands for a rare earth metal such as Y). The Q<sup>0</sup> values reported from this study are the highest of those in the table, but also the frequency is substantially lower than the other cavities, which likely contributes to smaller flux losses. Furthermore, there are yet-to-be published results from the team from Ref. [15] showing a Q<sup>0</sup> in the 10<sup>7</sup> range at 8 T [40]. Overall, superconducting cavities appear to be a promising avenue to increasing Q<sup>0</sup> in high magnetic fields, including the Nb3Sn coatings developed for accelerator applications. The difference in performance between the TESLA and cigar-shaped cavities show the importance of selecting an appropriate geometry that takes into account flux losses in the superconducting material.

Another recent result of a novel cavity developed to achieve high Q<sup>0</sup> in high magnetic fields is the 10.3 GHz dielectric resonator from di Vora et al. [7], which has a Q<sup>0</sup> of 9 × 10<sup>6</sup> at 4.2 K in a 8 T magnetic field. The use of cylindrical dielectrics to screen the field from the copper walls, meaning that there is a significant volume of the cavity with low RF field amplitude and therefore weak axion sensitivity. However, the high Q<sup>0</sup> may have a stronger impact, particularly at high frequencies ∼ 10 GHz and above.

Axion search scan rate scales with the fourth power of magnetic field, and fields significantly higher than 6 T are readily achievable, though not in the measurement system used in this paper. The Q<sup>0</sup> of the Nb3Sn cigar shaped cavity was decreasing relatively slowly with field at 6 T, and it may remain quite high up to higher magnetic fields. For example, between 4 T and 6 T, the Q<sup>0</sup> decreased from 6.6 × 10<sup>5</sup> to 5.3 × 10<sup>5</sup> . If this continued linearly, it would result in a Q<sup>0</sup> of 4.3 × 10<sup>5</sup> at 8 T. The upper critical field of Nb3Sn is substantially higher, ∼ 30 T for near-optimal stoichiometry [41].

There are some simple next steps for this experimental program. The first will be to test a new NbTi cavity with the cigar shape and compare its performance to the Nb3Sn cavities. The second will be to install a pump in the cryogenic system, which will allow future measurements to be performed also below 4 K, where pinning strength may be improved, and possibly also Q0.

For longer-term next steps, work is underway to develop a tunable superconducting cavity geometry with surface currents highly parallel to the applied field, as they are in the cigar shape. In addition, studies will be performed to modify the Nb3Sn coating process to try to increase pinning strength. Finally, procurement is underway for a millikelvin high magnetic field test stand, where studies can be performed under temperature and magnetic field conditions as close as possible to a dark matter search.

## VIII. SUMMARY

Nb3Sn SRF cavity technology developed for particle accelerators was studied at multi-tesla DC magnetic fields as a means to increase Q<sup>0</sup> in dark matter search applications and thereby improve the scan rate. A Nb3Sn coated cavity with a cigar-shaped geometry chosen to reduce flux dissipation had a Q<sup>0</sup> of (5.3 ± 0.3) × 10<sup>5</sup> at 6 T and 4.2 K. A Nb3Sn coated cavity with a geometry typical of accelerator applications had a Q<sup>0</sup> of (4.3±0.2)×10<sup>4</sup> at 6 T and 4.2 K. Results were compared to a theoretical model developed for SRF cavities in high magnetic fields. The model includes a figure of merit to guide geometry optimization and an estimate of frequency scaling.

## IX. ACKNOWLEDGEMENTS

The authors would like to thank the many people who contributed to the results presented in this paper. Thanks to Vadim Kashikin for extremely useful discussions of flux behavior in superconducting materials under multi-tesla fields, including important safety aspects of this experiment. Thanks to Vito Lombardo for sharing a model of the field inside the solenoid. Thanks to Eddie Pieszchala, Mike Foley, and the Fermilab machine shop for cavity fabrication. Thanks to Daniele Turrioni, Allen Rusy, and Simone Johnson, who run the lab with the 6 T magnet, and carried out the installation of the cavity insert to the solenoid and operation of the magnet and cryo system. Thanks to the ADMX collaboration, especially Andrew Sonnenschien and Daniel Bowring, who partially supported the fabrication of the cavity insert and who contributed to the investigations of the effects of putting large masses of superconducting material in the magnet. Thanks to Scott Adams and Tedd Ill for assembly of the cavities to the insert. Thanks to Brad Tennis for carrying out coating of the Nb3Sn cavities and initial setup of the cavity insert. Thanks to Anna Grassellino, Alex Romanenko, Raphael Cervantes, and Roni Harnik of Fermilab, Jim Sauls, Venkat Chandrasekhar, and Bill Halperin of Northwestern University and Phil O'Larey and Miles Naughton of ATI Metals for useful discussions. This work was supported by the United States Department of Energy, Office of High Energy Physics under Contracts DE-AC05-06OR23177 Fermilab. This material is based upon work supported by the U.S. Department of Energy, Office of Science, National Quantum Information Science Research Centers, Superconducting Quantum Materials and Systems Center (SQMS) under contract number DE-AC02-07CH11359.

## Appendix A: Measurement with 100 µT applied field

As an initial evaluation of Q<sup>0</sup> of an SRF cavity in a magnetic field, a 1.3 GHz Nb3Sn coated TESLA cavity was tested after cooldown with an applied magnetic field of 100 µT, generated by Helmholtz coils around the cavity (Helmholtz coils can be seen in Figure A.1). The cavity was cooled slowly and uniformly to try to fully trap the applied field [42] and avoid thermocurrents [43]. Other instrumentation around the cavity included a temperature map (or T-map), an array of 540 thermometers placed over the surface of the cavity (see [44] for more information about T-mapping of SRF cavities). Figure A.2 shows an azimuthal cross section of the cavity with the approximate location of the temperature sensors along its surface. The cavity and RF fields are axisymmetric about R = 0. There are 36 boards, each with 15 sensors, located 10 degrees apart azimuthally. The RF magnetic field strength is approximately the same (within ∼ 5%) for the middle 11 sensors, and somewhat lower for the two each at the top and bottom of the cavity (∼ 83% and ∼ 62% of the peak surface magnetic field value).

A temperature map measured at an accelerating gradient of 9.9 MV/m and a temperature of 1.6 K is shown in Figure A.3. The ∆T values are given relative to the ambient bath temperature. The quality factor was measured to be 2.1 × 10<sup>8</sup> , substantially lower than typical values between 10<sup>10</sup> and 10<sup>11</sup> at this temperature, indicating high surface resistance due to trapped flux. The T-map shows where the surface resistance was high and where it was low: surface temperature rise is proportional to the local RF power dissipation. Notice the consistency of the heating pattern with the shape of ˆz · ˆn|BRF| 2 from Fig. 5. As a result of these regions that show high dissipation, the TESLA geometry is expected to have substantial Q<sup>0</sup> degradation cooled in a large magnetic field.

![](0__page_9_Picture_1.jpeg)

**Caption:** Image depicting a 1.3 GHz superconducting RF cavity with Helmholtz coils configured around it. The coils generate a controlled magnetic field to study the cavity's performance under different magnetic field conditions, which is crucial for maximizing quality factors in contexts like dark matter axion searches where high sensitivity is needed【8:4†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. A.1. A 1.3 GHz cavity with Helmholtz coils around it.

![](0__page_9_Figure_3.jpeg)

**Caption:** Graph demonstrating loading quality factor measurements for TESLA and cigar-shaped Nb3Sn cavities against a background of references from notable research on dark matter axions. The data show improvements in cavity performance at substantial field strengths, offering insights into efficiency improvements achievable through advanced cavity designs【8:10†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


- [2] P. Sikivie, Physical Review Letters 51, 1415 (1983).
- [3] R. D. Peccei and H. R. Quinn, Physical Review Letters 38, 1440 (1977).
- [4] R. D. Peccei and H. R. Quinn, Physical Review D 16, 1791 (1977).
- [5] S. Weinberg, Physical Review Letters 40, 223 (1978).
- [6] J. Preskill, M. B. Wise, and F. Wilczek, Physics Letters B 120, 127 (1983).
- [7] L. Abbott and P. Sikivie, Physics Letters B 120, 133 (1983).
- [8] M. Dine and W. Fischler, Physics Letters B 120, 137 (1983).
- [9] P. W. Graham, I. G. Irastorza, S. K. Lamoreaux, A. Lindner, and K. A. Van Bibber, Annual Review of Nuclear and Particle Science 65, 485 (2015), arXiv:1602.00039.

![](0__page_9_Figure_12.jpeg)

**Caption:** This figure shows the temperature map sensor locations around an azimuthal slice of an SRF cavity. There are 36 slices, each equipped with temperature sensors to monitor the surface temperature, which helps to assess the local RF power dissipation and surface resistance characteristics. Such data are critical for understanding and minimizing quality factor degradation in high magnetic fields, as they provide insights into regions with higher dissipation due to trapped flux【8:10†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. A.2. Temperature map sensor locations around an azimuthal slice of the cavity. There are 36 such slices around the cavity.

In addition to T-map data, Q<sup>0</sup> vs Eacc data were measured. Measurements were made at high fields using typical SRF power balance methods [45] as well as with low power decays [46]. An example of a decay measurement is plotted in Figure A.4, in which the slope of the transmitted power vs time is used to determine QL. Then the Qext values determined from the power balance measurements are used to convert from Q<sup>L</sup> to Q0. The Q<sup>0</sup> vs Eacc data are plotted in Figure A.5.

There is a transition that occurs between ∼ 0.01 MV/m and ∼ 1 MV/m, in which the Q<sup>0</sup> decreases from ∼ 2 × 10<sup>9</sup> to ∼ 3 × 10<sup>8</sup> . This may be related to dissipative depinning of flux in the Nb3Sn (see for example measurements in [47]) at higher RF fields. The fields used here (even down to 0.01 MV/m) are significantly higher than would needed for axion applications, and higher than are excited by the network analyzer in the high magnetic field measurements of the 3.9 GHz cavities.

- [10] R. Cervantes, C. Braggio, B. Giaccone, D. Frolov, A. Grassellino, R. Harnik, O. Melnychuk, R. Pilipenko, S. Posen, and A. Romanenko, "Deepest sensitivity to wavelike dark photon dark matter with srf cavities," (2022).
- [11] T. Braine, R. Cervantes, N. Crisosto, N. Du, S. Kimes, L. J. Rosenberg, G. Rybka, J. Yang, D. Bowring, A. S. Chou, R. Khatiwada, A. Sonnenschein, W. Wester, G. Carosi, N. Woollett, L. D. Duffy, R. Bradley, C. Boutan, M. Jones, B. H. Laroque, N. S. Oblath, M. S. Taubman, J. Clarke, A. Dove, A. Eddins, S. R. O'kelley, S. Nawaz, I. Siddiqi, N. Stevenson, A. Agrawal, A. V. Dixit, J. R. Gleason, S. Jois, P. Sikivie, J. A. Solomon, N. S. Sullivan, D. B. Tanner, E. Lentz, E. J. Daw, J. H. Buckley, P. M. Harrington, E. A. Henriksen, and K. W. Murch, Physical Review Letters 124, 101303 (2020), arXiv:1910.08638.

![](0__page_10_Figure_1.jpeg)

**Caption:** Temperature map of a 1.3 GHz TESLA-style cavity, highlighting areas of elevated temperature relative to a 100 µT field application. The map underscores regions of high local surface resistance, informative for evaluating Q0 degradation during operation and reinforcing cavity design modifications to reduce power losses under experimental conditions relevant to axion search applications【4:11†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. A.3. Temperature map of the 1.3 GHz Nb3Sn cavity cooled in a 100 µT field, with 9.9 MV/m accelerating gradient at 1.6 K and a Q<sup>0</sup> of 2.1 × 10<sup>8</sup> .

![](0__page_10_Figure_3.jpeg)

**Caption:** This graph plots the transmitted power decay over time used to calculate the loaded quality factor (QL) of the cavity in high field conditions. The initial high power decay followed by stabilization is indicative of the cavity's ability to sustain performance levels, crucial in determining SRF cavity efficiency for targeted applications like axion haloscopes【8:17†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. A.4. Example of a low power decay from a known field. The red line shows where the data was fit to extract Q<sup>0</sup> values vs field at low Eacc.

- [12] L. Zhong, S. Al Kenany, K. M. Backes, B. M. Brubaker, S. B. Cahn, G. Carosi, Y. V. Gurevich, W. F. Kindel, S. K. Lamoreaux, K. W. Lehnert, S. M. Lewis, M. Malnou, R. H. Maruyama, D. A. Palken, N. M. Rapidis, J. R. Root, M. Simanovskaia, T. M. Shokair, D. H. Speller, I. Urdinaran, and K. A. Van Bibber, Physical Review D 97, 92001 (2018), arXiv:1803.03690.
- [13] D. Alesini, C. Braggio, G. Carugno, N. Crescini, D. D'Agostino, D. Di Gioacchino, R. Di Vora, P. Falferi, S. Gallo, U. Gambardella, C. Gatti, G. Iannone, G. Lamanna, C. Ligi, A. Lombardi, R. Mezzena, A. Ortolan, R. Pengo, N. Pompeo, A. Rettaroli, G. Ruoso, E. Silva, C. C. Speake, L. Taffarello, and S. Tocci, Physical Review D 99, 101101 (2019), arXiv:1903.06547.
- [14] J. Golm, S. A. Cuendis, S. Calatroni, C. Cogollos, B. D¨obrich, J. D. Gallego, J. M. G. Barcel´o, X. Granados, J. Gutierrez, I. G. Irastorza, T. Koettig, N. Lamas, J. Liberadzka-Porret, C. Malbrunot, W. L. Millar, P. Navarro, C. P. Carlos, T. Puig, G. J. Rosaz, M. Siodlaczek, G. Telles, and W. Wuensch, , 3 (2021), arXiv:2110.01296.
- [15] D. Ahn, O. Kwon, W. Chung, W. Jang, D. Lee, J. Lee, S. W. Youn, D. Youm, and Y. K. Semertzidis, arxiv:2002.08769 (2020), arXiv:2002.08769.
- [16] R. Di Vora, D. Alesini, C. Braggio, G. Carugno, N. Crescini, D. D. Agostino, D. Di Gioacchino, P. Falferi, U. Gambardella, C. Gatti, G. Iannone, C. Ligi, A. Lom-

![](0__page_10_Figure_10.jpeg)

**Caption:** This figure depicts the quality factor (Q0) of superconducting radiofrequency (SRF) cavities as a function of accelerating electric field (Eacc) using both low power decay and power balance methods. It shows a clear transition where Q0 drops from ~2 × 10^9 to ~3 × 10^8 as Eacc increases from 0.01 MV/m to 1 MV/m. This drop is attributed to dissipative behavior at higher RF fields, shedding light on the challenges of maintaining high Q0 under such conditions【12:7†temp_paper_52048bf61df497eddc920579fcbb1db3.txt】.


FIG. A.5. Q<sup>0</sup> vs Eacc data measured using two methods: power balance and low field decays. The observed change in Q<sup>0</sup> may be related to depinning of trapped flux at fields above Eacc ∼ 0.1 MV/m.

bardi, G. Maccarrone, A. Ortolan, R. Pengo, A. Rettaroli, G. Ruoso, L. Taffarello, and S. Tocci, , 1 (2022), arXiv:2201.04223.

- [17] D. Alesini, C. Braggio, G. Carugno, N. Crescini, D. D' Agostino, D. Di Gioacchino, R. Di Vora, P. Falferi, U. Gambardella, C. Gatti, G. Iannone, C. Ligi, A. Lombardi, G. Maccarrone, A. Ortolan, R. Pengo, C. Pira, A. Rettaroli, G. Ruoso, L. Taffarello, and S. Tocci, Nuclear Instruments and Methods in Physics Research, Section A: Accelerators, Spectrometers, Detectors and Associated Equipment 985, 164641 (2021), arXiv:2004.02754.
- [18] S. Posen and D. L. Hall, Supercond. Sci. Technol. 30, 033004 (2017).
- [19] D. C. Mattis and J. Bardeen, Physical Review 111, 412 (1958).
- [20] C. Benvenuti, S. Calatroni, I. E. Campisi, P. Darriulat, C. Durand, R. Russo, and A. Valente, Workshop on RF Superconductivity 1997, Tech. Rep. October 1997 (CERN EST/97-08 (SM), 1997).
- [21] M. Martinello, M. Checchin, A. Grassellino, O. Melnychuk, S. Posen, A. Romanenko, D. Sergatskov, and J. F. Zasadzinski, Proceedings of the Seventeenth International Conference on RF Superconductivity MOPB015 (2015).
- [22] M. Checchin and A. Grassellino, Phys. Rev. Applied 14, 0440181 (2020).
- [23] J. Bardeen and M. J. Stephen, Phys. Rev. 140, A1197 (1965).
- [24] A. Godeke, Supercond. Sci. Technol. 19, R68 (2006).
- [25] M. G. T. Mentink, M. M. J. Dhalle, D. R. Dietderich, A. Godeke, F. Hellman, and H. H. J. ten Kate, Supercond. Sci. Technol. 30, 025006 (2017).
- [26] R. M. Scanlan, W. A. Fietz, and E. F. Koch, J. Appl. Phys. 46, 2244 (1975).
- [27] B. J. Shaw, J. Appl. Phys. 47, 2143 (1976).
- [28] "Comsol,".
- [29] J. I. Gittleman and B. Rosenblum, Phys. Rev. Lett. 16, 734 (1966).
- [30] M. Checchin, M. Martinello, A. Grassellino, S. Aderhold, S. K. Chandrasekaran, O. S. Melnychuk, S. Posen, A. Romanenko, and D. A. Sergatskov, Appl. Phys. Lett. 112, 072601 (2018).
- [31] B. Aune, R. Bandelmann, D. Bloess, B. Bonin, A. Bosotti, M. Champion, C. Crawford, G. Deppe, B. Dwersteg, D. A. Edwards, H. T. Edwards, M. Ferrario, M. Fouaidy, P.-D. Gall, A. Gamp, A. G¨ossel, J. Graber, D. Hubert, M. H¨uning, M. Juillard, T. Junquera, H. Kaiser, G. Kreps, M. Kuchnir, R. Lange, M. Leenen, M. Liepe, L. Lilje, A. Matheisen, W.-D. M¨oller, A. Mosnier, H. Padamsee, C. Pagani, M. Pekeler, H.- B. Peters, O. Peters, D. Proch, K. Rehlich, D. Reschke, H. Safa, T. Schilcher, P. Schm¨user, J. Sekutowicz, S. Simrock, W. Singer, M. Tigner, D. Trines, K. Twarowski, G. Weichert, J. Weisend, J. Wojtkiewicz, S. Wolff, and K. Zapfe, Physical Review Special Topics - Accelerators and Beams 3, 092001 (2000).
- [32] R. Brinkmann, B. Faatz, K. Fl¨ottmann, J. Rossbach, J. R. Schneider, H. Schulte-Schrepping, D. Trines, T. Tschentscher, and H. Weise, TESLA XFEL, First Stage of the X-Ray Laser Laboratory, Technical Design Report, Tech. Rep. July (DESY, Hamburg, 2001).
- [33] J. N. Galayda (ed.), "LCLS-II Final Design Report, LCLSII-1.1-DR-0251-R0," (2015).
- [34] "Oxford Instruments,".
- [35] C. Bartram, T. Braine, R. Cervantes, N. Du, G. Leum, L. J. Rosenberg, G. Rybka, J. Yang, M. H. Awida, C. Boffo, D. Bowring, M. Checchin, A. S. Chou, M. Hollister, R. Khatiwada, S. Posen, A. Sonnenschein, W. Wester, O. S, N. Woollett, G. Carosi, L. D. Duffy, B. Mcallister, M. Goryachev, C. Boutan, M. Jones, N. S. Oblath, J. Clarke, A. Agrawal, A. V. Dixit, J. R. Gleason, S. Jois, P. Sikivie, N. Sullivan, D. B. Tanner, E. Lentz, E. J. Daw, J. H. Buckley, P. M. Harrington, E. A. Henriksen, K. W. Murch, and M. D. Bird, Snowmass 2021

(2021).

- [36] A. C. Crawford, Nuclear Instruments and Methods in Physics Research, Section A: Accelerators, Spectrometers, Detectors and Associated Equipment 849, 5 (2017).
- [37] A. A. Abrikosov, Zh. Eksp. Teor. Fiz. 32, 1442 (1957), [A. A. Abrikosov, Soviet Phys.—JETP 5, 1174 (1957)].
- [38] E. H. Brandt, Rep. Prog. Phys. 58, 1465 (1995).
- [39] S. Calatroni, CERN-ACC-NOTE-2020-0029 (2020).
- [40] D. Ahn, in 17th Patras Workshop on Axions, WIMPs and WISPs (2022).
- [41] A. Godeke, Superconductor Science and Technology 19, R68 (2006).
- [42] A. Romanenko, A. Grassellino, O. Melnychuk, and D. A. Sergatskov, Journal of Applied Physics 115, 184903 (2014).
- [43] M. Peiniger, M. Hein, N. Klein, G. M¨uller, H. Piel, and P. Thuns, in Proceedings of The Third Workshop on RF Superconductivity (Argonne National Laboratory, 1988).
- [44] J. Knobloch, Advanced thermometry studies of superconducting RF cavities, Ph.D. thesis, Cornell University (1997).
- [45] H. Padamsee, J. Knobloch, and T. Hays, RF superconductivity for accelerators (Wiley-VCH, New York, 2008) p. 521.
- [46] A. Romanenko and D. I. Schuster, Physical Review Letters 119, 1 (2017), arXiv:1705.05982.
- [47] A. Alimenti, N. Pompeo, K. Torokhtii, T. Spina, R. Flukiger, L. Muzzi, and E. Silva, IEEE Transactions on Applied Superconductivity 29 (2019), 10.1109/TASC.2019.2892584.