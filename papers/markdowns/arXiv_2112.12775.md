# arXiv:2112.12775

**Paper ID:** 9d00354737a198f13a76ac02c1b109d0

**PDF Path:** apl/Superconductors/arXiv:2112.12775.pdf

**Processing Status:** complete

**Captions Added:** 5

**Generated:** 2025-06-24T13:44:27.573933

---

# Impact of the superconductors properties on the measurement sensitivity of resonant-based axion detectors

Andrea Alimenti <sup>1</sup> , Kostiantyn Torokhtii <sup>1</sup> , Daniele Di Gioacchino <sup>2</sup> , Claudio Gatti <sup>2</sup> , Enrico Silva <sup>1</sup>,<sup>3</sup> and Nicola Pompeo <sup>1</sup>,<sup>3</sup>

<sup>1</sup> Universit`a degli Studi Roma Tre, Dipartimento di Ingegneria Industriale, Elettronica e Meccanica, Via Vito Volterra 62, 00146 Roma, Italy;

2 INFN, Laboratori Nazionali di Frascati, Frascati, Roma, Italy;

3 INFN, Sezione Roma Tre, Roma, Italy;

E-mail: andrea.alimenti@uniroma3.it

Dec 2021

Abstract. Axions, hypothetical particles theorized to solve the strong CP-problem, are presently being considered as strong candidates as cold dark matter constituents. The signal power of resonant-based axion detectors, known as haloscopes, is directly proportional to their quality factor Q. In this paper, the impact of the use of superconductors in the performances of the haloscopes is studied by evaluating the obtainable Q. In particular, the surface resistance R<sup>s</sup> of NbTi, Nb3Sn, YBa2Cu3O7−<sup>δ</sup> and FeSe0.5Te0.<sup>5</sup> is computed in the frequency, magnetic field and temperature ranges of interest, starting from the measured vortex motion complex resistivity and screening lengths of these materials. From R<sup>s</sup> the quality factor Q of a cylindrical haloscope with copper conical bases and superconductive lateral wall, operating with the TM<sup>010</sup> mode, is evaluated and used to perform a comparison of the performances of the different materials. Both YBa2Cu3O7−<sup>δ</sup> and FeSe0.5Te0.<sup>5</sup> are shown to improve the measurement sensitivity by almost an order of magnitude with respect to a whole Cu cavity, while NbTi is shown to be suitable only at lower frequencies (< 10 GHz). Nb3Sn can give an intermediate improvement in the whole spectrum of interest.

### 1. Introduction

The axion is a particle associated to the dynamic field introduced by Peccei-Quinn in 1977 [\[1,](#page-13-0) [2,](#page-13-1) [3\]](#page-13-2) to solve the 'strong CP problem', i.e. the missing experimental observation of the CP symmetry violation foreseen by the quantum chromodynamic theory. The characteristics of this particle make it particularly appealing also for the resolution of another unsolved physical problem: the nature and existence of dark matter. Thus, in the last years a strong interest emerged in the experimental detection of axions .

Due to the small cross sectionof axions with the ordinary baryonic matter and with photons, the different detection approaches exploit the Primakoff effect [\[4\]](#page-13-3), that is the conversion of axions in photons in presence of high magnetic fields B. The frequency of the produced photons γph is ν = Eph/h with Eph the total energy of the particle (including the rest-mass energy and the kinetic contribution) and h the Planck constant. Thus, Eph = mac <sup>2</sup> + 1 <sup>2</sup>mav 2 , where m<sup>a</sup> is the mass of the axions, c the speed of light in vacuum and v the particle speed, which, for axions and compatible with the cold dark matter properties, is < 10<sup>−</sup><sup>3</sup> c [\[5\]](#page-13-4). Hence, in practice, Eph ≈ mac 2 . m<sup>a</sup> is constrained by astronomical and cosmological considerations in the interval 1 < ma/(µeV) < 10<sup>3</sup> [\[5\]](#page-13-4). Hence, 2 × 10<sup>8</sup> ≤ ν/(Hz) ≤ 2 × 10<sup>11</sup> .

In 1983, P. Sikivie [\[6\]](#page-13-5) proposed an experimental test of the existence of the axions based on the use of resonant microwave cavities, known as haloscopes. When the axions enter the cavity, they can be converted into microwave photons by externally generated B fields and in this way they can be detected using very low noise electronics [\[7\]](#page-13-6). The signal power Pa→γph to be detected is shown to be [\[6,](#page-13-5) [7\]](#page-13-6):

<span id="page-1-0"></span>
$$P\_{a \to \gamma\_{ph}} \propto \left(B^2 V Q\right) \left(g\_{a \gamma\_{ph}}^2 \rho\_a \nu\right) \,, \tag{1}$$

where V is the volume of the cavity, Q the quality factor, gaγ the axion-photon coupling and ρ<sup>a</sup> the local density of axions. On the basis of existing experiments, one finds 10<sup>−</sup><sup>23</sup> < Pa→γph /(W) < 10<sup>−</sup><sup>22</sup>. . In the proportionality statement [\(1\)](#page-1-0), the terms in the right brackets do not depend on human will. Thus, the only change to improve the signal power, in order to reduce the integration time, consists in the optimization of the parameters in the left brackets of [\(1\)](#page-1-0).

The high Q needed to increase Pa→γph pushes toward the study of new methods to reduce the cavity losses. The best results are obtained with either superconductive [\[8,](#page-13-7) [9,](#page-13-8) [10\]](#page-13-9) or dielectric resonant cavities [\[11,](#page-13-10) [12\]](#page-13-11). Since in metallic cavities Q ∝ R<sup>−</sup><sup>1</sup> s , being R<sup>s</sup> the surface resistance of the conductive walls of the cavity, materials with low R<sup>s</sup> must be employed. Beyond copper, only superconductive materials can be explored. However, the choice is not obvious since when superconductors (SC) are used at high frequencies and in high magnetic fields, they are driven in the mixed state so that large losses emerge due to the dissipative motion of fluxons. Fluxonsare quanta of magnetic flux, of amplitude Φ<sup>0</sup> = h/(2e) ≈ 2.068 × 10<sup>−</sup><sup>15</sup> Wb, that penetrate type-II superconductors in presence of the high magnetic field levels needed for the haloscopes (i.e. of the order of some tesla). In these conditions the R<sup>s</sup> of technological superconductors is dominated by the electrodynamics of fluxons. The latter must then be accurately characterized in order to assess the suitability of these materials for the realization of highly sensitive haloscopes.

Different superconductive haloscopes have been recently developed and tested. In [\[8\]](#page-13-7) the Q-factor of copper and NbTi haloscopes (tuned at ∼ 9.08 GHz) have been measured, showing that with a NbTi cavity the figure of merit B<sup>2</sup>Q at ∼ 6 T is ∼ 5 times larger than what reached with a Cu cavity. The first RE-Ba2Cu3O7−<sup>δ</sup> (RE- BCO) cavity, with RE a rare earth element such as Y, was shown in [\[9\]](#page-13-8) to exhibit Q ≈ 3.25×10<sup>5</sup> at 6.93 GHz and up to 8 T, which is more than 6 times larger than with a Cu cavity. Recently [\[10\]](#page-13-9), a different configuration for a RE-BCO cavity was presented and its performances compared with one coated with Nb3Sn, both working at ∼ 9 GHz: the RE-BCO haloscope reached Q ∼ 7×10<sup>4</sup> at ∼ 12 T which was shown to be only 1.75 times larger than that obtained with the equivalent Cu cavity. Moreover, the Q-factor of the Nb3Sn cavity showed a strong field dependence that apparently prevents the use of this material for this kind of applications.

In this paper, the most promising superconductors for resonant axion detectors, i.e. Nb3Sn, NbTi, YBa2Cu3O7−<sup>δ</sup> (YBCO) and, as a perspective, FeSe0.5Te0.<sup>5</sup> (FeSeTe), are considered. Their R<sup>s</sup> is evaluated in the expected operative conditions of haloscopes, starting from the studies of the vortex motion properties performed on these materials. In particular, the data used to perform the analysis on Nb3Sn and NbTi are taken respectively from [\[13\]](#page-13-12) and [\[14\]](#page-14-0), whereas those used to analyse the performances of YBCO and FeSeTe were measured specifically for this study. From Rs, the figures of merit B<sup>2</sup>Q of perspective haloscopes, based on the cylindrical geometry shown in [\[8,](#page-13-7) [14\]](#page-14-0) and coated with these materials, are computed and compared.

## <span id="page-2-1"></span>2. Physical background

In this section the characteristics of the high frequency surface resistance R<sup>s</sup> of superconductors in the mixed state are introduced. First, the electrodynamics of vortex motion is presented and then its effects on R<sup>s</sup> are analysed in terms of the field and the frequency dependences.

### 2.1. High frequency vortex motion

In the microwave frequency range, fluxons are set in oscillation around their equilibrium positions by the impinging electromagnetic (e.m.) wave. With low excitations and high frequencies (i.e. > 8 GHz), as it is of interest in this case, the amplitude displacement of the vortices from their equilibrium position is generally smaller than the intervortex spacing. Thus, in this regime fluxons can be assumed to be non-interacting with each other, and a single-vortex motion approach can be followed to obtain the physical model [\[15,](#page-14-1) [16,](#page-14-2) [17,](#page-14-3) [18,](#page-14-4) [19\]](#page-14-5).

The force (per unit length) balance equation for a single fluxon is written down:

<span id="page-2-0"></span>
$$\mathbf{J} \times \left( \hat{n} \Phi\_0 \right) + \mathbf{F}\_{therm} = \eta \mathbf{v} + k\_p \mathbf{u} \,, \tag{2}$$

where J is the microwave-induced current density. Its interactionwith the fluxons oriented along the ˆn direction of the magnetic field B, gives rise to the driving Lorentz force that sets the fluxons in motion. Ftherm is a stochastic force on fluxons given by thermal agitation, v is the fluxon velocity vector, u the displacement vector of the fluxon by its pinning centre, η the viscous drag coefficient, and k<sup>p</sup> the elastic recall constant representing the pinning of the fluxon on its equilibrium position.

At the temperatures of interest for the haloscopes, thermal creep effects can be neglected. Thus, by solving Equation [\(2\)](#page-2-0) with respect to v and setting Ftherm → 0, one obtains the vortex-motion resistivity [\[16,](#page-14-2) [18,](#page-14-4) [19\]](#page-14-5):

<span id="page-3-0"></span>
$$
\tilde{\rho}\_v = \alpha \rho\_{ff} \frac{1}{1 - \mathrm{i}\nu\_p/\nu} \,, \tag{3}
$$

where ρf f = Φ0B/η represents the resistivity of completely free vortices, known as flux-flow resistivity, and α(φ) = sin<sup>2</sup> (φ) [\[20,](#page-14-6) [21\]](#page-14-7) is an adimensional correction coefficient taking into account the mutual orientation φ of the field and currents (φ = 0 when J k nˆ). Hence, ˜ρ<sup>v</sup> is a complex quantity characterized by the characteristic frequency 2πν<sup>p</sup> = kp/η which separates an elastic low-frequency vortex motion regime from a highly dissipative motion regime at higher frequencies. For ν νp, ˜ρ<sup>v</sup> → ρf f . The latter can be a rather large quantity at high fields, being a fraction ρf f ∼ ρnB/Bc<sup>2</sup> of the normal state resistivity ρn, with Bc<sup>2</sup> the upper critical field of the material [\[22,](#page-14-8) [23\]](#page-14-9).

In anisotropic materials (e.g. YBCO and FeSeTe) the vortex motion parameters exhibit an angular dependence given by the different properties of the material along the crystallographic directions. In case of uniaxial anisotropy, resulting by the layered structure of some superconductors, an anisotropic effective electron mass can be measured. This brings to the definition of the anisotropy coefficient m<sup>c</sup> = γ <sup>2</sup>mab, where m<sup>c</sup> and mab are the effective masses along the c-axis (the crystallographic axis of anisotropy) and the ab-planes (a and b are the crystallographic axes normal to the c-axis) respectively, and γ is the intrinsic material anisotropy coefficient. The anisotropy on the vortex motion parameters, for high-κ SC in the London limit [\[22\]](#page-14-8), is then derived following the Blatter-Geshkenbein-Larkin (BGL) scaling theory [\[24\]](#page-14-10) which states that for a field and angle dependent observable Q(H, θ), its H and θ (θ is the angle between H and the c-axis) dependences can be combined in an effective field H(θ):

<span id="page-3-1"></span>
$$\mathcal{Q}(H,\theta) = s\_{\mathcal{Q}}(\theta)\mathcal{Q}(H\epsilon(\theta))\tag{4a}$$

$$\epsilon(\theta) \qquad = (\gamma^{-2}\sin^2\theta + \cos^2\theta)^{1/2} \tag{4b}$$

where sQ(θ) is an a-priori known factor which depends on the particular Q(H, θ): sη(θ) = (θ) <sup>−</sup><sup>1</sup> and s<sup>k</sup><sup>p</sup> (θ) = (θ) −1 [\[21,](#page-14-7) [25,](#page-14-11) [26\]](#page-14-12). However, it should be noted that the scaling of k<sup>p</sup> with the BGL theory can be successfully performed only when isotropic pinning centres are present as discussed and experimentally shown in [\[25\]](#page-14-11). For this reason in the paper only samples without the addition of directional pinning centres are analysed.

Finally, when B k J, i.e. φ = 0, α = 0 since the driving force acting on the fluxon J × (ˆnΦ0) = 0. However, since fluxons are magnetic structures, inside the superconductor the direction of the magnetic flux density field B can differ from the direction of the outer static field H, hence B 6= µ0H. Indeed also with nominally H k J it was experimentally observed that ˜ρ<sup>v</sup> 6= 0 [\[14\]](#page-14-0). This can be explained considering that the finite line tension ε` = Φ<sup>2</sup> 0 /(4πµ0λ 2 ) of the fluxon lines, with λ the London penetration depth, allows them to locally deviate from the H orientation by an angle φloc, so that this finite angle between J and the flux lines yields α 6= 0. Indeed, vortex lines deform under the effects of the attractive action of pinning centres, of thermal fluctuations and of the Lorentz force exerted by the applied currents. The latter is well known to produce helicoidal current vortices, with the corresponding instabilities in DC regimes, and more generally flux cutting phenomena [\[27\]](#page-14-13). In the present situation we are considering vanishingly small, high frequency currents whose contribution to vortex deformation and tilting with respect to their nominal orientation can be neglected. On the other hand, the effect of pinning centres, here considered to be point-like, and thermal fluctuations, can be evaluated in a simplified way within the collective pinning theory [\[28\]](#page-14-14). Thanks to their finite elasticity, vortices accommodate on the underlying pinning landscape to minimize the total elastic and pinning energy. In so doing, they displace by an amplitude given by the interaction length with pins (∼ ξ) over the so called collective pinning length Lc. The latter identifies the length of the vortex segment which is independently pinned with respect to the other portions of the vortex. In a simple picture, this accommodation produces a tilting angle φloc,el ' arctan(ξ/Lc) between the flux line and the nominal H field orientation. The length L<sup>c</sup> is connected to the vortex collective pinning energy U<sup>c</sup> ' ε`ξ <sup>2</sup>/Lc. Moreover, the average amplitude of the thermal fluctuations uth can be estimated by equating the thermal energy per unit length kBT/L<sup>c</sup> to the elastic energy per unit length (1/2)ε`u 2 th due to these deformations. Thus, the tilting angle due to thermal fluctuations is φloc,th ' arctan(uth/Lc). Since the two mechanisms are independent, a combined tilting angle can be computed φloc ' arctan(p u 2 th + ξ <sup>2</sup>/Lc). The procedure used to estimate φloc from measured parameters will be detailed in Section [3.2.](#page-7-0)

This evaluation will prove useful in the following.

### 2.2. Surface impedance in the mixed state

The macroscopic material property that describes the electrodynamic response of (super)conductors is the surface impedance Z<sup>s</sup> [\[29\]](#page-14-15):

$$Z\_s \coloneqq E\_{\parallel}/H\_{\parallel} = R\_s + \mathrm{i}X\_s \,, \tag{5}$$

where E<sup>k</sup> and H<sup>k</sup> are the electric and magnetic field component parallel to the surface of the material, respectively, and R<sup>s</sup> := Re Z<sup>s</sup> and X<sup>s</sup> := Im Zs.

In the local limit for bulk good conductors, i.e. when the thickness d of the material is much larger than the screening characteristic lengths d min(δ, λ), with δ the skin penetration depth and λ the London penetration depth, from classical electromagnetism it can be shown that [\[29\]](#page-14-15):

$$Z\_s = \sqrt{\mathrm{i}\omega\mu\_0\tilde{\rho}}\ ,\tag{6}$$

where ω = 2πν is the angular frequency of the impinging e.m. field, µ<sup>0</sup> is the vacuum magnetic permeability and ˜ρ the resistivity of the material which, in the case of superconductors, is a complex quantity. The resistivity ˜ρ describes all the resistive and reactive phenomena in the material. Thus, for superconductors in the mixed state the contribution of both superconducting and normal charge carriers, as well as vortex motion, must be taken into account in ˜ρ. The interplay between the various contributions to ˜ρ , for small displacements u of the fluxons, yields an overall resistivity which can be written as [\[15\]](#page-14-1):

<span id="page-5-0"></span>
$$\tilde{\rho} = \frac{\tilde{\rho}\_v + \mathrm{i}/\sigma\_2}{1 + \mathrm{i}\sigma\_1/\sigma\_2} \,, \tag{7}$$

where σ2<sup>f</sup> := σ1−iσ<sup>2</sup> is the so called two-fluid conductivity of superconductors [\[22\]](#page-14-8). This conductivity can be described as the parallel between the real conductivity of the quasiparticles σ<sup>1</sup> and the inductive behaviour of the superfluid σ<sup>2</sup> ≈ 1/(ωµ0λ 2 ). Taking into account the low operative temperatures of haloscopes, since σ<sup>1</sup> ∝ x<sup>n</sup> with x<sup>n</sup> ≈ (T/Tc) β (with β = 4 and β = 2 for low-T<sup>c</sup> and high-T<sup>c</sup> SC, respectively), σ1/σ<sup>2</sup> 1. Hence, Equation [\(7\)](#page-5-0) can be approximated to ˜ρ ≈ ρ˜<sup>v</sup> + i/σ<sup>2</sup> which yields:

<span id="page-5-1"></span>
$$Z\_s \approx \sqrt{\omega \mu\_0 \left(-\frac{1}{\sigma\_2} + \mathrm{i}\tilde{\rho}\_v\right)}\,. \tag{8}$$

Equation [\(8\)](#page-5-1) will be used in the following to evaluate and compare the materials surface resistance R<sup>s</sup> starting from the measurements of the vortex motion parameters, ρf f and νp, which define ˜ρ<sup>v</sup> in Equation [\(3\)](#page-3-0), and λ.

### 3. Experimental section

In this section the surface resistance of NbTi, Nb3Sn, YBCO and FeSeTe is evaluated starting from the measurements of the vortex motion parameters. Then, the perspective performances of haloscopes coated with these materials are analysed.

To this purpose, we first show the structure of a typical haloscope taking as a reference the haloscope presented in [\[8\]](#page-13-7) in order to highlight the dependence of Q on the R<sup>s</sup> of the coating layer. Then, the properties of the superconductors needed for the evaluation of R<sup>s</sup> are collected from existing works. Finally, using the computed R<sup>s</sup> the sensitivity of haloscopes is evaluated to assess which of the SCs under scrutiny would give the best performances.

### 3.1. Haloscope design

In this work, the superconductive haloscope presented in [\[8\]](#page-13-7) within the QUAX experiment is used as <sup>a</sup> reference. The cavity has cylindrical shape ( = 26.1 mm) with conical bases and it is designed to operate with the applied magnetic field oriented along the main axis of the cavity. In order to facilitate the penetration of the magnetic

![](_page_6_Figure_1.jpeg)

**Caption:** Figure 1 shows a simulated electromagnetic field distribution in a superconducting haloscope operating under the TM_010 mode. Visualized are the electric field (E-field) in blue, magnetic field (H-field) in black, and surface current density (J_s) in red. The direction of an externally applied static magnetic field is parallel to the cavity's longitudinal axis. This simulation underscores the interaction dynamics between applied fields and the superconductive walls, providing insights into optimizing the haloscope design to maximize detection sensitivity of axion signals by influencing cavity Q-factors.


<span id="page-6-0"></span>Figure 1. Electromagnetic simulation of the haloscope excited with the TM<sup>010</sup> mode. The distributions of the radio frequency electric E-field (in blue), the magnetic H-field (in black) and the surface current density J<sup>s</sup> (in red) are shown. The direction of the externally applied static B field is shown to be parallel to the longitudinal axis of the cavity.

field, the conical caps are made of copper while the cylindrical SC body is divided in two halves by a copper thin (30 µm) layer to break the superconducting screening currents. The transverse magnetic TM<sup>010</sup> mode, which is usually employed for Primakoff axion detection in conjunction with a static magnetic field orientation parallel to the cavity axis, induces longitudinal currents on the lateral walls (thus H k J). The haloscope here considered resonates at ν<sup>0</sup> ∼ 9.08 GHz at 4 K [\[8\]](#page-13-7). The subscript indices of 'TM010' indicate the number of the peaks of the radio frequency H-field met along the three directions of a cylindrical coordinate system: the first index '0' indicates that the H-field is constant along the azimuth direction, the second index '1' indicates that the H-field exhibits a peak along the radial direction and the last index '0' that the H-field is constant along the axial direction (a sketch of the radio frequency fields and currents is reported in Figure [1\)](#page-6-0).

Due to the different materials used for the inner coating of the cavity elements, the overall quality factor is given by:

$$\frac{1}{Q} = \frac{R\_{s,cyl}}{G\_{cyl}} + \frac{R\_{s,cones}}{G\_{cones}} \,, \tag{9}$$

where for this cavity, Gcyl ' 482 Ω and Gcones ' 6270 Ω are the geometrical factors of the cylindrical body and conical caps, respectively, and Rs,cyl, Rs,cones their surface resistances. At 9.08 GHz and 4 K, in the anomalous skin depth regime [\[30\]](#page-14-16), the surface resistance of copper is Rs,Cu ' 4.9 mΩ. Thus, a cavity with this geometry entirely made of copper would have Qmin ' 9.1 × 10<sup>4</sup> at 9 GHz. This is the lower limit given by copper that we expect to be outperformed by the use of superconductive coatings. Whereas, in the ideal case for which Rs,cyl/Gcyl can be neglected, the upper limit Qmax ' 1.3 × 10<sup>6</sup> could be reached with this geometry. Thus, the highest performance increase in the B<sup>2</sup>Q figure of merit reachable with this haloscope geometry is Qmax/Qmin = Gcones/Gcyl + 1 ≈ 14.

### <span id="page-7-0"></span>3.2. Materials under investigation

The perspective application of four different superconductive materials to the realization of haloscopes is here investigated.

In Table [1](#page-7-1) the list of the samples, whose vortex motion parameter were measured in high magnetic fields, their main characteristics, and references where further information can be found, are reported.

| Material       | NbTi                          | Nb3Sn            | YBa2Cu3O7<br>− δ | FeSe0.5Te0.5     |
|----------------|-------------------------------|------------------|------------------|------------------|
| Thickness (nm) | 103<br>(3.5<br>±<br>0.5)<br>× | bulk             | 80<br>±<br>5     | 240<br>±<br>15   |
| Growing tech.  | RF sputtering                 | HIP1             | CSD2             | PLD3             |
| Substrate      | Cu                            | –                | LaAlO3           | CaF2             |
| Tc<br>(K)      | ±<br>8.0<br>0.2               | ±<br>18.0<br>0.2 | ±<br>89.9<br>0.5 | ±<br>18.0<br>0.2 |
| Ref.           | [8]                           | [31, 13]         | [32]             | [33]             |

<span id="page-7-1"></span>Table 1. List of the samples under investigation.

<sup>1</sup> High Isostatic Pressure

<sup>2</sup> Chemical Solution Deposition

<sup>3</sup> Pulsed Laser Deposition

The microwave vortex motion properties of NbTi were characterized through the study of the performance of the NbTi coated haloscope analysed in [\[8\]](#page-13-7). The Nb3Sn, YBCO and FeSeTe samples were studied through the measurement technique based on the use of the dielectric loaded resonator described and optimized in [\[34,](#page-14-20) [35,](#page-14-21) [36,](#page-14-22) [37\]](#page-14-23). In particular the analysis of the vortex motion parameters in Nb3Sn was reported in [\[13,](#page-13-12) [38\]](#page-14-24). The microwave surface impedance analysis of FeSeTe was detailed in [\[26,](#page-14-12) [39,](#page-14-25) [40\]](#page-14-26), while the YBCO measurements in high magnetic fields were presented in [\[41\]](#page-14-27).

For the comparison we choose to work in the same operative condition of the cavity analysed in [\[8\]](#page-13-7): T = 4 K, µ0H = 5 T parallel to the SC surface, and with the selected resonant mode inducing microwave currents J k H. Actually, the latter samples (Nb3Sn, YBCO and FeSeTe) were characterized with a different configuration of currents and field orientations (J ⊥ H with H k c-axis for anisotropic SC) with respect to the haloscope here considered. Hence, the vortex motion resistivity ˜ρ must be properly evaluated in the configuration of interest. First, the α coefficient must be evaluated, see Equation [\(3\)](#page-3-0). This is done through the first order model proposed in Section [2.](#page-2-1) The collective pinning length L<sup>c</sup> = p 2ε`/k<sup>p</sup> is evaluated from the experimentally measured k<sup>p</sup> by equating the pinning energy Uc/L<sup>c</sup> to the pinning energy (both per unit length) expressed in terms of kp, kpξ <sup>2</sup>/2. The data used to estimate L<sup>c</sup> and ε` , i.e. λ and kp, are reported in Table [2.](#page-10-0)

Second, for both YBCO and FeSeTe the intrinsic material anisotropy must be taken into account (see Equation [\(4\)](#page-3-1)), taking γ from [\[25\]](#page-14-11) and [\[39\]](#page-14-25) for YBCO and FeSeTe, respectively. The anisotropy enters also the evaluation of the α coefficient which, in
the geometry here considered, is influenced by the anisotropy which enhances tilting due to thermal fluctuations and reduces the one due to the elastic accommodation to pins. It is worth noting here that FeSeTe is a multiband superconductor. Indeed, it exhibits intricate anisotropy properties which are actually different among the various superconducting quantities λ, ξ, Bc<sup>1</sup> and Bc<sup>2</sup> [\[42\]](#page-14-0). Moreover, ρf f shows an unusual field dependence [\[43,](#page-14-1) [26,](#page-14-2) [39,](#page-14-3) [40\]](#page-14-4). Hence a direct experimental study of FeSeTe in the same configuration involved in the considered haloscope design would be particularly recommended.

In addition, for the sake of simplicity, we neglect the increased pinning when H k ab planes (the so called "intrinsic pinning"), connected to the layered structure of materials like YBCO. Moreover, it must be noted the in engineered materials the effects of the intrinsic pinning are essentially hidden by those of the artificial pinning centres [\[25\]](#page-14-5).

![](0__page_8_Figure_3.jpeg)

**Caption:** Figure 2 presents the pinning constant k_p as a function of reduced temperature t = T / Tc for YBCO (orange circles) and FeSeTe (blue squares) under a magnetic field of 5 T in zero field cooling (ZFC) condition, plotted in panel a. In panel b, the field dependence of k_p is shown for Nb3Sn (green triangles), YBCO, and FeSeTe at lower temperatures. The fitting model indicates the pinning regime involves strong pinning for YBCO, while FeSeTe and Nb3Sn exhibit a collective pinning regime. These data are critical in understanding the vortex behavior and optimally tuning the superconductor for higher B^2Q values in haloscope applications to enhance dark matter detection sensitivity.


<span id="page-8-0"></span>Figure 2. (a) pinning constant k<sup>p</sup> measured on YBCO (orange circles) and FeSeTe (blue squares) in zero field cooling condition (ZFC) at 5 T as a function of the reduced temperature t = T /Tc. The fit of the data is performed with the model developed in [\[44\]](#page-14-6): kp(t) = kp(0)(1 − t) 4/3 (1 + t) 2 e (−T /T0) . The best fit parameters are: kp(0) = 1.40 × 10<sup>5</sup> N m-2 and T<sup>0</sup> = 56 K for YBCO, kp(0) = 7.07 × 10<sup>4</sup> N m-2 and T<sup>0</sup> = T<sup>c</sup> for FeSeTe. The empty square symbols represent the points extrapolated from the fit at 4 K (which will be used in the next elaborations). (b) Field dependence of k<sup>p</sup> in Nb3Sn (green triangle), YBCO and FeSeTe samples measured at the lower temperatures. kp(H) in Nb3Sn and FeSeTe is shown to be ∝ H<sup>−</sup>0.<sup>5</sup> .

Finally, the T and H range of interest must be considered. Since measurements at T = 4 K and µ0H = 5 T are available only for Nb3Sn, for YBCO and FeSeTe the low temperature data must be extrapolated from currently existing measurements. However, the temperature dependences of both η and k<sup>p</sup> derive from those of λ and of the coherence length ξ [\[44\]](#page-14-6): hence, at low temperature these parameters saturate similarly to λ and ξ [\[22\]](#page-14-7). Thus, negligible uncertainties associated to the model choice are introduced in the mentioned extrapolation [\[44\]](#page-14-6).

![](0__page_9_Figure_1.jpeg)

**Caption:** Figure 3 illustrates the viscous drag coefficient η as a function of reduced temperature t for superconductors YBCO (orange circles) and FeSeTe (blue squares) under a magnetic field of 5 T. The y-axis represents η on a logarithmic scale, indicating the energy dissipation related to vortex motion. Empty square symbols denote extrapolated points at 4 K. Nb3Sn and NbTi curves (green and red, respectively) are obtained by considering η from flux-flow resistivity models. The analysis of η is crucial for evaluating how different superconducting materials can sustain the operational conditions in haloscopes, contributing to enhancing the Q-factor crucial for detecting axion-like particles.


<span id="page-9-0"></span>Figure 3. Viscous drag coefficient η measured on YBCO (orange circles) and FeSeTe (blue squares) at 5 T in ZFC as a function of the reduced temperature t. The measured data are approximated with η(t) = η(0)(1 − t 2 )/(1 + t 2 ) [\[44\]](#page-14-6), with η(0) = 3.59 × 10<sup>−</sup><sup>7</sup> N s m-2 for YBCO and η(0) = 2.02 × 10<sup>−</sup><sup>7</sup> N s m-2 for the FeSeTe sample. The empty square symbols represent the points extrapolated from the fit at 4 K (which will be used in the next elaborations). The green and red curves represent η(t) obtained by considering η(t) = Φ0Bc2(t)/ρ<sup>n</sup> (from ρf f = ρnB/Bc<sup>2</sup> and η = Φ0B/ρf f ) for Nb3Sn and NbTi respectively.

The pinning frequency is computed from k<sup>p</sup> and η as 2πν<sup>p</sup> = kp/η. In Figure [2\(](#page-8-0)a) the pinning constant kp(t), with t = T/Tc, measured at 5 T on YBCO and FeSeTe is shown. The measurements are obtained in zero field cooling (ZFC) condition by sweeping the magnetic field at fixed temperature values. The values at 4 K are extrapolated from these measurements through the empirical models shown in [\[44\]](#page-14-6) and reported in the caption of Figure [2.](#page-8-0) The field dependence of k<sup>p</sup> is shown in Figure [2\(](#page-8-0)b). Here also the Nb3Sn measurement, performed at 4 K (thus for it no extrapolation is needed), is reported. From the behavior of kp(H) on the different materials, different pinning regimes can be recognized. In YBCO, k<sup>p</sup> is field independent at 10 K and up to 10 T, indicating a pinning regime where fluxons are strongly pinned independently one from another. On the contrary, FeSeTe and Nb3Sn displaya k<sup>p</sup> ∝ H<sup>−</sup>0.<sup>5</sup> dependence, typical of the collective pinning regime (bundles of fluxons pinned on multiple pinning centres). These field dependences will be considered in the final performance analysis presented.

For what concerns η = Φ0B/ρf f , in Figure [3](#page-9-0) measurements at 5 T on YBCO and FeSeTe are reported. Also in this case, from the model shown in [\[44\]](#page-14-6), the value at 4 K is extrapolated. Resorting to the conventional Bardeen-Stephen behaviour [\[23\]](#page-14-8) exhibited by Nb3Sn [\[13\]](#page-13-0) and NbTi, the flux-flow resistivity is simply obtained from the normal state resistivity ρ<sup>n</sup> as ρf f = ρnB/Bc<sup>2</sup> with Bc<sup>2</sup> ≈ Bc2(0)(1 − t 2 ). We take ρ<sup>n</sup> = 14.8 µΩ cm and Bc<sup>2</sup> = 27 T for Nb3Sn [\[13\]](#page-13-0) and ρ<sup>n</sup> = 70 µΩ cm, Bc<sup>2</sup> = 13 T for NbTi [\[8\]](#page-13-1).

|                        | NbTi              | Nb3Sn             | YBCO               | FeSeTe             |
|------------------------|-------------------|-------------------|--------------------|--------------------|
| kp(N m-2) at 4 K, 5 T  | –                 | (9.9 ± 1.5) × 103 | (1.3 ± 0.2) × 105  | (5.9 ± 0.9) × 104  |
| pinning regime         | single            | collective        | single             | collective         |
| η(N s m-2) at 4 K, 5 T | –                 | –                 | (3.6 ± 0.5) × 10−7 | (1.8 ± 0.3) × 10−7 |
| ρn(µΩ cm)              | 70 ± 2            | 14.8 ± 0.2        | –                  | –                  |
| Bc2(0)(T)              | 13.0 ± 0.5        | 27.0 ± 0.5        | –                  | –                  |
| νp(GHz) at 4 K, 5 T    | 44 ± 7            | 4.2 ± 0.6         | 59 ± 9             | 52 ± 8             |
| meas. geometry         | H k J             | H ⊥ J             | H k c axis, H ⊥ J  |                    |
| γ                      | 1                 | 1                 | 5.3 ± 0.7          | 1.8 ± 0.2          |
| λ(0)(nm)               | 270 ± 20 [14, 45] | 160 ± 20 [46]     | 150 ± 10 [47, 48]  | 520 ± 50 [43]      |

<span id="page-10-0"></span>Table 2. Measured parameters of the SCs of interest used for the evaluation of Rs.

All the parameters of interest for the different materials are reported in Table [2.](#page-10-0)

# 4. Results and discussion

From the data collected in Table [2,](#page-10-0) the surface resistance R<sup>s</sup> of the selected SCs is evaluated to compare the figure of merit B<sup>2</sup>Q of haloscopes ( with the geometry shown in [\[8\]](#page-13-1)) coated with different materials. In particular, Equation [\(3\)](#page-3-0) is computed with the data in Table [2](#page-10-0) taking into account the α coefficient, the anisotropy γ for YBCO and FeSeTe, the collective pinning dependences in Nb3Sn and FeSeTe and the pair breaking effects on the superfluid σ<sup>2</sup> and, for NbTi, also on σ1. From this, the complex resistivity ρ˜ from Equation [\(7\)](#page-5-0) and R<sup>s</sup> from Equation [\(6\)](#page-4-0) are obtained. Finally, Q of the haloscope is computed using Equation [\(9\)](#page-6-0).

![](0__page_10_Figure_6.jpeg)

**Caption:** Figure 4 compares the figure of merit B^2Q as a function of the applied magnetic field B for different superconducting materials: YBCO, FeSeTe, Nb3Sn, and NbTi, evaluated at 4 K and 9.08 GHz. The y-axis represents the B^2Q factor on a logarithmic scale, while the x-axis shows the magnetic field B in Tesla. The B^2Q factors are plotted for worst-case (dotted lines, B ⊥ J) and realistic scenarios (continuous lines, taking α-angular coefficient into account). Copper (Cu) is used as a reference (star symbols). YBCO and FeSeTe outperform Cu even in the worst-case, while under realistic conditions, all studied superconductors show superior performance relative to Cu. The enhancement in B^2Q factor signifies improved haloscope sensitivity for axion detection.


<span id="page-10-1"></span>Figure 4. Comparison of the B<sup>2</sup>Q factors evaluated at 4 K and 9.08 GHz as a function of B. The dotted lines represent the worst lower limit obtained in the B ⊥ J configuration. The continuous lines represent a more realistic situation for which the α angular coefficient is taken into account as explained in the text. The "star" symbols represent the B<sup>2</sup>Q values obtained with a Cu cavity, the yellow area the values obtainable with YBCO, the light blue area those with FeSeTe, the green with Nb3Sn and the red with NbTi.

The performances comparison is based on the figure of merit B<sup>2</sup>Q. In Figure [4](#page-10-1) the

field dependences of B<sup>2</sup>Q of the different haloscopes are shown at 4 K and 9.08 GHz. Both a lower worst-case limit, obtained considering fluxons aligned in the ab planes (for the anisotropic SC) and B ⊥ J, and an upper limit, obtained with the α coefficients evaluated as shown in Section [2,](#page-2-0) are given. It can be seen that in the worst case both Nb3Sn and NbTi exhibit a lower B<sup>2</sup>Q than that given by a Cu cavity, while both YBCO and FeSeTe have, even in this case case, a B<sup>2</sup>Q higher than that of Cu. In the more realistic scenario given by the upper limits shown in Figure [4,](#page-10-1) all the SC here studied are better than Cu. Table [3](#page-11-0) reports the numerical evaluation of the performances at 5 T, where it can be seen that (B<sup>2</sup>Q)YBCO/(B<sup>2</sup>Q)Cu is close to the upper limit of 14 reachable with the analysed geometry. At the highest B, NbTi performance starts dropping due its lower Bc2. With respect to YBCO, FeSeTe exhibits similar performances despite its lower η and kp, because its large λ helps to reduce R<sup>s</sup> in the bulk limit, as it can be deduced from Equation [\(8\)](#page-5-1).

Since at fixed signal to noise ratio SNR, bandwidth detection and system temperature, the integration time is t<sup>i</sup> ∝ P −2 a→γph , then the increase of (B<sup>2</sup>Q) shown in Table [3](#page-11-0) can be directly read as an equivalent reduction in t<sup>i</sup> [\[7\]](#page-13-2).

The frequency dependence B<sup>2</sup>Q(ν), reported in Figure [5](#page-12-0) for 10<sup>9</sup> < ν/(Hz) < 10<sup>11</sup> at 5 T and 4 K, shows interesting features that can be used to assess the best material in the different bands. It can be seen that, given the low ν<sup>p</sup> of Nb3Sn, the Q of the associated haloscope is expected to decrease at frequencies lower than in the other materials. However, thanks to the low ρ<sup>n</sup> and high Bc<sup>2</sup> of Nb3Sn, with respect to NbTi the figure of merit of the Nb3Sn haloscope tends to settle on higher values at high frequency where ˜ρ<sup>v</sup> saturates at ρf f . Thus, Nb3Sn is more convenient than NbTi for ν > 10 GHz and, at even higher frequencies, its performances could be comparable with those of YBCO and FeSeTe. This result highlights clearly that for high frequency applications of SC in the mixed state, not only a high ν<sup>p</sup> is desirable but also a low ρf f is important.

<span id="page-11-0"></span>

|  | Material | (B2Q)/(B2Q)Cu | ti/ti,Cu |  |  |
|--|----------|---------------|----------|--|--|
|  | NbTi     | 4.3           | 0.05     |  |  |
|  | Nb3Sn    | 6.5           | 0.02     |  |  |
|  | FeSeTe   | 12            | 0.006    |  |  |
|  | YBCO     | 14            | 0.005    |  |  |

Table 3. Comparison of the (B<sup>2</sup>Q) figure of merit of the different materials with respect to that of Cu at 5 T, 4 K, 9.08 GHz.

For what concerns the frequency dependence of B<sup>2</sup>Q(ν) for FeSeTe and YBCO, it can be noted that the worst case lower limits of B<sup>2</sup>Q follow the same frequency behaviour due to their similar νp. The upper B<sup>2</sup>Q limits follow a ν <sup>2</sup>/<sup>3</sup> dependence (typical of the anomalous skin depth regime exhibited by Cu) for ν < 10 GHz for both YBCO and FeSeTe. In fact, with these materials, at the lower frequencies the cavity losses are dominated by the copper bases. At higher frequencies, the losses in YBCO still do not contribute substantially to the overall Q while with FeSeTe Q starts decreasing more rapidly than in the Cu cavity.

![](0__page_12_Figure_2.jpeg)

**Caption:** Figure 5 depicts the frequency dependence of the B^2Q factors for YBCO, FeSeTe, Nb3Sn, and NbTi, evaluated at 4 K and 5 T, with a frequency range from 10^9 Hz to 10^11 Hz on the x-axis and B^2Q on the y-axis. The continuous and dashed lines represent realistic and worst-case scenario calculations considering angular coefficients. Star symbols indicate B^2Q values for Cu cavities. YBCO maintains high sensitivity across frequencies, making it optimal for broad-spectrum haloscope applications, while NbTi becomes inefficient at higher frequencies. FeSeTe shows potential for improvement due to its intrinsic properties, offering competitive performance with further material optimization.


<span id="page-12-0"></span>Figure 5. Comparison of the B2Q factors evaluated at 4 K and 5 T as a function of ν and keeping the same geometrical factors as in Section [3.1.](#page-5-2) The dotted lines represent the worst lower limit obtained in the B ⊥ J configuration. The continuous lines represent a more realistic situation for which the α angular coefficient is taken into account as explained in the text. The "star" symbols represent the B2Q values obtained with a Cu cavity, the yellow area the values obtainable with YBCO, the light blue area those with FeSeTe, the green with Nb3Sn and the red with NbTi.

### 5. Conclusion

In this paper, the impact of vortex motion and screening properties of different superconducting materials (i.e. NbTi, Nb3Sn, YBCO and FeSeTe) on the performances of haloscopes, in terms of the power of the detected signal, was evaluated.

In particular, the comparison of the B<sup>2</sup>Q figure of merit of haloscopes in the hybrid geometry proposed in [\[8\]](#page-13-1) has been done by evaluating the Q(B, ν) as determined by the surface resistance Rs(B, ν) behaviour of the aforementioned SC.

In turn, R<sup>s</sup> of these SC was computed starting from the vortex motion parameters measured at microwave frequencies and high magnetic fields. The effect on fluxon pinning by the different orientations of the static magnetic field B with respect to the induced microwave currents imposed by the specific haloscope geometry was taken into account.

The results show that even at 5 T at low frequencies ν < 10 GHz all the studied materials allow to increase the haloscope sensitivity with respect to Cu while, at higher frequencies, NbTi becomes unsuitable and Nb3Sn, despite its lower νp, can still be a useful solution. Interestingly, FeSeTe, which here is for the first time considered for this application, despite its high vortex motion dissipation can be a promising material for the inner coating of haloscopes. Indeed, its large λ contributes to lower the bulk R<sup>s</sup> allowing to reach performances comparable to those obtained with YBCO. Moreover, one has to consider that Iron based SC are still far from the optimization and maturity levels reached with YBCO, thus allowing further room for improvements.

Finally, YBCO is shown to be the best choice, leading to an increase of the measurement sensitivity by a factor ∼ 13 with respect to the Cu cavity in the whole frequency range analysed (i.e. 10<sup>9</sup> < ν/(Hz) < 10<sup>11</sup>). This result is close to the maximum theoretical increase in the sensitivity reachable with the haloscope geometry studied.

To conclude, despite the expected high performances of YBCO in the operative conditions of haloscopes, this study shows the possibility to employ also other SC, in selected frequency bands, to increase the sensitivity of these axion detectors.

## Funding

Work partially supported by MIUR-PRIN project "HIBiSCUS" - grant no. 201785KWLE and by the INFN- Commissione Scientifica Nazionale 5 project "SAMARA".

### Acknowledgments

The authors acknowledge C. Pira for the NbTi coating of copper cavities, T. Spina and R. Fl¨ukiger for the Nb3Sn sample, V. Pinto and G. Celentano for the YBCO sample, G. Sylva and V. Braccini for the FeSeTe sample.

### References

- [1] Peccei R D and Quinn H R 1977 Phys. Rev. Lett. 38 1440
- [2] Wilczek F 1978 Phys. Rev. Lett. 40 279
- [3] Weinberg S 1978 Phys. Rev. Lett. 40 223
- [4] Primakoff H 1951 Phys. Rev. 81 899
- [5] Tanabashi M, Hagiwara K, Hikasa K, Nakamura K, Sumino Y, Takahashi F, Tanaka J, Agashe K, Aielli G, Amsler C et al. 2018 Phys. Rev. D 98 030001
- [6] Sikivie P 1983 Phys. Rev. Lett. 51 1415
- <span id="page-13-2"></span>[7] Braine T, Cervantes R, Crisosto N, Du N, Kimes S, Rosenberg L, Rybka G, Yang J, Bowring D, Chou A et al. 2020 Phys. Rev. Lett. 124 101303
- <span id="page-13-1"></span>[8] Alesini D, Braggio C, Carugno G, Crescini N, D'Agostino D, Di Gioacchino D, Di Vora R, Falferi P, Gallo S, Gambardella U et al. 2019 Phys. Rev. D 99 101101
- [9] Ahn D, Kwon O, Chung W, Jang W, Lee D, Lee J, Youn S W, Youm D and Semertzidis Y K 2020 arXiv preprint [arXiv:2002.08769](http://arxiv.org/abs/2002.08769)
- [10] Golm J, Cuendis S A, Calatroni S, Cogollos C, D¨obrich B, Gallego J, Barcel´o J, Granados X, Gutierrez J, Irastorza I et al. 2021 arXiv preprint [arXiv:2110.01296](http://arxiv.org/abs/2110.01296)
- [11] Alesini D, Braggio C, Carugno G, Crescini N, D'Agostino D, Di Gioacchino D, Di Vora R, Falferi P, Gambardella U, Gatti C et al. 2021 Nucl. Instrum. Methods Phys. Res. A 985 164641
- [12] Alesini D, Braggio C, Carugno G, Crescini N, D'Agostino D, Di Gioacchino D, Di Vora R, Falferi P, Gambardella U, Gatti C et al. 2020 Rev. Sci. Instrum. 91 094701
- <span id="page-13-0"></span>[13] Alimenti A, Pompeo N, Torokhtii K, Spina T, Fl¨ukiger R, Muzzi L and Silva E 2020 Supercond. Sci. Technol. 34 014003
- <span id="page-14-9"></span>[14] Di Gioacchino D, Gatti C, Alesini D, Ligi C, Tocci S, Rettaroli A, Carugno G, Crescini N, Ruoso G, Braggio C et al. 2019 IEEE Trans. Appl. Supercond. 29 1–5
- [15] Coffey M W and Clem J R 1991 Phys. Rev. Lett. 67 386
- [16] Gittleman J I and Rosenblum B 1966 Phys. Rev. Lett. 16 734
- [17] Wu D H, Booth J and Anlage S M 1995 Phys. Rev. Lett. 75 525
- [18] Pompeo N and Silva E 2008 Phys. Rev. B 78 094503
- [19] Pompeo N, Alimenti A, Torokhtii K and Silva E 2020 Low Temp. Phys. 46 343
- [20] Pompeo N 2015 J. Appl. Phys. 117 103904
- [21] Pompeo N and Silva E 2018 IEEE Trans. Appl. Supercond. 28 8201109
- <span id="page-14-7"></span>[22] Tinkham M 2004 Introduction to superconductivity (Dover, NY, USA)
- <span id="page-14-8"></span>[23] Bardeen J and Stephen M 1965 Phys. Rev. 140 A1197
- [24] Blatter G, Geshkenbein V B and Larkin A 1992 Phys. Rev. Lett. 68 875
- <span id="page-14-5"></span>[25] Pompeo N, Alimenti A, Torokhtii K, Bartolom´e E, Palau A, Puig T, Augieri A, Galluzzi V, Mancini A, Celentano G et al. 2020 Supercond. Sci. Technol. 33 044017
- <span id="page-14-2"></span>[26] Pompeo N, Torokhtii K, Alimenti A, Sylva G, Braccini V and Silva E 2020 Supercond. Sci. Technol. 33 114006
- [27] Campbell A M 2011 Supercond. Sci. Technol. 24 091001 ISSN 0953-2048
- [28] Blatter G, Feigel'man M, Geshkenbein V, Larkin A and Vinokur V 1994 Rev. Mod. Phys. 66 1125
- [29] Collin R E 2007 Foundations for microwave engineering (Hoboken, NJ, USA: John Wiley & Sons)
- [30] Reuter G and Sondheimer E 1948 Proc. R. Soc. Lond. 195 336
- [31] Fl¨ukiger R, Spina T, Cerutti F, Ballarino A, Scheuerlein C, Bottura L, Zubavichus Y, Ryazanov A, Svetogovov R, Shavkin S et al. 2017 Supercond. Sci. Technol. 30 054003
- [32] Pinto V, Vannozzi A, Angrisani Armenio A, Rizzo F, Masi A, Santoni A, Meledin A, Ferrarese F M, Orlanducci S and Celentano G 2020 Coatings 10 860
- [33] Braccini V, Kawale S, Reich E, Bellingeri E, Pellegrino L, Sala A, Putti M, Higashikawa K, Kiss T, Holzapfel B et al. 2013 Appl. Phys. Lett. 103 172601
- [34] Alimenti A, Torokhtii K, Silva E and Pompeo N 2019 Meas. Sci. Technol. 30 065601
- [35] Torokhtii K, Pompeo N, Silva E and Alimenti A 2021 Measurement: Sensors 18 100314
- [36] Torokhtii K, Alimenti A, Pompeo N, Leccese F, Orsini F, Scorza A, Sciuto S and Silva E 2018 Q-factor of microwave resonators: calibrated vs. uncalibrated measurements J. Phys. Conf. Ser. vol 1065 (IOP Publishing) p 052027
- [37] Alimenti A, Pompeo N, Torokhtii K and Silva E 2021 IEEE Instrum. Meas. Mag. 24 12
- [38] Alimenti A, Pompeo N, Torokhtii K, Spina T, Fl¨ukiger R, Muzzi L and Silva E 2019 IEEE Trans. Appl. Supercond. 29 3500104
- <span id="page-14-3"></span>[39] Pompeo N, Alimenti A, Torokhtii K, Sylva G, Braccini V and Silva E 2021 IEEE Trans. Appl. Supercond. 31 8000805
- <span id="page-14-4"></span>[40] Pompeo N, Alimenti A, Torokhtii K, Sylva G, Braccini V and Silva E 2020 Microwave properties of fe(se,te) thin films in a magnetic field: pinning and flux flow J. Phys. Conf. Ser. vol 1559 (IOP Publishing) p 012055
- [41] Alimenti A, Torokhtii K, Rizzo F, Pinto V, Augieri A, Celentano G, Silva E and Pompeo N 2020 Microwave surface impedance measurements in nanostructured YBCO up to high magnetic fields applied Superconductivity Conference - ASC2020 (virtual)
- <span id="page-14-0"></span>[42] Bendele M, Weyeneth S, Puzniak R, Maisuradze A, Pomjakushina E, Conder K, Pomjakushin V, Luetkens H, Katrych S, Wisniewski A et al. 2010 Phys. Rev. B 81 224520
- <span id="page-14-1"></span>[43] Okada T, Nabeshima F, Takahashi H, Imai Y and Maeda A 2015 Phys. Rev. B 91 054510
- <span id="page-14-6"></span>[44] Golosovsky M, Tsindlekht M and Davidov D 1996 Supercond. Sci. Technol. 9 1
- <span id="page-14-10"></span>[45] Benvenuti C, Calatroni S, Hauer M, Minestrini M, Orlandi G and Weingarten W 1991 (NbTi)N and NbTi coatings for superconducting accelerating cavities Proc. Fifth Workshop RF Supercond. pp 518–526
- <span id="page-14-11"></span>[46] Posen S and Liepe M 2014 Phys. Rev. Accel. Beams 17 112001
- <span id="page-14-12"></span>[47] Sonier J, Kiefl R, Brewer J, Bonn D, Carolan J, Chow K, Dosanjh P, Hardy W, Liang R,

MacFarlane W et al. 1994 Phys. Rev. Lett. 72 744

<span id="page-15-0"></span>[48] Tallon J, Bernhard C, Binninger U, Hofer A, Williams G, Ansaldo E, Budnick J and Niedermayer C 1995 Phys. Rev. Lett. 74 1008