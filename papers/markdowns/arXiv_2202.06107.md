# arXiv:2202.06107

**Paper ID:** 01b7df2f3d2e6d338c415e87fb8a5804

**PDF Path:** apl/Superconductors/arXiv:2202.06107.pdf

**Processing Status:** complete

**Captions Added:** 12

**Generated:** 2025-06-24T13:44:27.583356

---

# Nonlinear Meissner effect in Nb3Sn coplanar resonators.

J. Makita<sup>1</sup> , C. Sundahl<sup>2</sup> , G. Ciovati1,<sup>3</sup> , C.B. Eom<sup>2</sup> , and A. Gurevich<sup>1</sup>

<sup>1</sup>Department of Physics and Center for Accelerator Science,

Old Dominion University, Norfolk, Virginia 23528, USA

<sup>2</sup>Department of Materials Science and Engineering,

University of Wisconsin-Madison, Madison, Wisconsin 53706, USA

<sup>3</sup> Thomas Jefferson National Accelerator Facility, Newport News, Virginia 23606, USA

(Dated: February 15, 2022)

We investigated the nonlinear Meissner effect (NLME) in Nb3Sn thin film coplanar resonators by measuring the resonance frequency as a function of a parallel magnetic field at different temperatures. We used low rf power probing in films thinner than the London penetration depth λ(B) to significantly increase the field onset of vortex penetration and measure the NLME under equilibrium conditions. Contrary to the conventional quadratic increase of λ(B) with B expected in s-wave superconductors, we observed a nearly linear increase of the penetration depth with B. We concluded that this behavior of λ(B) is due to weak linked grain boundaries in our polycrystalline Nb3Sn films, which can mimic the NLME expected in a clean d-wave superconductor.

The Meissner effect is one of the fundamental manifestations of the macroscopic phase coherence of a superconducting state. Meissner screening current density J = −ensv<sup>s</sup> induced by a weak magnetic field is proportional to the velocity v<sup>s</sup> of the condensate. At higher fields, the superfluid density n<sup>s</sup> becomes dependent on v<sup>s</sup> due to pairbreaking effects, resulting in the nonlinear Meissner effect (NLME)[1–](#page-10-0)[8](#page-10-1). For a single band isotropic s-wave superconductor, the NLME is described by:

$$\mathbf{J} = -\frac{\phi\_0 \mathbf{Q}}{2\pi\mu\_0\lambda^2} (1 - \Upsilon\xi^2 Q^2),\tag{1}$$

where λ is the London penetration depth, ξ is the coherence length, Q = mvs/~ = ∇χ + 2πA/φ0, m is the quasiparticle mass, χ is the phase of the order parameter Ψ = ∆e iχ, A is the vector potential, φ<sup>0</sup> is the magnetic flux quantum, and the factor Υ(T, li) depends on the temperature T, the mean free path l<sup>i</sup> and details of pairing mechanisms. Ginzburg and Landau (GL) were the first who obtained the field dependent-correction to the penetration depth λ(B) of the magnetic field B parallel to a semi-infinite superconductor[1](#page-10-0) :

<span id="page-0-1"></span>
$$
\lambda(B) = \left[ 1 + \frac{\kappa(\kappa + 2^{3/2}) B^2}{8(\kappa + 2^{1/2})^2 B\_c^2} \right] \lambda,\tag{2}
$$

where B<sup>c</sup> = φ0/2 <sup>3</sup>/<sup>2</sup>πλξ is the thermodynamic critical field and κ = λ/ξ is the GL parameter.

In recent years, the NLME has attracted much attention as a probe of unconventional pairing symmetries of moving condensates. Particularly, Yip and Sauls [2](#page-10-2)[,3](#page-10-3) showed that in a clean d-wave superconductor at kBT < p<sup>F</sup> v<sup>s</sup> the supercurrent acquires a nonlinear singular term ∝ |Q|Q strikingly different from that in Eq. [\(1\)](#page-0-0), where p<sup>F</sup> is the Fermi momentum. Yet Eq. [\(1\)](#page-0-0) can describe a variety of nonlinear electromagnetic responses, both in conventional and unconventional superconductors. For instance Eq. [\(1\)](#page-0-0) describes a clean d-wave superconductor at high temperatures T > p<sup>F</sup> vs/k<sup>B</sup> or a d-wave superconductor with impurities [2](#page-10-2)[–7](#page-10-4). In a clean s-wave superconductor, the NLME is absent at T T<sup>c</sup> as Υ ∝ exp(−∆/kBT) [9](#page-10-5) , but occurs in the dirty limit (ξ . l) in which Υ ∼ 1 even at T → 0 [10](#page-10-6). In multiband superconductors the NLME could probe the proliferation of interband phase textures [11](#page-10-7) or the line nodes and interband sign change in the order parameter or mixed s − d pairing symmetries in iron pnictides [12](#page-10-8) .

<span id="page-0-0"></span>So far the observations of the NLME in high-T<sup>c</sup> cuprates have been inconclusive [13](#page-10-9)[–19](#page-10-10), mostly because of a small field region of the Meissner state in high-κ type-II superconductors and contributions of extrinsic materials factors, such as grain boundaries or local nonstoichiometry. Since the NLME becomes essential in fields B of the order of Bc, penetration of vortices above the lower critical field Bc<sup>1</sup> = (φ0/4πλ<sup>2</sup> )(ln κ + 0.5) B<sup>c</sup> [20](#page-10-11) limits the nonlinear correction in Eq. [\(2\)](#page-0-1) to (Bc1/Bc) <sup>2</sup>/8 ' ln κ/16κ <sup>2</sup> 1. Yet even small NLME terms in Eq. [\(1\)](#page-0-0) causes intermodulation effects[4,](#page-10-12)[5](#page-10-13) under strong ac fields, as it was observed in YBa2Cu3O7−<sup>x</sup> [17](#page-10-14)[,18](#page-10-15) .

In this paper we investigate the NLME in a thin film Nb3Sn coplanar resonator in a parallel dc magnetic field, which mitigates the problems of vortex penetration and nonequilibrium effects. We used the method of Ref. [22](#page-10-16) in which the resonant frequency ω = (LC) <sup>−</sup>1/<sup>2</sup> of a coplanar resonator is measured as a function of a parallel dc field B. Here C is the strip-to-ground capacitance and L = L<sup>g</sup> + L<sup>k</sup> is the total inductance containing both the geometrical inductance Lg, and the field-dependent kinetic inductance of the superconducting condensate, Lk(B) ∝ λ 2 (B) [21](#page-10-17). To extend the field region of the NLME, we performed our measurements on thin films of thickness d < λ for which Bc<sup>1</sup> = (2φ0/πd<sup>2</sup> ) ln(d/ξ) can be much higher than the bulk Bc<sup>1</sup> [23](#page-10-18). Rotating the field in the plane of the film gives rise to an orientational dependence of λ(B) [22](#page-10-16). Unlike the standard quadratic field correction to λ(B) observed on Nb [22](#page-10-16) and Al [24](#page-10-19) thin film resonators, we observed a nearly linear field dependence of λ(B) in polycrystalline Nb3Sn films. The fact that the NLME in the s-wave superconductor Nb3Sn exhibits the behavior expected from a

clean d-wave superconductor[2,](#page-10-2)[3](#page-10-3) shows the importance of materials factors, particularly local nontoichiometry and weakly coupled grain boundaries characteristic of Nb3Sn, cuprates and pnictides [25,](#page-10-20)[26](#page-11-0) .

The paper is organized as follows. In Sec. II we describe the experimental setup and Nb3Sn coplanar resonator used in the measurements of NLME. Sec. III contains the main experimental results. Sec. IV summarizes the essential mechanisms of NLME and theoretical results necessary for the comparison of theory with experiment. Sec. V contains discussion of our results.

## I. EXPERIMENTAL

### A. Film Deposition and Patterning

The coplanar resonator was fabricated from a 50 nm thick Nb3Sn film on a 10 mm × 10 mm × 1 mm Al2O<sup>3</sup> substrate. The film was prepared with magnetron cosputtering using both Nb and Sn targets in a growth chamber at University of Wisconsin-Madision, as described in Refs. [27](#page-11-1) and [28.](#page-11-2) Figure [1](#page-1-0) shows the resistive transition in a film grown under a similar condition. The film had a midpoint T<sup>c</sup> ≈ 17.2 K, normal state sheet resistance of 5.1 Ω, and a residual resistance ratio (RRR), Rs(300 K)/Rs(18 K) ≈ 3.2.

![](_page_1_Figure_5.jpeg)

**Caption:** Figure 7 illustrates the orientation of the magnetic field with respect to the coplanar strip line in the tested Nb3Sn resonator. The figure aids in interpreting experimental results concerning field alignment and its impact on quality factor and frequency shifts during magnetic field cycling. This setup is crucial for achieving precise alignment and understanding field orientation effects on superconducting properties crucial for optimizing Nb3Sn thin film resonator performance .


<span id="page-1-0"></span>FIG. 1. Temperature dependence of the resistance of the film Rs(T) with a midpoint critical temperature T<sup>c</sup> = 17.2 K.

The film has a polycrystalline structure with rigid grains along the [-1011] direction of the Al2O<sup>3</sup> substrate as revealed by the atomic force microscopy shown in Fig. [2.](#page-1-1) Those grains contributed to an RMS roughness of approximately 10 nm[27,](#page-11-1)[28](#page-11-2) .

The sample was patterned into a half-wave coplanar waveguide resonator using contact lithography followed by Ar ion milling. The optical image of the resonator is shown in Fig. [3.](#page-2-0) The meandered resonator has a total

![](_page_1_Figure_9.jpeg)

**Caption:** Figure 9 presents the surface morphology of a Nb3Sn thin film deposited on an Al₂O₃ substrate, characterized by scanning tunneling microscopy. The image reveals a textured grain boundary pattern illustrated with scale bars indicative of nanometer-scale resolution. These grain boundaries, arising from polycrystalline structures and Sn depletion, are pivotal in dictating the superconducting behavior observed under varying magnetic field conditions. Understanding these structural features aids in comprehending the enhanced grain boundary effects seen in Nb3Sn used for high-performance superconducting applications.


<span id="page-1-1"></span>FIG. 2. AFM image showing a polycrystalline structure of our films.

length l ≈ 24.6 mm corresponding to the fundamental resonant frequency f<sup>0</sup> = 2.236 GHz. The center conductor has a width w = 15 µm, and a gap width s = 8.8 µm between the center strip and the ground plane. The s/w ratio was set to achieve a characteristic line impedance 50 Ω. The resonator is coupled to input and output RF probes by interdigital capacitors patterned on the strip. At the ends of the transmission line, landings pads for ground-signal-ground (GSG) probes were fabricated, shown as the lightly shaded region in Fig. [3\(](#page-2-0)a). These were made by first removing few nanometers of oxide layers on the surface of Nb3Sn using Ar ion milling and then depositing a 20 nm thick layer of Pd in-situ using a lift-off technique. The landing pads made of Pd serve to prevent oxidation and damage of the film from repeated touchdown of the probes and ensure ohmic contact between the probe and the sample.

### B. Measurement Setup

The patterned sample was mounted inside a cryogenic probe station equipped with a closed cycle cryocooler [29](#page-11-3) , as shown in Fig. [4.](#page-2-1) The complex transmission coefficient S21,<sup>12</sup> was measured as functions of temperature and the external in-plane magnetic field B. The resonant frequencies and the London penetration depth were extracted by fitting the dependence of the phase on frequency of the transmission spectra[30–](#page-11-4)[36](#page-11-5). The temperature of the sample was varied using a resistive heater underneath the sample stage, and a parallel dc field up to 200 mT was produced with a NbTi superconducting magnet. This magnet was mounted on a six-motor hexapod system that allowed for fine tuning of magnet orientation by ±7 ◦ in three axes while taking the sample measure-

![](_page_2_Figure_1.jpeg)

**Caption:** Figure 3 illustrates (a) the optical image of a Nb3Sn coplanar half-wave resonator with a resonant frequency of f₀ = 2.236 GHz, showing its center meandered resonator and (b) a zoomed-in section detailing the strip and gap dimensions. The image indicates design features crucial for achieving the desired 50 Ω line impedance, involving precise fabrications such as interdigital capacitors for RF coupling and palladium landing pads for maintaining probe integrity. This structural information is vital for understanding the resonator’s fabrication and its performance under varying experimental conditions.


<span id="page-2-0"></span>FIG. 3. (a) The image of the Nb3Sn coplanar half-wave resonator with f<sup>0</sup> = 2.236 GHz. The meandered resonator in the center is terminated capacitively on both ends which tapers out to the input and output landing pads. (b) Zoomed in section of the coplanar resonator where the width of the strip is w = 15 µm and the gap between the signal strip and the ground is s = 8.8 µm.

ments. The temperature of the sample was measured using a calibrated Cernox (CX-1050-CU-HT, Lakeshore Cryotronics) resistance-temperature device fixed on the stage next to the sample. The output port of a Vector Network Analyzer (VNA) provided rf power that was delivered to the resonator by landing two GSG probes to the contacts. These probes and the cables connecting to the network analyzer were calibrated using a Short-Open-Load-Through calibration substrate mounted on the sample stage at 7 K. The drive power of VNA was selected to be -30 dBm to maximize the signal-to-noise ratio while avoiding distortion of the Lorentzian shape in transmission signal observed at higher power due to nonlinear heating effects [33,](#page-11-6)[37](#page-11-7) as shown in Fig. [5.](#page-2-2) To minimize the number of vortices trapped in the sample during its cooldown through Tc, we used three pairs of Helmholtz coils to reduce the ambient field Ba. The magnitude of B<sup>a</sup> was measured by a magnetometer while adjusting the coil currents to achieve B<sup>a</sup> <4 mG in an optimum configuration.

The resonance frequency f<sup>0</sup> = (CL) <sup>−</sup>1/<sup>2</sup>/2π is determined by the ground capacitance C and the resonator inductance L. Here L = L<sup>g</sup> + L<sup>k</sup> contains a geometrical inductance L<sup>g</sup> and a kinetic inductance Lk(T, B) associated with the inertia of supercurrents. The geometric inductance for the parameters of our sample L<sup>g</sup> = 420.5 nH/m was calculated in Appendix [A](#page-9-0) following Ref. [34.](#page-11-8) The kinetic inductance for a thin film strip of thickness d < λ(T) and width w is given by [21,](#page-10-17)[31,](#page-11-9)[32](#page-11-10)

$$L\_k \approx \frac{\mu\_0 \lambda(T)}{w} \coth\left[\frac{d}{\lambda(T)}\right] \approx \frac{\mu\_0 \lambda^2(T)}{wd}.\tag{3}$$

Temperature dependencies of Lk(T) and λ(T) were inferred from the measured frequency shift δf /f<sup>0</sup> =

![](_page_2_Picture_7.jpeg)

**Caption:** Figure 4 shows an image of the cryogenic probe station used for characterizing the Nb3Sn resonators. The setup includes a sample chip mounted on a stage with connected thermal anchors and a thermometer to monitor temperature changes. This setup is crucial for accurate temperature control and minimizing vibration during measurements, thereby improving the stability and reliability of data collected on superconducting properties as functions of magnetic field and temperature.


FIG. 4. A setup of the sample stage with two GSG probes. The sample is mounted using silver paint, and the thermometer and the OFHC bobbin for thermal anchoring of the lead wires are screwed onto the sample stage.

<span id="page-2-1"></span>![](_page_2_Figure_9.jpeg)

**Caption:** Figure 9 shows the field dependency of the normalized frequency shift δf/f₀ for varying orientations of the in-plane magnetic field on a Nb₃Sn film. The data indicates that at higher fields, the shift in the parallel B orientation displays a steeper decrease in frequency compared to the perpendicular configuration. Such disparities are reflective of the predominant influence exerted by grain boundaries on the kinetic inductance and magnetic response, distinctly observable in the Nb3Sn superconducting thin films.


<span id="page-2-2"></span>FIG. 5. A typical transmission spectrum near resonance displays a symmetric Lorentzian line shape with VNA power output of -30 dBm. Higher input power distorts the Lorentzian shape at -5 dBm.

<span id="page-2-4"></span> $[f\_0(T) - f\_0(7K)]/f\_0(7K)$ : 
$$\frac{\delta f(T)}{f\_0(7K)} = \frac{\sqrt{L\_g + L\_k(7K)}}{\sqrt{L\_g + L\_k(T)}} - 1,\tag{4}$$

where C and L<sup>g</sup> are assumed independent of T. By fitting the observed δf(T)/f0(7K) to Eqs. [\(3\)](#page-2-3) and [\(4\)](#page-2-4), we obtained λ(T) as described in the next subsection.

<span id="page-2-3"></span>For the NLME measurements, the alignment of the dc field B to the plane of the strip is crucial to keep the superconductor in the Meissner state and avoid perpendicular vortices penetrating from the film edges. These vortices caused by the misaligned field reduce the quality factor and give rise to an additional field dependence of δf(B, T) unrelated to the NLME. To find the orientation of the magnet which produces B parallel to the film plane and the minimum amount of trapped flux, the loaded quality factor QL(B) and δf(B) were measured as functions of the out-of-plane field angle ζ. We first measured the initial values of f0<sup>i</sup> and QLi at zero field, ramped the field up to 60 mT and down to zero, and measured the resulting values of f0<sup>a</sup> and QLa affected by the number of vortices trapped in the process. The sample was then thermal cycled above T<sup>c</sup> = 17.2 K at zero field to flush out trapped vortices. Measurements were repeated after the magnet was adjusted to a new angle. Shown in Fig. [6](#page-3-0) are the normalized shifts δf0/f<sup>0</sup> = (f0<sup>a</sup> − f0i)/f<sup>0</sup> and δQL/Q<sup>L</sup> = (QLa − QLi)/Q<sup>L</sup> as functions of the magnet angle ζ. Both δf0(ζ)/f<sup>0</sup> and δQL(ζ)/QL(0) peaked at ζ = 3.8 ◦ which we adopted as a magnet orientation producing the dc field parallel to the plane of the film. This procedure is similar to that which was used in Ref. [22.](#page-10-16)

![](_page_3_Figure_1.jpeg)

**Caption:** Figure 6 illustrates the normalized shifts in (a) resonant frequencies δf/f₀ and (b) loaded quality factors δQ_L/Q_L as functions of the field misalignment angle ζ (degrees) for a Nb3Sn superconducting resonator. The experimental context involves cycling the magnetic field from 0 to 60 mT and back to zero, measuring resultant changes in the loaded quality factor and frequency to determine the optimal magnet alignment. Notably, both δf(ζ)/f₀ and δQ_L(ζ)/Q_L(0) peak at ζ = 3.8°, suggesting a configuration in which the magnetic field is parallel to the film plane. This behavior is crucial for minimizing trapped vortices and optimizing device performance, confirming field alignment within the experimental setup .


<span id="page-3-0"></span>FIG. 6. Normalized shifts in (a) resonant frequencies and (b) loaded quality factors after cycling B from 0 to 60 mT and back to 0 as a function of the offset angle ζ. Both δf(ζ) and δQL(ζ) are peaked at ζ = 3.8 ◦ .

Having aligned the magnet, we measured f0(T, B) as a function of in-plane dc field up to 200 mT at parallel ϕ = 0◦ and perpendicular ϕ = 90◦ field orientations with respect to the strip indicated by Fig. [7](#page-3-1) at temperatures between 7 K and 12 K. After measurements at a given temperature were completed, the sample was warmed up above T<sup>c</sup> to expel any trapped vortices. For each field and temperature point, we repeated the measurement over 50 times, and the average f0(B, T) was calculated.

![](_page_3_Figure_5.jpeg)

**Caption:** Figure 7 illustrates the schematic diagram of the experimental configuration used to determine the orientation of the magnetic field with respect to the direction of the RF current in a Nb3Sn coplanar resonator. This setup is essential for experiments investigating the nonlinear Meissner effect, highlighting the importance of precise field alignment to minimize vortex penetration and its adverse effects on resonator performance. Notably, achieving accurate alignment aids in maintaining high quality factors and stable resonant frequencies during superconductivity studies in Nb3Sn films.


<span id="page-3-1"></span>FIG. 7. The coordinate system for the magnet orientation with respect to the direction of the rf current on the coplanar resonator.

### C. Temperature Dependence

The temperature-dependence of λ(T) can be obtained from the measurements of the resonance frequency f0(T). Using Eqs. [\(3\)](#page-2-3) and [\(4\)](#page-2-4) we extracted λ(T) from the measured f0(T) by fitting the temperature dependence of a relative frequency shift δf /f<sup>0</sup> = [f0(T)−f0(7K)]/f0(7K) with the conventional two-fluid approximation of λ(T) = λ(0)[1 − (T /Tc) 4 ] −1/2 . Shown in Fig. [8](#page-4-0) is the temperature dependent part of f0(T) along with the fit with Eq. [4.](#page-2-4) The fit gives λ(0) = 353 nm, well above the London penetration depth λ(0) ≈ 90 nm for a clean stoichiometric Nb3Sn [35](#page-11-11). The latter may result from nonstoichimetric inclusions which cause a slight reduction of T<sup>c</sup> in our films[27](#page-11-1)[,28](#page-11-2). For λ(0) = 352 nm, w = 15 µm and d = 50 nm, the kinetic inductance L<sup>k</sup> = µ0λ 2 (T)/wd ' 200 nH/m at 9 K accounts for about 1/3 of the total inductance L = L<sup>g</sup> + L<sup>k</sup> with L<sup>g</sup> = 420.5 nH/m.

Another factor contributing to the large value of λ(0) is the grain boundary structure of our Nb3Sn films shown in Fig. [2.](#page-1-1) It has been well-established that Sn depletion at GBs [40](#page-11-12)[–43](#page-11-13) results in weak Josephson coupling of crystalline grains, which has been used to optimize pinning of vortices by GBs in Nb3Sn conductors[39](#page-11-14)[,44](#page-11-15). In turn, the weakly-coupled GBs facilitate preferential penetration of the magnetic field along the GB network causing an increase of the global λ, as is characteristic of many superconductors with short coherence length, including Nb3Sn, cuprates and pnictides [25](#page-10-20)[,26](#page-11-0) .

![](_page_4_Figure_1.jpeg)

**Caption:** Figure 8 displays the temperature-dependent shifts in the resonant frequency of a thin Nb3Sn film on an Al₂O₃ substrate. The resonant frequency f₀(T) is portrayed against temperature, with empirical data fitted to a model yielding a London penetration depth λ(0) = 353 nm. This deviation from the expected 90 nm in clean stoichiometric conditions is attributed to the Sn depletion at grain boundaries, illustrating the pronounced effect of grain boundary microstructure on superconductive properties, facilitating preferential field penetration along the grain boundaries.


<span id="page-4-0"></span>FIG. 8. The normalized temperature-dependent part of the resonant frequency. The fit of the data to Eq. [4](#page-2-4) gives λ(0) = 353 nm.

### D. Field Dependence

Shown in Fig. [9](#page-4-1) are the observed field dependencies of the frequency shifts for in-plane B parallel and perpendicular to the strip. In both cases δf(B) decreases nearly linearly with B above 30 − 40 mT but flattens at lower fields. Here the slope of δf(B) for in-plane B parallel to the strip is about twice of the slope of δf(B) for in-plane B perpendicular to the strip. In the field range 0 < B < 200 mT of our measurements the Nb3Sn film of thickness 50 nm is in the Meissner state as the parallel field B remains below the nominal lower critical field of a vortex in a thin film. Indeed, an estimate of Bc<sup>1</sup> = (2φ0/πd<sup>2</sup> ) ln(d/ξ) in the London approximation [23](#page-10-18) with d = 50 nm and ξ = 5 nm[38](#page-11-16)[,39](#page-11-14) yields Bc<sup>1</sup> ' 1.17 tesla exceeding B<sup>c</sup> = 0.54 tesla of Nb3Sn [38](#page-11-16)[,39](#page-11-14). The slope of δf(B) in Fig. [9](#page-4-1) increases with increasing temperature, consistent with the temperature dependence of λ(T).

Our δf(B) data exhibit significant scatter, as has also been observed in NLME experiments on cuprate [15](#page-10-21). The error bars in Fig. [9](#page-4-1) represent a standard deviation from repeated measurements at each data points, δf(B) data for Bky having larger error bars as compared to Bkz. The main contribution to the error bars comes from vibrations of the sample stage and GSG probes originating from the cryocooler. For B perpendicular to the strip, longer probe arms had to be used in the NLME measurements, which increases the amplitude of vibrations.

### II. CONTRIBUTIONS TO NLME

In this section we consider different contributions to the resonant frequency shift δf(T, B) caused by the inplane dc magnetic field B.

![](_page_4_Figure_8.jpeg)

**Caption:** Figure 8 provides the normalized temperature-dependent aspect of resonatore frequency shifts, fitted to a model indicating λ(0) = 353 nm in the presence of Sn depletion at grain boundaries, revealing the impactful structural variation on Nb3Sn film superconductivity. This further correlates to field-induced inductance variations as detailed in coupled experimental and theoretical assessments on temperature impacts.


<span id="page-4-1"></span>FIG. 9. Normalized frequency shift δf0(T, B)/f0(T, 0) as a function of the in-plane dc magnetic field (a) parallel and (b) perpendicular to the strip.

### A. Meissner current pairbreaking

We start with the calculation of the contribution of pairbreaking Meissner currents to δf(H) using the TDGL equations for a dirty s-wave superconductor[45](#page-11-17)[,46](#page-11-18):

$$\tau\_{GL}(1+4\tau\_E^2\Delta^2)^{-1/2}\left(\frac{\partial}{\partial t}+2ie\Phi+2\tau\_E^2\frac{\partial\Delta^2}{\partial t}\right)\Psi$$

$$=\left(1-\frac{\Delta^2}{\Delta\_0^2}\right)\Psi+\xi^2\left(\nabla-2ie\mathbf{A}\right)^2\Psi,\tag{5}$$

<span id="page-4-3"></span><span id="page-4-2"></span>
$$\mathbf{J} = -\frac{\pi \sigma\_0}{4eT\_c} \Delta^2 \mathbf{Q} - \sigma\_0 \left(\nabla \Phi + \frac{\partial \mathbf{A}}{\partial t}\right). \tag{6}$$

Here τGL = π~/8kB(T<sup>c</sup> − T), ξ = [π~D/8kB(T<sup>c</sup> − T)]<sup>1</sup>/<sup>2</sup> is the coherence length, D the electron diffusivity, Φ is a scalar potential, τ<sup>E</sup> is an energy relaxation time due to electron-phonon scattering[46](#page-11-18), ∆<sup>2</sup> <sup>0</sup> = 8π 2k 2 <sup>B</sup>Tc(T<sup>c</sup> − T)/7ζ(3), σ<sup>0</sup> = 2e <sup>2</sup>DN(0) is the normal state conductivity, N(0) is the density of states at the Fermi surface,

and −e is the electron charge. Equations [\(5\)](#page-4-2) and [\(6\)](#page-4-3) were derived from the kinetic BCS theory assuming that Q(r, t) and ∆(r, t) vary slowly over ξ<sup>0</sup> ' (~D/kBTc) 1/2 , the diffusion length L<sup>E</sup> = (DτE) <sup>1</sup>/<sup>2</sup> and τ<sup>E</sup> [45](#page-11-17)[,46](#page-11-18), where

$$\tau\_E = \frac{8\hbar}{7\pi\zeta(3)\gamma k\_B T\_F} \left(\frac{c\_s}{v\_F}\right)^2 \left(\frac{T\_F}{T}\right)^3. \tag{7}$$

Here c<sup>s</sup> is the speed of longitudinal sound, v<sup>F</sup> and T<sup>F</sup> = <sup>F</sup> /k<sup>B</sup> are the Fermi velocity and temperature, respectively, and γ is a dimensionless electron-phonon coupling constant. For cs/v<sup>F</sup> ' 10−<sup>3</sup> , T<sup>F</sup> ∼ 10<sup>5</sup> K, T<sup>c</sup> = 17 K and γ ' 1.5 [38](#page-11-16), Eq. [\(7\)](#page-5-0) yields τE(Tc) ∼ 10 ps.

For a wide film in a parallel magnetic field, Q(x) and ∆(x) depend only on the coordinate x across the film, and the TDGL equations in the gauge Φ = 0 can be written in the dimensionless form:

$$(1 + g^2 \psi^2)^{1/2} \dot{\psi} = (1 - q^2) \psi + \psi'' - \psi^3,\qquad(8)$$

$$j = -u\psi^2 q - \dot{q},\tag{9}$$

where ψ = ∆/∆0, q = Qξ, g = 2∆0τE/~, j = J/J0, t is in units of τGL, J<sup>0</sup> = σ0/2eξτGL, x is in units of ξ, the prime and overdot denotes differentiation with respect to x and t, respectively, and u = π <sup>4</sup>/14ζ(3) ≈ 5.79.

For a coplanar resonator of thickness d < λ and width w λ in a parallel dc field B inclined by the angle ϕ to the z-axis along the strip, we have:

$$q\_z = -hx\sin\varphi + a\_\omega e^{i\omega t}, \qquad q\_y = hx\cos\varphi,\qquad(10)$$

where h = B/Bc2, Bc<sup>2</sup> = φ0/2πξ<sup>2</sup> , a<sup>ω</sup> = Aωe iωt/A0, A<sup>ω</sup> is a small rf vector potential excited along the strip, A<sup>0</sup> = φ0/2πξ, x = 0 is taken in the middle of the film and the London screening at d λ is disregarded.

At a<sup>ω</sup> = 0 the dc field causes a reduction of ψ(x) = 1−ψ1(x). The equation for a field-induced correction ψ<sup>1</sup> is obtained from Eq. [\(8\)](#page-5-1) in the first order in h <sup>2</sup> 1:

$$
\psi\_1'' - 2\psi\_1 = -h^2 x^2, \qquad \psi\_1'(\pm d/2) = 0. \tag{11}
$$

where d is in units of ξ. The squared order parameter averaged over the film thickness, ψ¯<sup>2</sup> ' 1−2d −1 R d/<sup>2</sup> −d/2 ψ1dx, is obtained by integrating Eq. [\(11\)](#page-5-2) over x:

<span id="page-5-5"></span>
$$
\bar{\psi}^2 = 1 - \frac{d^2 h^2}{12}.\tag{12}
$$

To calculate a linear response current δIωe iωt induced by a weak aωe iωt in the presence of a parallel dc field, we linearize Eq. [\(9\)](#page-5-3) in aω:

$$\frac{\delta I\_{\omega}}{wJ\_{0}} = -u d\bar{\psi}^{2} a\_{\omega} + 2hu \sin \varphi \int\_{-d/2}^{d/2} x \delta \psi(x) dx - id\omega a\_{\omega}. \tag{13}$$

This shows that δψ is coupled with the dc Meissner current flowing along the strip due to the field component B<sup>y</sup> = B sin ϕ. Here δψ(x) ∝ a<sup>ω</sup> satisfies the following equation obtained from Eq. [\(8\)](#page-5-1) linearized with respect to small δψ and aω:

<span id="page-5-0"></span>
$$
\delta \psi'' - k\_\omega^2 \delta \psi = -2a\_\omega hx \sin \varphi, \qquad \delta \psi'(\pm d/2) = 0,\tag{14}
$$

$$k\_{\omega}^2 = 2 + i\omega\tau, \qquad \tau = \tau\_{GL}\sqrt{1 + (2\tau\_E \Delta/\hbar)^2}. \tag{15}$$

The solution of Eq. [\(14\)](#page-5-4) is given by:

$$\delta\psi(x) = \sum\_{n=0}^{\infty} A\_n \sin q\_n x, \qquad q\_n = \frac{\pi}{d}(2n+1), \qquad (16)$$

<span id="page-5-4"></span>
$$A\_n = \frac{8(-1)^n a\_\omega h \sin \varphi}{d q\_n^2 (k\_\omega^2 + q\_n^2)}. \tag{17}$$

<span id="page-5-6"></span>Inserting δψ(x) in Eq. [\(13\)](#page-5-5) and integrating over x yields:

<span id="page-5-1"></span>
$$\frac{\delta I\_{\omega}}{dwJ\_{0}} = -\left[u\bar{\psi}^{2} - \sum\_{n=0}^{\infty} \frac{32ud^{2}h^{2}\sin^{2}\varphi}{\pi^{4}(2n+1)^{4}(2+i\omega r+q\_{n}^{2})} + i\omega\right]a\_{\omega} \tag{18}$$

<span id="page-5-3"></span>The sum in Eq. [\(18\)](#page-5-6) converges rapidly, so q 2 <sup>n</sup> ∝ (ξ/d) <sup>2</sup> 1 in the denominator can be neglected in films with ξ d λ. Using P<sup>∞</sup> <sup>n</sup>=0(2n + 1)<sup>−</sup><sup>4</sup> = π <sup>4</sup>/96 and restoring the original units, we obtain the linear response current δI<sup>ω</sup> induced by the ac vector potential Aω:

<span id="page-5-8"></span><span id="page-5-7"></span>
$$
\delta I\_{\omega} = -\left(\frac{1}{\mu\_0 \tilde{\lambda}\_{\omega}^2} + i\omega \sigma\_0\right) dw A\_{\omega}, \tag{19}
$$

where the complex London penetration depth is given by:

$$\frac{1}{\tilde{\lambda}\_{\omega}^{2}} = \frac{1}{\lambda^{2}} \left[ 1 - \frac{1}{3} \left( \frac{2\pi\xi dB}{\phi\_{0}} \right)^{2} \left( \frac{1}{4} + \frac{\sin^{2}\varphi}{2 + i\omega\tau} \right) \right]. \tag{20}$$

Here the NLME term ∝ B<sup>2</sup> depends on ω and the field orientation angle ϕ. The maximum NLME contribution to λ<sup>ω</sup> occurs at ϕ = π/2 when B ⊥ z and the rf current is parallel to the dc Meissner current. At ωτ 1 the angular dependence of λω(ϕ) in Eq. [\(20\)](#page-5-7) reduces to that was obtained previously in a quasi-static limit [22](#page-10-16) .

<span id="page-5-2"></span>The imaginary part of λ˜<sup>2</sup> <sup>ω</sup> contributes to the dynamic conductivity σω. Denoting λ 2 <sup>ω</sup> = Reλ˜<sup>2</sup> <sup>ω</sup>, and separating real and imaginary parts in Eqs. [\(19\)](#page-5-8) and [\(20\)](#page-5-7), yields:

$$
\lambda\_{\omega}^{2} = \left[1 + \frac{1}{3} \left(\frac{2\pi\xi dB}{\phi\_{0}}\right)^{2} \left(\frac{1}{4} + \frac{2\sin^{2}\varphi}{4 + \omega^{2}\tau^{2}}\right)\right] \lambda^{2}, \quad (21)
$$

<span id="page-5-10"></span><span id="page-5-9"></span>
$$
\sigma\_{\omega} = \sigma\_0 + \frac{1}{3\mu\_0 \lambda^2} \left(\frac{2\pi\xi dB}{\phi\_0}\right)^2 \frac{\tau \sin^2 \varphi}{4 + \omega^2 \tau^2}. \tag{22}
$$

The NLME field correction to the kinetic inductance L<sup>M</sup> <sup>k</sup> = µ0λ 2 (B)/dw is given by:

$$
\delta L\_k^M = \frac{\mu\_0 \lambda^2}{3dw} \left(\frac{\pi \xi dB}{\phi\_0}\right)^2 \left[1 + \frac{2\sin^2\varphi}{1 + (\omega \tau/2)^2}\right].\tag{23}
$$

As follows from Eqs. [\(22\)](#page-5-9) and [\(23\)](#page-5-10), the NLME correction to σ<sup>ω</sup> remains finite at Tc, while δL<sup>M</sup> <sup>k</sup> ∝ (T<sup>c</sup> − T) −2 increases stronger than the zero-field kinetic inductance
L<sup>M</sup> <sup>k</sup> = λ <sup>2</sup>/d ∝ (T<sup>c</sup> − T) <sup>−</sup><sup>1</sup> as T → Tc. The dependencies of λ<sup>ω</sup> and σ<sup>ω</sup> on the orientation of B persist as long as ωτ . 1 and disappear at ωτ 1. The latter occurs both at T → T<sup>c</sup> where τGL(T) diverges and at low temperatures where τE(T) ∝ T −3 increases strongly.

We estimate δλ(B) = λ(B) − λ = (πξdB/φ0) <sup>2</sup>/2 at ϕ = π/2 and ωτ 1 for d = 50 nm and ξ = 5 nm. Here δλ/λ = 7.7 × 10−<sup>4</sup> at B = 100 mT, which translates to δλ = 0.27 nm at λ = 350 nm. If the total L is dominated by the kinetic inductance, Eq. [\(23\)](#page-5-0) yields the maximum NLME frequency shift:

$$\frac{\delta f}{f} = -\frac{1}{6} \left( \frac{\pi \xi dB}{\phi\_0} \right)^2 \left[ 1 + \frac{2 \sin^2 \varphi}{1 + (\omega \tau/2)^2} \right]. \tag{24}$$

In superconductors with κ 1 the maximum δf(Bc1)/f<sup>0</sup> in a thin film is much greater than the maximum NLME bulk shift δλ/λ = (Bc1/Bc) <sup>2</sup>/8 = (ln κ/4κ) <sup>2</sup> which follows from Eq. [\(2\)](#page-0-0). For d = 50 nm ξ = 5 nm, ϕ = π/2 and B = 100 mT, we obtain δf(B)/f<sup>0</sup> = (πξdB/φ0) <sup>2</sup>/2 ' 8 × 10<sup>−</sup><sup>4</sup> , well below the observed δf /f0. Taking the geometrical inductance into account further reduces δf /f<sup>0</sup> by a factor ' 3. Moreover, according to Eq. [\(21\)](#page-5-1) the field-induced shift δλ at ϕ = π/2 for which the rf currents are parallel to the dc Meissner currents is 3 times larger than δλ at ϕ = 0. This is inconsistent with the experimental data shown in Fig. [9](#page-4-0) where the slope of δf(B)/f<sup>0</sup> at ϕ = 0 is about 2 times larger than for δf(B)/f<sup>0</sup> at ϕ = π/2. Thus, not only is the Meissner pairbreaking too weak to account for the observed δf /f<sup>0</sup> but it yields the field and orientational dependencies of δf(B)/f<sup>0</sup> inconsistent with our experimental data on Nb3Sn.

## B. Grain boundary contribution

A significant contribution to L<sup>k</sup> can come from local non-stoichiometry, strains, and grain boundaries (GBs) in polycrystalline Nb3Sn. Particularly, the well-known Sn depletion at GBs [40–](#page-11-0)[43](#page-11-1) results in weak Josephson coupling of grains in Nb3Sn. Our polycrystalline films have lateral grain sizes l<sup>2</sup> ∼ 0.1 − 1 µm (see Fig. [2\)](#page-1-0) and local nonstoichiometry causing inhomogeneities of superconducting properties on the same length scales [27,](#page-11-2)[28](#page-11-3). If weakly coupled GBs are regarded as planar Josephson junctions (JJs), each GB has a kinetic inductance [47](#page-11-4)

$$L\_J = \frac{\phi\_0}{2\pi I\_c \cos\theta},\tag{25}$$

where Ic(T) is a critical current of the JJ. The field dependence of L<sup>J</sup> (B) is determined by the phase difference θ(r) induced by the dc field on a GB. Because GBs in Nb3Sn can have broad distributions of sizes and I<sup>c</sup> values [25](#page-10-0), low-I<sup>c</sup> GBs can significantly increase L<sup>J</sup> if they form interfaces blocking the cross-section of the film.

An array of weakly-coupled GBs can be modeled by a Hamiltonian of a granular superconductor [48–](#page-11-5)[50](#page-11-6)

$$\mathcal{H} = -\sum\_{ij} J\_{ij}\cos(\chi\_i - \chi\_j - A\_{ij}),\tag{26}$$

where the coupling energies Jij = ~I ij <sup>c</sup> /2e of the i-th and j-th grains are proportional to the respective intergrain Josephson critical currents I ij c , χ<sup>i</sup> and χ<sup>j</sup> are phases of the superconducting order parameters in the grains, and the magnetic phase factors Aij are given by:

$$A\_{ij} = \frac{2\pi}{\phi\_0} \int\_i^j \mathbf{A} \cdot d\mathbf{l}.\tag{27}$$

Monte-Carlo simulations of the Hamiltonian [\(26\)](#page-6-0) of a disordered XY model have shown that the helicity moduli and the global kinetic inductance change nearly linearly with the dc magnetic field at B . φ0/l<sup>2</sup> <sup>2</sup> and exhibit strong fluctuations as functions of B due to transitions between many metastable states in finite JJ arrays [49,](#page-11-7)[50](#page-11-6) . These results appear qualitatively consistent with the observed field dependence of δf(B)/f<sup>0</sup> shown in Fig. [9.](#page-4-0)

To get an insight into the nearly linear decrease of δf(B) with B shown in Fig. [9,](#page-4-0) we use a mean-field model in which cos(χ<sup>i</sup> − χ<sup>j</sup> − Aij ) is replaced with an averaged value hcos θi. Then a disturbance δθ induced by a weak rf current Jωe iωt on an overdamped GB is described by the dynamic equation of the RSJ model

<span id="page-6-2"></span>
$$
\tau\_J \delta \dot{\theta} + \langle \cos \theta \rangle \delta \theta = (J\_\omega / J\_c) e^{i\omega t},\tag{28}
$$

where τ<sup>J</sup> is a relaxation time caused by ohmic quasiparticle current through JJ, and h...i denotes averaging over orientations of the GB planes:

<span id="page-6-1"></span>
$$\langle \cos \theta \rangle = \frac{1}{NS} \sum\_{s} n\_{zs}^{2} \int\_{S} \cos \theta\_{s}(\mathbf{r}) dS \tag{29}$$

Here the summation goes over all GBs, and θs(r) on a planar GB smaller than the Josephson length λ<sup>J</sup> = (φ0/2πµ0dJc) 1/2 is determined by [47](#page-11-4)

<span id="page-6-6"></span><span id="page-6-4"></span>
$$
\nabla \theta = \frac{2\pi\Lambda}{\phi\_0} [\mathbf{B} \times \mathbf{n}],\tag{30}
$$

<span id="page-6-5"></span>where n is a unit vector normal to the GB surface, and Λ ∼ d for a JJ in a thin film (d . λ) in a parallel magnetic field [47](#page-11-4)[,51,](#page-11-8)[52](#page-11-9). The factor n 2 z in Eq. [\(29\)](#page-6-1) takes into accounts that only the perpendicular component of the rf current nzJ<sup>ω</sup> = Jcδθ causes the Josephson voltage Vω~δ ˙θnz/2e along Jω. From Eq. [\(28\)](#page-6-2), we obtain δθ = Jωe iωt/(hcos θi + iωτ<sup>J</sup> )J<sup>c</sup> and the impedance Z<sup>J</sup> = NVω/I<sup>ω</sup> of N grain boundaries:

$$Z = \frac{i\omega NR \langle L\_J \rangle}{R + i\omega \langle L\_J \rangle}, \qquad \langle L\_J \rangle = \frac{\phi\_0}{2\pi I\_c \langle \cos \theta \rangle} \tag{31}$$

<span id="page-6-0"></span>We calculate hcos θi for randomly-oriented planar GBs parameterized by the Euler angles α and β shown in Fig. [10.](#page-7-0) As shown in Appendix [B,](#page-9-0) averaging cos θ over the area S of a tilted GB in a magnetic field gives:

<span id="page-6-3"></span>
$$\bar{c} = \frac{1}{S} \int \cos \theta dS = \frac{\sin(q\_1 l\_1) \sin(q\_2 l\_2)}{q\_1 l\_1 q\_2 l\_2},\qquad(32)$$

![](0__page_7_Figure_1.jpeg)

**Caption:** Figure 10 schematically represents the laboratory setup for magnetic field orientation in relation to the applied rf current on the coplanar resonator. This illustration is essential for understanding the experimental methodology employed in assessing resonator behavior under different field alignments, crucial for interpreting data on magnetic flux entry, resonator quality, and frequency shifts.


<span id="page-7-0"></span>FIG. 10. Geometry of a rectangular tilted GB (yellow). The red arrow shows the normal to the GB plane.

where

$$q\_1 = \frac{\pi B}{\phi\_0} \Lambda \sin \alpha, \quad q\_2 = \frac{\pi B}{\phi\_0} \Lambda \sin \beta \cos \alpha, \quad \mathbf{B} \| z \quad \text{(33)}$$

$$q\_1 = \frac{\pi B}{\phi\_0} \Lambda \cos \alpha, \quad q\_2 = \frac{\pi B}{\phi\_0} \Lambda \sin \beta \sin \alpha, \quad \mathbf{B} \parallel y \quad (34)$$

There is a significant difference of GB lengths l<sup>1</sup> and l<sup>2</sup> in our Nb3Sn coplanar resonator with d w. Here l<sup>1</sup> is smaller or of the order of the film thickness, l<sup>1</sup> . d ' 50 nm, whereas lateral GB lengths l<sup>2</sup> are in a submicron range l<sup>2</sup> ∼ 0.1 − 1 µm (see Fig. [2\)](#page-1-0) so that l<sup>1</sup> ∼ (10<sup>−</sup><sup>2</sup> − 10<sup>−</sup><sup>1</sup> )l2. As a result, sin(q1l1)/l1q<sup>1</sup> in Eq. [\(32\)](#page-6-3) remains close to 1 in the field region B . φ0/πΛl<sup>1</sup> ∼ φ0/πd<sup>2</sup> ∼ Bc<sup>1</sup> of our measurements, while sin(q2l2)/l2q<sup>2</sup> has a strong field dependence at B > B<sup>0</sup> ' φ0/πdl2. Here B<sup>0</sup> ' 25 mT at l<sup>2</sup> = 0.5 µm and d = 50 nm. As a result, Eq. [\(32\)](#page-6-3) at B < Bc<sup>1</sup> simplify to

$$\bar{c} = \frac{\sin(b \cos \alpha \sin \beta)}{b \cos \alpha \sin \beta}, \qquad \mathbf{B} \| z \tag{35}$$

$$\bar{c} = \frac{\sin(b \sin \alpha \sin \beta)}{b \sin \alpha \sin \beta}, \qquad \mathbf{B} \| y \tag{36}$$

$$b = B/B\_0, \qquad B\_0 = \phi\_0/\pi \Lambda l\_2. \tag{37}$$

Here only tilted GBs which are not perpendicular to the film plane (β 6= 0) contribute to the strong field dependence of ¯c(B). The average hcos θi over all GB orientations in Eq. [\(29\)](#page-6-1) can be written in the form

$$\langle \cos \theta \rangle = \sum\_{i,j} P\_{ij} \bar{c}\_{ij} \cos^2 \alpha\_i \cos^2 \beta\_j,\tag{38}$$

where ¯cij is given by either Eq. [\(35\)](#page-7-1) or [\(36\)](#page-7-2), depending on the direction of B, i and j label different GBs, n ij <sup>z</sup> = cos α<sup>i</sup> cos β<sup>j</sup> , and Pij is a probability distribution of α<sup>i</sup> and β<sup>j</sup> normalized by P ij Pij = 1.

The angular distributions of GBs is affected by the crystalline texturing during the film growth and other materials factors. We consider here a simple case of randomly-oriented GBs with equal probabilities of all α<sup>i</sup> and β<sup>j</sup> . Then Eq. [\(38\)](#page-7-3) reduces to the integrals [\(B12\)](#page-10-1) and [\(B13\)](#page-10-2) given in Appendix [B.](#page-9-0) Numerical calculation of these integrals for hcos θi yields the field dependencies of hL<sup>J</sup> i shown in Fig. [11.](#page-7-4)

![](0__page_7_Figure_14.jpeg)

**Caption:** Figure 11 depicts the field dependency of kinetic inductance 〈L_J(B)〉 normalized to 〈L_J(0)〉 when subjected to a dc magnetic field, either parallel (B||z, blue) or perpendicular (B||y, red) to the strip in a Nb3Sn film. The graph displays a notably stronger increase in kinetic inductance for B||z as compared to B||y, diverging from the expected behavior observed in Nb coplanar resonators. This observation highlights the impact of grain boundaries, which alter the magnetic field's effect on superconducting kinetics differently from the conventionally predicted models reliant solely on Meissner pairbreaking.


<span id="page-7-6"></span><span id="page-7-4"></span>FIG. 11. The field-dependencies of the kinetic inductance hL<sup>J</sup> (B)i calculated from Eqs. [\(31\)](#page-6-4), [\(B12\)](#page-10-1) and [\(B13\)](#page-10-2) for random orientation of GB planes, and the dc magnetic field applied parallel (Bkz) and perpendicular (Bky) to the strip line.

As follows from Fig. [11,](#page-7-4) grain boundaries can radically change the field dependence of the kinetic inductance as compared to the NLME caused by the Meissner pairbreaking. First, the GB contribution hL<sup>J</sup> (B)i is quadratic in B only at very low fields B . B<sup>0</sup> B<sup>c</sup> and exhibits a nearly linear field dependence at B & B<sup>0</sup> Bc, whereas the Meissner pairbreaking gives δL<sup>k</sup> ∝ B<sup>2</sup> all the way to B = Bc1. Second, the field Bkz applied along the strip causes stronger increase of hL<sup>J</sup> (B)i than the transverse field Bky. This is the opposite of the orientational field dependence of δL<sup>M</sup> k (B) described by Eq. [\(23\)](#page-5-0) and observed on Nb coplanar resonator [22](#page-10-3). Yet both features of hL<sup>J</sup> (B)i are in agreement with our experimental data on polycrystalline Nb3Sn shown in Fig. [9.](#page-4-0)

## III. DISCUSSION

<span id="page-7-3"></span><span id="page-7-2"></span><span id="page-7-1"></span>To compare the contributions of Meissner pairbreaking and weakly-coupled GBs to δf(B)/f0, we evaluate the GB kinetic inductance L J k per unit length of the strip in the above mean-field model. If GBs have the same critical current density Jc, dimensions d×l<sup>2</sup> but different orientations, L J <sup>k</sup> ∼ hL<sup>J</sup> i/l2, where hL<sup>J</sup> i is given by Eq. [\(25\)](#page-6-5) with I<sup>c</sup> ∼ dwJc. As a result,

<span id="page-7-5"></span>
$$L\_k^J \sim \frac{\phi\_0}{2\pi J\_c w d l\_2 \langle \cos \theta \rangle} \sim L\_k^M \times \frac{\xi J\_d}{l\_2 J\_c} \tag{39}$$

Here J<sup>d</sup> = φ0/2πµ0λ 2 ξ is of the order of the GL depairing current density. The kinetic inductance is dominated by weakly-coupled GBs if J<sup>c</sup> ξJd/l2. For our Nb3Sn films with ξ ' 5 nm and l<sup>2</sup> ' 200 nm, the GB contribution dominates if J<sup>c</sup> . 10<sup>−</sup><sup>2</sup>Jd.

The effective penetration depth λ extracted from the measured kinetic inductance is affected by GBs. The

Meissner contribution L<sup>M</sup> k is determined by the London penetration depth λ in the dirty limit[46](#page-11-10):

$$L\_k^M = \frac{\mu\_0 \lambda^2}{dw} = \frac{\hbar \rho\_s}{\pi dw \Delta(T)} \coth \frac{\Delta(T)}{2k\_B T},\qquad(40)$$

where ρ<sup>s</sup> is the normal state resistivity. If GBs can be modeled as S-I-S Josephson junctions [47](#page-11-4), their contribution to L<sup>k</sup> can be evaluated from Eq. [\(39\)](#page-7-5):

$$L\_k^J \sim \frac{\hbar}{eJ\_c w dl\_2} \simeq \frac{\hbar R\_\perp}{dw l\_2 \Delta(T)} \coth \frac{\Delta(T)}{2k\_B T},\qquad(41)$$

where R<sup>⊥</sup> is the tunneling resistance of the JJ per unit area. Defining the effective penetration depth λ˜ in the total kinetic inductance L<sup>k</sup> = L<sup>M</sup> <sup>k</sup> + L J <sup>k</sup> <sup>=</sup> <sup>λ</sup>˜2/dw and combining Eqs. [\(40\)](#page-8-0) and [\(41\)](#page-8-1), we obtain:

$$
\tilde{\lambda}^2(T) = \frac{\hbar}{\pi \Delta(T)} \left[ \rho\_s + C\_1 \frac{R\_\perp}{l\_2} \right] \coth \frac{\Delta(T)}{2k\_B T}, \qquad (42)
$$

where the factor C<sup>1</sup> ∼ 1 accounts for details of the shape and angular distributions of GBs. In the S-I-S model the temperature dependencies of the GB and Meissner contributions to λ˜ are the same. This is no longer the case if GBs are proximity-coupled S-N-S junctions [47](#page-11-4) for which J<sup>c</sup> ∝ (1 − T /Tc) 2 . The S-N-S scenario may be more relevant for Nb3Sn in which strongly-coupled GBs can transmit high current densities which are still well below the depairing limit [39–](#page-11-11)[44](#page-11-12) .

The above estimate pertains to B = 0 but the fieldinduced frequency shift δf /f<sup>0</sup> is determined by small field - dependent corrections δL<sup>M</sup> k and δL<sup>J</sup> k . Here δL<sup>M</sup> k is given by Eq. [\(23\)](#page-5-0) and δL<sup>J</sup> <sup>k</sup> ∼ (µ0λ <sup>2</sup>/dw)(ξJd/l2Jc)(B/B0) at B & B<sup>0</sup> ' φ0/3πdl2, as shown in Fig. [11.](#page-7-4) Hence,

$$\frac{\delta L\_k^M}{\delta L\_l^J} \sim \frac{B\xi dJ\_c}{\phi\_0 J\_d}, \qquad B \gtrsim B\_0 \tag{43}$$

This ratio is independent of the lateral GB sizes and is much smaller than 1 since J<sup>c</sup> Jd, and Bdξ/φ<sup>0</sup> < 2 × 10<sup>−</sup><sup>2</sup> at B < 200 mT, ξ = 5 nm and d = 50 nm. Thus, the NLME field-dependent frequency shift in our polycrystalline Nb3Sn coplanar resonators is dominated by grain boundaries, even if their contribution to the total kinetic inductance at B = 0 is much smaller than that of the Meissner currents. This conclusion is consistent not only with the observed nearly linear field dependence of δf /f<sup>0</sup> but also with the fact that the slope of δf /f<sup>0</sup> at B along the strip is larger than the slope of δf /f<sup>0</sup> at B perpendicular to the strip.

The fit of the observed δf(T, B)/f0(T, 0) to the GB model depends on many uncertain parameters such as distribution of orientations and local J<sup>c</sup> values of GBs, their geometrical sizes and mechanisms of current transport through GBs. Shown in Fig. [12](#page-8-2) is an example of δf(T, B)/f0(T, 0) calculated for uniform distributions of the Euler angles of GBs:

<span id="page-8-3"></span>
$$\frac{\delta f(T, B)}{f\_0(T, 0)} = \left[ \frac{1 + a\epsilon(T) / \langle \cos \theta(0) \rangle}{1 + a\epsilon(T) / \langle \cos \theta(B) \rangle} \right]^{1/2} - 1,\tag{44}$$

<span id="page-8-1"></span><span id="page-8-0"></span>where a = L J k (0, 0)/L, the factor (T) = [1 − (T /Tc) 4 ] −2 approximates the S-N-S temperature dependence of Jc(T) in Eq. [\(39\)](#page-7-5) at 7K < T < Tc, and hcos θ(B)i is given by Eqs. [\(B12\)](#page-10-1) and [\(B13\)](#page-10-2). We took a = 3.5×10−<sup>4</sup> , B<sup>0</sup> = 10 mT and different temperatures corresponding to those in Fig. [9.](#page-4-0) As follows from Fig. [12,](#page-8-2) the model captures the observed features of δf(T, B)/f0(T, 0) for both orientations of B shown in Fig. [9,](#page-4-0) although one can hardly expect a perfect fit from such a crude model. For instance, the difference between the slopes of δf(T, B)/f0(T, 0) for two field orientations in Fig. [12](#page-8-2) is about 30% higher than in Fig. [9,](#page-4-0) which can occur because the GB orientations shown in Fig. [2](#page-1-0) are not completely random. Yet the GBs can dominate the fieldinduced frequency shift even if they only contribute less than 10−<sup>3</sup> to the total inductance of the strip.

![](0__page_8_Figure_14.jpeg)

**Caption:** Figure 12 shows the calculated normalized shift in δf₀(T, B)/f₀(T, 0) of a Nb3Sn coplanar resonator under an in-plane magnetic field. Data is presented for (a) a magnetic field parallel to the strip (Bkz) and (b) perpendicular (Bky), calculated using mean-field models involving grain boundary orientations at low fields, B₀ = 10 mT. Across various temperatures, the model effectively captures the field-induced effects on resonance frequency shifts observed experimentally. This suggests significant contributions of grain boundaries to the nonlinear Meissner effect (NLME) seen in these polycrystalline superconducting films, providing insights into the field-dependent inductance behavior.


<span id="page-8-2"></span>FIG. 12. Normalized shift in δf0(T, B)/f0(T, 0) as a function of field calculated from Eqs. [\(44\)](#page-8-3), [\(B12\)](#page-10-1) and [\(B13\)](#page-10-2) at B<sup>0</sup> = 10 mT, L J <sup>k</sup> (0, 0)/L = 3.5 × 10<sup>−</sup><sup>4</sup> and different temperatures corresponding to those in Fig. 9 for: (a) Bkz and (b) Bky.

Grain boundaries can also contribute to the fielddependent resistance R<sup>J</sup> (B). Indeed, the coplanar resonator impedance per unit length, Zg(B) = R<sup>J</sup> + iL<sup>J</sup> k estimated from Eq. [\(31\)](#page-6-4) is given by:

<span id="page-9-1"></span>
$$Z\_g \sim \frac{i\omega \bar{L}\_{\square} \bar{R}\_{\square}}{wdl\_2 (i\omega \bar{L}\_{\square} + \bar{R}\_{\square})},\tag{45}$$

where L¯ = φ0/2πJchcos θi, and R¯ is the GB quasiparticle resistance per unit area averaged over the GB orientations. The field dependencies of R<sup>J</sup> (B) and L J k (B) can be strongly affected by ω and T. At high frequencies, ω ω<sup>c</sup> = R¯/L¯ the impedance Z<sup>g</sup> ∼ R¯/wd becomes independent of B, that is, GBs do not contribute to the NLME. For S-I-S GBs, the crossover frequency ω<sup>c</sup> is,

$$
\omega\_c \sim \langle \cos \theta \rangle \frac{\Delta}{\hbar} \tanh \left( \frac{\Delta}{2k\_B T} \right) \tag{46}
$$

Here ω<sup>c</sup> ∼ ∆/~ if T is not very close to T<sup>c</sup> and B . B0, but ωc(T, B) decreases as B increases beyond B<sup>0</sup> and T approaches Tc. For proximity-coupled GBs, ω<sup>c</sup> can be smaller than ∆/~ even at T T<sup>c</sup> and B < B<sup>0</sup> since JcR for S-N-S JJs can be much smaller than ∆/e [47](#page-11-4) . For Nb3Sn in which ∆/~ ∼ 1 THz is some 2 orders of magnitude higher than the frequency 2πf<sup>0</sup> of our resonator, it appears that the observed behavior of δf(T, B) at 7 < T < 12 K is consistent with the low-frequency limit ω ωc(T, B) in which Eq. [\(45\)](#page-9-1) gives:

$$R\_J(B) \sim \frac{\phi\_0^2 \omega^2}{4\pi^2 R\_\square J\_c^2 w d l\_2 \langle \cos \theta \rangle^2} \tag{47}$$

For hL<sup>J</sup> (B)i calculated above (see Fig. [11\)](#page-7-4), R<sup>J</sup> (B) ∝ B<sup>2</sup> in Eq. [\(47\)](#page-9-2) increases quadratically with B at B & B0. This field dependence is the same as for the Meissner contribution to σω(B) in Eq. [\(22\)](#page-5-2), but the orientational dependence of R<sup>J</sup> (B) is opposite to that of σω(B). Thus, GBs can give rise to a strong nonlinearity of the electromagnetic response of polycrystalline films, which can be essential for Nb3Sn thin film coatings of high-Q resonator cavities in particle accelerators [28](#page-11-3)[,42,](#page-11-13)[43,](#page-11-1)[53](#page-11-14) .

In conclusion, grain boundaries and local nonstoichiometry on nanometer scales can significantly contribute to the NLME in polycrystalline Nb3Sn. Particularly, GBs can cause the linear field-dependence of the magnetic penetration depth expected from a clean dwave superconductor at low temperatures. By contrast, for elemental superconductors such as Nb [22](#page-10-3) and Al [24](#page-10-4) with large coherence lengths, δλ(T, B) ∝ B<sup>2</sup> is described well by the Meissner pairbreaking. However, extended crystalline defects in superconductors with short coherence lengths can radically change the field-dependence of δλ(T, B), even if their contribution to the kinetic inductance at zero field is small. This feature can impose stringent requirements for the quality of single crystals used for the observation of manifestations of d-wave pairing in λ(T, B), particularly in cuprates and pnictides which are prone to the weak-link behavior of grain boundaries.

## IV. ACKNOWLEDGMENTS

This work was supported by DOE under Grant DE-SC 100387-020.

# Appendix A: Geometric inductance of the coplanar waveguide

The geometric inductance per unit length L<sup>g</sup> of the coplanar resonator is calculated by conformal mapping of the cross section of the coplanar resonator into parallel plates [34](#page-11-15). For a thin film with d w, this yields

<span id="page-9-3"></span>
$$L\_g = \frac{\mu\_0 K(k')}{4K(k)},\tag{A1}$$

where k = s/(2w + s), k <sup>0</sup> = √ 1 − k <sup>2</sup>, and K(k) is a complete elliptic integral of the first kind. For w = 15 µm and s = 8.8 µm, Eq. [\(A1\)](#page-9-3) gives L<sup>g</sup> = 420.5 H/m.

## <span id="page-9-4"></span><span id="page-9-0"></span>Appendix B: tilted GB

<span id="page-9-2"></span>For a rectangular JJ of width l<sup>1</sup> and length l<sup>2</sup> shown in Fig. [10,](#page-7-0) the solution of Eq. [\(30\)](#page-6-6) is

$$\theta = \frac{2\pi B\Lambda}{\phi\_0} \left( n\_x y - n\_y x \right), \qquad \mathbf{B} \| z \tag{\text{B1}}$$

<span id="page-9-5"></span>
$$\theta = \frac{2\pi B\Lambda}{\phi\_0} \left( n\_z x - n\_x z \right), \qquad \mathbf{B} \| y \tag{\text{B2}}$$

We rotate the coordinate system by the Euler angles α and β about the x and the y axes to a new cartesian system (X, Y, Z) in which Z is perpendicular to the GB plane and X and Y are directed along the GB sides of lengths l<sup>1</sup> and l2, respectively. Here X<sup>i</sup> = Rijx<sup>j</sup> , where the rotation matrix Rij = R y ikR<sup>x</sup> kj is given by:

$$R\_{ij} = \begin{pmatrix} \cos \beta & 0 & \sin \beta \\ 0 & 1 & 0 \\ -\sin \beta & 0 & \cos \beta \end{pmatrix} \cdot \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos \alpha & -\sin \alpha \\ 0 & \sin \alpha & \cos \alpha \end{pmatrix} . \tag{B3}$$

The transpose matrix R<sup>T</sup> ij is then:

$$R\_{ij}^T = \begin{pmatrix} \cos \beta & 0 & -\sin \beta \\ \sin \alpha \sin \beta & \cos \alpha & \sin \alpha \cos \beta \\ \cos \alpha \sin \beta & -\sin \alpha & \cos \alpha \cos \beta \end{pmatrix}. \tag{B4}$$

Using x<sup>i</sup> = R<sup>T</sup> ijX<sup>j</sup> we obtain:

<span id="page-9-7"></span><span id="page-9-6"></span>
$$x = X\cos\beta - Z\sin\beta,\tag{B5}$$

$$y = X \sin \beta \sin \alpha + Y \cos \alpha + Z \cos \beta \sin \alpha,\qquad \text{(B6)}$$

$$z = X \sin \beta \cos \alpha - Y \sin \alpha + Z \cos \beta \cos \alpha,\qquad \text{(B7)}$$

n<sup>x</sup> = − sin β, n<sup>y</sup> = cos β sin α, n<sup>z</sup> = cos β cos α (B8)

Equations [\(B1\)](#page-9-4)-[\(B2\)](#page-9-5) and [\(B5\)](#page-9-6)-[\(B7\)](#page-9-7) with Z = 0 give:

$$\theta = -\frac{2\pi B\Lambda}{\phi\_0} \left[ X \sin \alpha + Y \sin \beta \cos \alpha \right], \quad \mathbf{B} || z \qquad \text{(B9)}$$

$$\theta = \frac{2\pi B\Lambda}{\phi\_0} \left[ X \cos \alpha - Y \sin \beta \sin \alpha \right], \quad \mathbf{B} \| \mathbf{y}. \tag{\text{B10}}$$

Next, we calculate ¯c = S −1 R cos θ(X, Y )dS:

$$\bar{c} = \frac{1}{l\_1 l\_2} \text{Re} \int\_{-l\_1/2}^{l\_1/2} dX \int\_{-l\_2/2}^{l\_2/2} dY e^{i\theta(Y, Z)},\qquad \text{(B11)}$$

which leads to Eqs. [\(32\)](#page-6-3)-[\(34\)](#page-7-6).

For uniform distributions of the GB orientation angles, Eqs. [\(35\)](#page-7-1) and [\(38\)](#page-7-3) yield at Bkz:

$$\langle \cos \theta \rangle = $$

$$\frac{1}{\pi^2 b} \int\_0^\pi d\alpha \int\_0^\pi d\beta \sin(b \cos \alpha \sin \beta) \frac{\cos \alpha \cos \beta}{\tan \beta}. \quad \text{(B12)}$$

- <sup>1</sup> V.L. Ginzburg and L.D. Landau, On the theory of superconductivity, Zh. Exp. Teor. Fiz. 20, 1064 (1950).
- <sup>2</sup> S.K. Yip and J.A. Sauls, Nonlinear Meissner effect in CuO superconductors, Phys. Rev. Lett. 69, 2264 (1992).
- <sup>3</sup> D. Xu, S.K. Yip, and J.A. Sauls, Nonlinear Meissner effect in unconventional superconductors, Phys. Rev. B 51, 16233 (1995).
- <sup>4</sup> T. Dahm and D.J. Scalapino, Theory of intermodulation in a superconducting microstrip resonator, J. Appl. Phys. 81, 2002 (1997).
- <sup>5</sup> T. Dahm and D.J. Scalapino, Nonlinear current response of a d-wave superfluid, Phys. Rev. B 60, 13125 (1999).
- <sup>6</sup> M.-R. Li, P.J. Hirschfeld, and P. W¨olfle, Is the nonlinear Meissner effect unobservable? Phys. Rev. Lett. 81, 5640 (1998).
- <sup>7</sup> M.-R. Li, P.J. Hirschfeld, and P. W¨olfle, Free energy and magnetic penetration of a d-wave superconductor in the Meissner state, Phys. Rev. B 61, 648 (2000).
- <sup>8</sup> R. Prozorov and R.W. Giannetta, Magnetic penetration depth in unconventional superconductors, Supercond. Sci. Technol. 19, R41 (2006).
- 9 J. Bardeen, Critical fields and currents in superconductors, Rev. Mod. Phys. 34, 667 (1962).
- <sup>10</sup> K. Maki, On persistent currents in a superconducting alloy. II, Prog. Theor. Phys. 29, 333 (1963).
- <sup>11</sup> A. Gurevich and V.M. Vinokur, Phase textures induced by dc current pairbreaking in multilayer structures and twogap superconductors, Phys. Rev. Lett. 97, 137003 (2006).
- <sup>12</sup> P. Hirschfeld, M. Korshunov, and I. Mazin, Gap symmetry and structure of Fe-based superconductors, Rep. Prog. Phys. 74, 124508 (2011).
- <sup>13</sup> C. P. Bidinosti, W. N. Hardy, D. A. Bonn, and R. Liang, Magnetic field dependence of λ in YBa2Cu3O6.95: Results as a function of temperature and field orientation, Phys. Rev. Lett. 83, 3277 (1999).
- <sup>14</sup> A. Bhattacharya, I. Zutic, O. T. Valls, A. M. Goldman, U. Welp, and B. Veal, Angular dependence of the nonlinear transverse magnetic moment of YBa2Cu3O6.<sup>95</sup> in the

Likewise, we get from Eqs. [\(36\)](#page-7-2) and [\(38\)](#page-7-3) at Bky:

$$\langle \cos \theta \rangle = $$

$$\frac{1}{\pi^2 b} \int\_0^\pi d\alpha \int\_0^\pi d\beta \sin(b \sin \alpha \sin \beta) \frac{\cos \alpha \cos \beta}{\tan \alpha \tan \beta}. \quad (\text{B13})$$

At b = B/B<sup>0</sup> 1 we obtain from Eqs. [\(B12\)](#page-10-1)-[\(B13\)](#page-10-2):

<span id="page-10-2"></span>
$$
\langle \cos \theta \rangle = \frac{1}{4} \left( 1 - \frac{b^2}{32} \right), \quad \mathbf{B} \| z, \tag{\text{B14}}
$$

$$
\langle \cos \theta \rangle = \frac{1}{4} \left( 1 - \frac{b^2}{96} \right), \quad \mathbf{B} \| y. \tag{\text{B15}}
$$

<span id="page-10-1"></span>The field Bkz applied along the strip causes stronger reduction of hcos θi than the transverse field Bky.

Meissner state, Phys. Rev. Lett. 82, 3132 (1999).

- <sup>15</sup> A. Carrington, R.W. Giannetta, J.T. Kim, and J. Giapintzakis, Absence of nonlinear Meissner effect in YBa2Cu3O6.95, Phys. Rev. B 59, R14173 (1999).
- <sup>16</sup> K. Halterman, O.T. Valls, and I. Zuti´c, Reanalysis of the magnetic field dependence of the penetration depth: Observation of the nonlinear Meissner effect, Phys. Rev. B 63, 180405R (2001).
- <sup>17</sup> D.E. Oates, S.-H. Park, and G. Koren, Observation of the nonlinear Meissner effect in YBCO thin films: Evidence for a d-wave order parameter in the bulk of the cuprate superconductors, Phys. Rev. Lett. 93, 197001 (2004).
- <sup>18</sup> D.E. Oates, Overview of nonlinearity in HTS: What we have learned and prospects for improvement, J. Supercond. Novel Magn. 20, 3 (2007).
- <sup>19</sup> A. Carrington, Studies of the gap structure of ironbased superconductors using magnetic penetration depth, Comptes Rendus Physique 12, 502 (2011).
- <sup>20</sup> E.H. Brandt, The flux line lattice in superconductors, Rep. Prog. Phys. 58, 1456 (1995).
- <sup>21</sup> T.P. Orlando and K.A. Delin, Foundations of Applied Superconductivity, (Prentice Hall, 1991).
- <span id="page-10-3"></span><sup>22</sup> N. Groll, A. Gurevich, and I. Chiorescu, Measurements of the nonlinear Meissner effect in superconducting Nb films using resonant microwave cavity: A probe of unconventional order parameter, Phys. Rev. B81, 020504(R) (2010).
- <sup>23</sup> A.A. Abrikosov, On the lower critical field of thin layers of superconductors of the second group, Zh. Exp. Teor. Fiz. 46, 1464 (1964); [Sov. Phys. JETP 19, 988 (1964)].
- <span id="page-10-4"></span><sup>24</sup> K. Borisov, D. Rieger, P. Winkel, F. Henriques, F. Valenti, A. Ionita, M. Wessbecher, M. Spiecker, D. Gusenkova, I. M. Pop, and W. Wernsdorfer, Superconducting granular aluminum resonators resilient to magnetic fields up to 1 Tesla, Appl. Phys. Lett. 117, 120502 (2020).
- <span id="page-10-0"></span><sup>25</sup> J.H. Durrell, C.B. Eom, A. Gurevich, E.E. Hellstrom, C. Tarantini, A. Yamamoto, and D.C. Larbalestier, The behavior of grain boundaries in the Fe-based superconductors, Rep. Prog. Phys. 74, 124511 (2011).
- <sup>26</sup> T.L. Hylton and M.R. Beasley, Effect of grain boundaries on magnetic field penetration in polycrystalline superconductors, Phys. Rev. B 39, 9042 (1989).
- <span id="page-11-2"></span><sup>27</sup> C. Sundahl, Synthesis of superconducting Nb3Sn thin film heterostructures for the study of high-energy rf physics, Ph.D Thesis, University of Wisconsin-Madison, 2019.
- <span id="page-11-3"></span><sup>28</sup> C. Sundahl, J. Makita, P.B. Welander, Y-F. Su, F. Kametani, L. Xie, H. Zhang, L. Li, A. Gurevich, and C.B. Eom, Development and characterization of Nb3Sn/Al2O<sup>3</sup> superconducting multilayers for particle accelerators, Sci. Rep. 11, 7770 (2021).
- <sup>29</sup> SCM-50-CF Closed Cycle Cryomagnetic Probe Station equipped with Sumitomo RDK-415D2 Cold Head and Sumitomo Compressor Model F-70L by MicroXact, Inc.
- <sup>30</sup> P.J. Petersan and S.M. Anlage, Measurement of resonant frequency and quality factor of microwave resonators: Comparison of methods, J. Appl. Phys. 84, 3392 (1998).
- <sup>31</sup> S.M. Anlage, B.W. Langley, H.J. Snortland, C.B. Eom, T.H. Geballe, and M.R. Beasley, Magnetic penetration depth measurements with the microstrip resonator technique, J. Supercond. 3, 331 (1990).
- <sup>32</sup> A. Porch, P. Mauskopf, S. Doyle, and C. Dunscombe, Calculation of the characteristics of coplanar resonators for kinetic inductance detectors, IEEE Trans. Appl. Supercond. 15, 552 (2005).
- <sup>33</sup> D. Hafner, M. Dressel, and M. Scheffler, Surface-resistance measurements using superconducting stripline resonators, Rev. Sci. Instrum. 85, 014702 (2014).
- <span id="page-11-15"></span><sup>34</sup> R.E. Collin, Foundations for Microwave Engineering, (IEEE Press, New York, 2000).
- <sup>35</sup> M. Hein, High-Temperature-Superconductor Thin Films at Microwave Frequencies, Springer-Verlag Berlin Heidelberg, (1999).
- <sup>36</sup> H.R. Mohebbi, O.W.B. Benningshof, I.A.J. Taminiau, and G.X. Miao, Composite arrays of superconducting microstrip line resonators, J. Appl. Phys. 115, 094502 (2014).
- <sup>37</sup> B. Abdo, E. Segev, O. Shtempluck, and E. Buks, Nonlinear dynamics in the resonance line shape of NbN superconducting resonators, Phys. Rev. B 73, 13 (2006).
- <sup>38</sup> T.P. Orlando, W.J. McNiff, S. Foner, and M.R. Beasley, Critical fields, Pauli paramagnetic limiting, and material parameters of Nb3Sn and V3Si, Phys. Rev. B 19, 4545 (1979).
- <span id="page-11-11"></span><sup>39</sup> A. Godeke, A review of the properties of Nb3Sn and their variation with A15 composition, morphology and strain state, Supercond. Sci. Technol. 19, R68 (2006).
- <span id="page-11-0"></span><sup>40</sup> M. Suenaga and W. Jansen, Chemical compositions at and near the grain boundaries in bronze-processed superconducting Nb3Sn, Appl. Phys. Lett. 43, 791 (1983).
- <sup>41</sup> M.J.R. Sandim, D. Tytko, A. Kostka, P. Choi, S. Awaji, K. Watanabe, and D. Raabe, Grain boundary segregation in a bronze-route Nb3Sn superconducting wire studied by atom probe tomography, Supercond. Sci. Technol. 26, 055008 (2013).
- <span id="page-11-13"></span><sup>42</sup> J. Lee, S. Posen, Z. Mao, Y. Trenikhina, K. He, D.L Hall, M. Liepe, and D.N, Seidman, Atomic-scale analyses of Nb3Sn on Nb prepared by vapor diffusion for superconducting radiofrequency cavity applications: a correlative study, Supercond. Sci. Technol. 32, 024001 (2019).
- <span id="page-11-1"></span><sup>43</sup> J. Lee, Z. Mao, K. He, Z.H. Sung T. Spina S-I. Baik, D.L.Hall, M. Liepe, D.N. Seidman, and S. Posen, Grainboundary structure and segregation in Nb3Sn coatings on Nb for high-performance superconducting radiofrequency cavity applications, Acta Mater. 188, 155 (2020).
- <span id="page-11-12"></span><sup>44</sup> R.M. Scanlan and W.A. Fietz, Flux pinning centers in superconducting Nb3Sn, J. Appl. Phys. 46, 2244 (1975).
- <sup>45</sup> L. Kramer and R.J. Watts-Tobin, Theory of dissipative current-carrying states in superconducting filaments, Phys. Rev. Lett. 40, 1041(1978).
- <span id="page-11-10"></span><sup>46</sup> N. Kopnin, Theory of Nonequilibrium Superconductivity.(Oxford University Press, New York, 2001).
- <span id="page-11-4"></span><sup>47</sup> A. Barone and G. Paterno, Physics and Applications of Josephson Effect (Wiley, New York, 1982).
- <span id="page-11-5"></span><sup>48</sup> R. Rammal, T.C. Lubensky, and G. Toulouse, Superconducting networks in a magnetic field, Phys. Rev. B 27, 2820 (1983).
- <span id="page-11-7"></span><sup>49</sup> W.Y. Shih, C. Ebner, and D. Stroud, Frustration and disorder in granular superconductors, Phys. Rev. B 31, 134 (1984).
- <span id="page-11-6"></span><sup>50</sup> C. Ebner and D. Stroud, Diamagnetic susceptibility of superconducting clusters: Spin glass behavior, Phys. Rev. B 30, 165 (1985).
- <span id="page-11-8"></span><sup>51</sup> P.A. Rosenthal, M.R Beasley, K. Char, M.S. Colclough, and G. Zaharchuk, Flux focusing effects in planar thinfilm grain-boundary Josephson junctions, Appl. Phys. Lett. 59, 3482 (1991).
- <span id="page-11-9"></span><sup>52</sup> J.R. Clem, Josephson junctions in thin and narrow rectangular superconducting strips, Phys. Rev. B 81, 144515 (2010).
- <span id="page-11-14"></span><sup>53</sup> A. Gurevich, Theory of RF superconductivity for resonant cavities, Supercond. Sci. Technol. 30, 034004 (2017).