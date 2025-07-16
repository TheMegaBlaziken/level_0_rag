# arXiv:2203.08132

**Paper ID:** 562797f265db8ca9129e1af234a897ba

**PDF Path:** apl/Superconductors/arXiv:2203.08132.pdf

**Processing Status:** complete

**Captions Added:** 10

**Generated:** 2025-06-24T13:44:27.596229

---

# **Hybrid conformal REBCO dipole for a next hadron collider**

#### **Peter M McIntyre\*<sup>1</sup>**

Texas A&M University and Accelerator Technology Corp. p-mcintyre@tamu.edu

**Abstract**. **A strategy is presented by which a tape-stack REBCO cable may be configured in a** *conformal winding* **in such a way that the favorable** ∥ **orientation is sustained everywhere in the winding. A method for** *dynamic current sharing* **is discussed, in which current would naturally re-distribute among the tapes of each cable turn as the winding current is increased without inducing premature quench. A dipole geometry is presented in which such a conformal REBCO insert winding is configured within a** *cable-in-conduit Nb3Sn outsert winding* **so that the sub-windings can be separately fabricated, and then assembled in a common structure and preloaded for effective stress management. The total cost of superconductor in this dipole is likely a minimum among the design options for 18 T. Plans for development of the coil technologies and a model dipole are presented.**

### **Introduction**

The total cost/TeV for the superconducting magnets and the tunnel must be reduced by a factor ~five if any future hadron collider proposal is to be feasible.The FCC-hh proposal [1] highlighted the challenges of magnet technology that dominate the cost of a ~100 TeV, 100 km circumference collider. It has inspired many design efforts to push superconducting dipole technology to 16-20 T [2]. Those challenges have inspired many efforts worldwide to improve the cost/TeV of superconducting dipoles.

The recent review by van Nugteren *et al*. [3] examines the several challenges in developing hybrid dipoles that incorporate an inner winding of REBCO and an outer winding of Nb3Sn to produce ~20 T bore field. Among the challenges is that REBCO is ruinously expensive ~ 100/m for a single thin tape. A related challenge is that REBCO is a highly anisotropic superconductor [4] with a critical current that is strongly dependent on the magnitude and direction of the magnetic field in which it is operating (Figure 1). The critical current is approximately 3x greater when the magnetic field at the tape is parallel to the tape face (∥, q=90°), corresponding to ~2000 A/mm2 @ 20 K, than when it is perpendicular to the tape face (\$, q=0°), corresponding to ~700 A/mm2 @ 4.2 K.

In recent papers [5] we presented a conformal winding method in which a REBCO insert winding can be configured in a hybrid dipole so that the tapes within the winding are everywhere oriented parallel to the field at the tape, so that the winding can use the full capacity of the REBCO tapes. Figure 2 shows an example field design for such a dipole. The insert winding consists of a tape-stack cable in which 25 REBCO tapes are stacked face-to-face, and each turn of the cable is oriented closely parallel to the field at that location.

Clustering REBCO tapes has been used by several authors, including Roebel cable [6], CORC [7], cable-in-conduit [8], and stacked tapes [9]. In most cases REBCO tapes are stacked in a face-on cluster and multiple clusters are cabled with a twist pitch so that each tape spends equal length on the inside and outside of the cable in a winding. Yagotintsev *et al*. [10] review the measurements of AC loss and contact resistance in those several forms of clustered-tape

<sup>1</sup> The ARL/ATC collaboration includes Cannon Coates, Tim Elliott, Ray Garrison, Gareth May, John Rogers, Christian Ratcliffe, and Daya Rathnayaka.

![](_page_1_Figure_0.jpeg)

**Caption:** This figure presents two key visualizations for an 18 Tesla dipole containing a conformal REBCO insert winding. On the left, a graph displays the dependence of the critical current (Ic) on the angle between the tape face normal and the magnetic field for various field strengths at a temperature of 30 K. The plot highlights that the critical current is maximized when the magnetic field is parallel to the tape face (θ = 90°), decreasing substantially when perpendicular (θ = 0°). Notably, at a parallel orientation and 30 K, Ic approaches 2000 A/mm². On the right, the field distribution is shown in one quadrant of the 18 T dipole. The visualization reveals how the conformal REBCO insert winding is oriented to optimize field interaction and minimize quenching risks. A sextupole correction winding addresses magnetic field homogeneity and sextupole harmonics, critical for the dipole's operation in particle accelerators. Together, these visualizations support the design's strategy to maximize performance by aligning REBCO tapes with the magnetic field, crucial for achieving high-field strength with stability.


*Figure 1. Dependence of Ic on the angle between the tape face normal and the magnetic field, for various field strengths (30 K temperature) (from Ref. 4).*

conductors. But all of the present approaches to tape clustering involve twisting the cluster and so forego any possibility to sustain the ∥ condition that would make it possible to use REBCO tape to its full potential. That is the goal of the conformal winding method.

There are two challenges in making the conformal winding realizable:

- Can uniformly high critical current be sustained in the end windings of the dipole insert (where the magnetic field and the winding must both be flared) and in the leads, just as it is in the body?
- Can the tape-cluster cable support dynamic current-sharing within each cable turn so that inductive forces within the non-insulated cable do not concentrate its current in outer tapes and drive premature quench?

# **Optimization of conformal winding**

The hybrid dipole shown in Figure 1 contains an insert winding made from rectangular REBCO tape-stack cable and an outsert winding made from Nb3Sn Cable-in-Conduit 'Super-CIC' [11]. Table 1 summarizes the main parameters of both windings. The tape-stack cable contains 25 6-mm-wide REBCO tapes with 100 µm Cu clad to one side. The SuperCIC cable contains 17 0.85 mm-diameter wires of Hi-Lumi-class 108/127 Nb3Sn/Cu wires, spiralwrapped as a single layer around a thin-wall perforated center tube, then pulled as loose fit through a bronze sheath tube and drawn to compress the wires against the center tube to immobilize them.

The windings are operated in series, and the number of wires in each winding are chosen so that a cable current of 17.5 kA produces ~18 T bore field and corresponds approximately to critical current in each winding. The REBCO insert can operate at up to 20K, while the Nb3Sn winding operates at ~5K.

# **Conformal geometry at injection field**

The field distribution in the winding region is significantly different at injection field and at collision field, due to progressive saturation in the flux return. Figure 3 shows the field distribution at injection field (1.75 kA winding current). There is a significant angle q between the tape face and the field direction in several blocks, but the value of || is low enough that the critical current '(||, ) is sufficient that there is no risk of quenching providing that currentsharing is dynamically stable (the subject of the next section).

*Table 1. Main parameters of the 18 T hybrid dipole.*

| Bore field @ 4.2K short-sample                | 18.3       | T     |
|-----------------------------------------------|------------|-------|
| Cable current (windings in series)            | 17.5       | kA    |
| Aperture: horizontal, vertical                | 40, 30     | mm    |
| REBCO: #tapes/cable x #cables/shell x #shells | 25x 16 x 3 |       |
| Nb3Sn CIC: #wires/CIC x #cables/layer         | 17x 16 x 8 |       |
| x#layers                                      |            |       |
| Bmax in REBCO, Nb3Sn CIC                      | 20.0, 11.8 | T     |
| Sextupole @ full field, injection field       | -5, -21    | units |
|                                               |            |       |

![](_page_2_Figure_2.jpeg)

**Caption:** Field distribution at the injection field for a dipole designed with a conformal REBCO winding is shown. The plot illustrates how the rectangular REBCO tape-stack cables conform to the magnetic field's contour within the insert winding region. Cylindrical CIC cables in the outsert winding form a block-coil array, and a steel flux plate above the bore tube is employed to suppress multipoles at the injection field, ensuring stable dynamic current share.


#### **Body field simulations**

*Figure 3. Field distribution at injection field.*

The rectangular REBCO tape-stack cables are oriented along a curving contour to conform with the flaring behavior of the magnetic field in the region of the insert winding. The cylindrical CIC cables are oriented in a rectangular block-coil array in the outsert winding. There is also a 'sextupole correction' sub-winding of REBCO cables located inside and above the main REBCO insert, which is operated in series to correct sextupole in the bore field. Immediately above the bore tube is a steel flux plate that suppresses multipoles at injection field to suppress the effects of persistent currents and snap-back at injection field. The REBCO insert contributes ~8 and the CIC winding ~10 to the bore field.

Figure 4 shows a detailed cross-section of three particular 3-turn blocks within the REBCO insert winding: the top-right block, the mid-plane block, and the sextupole block. These blocks exhibit variously the highest field or the maximum deviation from the ∥ orientation for some of the constituent tapes. The current capacity of the *n*th tape in each cluster can be estimated by extracting the local sheet critical current density *Kn(u)* as a function of the location *u* across the width of that tape*,* using the local values for |(, )| and q, and adding the contributions for segments spanning the entire width *<sup>w</sup>* of that tape: '4 <sup>=</sup> <sup>∫</sup> 4() : ; . The total cable current capacity is then obtained by adding the maximum current capacity of the 25 tapes in that cable segment. The maximum cable current for each cable is shown in the figure for each of the inner and outer cable turns.

#### **End Field Simulations**

An additional challenge is to design a winding strategy for the magnet ends that does not compromise the operational current capacity. In the 3D end regions of the dipole, fields flare and bend in a disorderly fashion raising concerns about the critical current of the cables. The flared ends of the tape-stack winding are formed using the method first pioneered by Willy Sampson (BNL) in the 1970s. Figure 5 shows a flared-end quadrupole containing a stackedtape winding of dip-process Nb3Sn tape. Each tape-stack cable is twisted about its axis as it is flared vertically for form a catenary in which all tapes remain as a stack but no tape is bent in the hard direction. With this approach each tape face naturally follows closely the flaring of the magnetic field in the end.

This twisted-flare end has been modeled in a 3-D CAD design of the flared end, and the fields have been modeled using 3-D COMSOL. Figure 4 shows the results in three example cross sections through the end region: a) a y/z cross section through the vertical midplane; b) an x/y cross section at the transition from body to flared end; and c) an x/y cross section at a location 3 cm into the body from that transition. We are estimating '(||, ) for each tape

![](_page_3_Figure_0.jpeg)

**Caption:** A cross-sectional simulation illustrating the orientation of magnetic fields and tape faces in the REBCO insert winding. The analysis includes simulations for top right, mid-plane, and sextupole 3-turn blocks, emphasizing different field directions and tape alignments. This representation aids in estimating the critical current Ic at various tape-stack locations, ensuring no 'weak sister' limits affect the tapes through the flared ends .


*Figure 4. Detailed simulation of the orientation of magnetic field and tape faces for the a) top right; b) mid-plane; and c) sextupole 3-turn blocks of tape-stack cables in the REBCO insert winding.*

within each layer of each cross section using the data of Figure 1, and adding them to obtain the *Ic* in each tape-stack cable at that location.

There is a compensating interplay that sustains *Ic* with little or no degradation through the flared ends: the inner tapes are well-aligned to with the field direction but have the strongest value, while the outer tapes flare with significant angle with respect to the field but the field strength is low enough that the angular dependence of Ic is also broadened. *The flared-end regions do not present a 'weak sister' limit to the Ic of the cable in any turn of the winding.*

## **Dynamic current-sharing in the conformal winding**

As the dipole is ramped to increase the bore field from its value Bi at injection energy to Bc at collision energy, the current within each tape-cluster cable is increased proportionately. As shown in Figure 6, the magnetic field at each cable produces a Lorentz force ⃗ that pushes current flowing within each tape-cluster cable away from the dipole bore. Thus, even if current were injected so that it was initially distributed equally among all tapes within a first turn of cable, the Lorentz force would re-distribute current to the outermost tapes within the winding. It would therefore seem that as coil current were increased the outermost tape would quench when the overall cable current was still only a fraction of the desired current!

But REBCO can operate at 30 K, where the heat capacity of the tape (~T3 ), and the conduction to remove heat ~(Thot-T) are both much greater than at liquid helium temperature, so we conjecture that it may be possible to operate a cable of stacked non-insulated tapes *without transposition* and rely upon the 'soft' approach to quench in each tape layer to force re-distribution of current to neighboring tapes within the cable as the cable current is further increased. This strategy has been used to good effect in 'no-insulator' (*NI*) pancake windings for highfield solenoids [12], but never in a dipole. We now analyze the dynamics of quasi-equilibrium current-sharing among the tapes within a non-transposed cable in the dipole of Figure 2, using the modelling approach of Noguchi [13].

The REBCO layer within each exhibits a retarding electric field *Ez* that is current dependent:

$$E\_z = E\_0 (I/I\_c)^n \tag{1}$$

where *E0* = 10-6 *V/m* is the quench criterion, '(||, , ) is the critical current for the conditions (*B,*q,T) for that tape, and *n*~30 is the index that characterizes the power-law dependence of the superconductornormal transition for REBCO.

![](_page_3_Picture_10.jpeg)

**Caption:** The photograph displays a flared-end quadrupole using Sampson's design technique, with a tape-stack cable composed of Nb3Sn superconducting tape. The twisted-flare end method ensures each tape aligns with magnetic field flaring ends without significant bending in the hard direction. Such design helps maintain strong magnetic field and current capacity without degradation through flared ends .


*Figure 5. Example of flared ends in a tape-stack cable: Sampson's flared-end quadrupole using dip-coated Nb3Sn superconducting tape.*

![](_page_4_Figure_0.jpeg)

**Caption:** The illustration depicts current-sharing dynamics in a conformal winding of a tape-stack REBCO cable. Arrows indicate the Lorentz force acting on the superconducting transport system, akin to the Hall effect. The potential difference created when neighboring tapes carry different currents, along with contact resistivity Rc between adjacent tapes, leads to a displacement of current by Lorentz forces. This model employs a series-parallel L/R network to estimate the dynamic current distribution in a tape-stack cable, considering induced inductance per unit length and power-law series resistance per unit length .


*Figure 6. Schematic illustration of current-sharing dynamics in a conformal winding of tape-stack REBCO cable.*

![](_page_4_Figure_2.jpeg)

**Caption:** A detailed view of a block of 3 tape-stack tapes within a conformal REBCO winding, each comprising 25 tapes. A laminar spring enforces uniform ~1 MPa compression across all tapes, supporting effective current-sharing and suppressing quench risks. It draws parallels with the Hall effect dynamic, where transverse Lorentz forces and contact resistivity Rc facilitate current redistribution .


*Figure 7. Detail showing a block of 3 tape-stack tapes, each containing 25 tapes, with a laminar spring to provide uniform ~1 MPa compression to all cables.*

The dynamics is analogous to the Hall effect, in which the superconducting transport is acted upon by the transverse Lorentz force, by a transverse electric field produced by the potential difference between neighboring tapes when they carry different currents, and by a contact resistivity *Rc* between adjacent tapes through which current is displaced by Lorentz forces.

Following Ref. 13, the time dependent distribution of current in a tape-stack cable can be estimated in a simple model in which the full length of one half-turn of the tape-stack cable is treated as a series-parallel L/R network. Each tape within a half-turn of one cable has a self-

$$\begin{aligned} \text{induced per unit length} \qquad & \tilde{\mathcal{L}} = \frac{\mu\_0 x}{\omega g} = 4 \times 10^{-4} H/m \qquad (2) \\ \text{and power-law series resistance/length} \qquad & \tilde{R}\_s = \frac{E}{I} = \frac{E\_0}{I\_0} \left(\frac{I}{I\_0}\right)^{\mathbf{n}-1} = \{0.7 n\Omega/m\} \left(\frac{I}{I\_0}\right)^{\mathbf{n}-1} \qquad (3) \\ \text{where } \omega = \omega\_0 \qquad \dots \qquad \omega\_0 = \omega\_0 \qquad \dots \qquad \omega\_0 = \omega\_0 \qquad (4) \end{aligned}$$

where *w*=6 mm is the tape width, *g*=10 cm is the vertical gap in the steel flux return, and *x*~10 cm is the horizontal width of the tape loop.

Lu *et al*. [14] measured the dependence of the contact surface resistance *Rc* upon the compression among the tapes in the stack, shown in Figure 8. As shown in Figure 7, each turn of tape-cluster cable in the conformal winding is supported by a laminar spring that provides ~1 MPa uniform compression all tapes of the tape-cluster, corresponding to *Rc* ~35 µW-cm2. The parallel resistance Rp by which a tape segment shares current with its neighbors is obtained by dividing *Rc* by the segment area: B <sup>=</sup> CD ℒ: <sup>=</sup> (;.GHI∙K) <sup>ℒ</sup> (4)

From these quantities, we can extract two reults that characterize the scale of current-sharing. First, the scale length l over which this homogenization operates is the winding length for which *Rp ~ Rs*l:

$$
\lambda = \sqrt{\frac{R\_c}{\omega \tilde{R}\_s}} = 2 \,\mathrm{9m} \left(\frac{l}{l\_c}\right)^{-11.5} \tag{5}
$$

l is much longer than any reasonable winding length, so the current distribution would relax uniformly along the winding.

Second, we can estimate the time constant with which a difference in current between successive tapes in a tape-stack cable relaxes to an equilibrium governed by the Lorentz force and the 2-D distribution of resistance within a tape-stack cable. The change in *Figure 8. Contact resistance between* 

![](_page_4_Figure_13.jpeg)

**Caption:** This graphical analysis presents contact resistivity Rc among copper-clad REBCO tapes as a function of compression, crucial for understanding the dynamics of current sharing in tape-stack cables. Measures such as applying uniform compression through laminar springs are emphasized to ensure homogeneous current distribution and reliability under high-stress environments .


*two copper-clad REBCO tapes as a function of compression.*

inductance *dL* along one winding length ℒ between one tape and the next is

$$dL = \frac{\mu\_0}{\omega g} \mathcal{L}; \qquad dL[\mu H] = 0.4 \mathcal{L}[m] \tag{6}$$

So the time constant for relaxation between tapes is = bc Cd ; []~(1 second) ℒ[]l (7)

In a conformal winding for a collider dipole, the current re-distributes rapidly enough that no tape should reach *Ic* prematurely. From this simple model, we predict that, as coil current is increased from zero, current would accumulate in the outermost tape of each tape-stack cable until the coil current approaches a limit *I0* ~ 0.8 *NIc(B,T,*q) for that tape. Then as coil current is further increased, current shares to the neighboring tape until the coil current reached *2 Ic* in the two tapes. Then as coil current is further increased, current would share to the 3rd tape, *etc*., until finally current would become ~homogeneous throughout the cable as the current approached an ultimate limit of *~NI0*.

#### **Field homogeneity for collider requirements**

Field homogeneity is of particular concern for the dipole magnets of an accelerator or collider. The sextupole harmonic can be selectively canceled by placement of one correction turn in the winding, at the location shown in Figure 2. The particular example magnetic design shown has been optimized to produce nearly pure dipole field over a dynamic range of field 0.2-4 T, in which the amplitudes *bn* are all <10-4 over that range.

Current-sharing poses a further challenge for field homogeneity, however. At injection field, the current in each tape cluster is located mainly in the outermost tape; at collision field, the current is ~equally shared, so the 'current position' for that cable turn is shifted inwards by half the cluster width.

$$\text{Conventional the n}^{\text{th}} \text{ multiplier of a dipole field distribution is defined at each location}$$

$$\vec{r} = r(\ $\cos\theta + \$ \sin\theta) \text{ by the expansion } B\_x + iB\_y = \sum b\_n e^{-n\ell\theta} \left(\frac{r}{n}\right)^{n-1} \tag{8}$$

The multipoles have been evaluated for the magnetic design of Figure 2 for the maximum field ( limiting cases. The difference in the calculated multipoles is Dbn <0.5x10-4 for all multipoles! This remarkable result is a consequence of the conformal design strategy: because each tape in the cable is oriented so its face is closely parallel to |⃗, the field distribution is insensitive to the horizontal position of the 'current center position' of that cluster.

Prestemon *et al*. [15] modeled current transfer in a stacked-tape cable and found that a region of cable with coupling resistance ~10 Ω is effective in equalizing current among its tapes. The laminar spring sustains 1 MPa compression, so the characteristic length of cable for stability is

$$L \sim \frac{35\mu\Omega \cdot cm^2}{10n\Omega \cdot 6mm} = 60\text{m} \tag{9}$$

This is conveniently the approximate length for one turn of a collider dipole so ramping current should stabilize turn-by-turn.

![](_page_6_Picture_0.jpeg)

**Caption:** Depicts detailed design of an NbTi cable-in-conduit configuration within a 3 T large-aperture dipole for JLEIC. It includes an example of flared cable ends supporting the interfacing of Nb3Sn CIC structures. The design emphasizes stress management and the robustness of repeated bending without degradation, crucial attributes for high-performance dipole applications .


*Figure 9. a) Cross-section of cable-in-conduit; b) NbTi CIC winding for a 3 T large-aperture dipole for JLEIC, showing flared ends.*

# **Conductor cost for the 18 T dipole**

The importance of the REBCO-hybrid dipole technology comes in its impact on the cost of the superconductor required for an 18 T collider dipole. The total inventory of REBCO tape and Nb3Sn wire required for the design of Figure 2 and Table 1 is summarized in Table 2.

*Table 2. Superconductor cost/bore/m for the 18 T dipole. Prices are small-quantity prices actually paid by ATC in recent purchases.*

|            |                                             |              | \$224K/bore/m |
|------------|---------------------------------------------|--------------|---------------|
| Nb3Sn CIC: | 17 wires/CIC x 16 cables/layer x # 8 layers | 0.85 mm dia. | \$39K/bore/m  |
| REBCO:     | 25 tapes/cable x 16 cables/shell x 3 shells | 6 mm wide    | \$185K/bore/m |

For a collider with √ = 100 , the total dipole bore length required is 116 km, corresponding to a superconductor cost of \$26 billion. That is far more than has ever been invested in an facility for high-energy physics, but the prospect for cost reduction in huge quantity just might reduce it to feasible scale. In any case the superconductor cost for

Of course the superconductor is only one component of the cost of a dipole, but for all conceptual designs of dipoles to date it is the *dominant* component. Another important element of cost and performance is the integration of REBCO insert and Nb3Sn outsert into the dipole structure. The Nb3Sn outsert must be completed and heat-treated as a separate structure, and then assembled with the REBCO insert in a structure that provides for preload and stress management between the two sub-windings. The structure shown in Figure 2 provides a clean separation of the two sub-windings: the REBCO insert is wound onto a structure that embraces the beam tube; the Nb3Sn outsert is fabricated in two half-shells; and the half-shells are assembled onto the insert. We have successfully fabricated flared-end dipoles using cable-in-conduit (for the JLEIC dipole. Figure 9 shows a cross-section of Nb3Sn CIC, fabricated at Accelerator Technology Corp. (ATC) for a 3 T large-aperture dipole for JLEIC. Also shown is the completed winding, fabricated at the Accelerator Research Lab (ARL) at Texas A&M University. An important attribute of CIC is that it supports bending flared ends in a compact geometry, with no degradation of the internal support and registration of the wires, and it provides robust structure for stress-management in the end winding. Those provisions are unique among all conceptual designs for >16 T hybrid dipoles, and should make fabrication and assembly workable.

![](_page_7_Figure_0.jpeg)

**Caption:** Figure illustrates a 15 T model dipole with a 1 m body length designed for development purposes. The diagram includes field design and portrays the dipole's isometric view. Achieving 14.8 T bore field, the REBCO insert reaches a maximum field of 16.5 T, while the Nb3Sn outsert peaks at 8.9 T. These fields suggest a series winding current of 17.5 kA, underscoring its design for high-field applications without reaching its short-sample limits .


# **Plans for development**

The ARL and ATC have teamed to develop the coil technology for the REBCO stacked-tape cable and for the Nb3Sn cable-in-conduit. ATC received SBIR funding to develop the Nb3Sn CIC cable [ 16]. That project should produce the quantity of CIC that is required for the 1 m prototype hybrid dipole that is described below. ARL has submitted a proposal for 3-year funding to develop the REBCO stacked-tape cable of Figure 7 and to build and test the model dipole described below.

#### **1-meter 15 T model dipole**

Developing the 18 T hybrid dipole requires a succession of buildand-test developments. For this purpose we have designed a 15 T model dipole that is identical to the full design, but with fewer layers of the outsert Nb3Sn CIC winding. Figure 11 shows the field design and an isometric view of the dipole with 1 m body length. The bore field is 14.8 T, the maximum field in the REBCO insert winding is 16.5 T, the maximum field in the Nb3Sn outsert winding is 8.9 T. Those fields assume a series winding current of 17.5 kA (just as for the 18 T dipole), and neither winding is at its short-sample limit.

![](_page_7_Picture_5.jpeg)

**Caption:** This figure shows ATC's cryogen-free test cryostat, highlighting the mounting of the 15 T model dipole and a current transformer. Two 2-stage cryocoolers (4.2 K/50K) for HTS current leads and a single-stage cryocooler (120 K) for heat shields are utilized within this setup. It is part of a sequence of build/test stages aimed at developing REBCO stacked-tape cable technology and integrating these into a model hybrid dipole .


*Figure 10. ATC's cryogen-free test cryostat, showing the mounting of the 15 T model dipole and its current transformer.*

# **Testing in cryogen-free test cryostat**

ATC is building a cryogen-free test cryostat for use in testing a REBCO-based dipole for another project. The cryostat utilizes two 2-stage 4.2 K/ 50K cryocoolers to cool HTS current leads, and a single-stage l20 K cryocooler to cool heat shields. The cryostat will be equipped to test the model dipole, as shown in Figure 10.

A sequence of build/test stages is planned to mature the coil technologies and their integration into the model hybrid dipole:

- Fabricate few-turn winding of Nb3Sn CIC, form the flared ends, heat-treat them, and test to establish whether CIC preserves the short-sample limit of the wire. Study quench detection and quench protection.
- Fabricate the 4-layer outsert winding, heat-treat, install and pre-load in steel flux return. Study parameters of operation, quench detection, and quench protection.
- Fabricate few-turn winding of REBCO stacked-tape cable, install inside the CIC dipole with independent currents, and preload the structure. Study operation, current-sharing, and quench dynamics.
- Fabricate full REBCO insert winding, assemble Nb3Sn half-shell windings in series and preload. Study operation, current-sharing, quench dynamics, and inductive couplings.

## **Conclusion**

A strategy is presented by which a tape-stack REBCO cable may be configured in a *conformal winding* in such a way that the favorable ∥ orientation is sustained everywhere in the winding. A method for *dynamic current sharing* is discussed, in which current would naturally re-distribute among the tapes of each cable turn as the windng current is increased without inducing premature quench. A dipole geometry is presented in which such a conformal REBCO insert winding is configured within a *cable-in-conduit Nb3Sn outsert winding* so that the sub-windings can be separately fabricated, and then assembled in a common structure and preloaded for effective stress management.

A design is presented for an 18 T hybrid conformal REBCO dipole in which the Nb3Sn CIC outsert provides 12 T and the REBCO insert provides 8 T. The total cost of superconductor in this dipole is likely a minimum among the design options for 18 T.

Plans for development of the coil technologies and a model dipole are presented.

# **References**

- [1] Adaba A *et al*., 'FCC-hh: The Hadron Collider', *European Physical J. Special Topics* **228**, 755–1107 (2019).
- [2] Rossi L and Todescu E, 'Conceptual design of 20 T dipoles for high-energy LHC', CERN Report, https://arxiv.org/abs/1108.1619

Schoerling D *et al*., 'The 16 T Dipole Development Program for FCC and HE-LHC' IEEE Trans. Appl. Superconduct. **29**, 5, 4003109 (2019).

- [3] van Nugeren J *et al*., 'Toward REBCO 20 T+ dipoles for accelerators', Proc. MT27: Int'l Conf. on Magnet Technology (2022).
- [4] http://www.superpower-inc.com/content/2g-hts-wire
- [5] McIntyre PM *et al*., 'REBCO-based dipole optimized for an ultimate-energy hadron collider', Proc. Appl. Superconductivity Conf. (2021).

McIntyre PM *et al*., 'Strategies for conformal REBCO windings', Proc. Int'l Cryomaterials Conf. (2022).

- [6] Goldacker W, Grilli F, Pardo E, Kario A, Schlachter SI, and Vojenciak M 2014 *Supercond. Sci. Technol.* **27** 093001
- [7] Van der Laan DC: 2015 Patent *WO2015006350A3*
- [8] Shun Y *et al*., 2017 Patent CN107564623A
- [9] Iwasa Y, Bascuñán J, and Hahn S 2018 Patent US10079092B2
- [10] Yagotintsev K, Anvar VA, Gao P, Dhalle MJ, Haugan TJ, van der Laan DC, Weiss JD, Hossain MSA, and Nijhuis A 2020 *Supercond. Sci. Technol.* **33** 085009
- [11] Breitschopf J, Chavez D, Elliott T, Kellams J, McIntyre P, and Sattarov A 2020 *IOP Conf. Ser.: Mater. Sci. Eng.* **756** 012031
- [12] Choi S, Jo HC, Hwang YJ, Hahn S, and Ko TK 2012 *IEEE Trans. Appl. Superconduct.* **2**, 3, 4904004
- [13] Noguchi S 2019 *IEEE Trans. Appl. Supercond.* **29**, 5, 4602607
- [14] Lu J, Goddard R, Han K, and Hahn S 2017 *Physics: Instruments and Detectors*  1701.00447
- [15] Martinez ACA, Ji Q, Prestemon SO, Wang X, and Mary-Cuna GHI 2020 *IEEE Trans. Appl. Superconduct.* **30**, 4, 1-5, 6600605
- [16] 'Nb3Sn SuperCIC outsert and REBCO conformal insert for an 18 T collider dipole', DOE SBIR grant DE-SC0021688.