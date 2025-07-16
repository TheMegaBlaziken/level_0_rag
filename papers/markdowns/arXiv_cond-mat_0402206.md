# arXiv:cond-mat_0402206

**Paper ID:** b60b470f3982c60d58592d4bf410a8ea

**PDF Path:** apl/Superconductors/arXiv:cond-mat_0402206.pdf

**Processing Status:** complete

**Captions Added:** 10

**Generated:** 2025-06-24T13:44:28.147659

---

L. D. Cooley1,2, C. M. Fischer1 , P. J. Lee1 , and D. C. Larbalestier1

*1 Applied Superconductivity Center, University of Wisconsin, Madison, Wisconsin 53706 2*

 *Materials Science Department, Brookhaven National Laboratory, Upton, New York, 11973* 

In powder-in-tube (PIT) Nb3Sn composites, the A15 phase forms between a central tin-rich core and a coaxial Nb tube, thus causing the tin content and superconducting properties to vary with radius across the A15 layer. Since this geometry is also ideal for magnetic characterization of the superconducting properties with the field parallel to the tube axis, a system of concentric shells with varying tin content was used to simulate the superconducting properties, the overall severity of the Sn composition gradient being defined by an index *N*. Using well-known scaling relationships and property trends developed in an earlier experimental study, the critical current density for each shell was calculated, and from this the magnetic moment of each shell was found. By summing these moments, experimentally measured properties such as pinning-force curves and Kramer plots could be simulated. We found that different tin profiles have only a minor effect on the shape of Kramer plots, but a pronounced effect on the irreversibility fields defined by the extrapolation of Kramer plots. In fact, these extrapolated values *HK* are very close to a weighted average of the superconducting properties across the layer for all *N*. The difference between *HK* and the upper critical field commonly seen in experiments is a direct consequence of the different ways measurements probe the simulated Sn gradients. Sn gradients were found to be significantly deleterious to the critical current density *Jc*, since reductions to both the elementary pinning force and the flux pinning scaling field *HK* compound the reduction in *Jc*. The simulations show that significant gains in *Jc* of Nb3Sn strands might be realized by circumventing strong compositional gradients of tin.

74.70.Ad, 74.25.Ha

### **I. Introduction**

Superconductivity in the A15 phase of the Nb-Sn system exists over the whole composition range from ~18 to ~25 at.% Sn, with better properties being found as stoichiometry is approached. Since all Nb3Sn conductors are made by a diffusion reaction, the influence of not being at thermodynamic equilibrium should be considered, though in fact it seldom is. In reality it is almost certain that no filamentary Nb3Sn composite of any design has ever been made to attain an equilibrium, that is uniform composition, because the need to maintain a small A15 grain size of 150 nm or less limits both the time and temperature of reaction. The consequence of restricting reaction, then, is that there is always a composition gradient across the superconducting layer. Devantay *et al.*<sup>1</sup> found that the critical temperature *Tc* varies approximately linearly from about 6 to about 18 K. The variation of the upper critical field *Hc2* with composition is less certain but also significant.1,2,3 Since most modern high-field Nb3Sn conductors introduce alloying elements to raise the resistivity and suppress the martensitic transformation4 , this variation may also be approximately linear with tin content too. Many years ago it was realized that compositional variations across the A15 phase would affect the observed superconducting properties5,6,7,8,9,10,11. However, neither the geometry of the largely bronze-route composites available in the 1980s, nor the analytical techniques of that time, made it feasible to survey in detail the actual composition gradients that were present across whole A15 layers.

Recently we returned to study this problem, incited by the availability of high quality powder-in-tube (PIT) Nb3Sn composites made by Lindenhovius *et al*. 12,13 We find these PIT conductors to be of high quality both because of the very uniform nature of their filament tubes and because of their high critical current density *Jc*. These PIT conductors have two other important benefits for study, one being that the A15 phase forms layers that are 5-10 µm thick, suitable for accurate electron-probe microanalysis. The second advantage is a consequence of the special architecture of PIT wires, namely that the A15 layer forms by reaction between a central tin-rich core and a coaxial Nb tube, as shown in fig. 1. Thus the Sn content of the A15 layer is greater on the inside of the layer and superconductivity becomes weaker with radius outward from the core, making the system magnetically transparent in terms of superconducting shielding currents as a function of increasing field at given temperature. Magnetization measurements with the field coaxial to the strand are also relatively simple to interpret, because the superconducting A15 layer is a cylinder, which can be approximated by an array of coaxial tubes. Hawes *et al.*14,15 used this geometry to estimate the radial variation of tin content in a binary PIT strand series by inverting the temperature dependence of the magnetic moment *m*, finding that the variation of *Tc* with radius *r* was consistent with the radial variation of the tin content determined by energy-dispersive x-ray spectroscopy (EDX) in a scanning electron microsocpe.

In a subsequent study, Fischer *et al.* 16,17 studied the development of flux pinning and high-field superconducting properties by magnetization measurements in a vibrating sample magnetometer (VSM) at fields up to 14 T, which thus allowed a large range of magnetic field-temperature (*H*-*T*) space to be studied. A puzzling early finding was a large difference between the irreversibility field *H\**, determined by either Kramer function extrapolation (*HK*) or magnetization loop closure, and the upper critical field *Hc2* for all heat treatment times. The difference between *Hc2* and *HK* was always substantial but did decrease with increasing reaction, suggesting substantial changes in the tin composition gradient with increasing reaction. A curious feature of the Kramer plots, even those for very short reaction times when concentration gradients are largest, was that they were very linear all the way to zero bulk *Jc* when measured at temperatures above ~12 K, at which the 14 T field of the VSM could completely suppress superconductivity in the wires.

This behavior is rather different from that of many bronze wires, which generally exhibit significantly curved Kramer plots. However, the normal architecture of bronze conductors places the highest Sn content at the *outside* of the filament, making the superconducting inhomogeneity magnetically invisible because the strongest shielding layer surrounds the weaker ones. Bronze conductors also have thin A15 layers (~1-2 µm) making accurate measurement of their Sn gradient difficult. The few such measurements5,6,8,10,11 are all consistent with strong Sn gradients, making the association of a curved Kramer plot with strong Sn gradients across the whole layer plausible. Thus we were initially motivated to believe that the linear Kramer plots observed with PIT composites16,18 resulted from rather flat Sn gradients. And in fact Hawes *et al.* showed that the Sn gradient in the PIT conductor was rather flat over most of the ~5 µm radius of the A15 layer, ranging from about 1 at.%Sn·µm-1 at short reactions to as small as 0.4 at.%Sn.µm-1 at longer times, the Sn content falling steeply to 18 at.% only near the Nb3Sn/Nb interface.14 Thus, while thermodynamics enforces the terminal compositions of the inner and outer radii of the A15 layer to be those in equilibrium with the Sn-rich phase and with Nb respectively, it does *not* dictate the nature of the gradient, which could vary from one architecture to another.

Based on EDX analyses, magnetic *Tc* analyses and a volumetric specific heat *Tc* analysis, Hawes *et al.* found that the Sn and *Tc* gradients were both declining as the reaction proceeded. Fischer *et al.* and Godeke *et al.*<sup>19</sup> subsequently extended this zero-field study to make careful studies of how different in-field measurement techniques average the range of superconducting properties generated by the tin composition profile. The reasonable assumption made by Fischer *et al.* was that extrapolating Kramer plots from low fields defines an irreversibility field *HK* that is characteristic of a weighted average, rather than the best regions, of the A15 layer carrying the current. *Hc2* can also be determined in the VSM by observing the change in the slope of the reversible magnetization as superconductivity is destroyed. However, inhomogeneity in the tin composition smears this transition such that the experimentally observed discontinuity at *Hc2* represents the best region of the A15 layer. The difference between these two measurements is not trivial—at optimum reaction for highest *Jc* and *HK*, the actual measurements at 12 K showed that *HK* was 9.7 T while *Hc2* was 13.1 T. Similarly large differences between *Hc2* and *H\** had been seen earlier by Suenaga *et al.*20 in bronze conductors, although in that study the difference between *HK* and *Hc2* was attributed to vortex melting effects in what was otherwise implied to be a homogeneous conductor.

To address the general influence of radial superconducting property gradients, we decided to simulate PIT strands using a system of concentric shells to represent various tin concentration profiles, drawn schematically in fig. 1. Current-density profiles associated with the input composition profiles were calculated based on literature values for *Tc* and *Hc2* and reasonable use of the Fietz and Webb scaling relations.21 The magnetic moment of each

![](_page_1_Figure_7.jpeg)

**Caption:** This figure, extracted from scanning electron microscopy data, displays the layer structure of a PIT Nb3Sn composite with labels identifying key components: Nb, Nb3Sn, and the Sn-rich powder core. The visual provides a microscopic view at 20 µm scale of the distribution and arrangement of the prevalent phases within the composite. This imaging underscores the heterogeneous nature of PIT composites, which affects the conductivity and critical magnetic properties crucial to the design and manufacturing of advanced superconducting materials.


Magnetization current

![](_page_1_Figure_9.jpeg)

**Caption:** Figure 8 showcases various aspects of a PIT Nb3Sn composite strand with emphasis on radial Tin diffusion: (a) features a schematic illustrating the diffusion process, wherein Tin migrates through concentric shells under an applied magnetic field H. The radial diffusion is highlighted as a critical mechanism influencing the superconducting properties of Nb3Sn composites by affecting the continuity and effective use of the superconducting layer. The visualization signifies the role of radial diffusion in optimizing the composite's electromagnetic characteristics.


Fig. 1. Top: Scanning electron microscope image of a crosssection of a PIT Nb3Sn composite. Bottom: Schematic of a typical configuration for electromagnetic measurements, showing the arrangement of magnetization current shells with respect to the applied field and the radial direction of tin diffusion.

shell could then be obtained from the current density and shell geometry and, by summing the contribution from all shells, we could estimate quantities that are obtained in magnetization experiments with the field parallel to the strand axis.

The general conclusion we draw from these simulations is that *H\*,* as determined by *HK*, reflects a weighted *average,* while *Hc2* reflects the *best* portion of the filaments, thus validating the assumption of Fischer *et al.* above. The difference between *Hc2* and *HK* indeed diminishes as the Sn concentration gradient flattens. Broadly speaking, these simulations suggest that the *tin composition profile* determines *H\**, suggesting that *H\** and *Hc2* become *equal* in the absence of composition gradients. We here neglect thermal fluctuation effects. A very striking conclusion from the simulations is that the bulk flux pinning force is correlated with the properties of the weakest shells, even very far from *HK* or *Hc2*. This conclusion demonstrates that significant improvement of the *Jc* of Nb3Sn will result from reducing or in the best of circumstances eliminating macroscopic tin composition gradients provided that the grain size can be maintained small at the same time.

### **II. Experiment**

A series of tin concentration profiles was generated using a spreadsheet. It was assumed that the inner radius was next to the tin source, anchoring the composition of the A15 layer at Nb0.75Sn0.25 at this interface, while the outer radius at the Nb/A15 interface was anchored at Nb0.82Sn0.18. Between these endpoints, the tin content was allowed to vary according to the expression

$$\text{\textbulletSn } (r) = 18 + \text{\textbullet } [1 - r^N + (1 - r)^N]. \tag{1}$$

The index *N* indicates the severity and steepness of the overall gradient, where *N* = 1 represents a linear variation of tin composition and *N* → ∞ describes a flat profile with an abrupt falloff near the Nb3Sn / Nb interface. A plot of the assumed tin composition profiles is shown in fig. 2.

The critical temperature at each shell was mapped according to each profile, assuming the linear dependence found by Devantay *et al.*<sup>1</sup> :

$$T\_{c\text{-sim}}(r) = 6 + 12 \left[ \left( \% \text{Sn}(r) - 18 \right) / 7 \right] \text{ kelvin.} \tag{2}$$

These *Tc-sim* profiles were used to generate profiles of upper critical field values *Hc2-sim*(*T,r*) at *T* = 0 K,

$$
\mu\_0 H\_{c2\cdot \text{sim}}(0, r) = 0.69 \cdot 2.4 \cdot T\_{c\cdot \text{sim}}(r) \text{ tesla.} \tag{3}
$$

![](_page_2_Figure_12.jpeg)

**Caption:** Figure 12 depicts the input profiles of tin concentration as a function of radius (r) for various gradient indices (N = 1, 2, 3, 5, 8, 10, 15, and 20), alongside an ideal profile with a constant tin content of 25%. The purpose is to analyze how radial tin concentration affects superconducting properties in Powder-in-Tube (PIT) Nb3Sn strands. Labels on the right axes show local critical temperatures and upper critical field values corresponding to these compositions. Profiles ranging from linear to a flat plateau with an abrupt transition near the Nb3Sn/Nb interface are considered. These profiles influence the superconducting properties, with higher N values representing a flatter gradient indicative of extended heat treatments. This visualization is critical for understanding how tin gradients impact the field and temperature-dependent superconducting behavior, offering insights crucial for optimizing heat treatment processes in Nb3Sn-based superconductors.


Fig. 2. Input profiles of tin composition as a function of radius. The index *N* depicts the sharpness of the profile, where *N* = 1, 2, 3, 5, 8, 10, 15, and 20 are shown here and in figs. 4 to 7. An ideal profile, in which the tin content is constant at 25%, is also shown. Local values of the critical temperature and upper critical field (at *T* = 0) that correspond to the local A15 composition are indicated on the right axes.

Equation (3) is the usual dirty-limit expression *Hc2*(0) = 0.69 *Tc* (-d*Hc2*/d*T*|*Tc*) assuming the slope µ0d*Hc2*/d*T* at *Tc* is –2.4 T/K. The particular value of the slope of *Hc2* at *Tc*, being a bit higher than values in the literature,22,23 was chosen here to make *Hc2-sim*(0,0) close to 30 T, as suggested by recently observed values for unalloyed Nb3Sn prepared by hot isostatic pressing.<sup>3</sup> It should be noted that the variation of *Hc2* with composition is not as well known as that of *Tc*. Finally, a temperature dependent factor of 1– (*T*/*Tc-sim*) 1.5 was multiplied by *Hc2-sim*(0,*r*) used to get *Hc2-sim*(*T*,*r*). This fit is an excellent approximation of the more complicated Maki-deGennes-Werthamer dirty-limit expression for *Hc2*(*T*) 24,25. In contrast to experiments, in which *Tc* and *Hc2* are single values determined by measurement, in the simulations the critical temperature and the upper critical field are *local* values; a hypothetical resistive or magnetization measurement would always produce experimental *Tc* and *Hc2* value that correspond to the maximum *Tc-sim* value for all shells, *i.e.* 18 K and close to 30 T at 0 K for the shell next to the powder core.

As mentioned earlier, an interesting feature of the *Jc* data on PIT composites is the observation of field and temperature scaling for a wide range of heat treatment conditions. Plots of *Jc* 1/2*H*1/4 vs. field *H* (Kramer plots) are linear for almost all heat treatment conditions15 and measurement temperatures.26 This permits the flux pinning force for each shell to be written

$$\begin{cases} F\_{p \cdot \text{sim}}(h, T, r) = \text{3.5 } F\_{\text{max} \cdot \text{sim}}(T, r) \text{ } h^{1/2} \text{ } (1 - h)^2 \end{cases} \tag{4}$$

where *h* = *H* / *Hc2-sim*(*T*,*r*) and the factor 3.5 normalizes the field dependence. Further, analyses of the temperature

![](_page_3_Figure_1.jpeg)

**Caption:** Figure 1 illustrates the maximum bulk pinning force (Fp_max) plotted against the irreversibility field (µ0 H_K). The data points represent measurements from a series of ten Powder-in-Tube (PIT) samples, conducted across temperatures from 4.2 K to 16 K. The figure captures data that collapse onto a single linear fit described by a mathematical equation indicated in the study. Notably, despite varying heat treatment durations (0.5 to 128 hours at 675°C) and temperatures, the data consistency underscores a universal behavior. Low-temperature data align with those obtained at 15 to 16 K (denoted by open symbols), which yield low H_K values but remain consistent with the lower-temperature data set under the defined equation. The significance of this figure lies in demonstrating the robustness and repeatability of bulk pinning force characteristics across varied experimental conditions, presenting critical insights into the performance of PIT superconductors.


Fig. 3. Maximum bulk pinning force plotted against the irreversibility field, derived by extrapolating Kramer plots, for a series of 10 PIT samples, measured at 4.2 to 16 K. Despite differences in heat treatment duration (0.5 to 128 hours at 675°C ) and temperature, these data collapse onto the single linear fit given by Eq. 5. Data taken at 15 to 16 K (open symbols) yield very low values of *HK*, but are still consistent with those at lower temperatures used for Eq. 5.

dependence of the bulk pinning force showed that the temperature scaling could be expressed as *Fmax*(*T*) = 0.1 [*HK*(*T*)]2.0, as shown in fig. 3. Therefore, the same temperature scaling for the bulk pinning force in each shell was assumed, namely:

 $F\_{max\text{-}sim}(T, r) = 0.1$   $\left[\mu\_0 H\_{c2\text{-}sim}(T, r)\right]^2$   $\text{GN/m}^3,$ 

with µ0*Hc2-sim* in tesla. To obtain the critical current density for each shell, the definition of the Lorentz force was inverted,

$$J\_{c\text{-sim}}(h, T, r) = F\_{p\text{-sim}}(h, T, r) \;/\; h \; \text{A/m}^2,\tag{6}$$

and *Jc-sim*(*h*,*T*,*r*) was mapped to *Jc-sim*(*H*,*T*,*r*) using *H* = *hHc2 sim*(*T*,*r*). At this point, it is important to distinguish that, while experimental data commonly exhibit scaling of the pinning force with *H\** or *HK*, there is no *a priori* way to determine the scaling field from *Hc2* in the absence of thermal fluctuations of the vortices. Moreover, fluxpinning models27 suggest that the superconducting condensation energy, vortex density, and flux-lattice behavior should determine the pinning force, which are all tied to *Hc2*. Thus, eqs. 4 and 5 should be appropriate for simulating the local properties of each shell. This point will be discussed again later.

To get reasonable estimates for experimental quantities, the simulations were carried out by mapping the coordinate *r* to a radius *R* = *Rmin +* 5*r* [µm]. This corresponds to a tin gradient that varies between a powder core located inside *Rmin* = 10 µm and a Nb3Sn/Nb interface located at *Rmax* = 15 µm. These choices are typical radii observed by scanning electron microscopy (*e.g.* fig. 1). The magnetic moment *msim*(*H*,*T*,*R*) for each shell was calculated from the critical current density by using the definition of a magnetic moment,

$$m\_{sim}(H,T,\mathcal{R}) = \pi \mathcal{R}^2 \, I\_{c \cdot sim}(H,T,\mathcal{R})$$

$$= \pi \mathcal{R}^2 \, t \, L \, J\_{c \cdot sim}(H,T,\mathcal{R}) \, \text{A} \cdot \text{m}^2 \tag{8}$$

where *Ic-sim*(*H*,*T*,*R*) is the current flowing around each shell, *t* = (*Rmax* – *Rmin*)/100 is the thickness of each of the 100 shells. The total moment for all shells *m*(*H*,*T*) was then obtained by summing the contributions from each shell.

Since the total magnetic moment is the physical property experimentally measured by magnetometry, the simulated profiles could be directly identified with experimental quantities. In particular, an estimate for the experimental critical current density *Jc* could be obtained by applying the critical state model to the total moment and the input sample dimensions,

$$J\_c(H, T) = \text{3 } m(H, T) \text{ / [} \text{\textdegree L } (R\_{max} \stackrel{3}{\text{}} - R\_{min} \stackrel{3}{\text{}}) \text{] A/m}^2. \tag{9}$$

This expression duplicates that used by Fischer *et al.* to determine the critical current density from vibrating sample magnetometer (VSM) measurements with the field parallel to the strand axis. Estimates for other experimental quantities, such as the bulk pinning force *Fp* = µ0*H*·*Jc* and the Kramer function *FK* = *Jc* 1/2*H*1/4 were then derived.

Fig. 4 shows the simulated moment as a function of radius derived at 4.2 K, 12 T (plot a) and 12 K, 5 T (b), together with the mapping from *r* to *R*. These temperatures correspond to temperatures investigated in the experiments of Fischer *et al*., and the first plot also corresponds to a benchmark for recent R&D composite development.28 Since a typical value for *L* is a few millimeters, the simulated moment for each data point in fig. 4 corresponds to a real moment on the order of 10-10 A·m2 . The total moment for 100 shells in each PIT filament and ~200 filaments in a typical sample is thus about 2 × 10-6 A·m2 = 2 × 10-3 emu, which is comparable to observed values for PIT composites in magnetometry experiments. The curve labeled "ideal" represents the so-far experimentally unattainable constant tin content of 25% across the entire A15 layer (*i.e.*, *N*→∞). Notice that there is no contribution to *m* from many outer shells, even for very high *N*, when the chosen field and temperature exceed *Tc-sim* and *Hc2-sim*.

![](_page_4_Figure_1.jpeg)

**Caption:** Figure 3 lays out the simulated moment of concentric shells as a function of radius derived under conditions of 4.2 K and 12 T (part a) and 12 K and 5 T (part b). Each profile shows the moment reduction progressing to outer shells due to decreased Tin concentration gradients and superconductivity. This depletion effect is more prevalent at higher temperatures, where the outer sections do not exhibit superconductivity in zero fields. These plots highlight the necessity of maintaining an optimal Tin gradient to ensure maximum superconducting moment and efficient performance of PIT strands.


Fig. 4. Moment of the shells as a function of radius simulated at 4.2 K and 12 T (a), and at 12 K and 5 T (b). The indices for the curves are the same as for Fig. 2. Note that the coordinate *r* has been mapped to a radius *R* to simulate actual composites.

This is especially prominent at 12 K, where the outer sections of the layer are not even superconducting in zero field. The quantitative importance of the gradient to the moment accessible by experiments is very clear: If the tin concentration profile were indeed just a linear function of the thickness of the layer (*N* = 1), then the superconducting moment at 12 K would be small indeed due to the loss of many shells with *msim* = 0.

Figs. 5 through 7 present the experimental equivalent of *Jc*(*H*), *Fp*(*H*), and *FK*(*H*) for the same set of index values, with curves derived for 4.2 K and 12 K plotted in each figure. Consistent with Fig. 4, there is an evident loss of critical current density and bulk pinning force with decreasing *N*. The simulations also produce a shift of the peak of *Fp*(*H*), and also in the field at which *Fp*(*H*) goes to zero, towards higher fields as *N* increases. After examining many such *msim*(*H,T,R*) curves, we find that the curvature in the Kramer plots first appears for simulated fields that exceed *Hc2-sim*(*T*,*Rmax*); that is, when the outermost shell is no longer superconducting. This is always true at 12 K, since *Tc-sim* at *Rmax* is only 6 K for Nb18%Sn.

![](_page_4_Figure_5.jpeg)

**Caption:** Figure 5 presents the critical current density (J_c) derived from magnetization simulations of actual PIT strands, plotted across magnetic fields at (a) 4.2 K and (b) 12 K. When inset, the figure enhances mid-field data visibility, further elucidating the complex interrelation between composition profiles and current capacity under varying conditions. These representations underscore the impact of Tin gradients on J_c, and how simulation indices corroborate with experimental data, indicating the level of influence exerted by compositional heterogeneity.


Fig. 5. Critical current density, as would be determined from a magnetization experiment of an actual PIT strand, as a function of field simulated at (a) 4.2 K, and (b) 12 K. The inset of plot (a) shows a magnified view of the mid field data. The indices for the curves follow the same sequence as in Fig. 2.

### **III. Discussion**

# **A. Effect of tin gradients on the apparent irreversibility field**

An important result of Fischer *et al.* was the finding of strong changes in, and large differences between, the values of *Hc2* and *HK* as a function of heat treatment time.16,17 The difference between *Hc2* and *HK* at 12 K ranged from ~5 T at short times to ~3 T at long times, and both values increased with heat treatment time until saturating when the reaction was complete. It was argued that these differences were a direct consequence of the different ways in which *HK* and *Hc2* are measured and tin composition profile, which was steep (low *N*) during the early stages of reaction and flattened (high *N*) as the reaction proceeded. The upper critical field was thought to represent the best (most tin-rich) regions of the A15 layer because *Hc2* was defined by the return of the magnetization to the normal-state paramagnetic background at very high field, which occurs when the best potions of the A15 layer lose superconductivity. This was later found to be

![](_page_5_Figure_1.jpeg)

**Caption:** This figure juxtaposes simulated data through Kramer plots, reflecting linear characteristics across a wide field range and different composition gradients. As seen, for high N values, linearity extends nearly to the field axis. These plots serve to showcase how compositional changes affect extrapolation thus revealing pivotal insights into the superconducting quality and uniformity, captured via typical electromagnetic measurements in the simulated context.


![](_page_5_Figure_2.jpeg)

**Caption:** Figure 2 presents two graphs depicting the radial distribution of the critical current density per unit length (m_sim) plotted as a function of radius (R) at two conditions: (a) 4.2 K, 12 T, and (b) 12 K, 5 T. The plots also juxtapose these data with an 'Ideal' line, representing the theoretical maximum current density without any compositional inhomogeneities. Multiple profiles are shown, indexed by their radial position, reflecting varying heat treatments during simulations. The observed trends indicate that the current density diminishes rapidly close to the outer radius when radial composition gradients are present, especially under higher temperature conditions (12 K). This figure underscores how radial tin gradients critically alter superconductor performance by reducing current flow in outer layers, particularly at elevated temperatures.


comparable to resistive measurements24, which sample the best electrical pathway. By contrast, *HK*, being derived from the full width of the hysteresis loop, was determined by the critical state across the entire A15 layer at lower fields, thus providing a weighted average of the superconducting properties across the layer. Since most Kramer plots obtained by Fischer *et al.* were highly linear, even close to the field axis, it was further assumed both that *HK* was well determined, and that the tin concentration profile was often rather flat except near the A15/Nb interface (equivalent to high *N*). However, it was not clear why this should be, especially for short heat treatment times. It is important to note that the measurement geometry used by Fischer *et al.* was such that currents enclose the powder core for *H* < *HK* and thus make large contributions to the measured magnetization, while for *HK* < *H* < *Hc2* hysteretic current loops might still be present, but their contribution to the overall magnetization, which includes reversible components, would hardly be detected. This makes confusion between these values in their experiment very unlikely.

The Kramer plots derived from the simulation follow s traight lines when all of the shells are superconducting. In

![](_page_5_Figure_5.jpeg)

**Caption:** Figure 7 represents simulated Kramer function curves as a function of magnetic field strength, with data obtained at (a) 4.2 K and (b) 12 K. The figures include high-resolution insets for near-field axis data, illustrating how high gradient indices (N values) lead to persistently linear Kramer plots down to the field axis, specifically at lower temperatures. These plots emphasize the correlation between N values and the superconducting shell activity, further providing evidence that Kramer plot extrapolations can significantly differ based on underlying composition profiles.


Fig. 7. Kramer function curves as a function of field simulated at (a) 4.2 K and (b) 12 K. The indices for the curves follow the same sequence as in Fig. 2. The insets in both plots show a magnified view of the region near the field axis.

many cases, especially when *N* is high and *T* is low, linear plots persist almost down to the field axis. Even when some shells are not superconducting because *H* > *Hc2 sim*(*T,r*), as is often the case at high temperature, the curvature for large *N* is not striking. On the other hand, the linear portions of the Kramer plots extrapolate to *HK* values less than the maximum value of *Hc2-sim*(*T*) (being that for the inner shell with 25% Sn), even for high index values. This maximum value is clearly indicated on each plot by the curve labeled "ideal", since the ideal profile has no variation in tin composition and, by definition, produces *HK* = *Hc2* at all temperatures. Thus, the *primary influence of the different composition profiles is to alter the Kramer plot extrapolation*, and not to change greatly the Kramer plot shape itself. Moreover, the slopes of the linear portion of the Kramer plots at 12 K fall off with decreasing *N*, whereas they remain nearly constant at 4.2 K. This behavior is the same as that seen in the experimental curves of Fischer *et al.* The slope change is another effect of the loss of superconductivity in the outer shells far away from the tin source.

| Gradient index | Weighted mean of Hc2-sim(r) at 4.2 K | HK at 4.2 K | Weighted mean of Hc2-sim(r) at 12 K | HK at 12 K |
|----------------|--------------------------------------|-------------|-------------------------------------|------------|
| N              | (T)                                  | (T)         | (T)                                 | (T)        |
| 1              | 15.7                                 | 19.1        | 7.1                                 | 9.6        |
| 2              | 19.4                                 | 21.1        | 8.5                                 | 10.3       |
| 3              | 21.2                                 | 22.1        | 9.4                                 | 10.8       |
| 5              | 23.0                                 | 23.3        | 10.5                                | 11.4       |
| 8              | 24.2                                 | 24.2        | 11.3                                | 11.9       |
| 10             | 24.6                                 | 24.6        | 11.7                                | 12.1       |
| 15             | 25.2                                 | 25.1        | 12.2                                | 12.5       |
| 20             | 25.5                                 | 25.4        | 12.6                                | 12.7       |
| Ideal          | 26.5                                 | 26.4        | 13.6                                | 13.6       |

Table I. Comparison of the average value of the upper critical field with extrapolations of simulated Kramer functions, at 4.2 and 12 K.

To test these predictions of Fischer *et al.* more explicitly, the *HK* values derived from the simulation are compared in Table I with the average of the *Hc2-sim*(*r*) values across the simulated tin profiles at 4.2 and 12 K. The averaging is weighted so that only superconducting shells are considered; shells for which *T* > *Tc-sim*(*r*) do not figure into the mean. Notice that there is very good agreement between *HK* and the weighted mean for all *N* at 4.2 K. This is true even though the minimum value of *Hc2-sim*(*r*) is only about 5 T at 4.2 K for *N* = 1, due to the low critical temperature of the tin-poor outer shells. At 12 K, the agreement is still reasonably good, and becomes better for increasing *N*. Thus, these simulations show that extrapolations of Kramer plots from low fields give good estimates of the *average* upper critical field across the active A15 layer, in agreement with the assertion by Fischer *et al.*.

In addition, at 4.2 K the difference between *Hc2* (= *HK* for the ideal profiles) and *HK* is ~10 T for *N* = 1, and then rapidly drops to 3-4 T for intermediate *N* and is only 1-2 T for *N* ≥ 10. At 12 K, the separation is ~6 T for *N* = 1 and likewise falls to ~1 T for *N* ≥ 15. This trend supports the conjecture by Fischer *et al.* that the tin composition profile flattens with increasing heat treatment time, which decreases the separation between *Hc2* and *HK.* This conclusion was also reached by Hawes *et al.*, 14,15 as we discuss in more detail later. Since the terminal separation of *Hc2* and *HK* found by Fischer *et al.* was ~3 T at 12 K, the data in Table I suggest that a profile index of between 5 and 10 might apply to their optimum wires.

It is important to note that we did not simulate any effects of thermal fluctuations—the differences between *Hc2* and *HK* are *entirely* due to the variation of *Hc2-sim*(*T,r*) related to the simulated tin composition profile. Keeping in mind that different experiments probe local properties in different ways, this result suggests that the experimental difference between *Hc2* and *H\** for real wires is mostly due to the effects of *tin composition inhomogeneity*. This implies that the *H\**(*T*) line is not determined by flux lattice properties in Nb3Sn, as it is for high-temperature superconductors.

# **B. Effects on flux pinning and the critical current density**

The tin composition profile can influence the global flux-pinning force *FP* by changing the magnitude of the local elementary pinning interaction and by shifting the scaling field for the bulk pinning-force curve.27 The former effect is expressed in eq. (5), where the temperature dependence of the pinning force, derived from *Hc2-sim*(*T*), reflects the condensation energy difference for a pinning interaction or the vortex line tension. The latter effect reflects flux density-dependent changes in summation and is expressed in eq. (4). Because these independent sources are compounded, the tin composition inhomogeneity actually has a significantly stronger effect on the flux pinning magnitude than on the critical field magnitudes, even though *Jc* is generally evaluated far from *Hc2*. To illustrate this point, the maximum value of *Fp* is summarized in Table II at 4.2 and 12 K. At 4.2 K, the maximum values are reduced from that of the ideal curve by ~28% for *N* = 5 and ~8% for *N* = 20 at 4.2 K. The corresponding decreases of *HK* are ~13% and ~4% for these respective indices in Table I, so the variations of the maximum value of *Fp* are consistent with eq. 5. However, since *HK* is also lower for smaller *N*, the value of *Fp* at *fixed* field suffers an additional loss. At 12 T, for example, *Fp* and *Jc* for *N* = 5 are ~40% less than their corresponding values for the ideal curve. The 12% additional subtraction is due to the shift in the scaling function (eq. 4) of each of t he simulated shells because of the corresponding lower *Hc2 sim*(*r*) values.

It should be noted that, although flux-pinning theories express scaling functions in terms of *Hc2*, it is often useful to choose *HK* or *H\** as the scaling field (i.e. *h* = *H / HK* in eq. 4) in experiments. While this choice might seem to be the more rational one, properties of the flux-line lattice, from which various scaling relations are derived,27 can be traced back directly to the Ginzburg-Landau theory,29 suggesting *Hc2* is the proper choice from a theoretical point of view. In the present simulation, these contradictory views are bridged because the *local* scaling function for

| Gradient | Maximum Fp(H) at 4.2 K |            | Maximum of Fp(H) at 12 K |            | Jc at 12 T, 4.2 K |            |
|----------|------------------------|------------|--------------------------|------------|-------------------|------------|
| index N  |                        |            |                          |            |                   |            |
|          | (GN/m3<br>)            | % of ideal | (GN/m3<br>)              | % of ideal | (A/mm2<br>)       | % of ideal |
| 1        | 23                     | 33         | 2.3                      | 12         | 771               | 19         |
| 2        | 35                     | 51         | 4.9                      | 26         | 1460              | 36         |
| 3        | 42                     | 61         | 6.8                      | 37         | 1930              | 47         |
| 5        | 50                     | 72         | 9.5                      | 51         | 2500              | 61         |
| 8        | 56                     | 81         | 12                       | 64         | 2960              | 72         |
| 10       | 59                     | 84         | 13                       | 69         | 3140              | 77         |
| 15       | 62                     | 89         | 14                       | 77         | 3420              | 84         |
| 20       | 64                     | 92         | 15                       | 82         | 3570              | 87         |
| Ideal    | 70                     | --         | 18                       | --         | 4090              | --         |

Table II. Variation of flux-pinning quantities with tin profile index.

each current shell, *Fp-sim*(*H,T,r*) is tied to the corresponding local value of *Hc2-sim*(*T,r*), whereas the *global* scaling curves *Fp*(*H,T*) that emerge from the simulation (Fig. 6) appear to scale with *H/HK*. This is an additional averaging effect of the current shells, where, as discussed above, *HK* represents the weighted mean of *Hc2-sim*(*r*) values. Hence, *the averaging effect is why the irreversibility field is the appropriate scaling field for flux pinning*.

Because these simulations directly implicate the gradient of Sn in determining *Jc*, it is interesting to speculate how flattening the tin composition profile (while retaining similar grain size and grain-boundary density) could lead to gains in the performance of Nb3Sn wires at 4.2 K and 12 T. Fischer *et al.* found that the highest critical current density of the A15 layer approached 3500 and 4300 A/mm2 at 12 T, 4.2 K for a binary Nb3Sn and a ternary (Nb,Ta)3Sn PIT wire, respectively. When normalized to the entire tubular filament area, including the spent powder core and any unreacted Nb, these current densities are 1460 and 1780 A/mm2 respectively. Since the simulation does not consider unreacted Nb but does include the core, the value of *Jc-sim* should lie between these experimental values. As seen in Table II, this comparison suggests that the gradient index is perhaps 5 to 8, about the same range as that concluded from the critical field analyses discussed earlier. It should be kept in mind that the simulated values of *Hc2* and *HK* (26.5 and >23 T) are close to the estimates of *HK*(4.2 K) ~ 23 T for the binary composite and ~25 T for the ternary composite by Fischer *et al.*. While more experiments are needed to verify the actual values of *Hc2* and *HK* at 4.2 K, this analysis suggests that modern PIT strands are operating at perhaps only 60% of the maximum *Jc* attainable.

### **C. Features of real tin composition profiles**

More experimental focus has been given to electromagnetic characterization than to microchemical characterization of PIT strands, and particular information about the tin profiles in present state-of-the-art strands is lacking. Hawes *et al.*14,15 conducted EDX analyses at various positions across the A15 layer for several strands from an earlier PIT conductor. The microprobe analyses showed that the composition fell off approximately linearly, from ~25% Sn near the powder core to ~22% near the A15/Nb interface. The final 4% drop to the tin-poor boundary of the A15 phase range could not be resolved by the instrument, suggesting that the moderate slope falls off more abruptly at the A15/Nb interface.

We simulated this profile by multiplying a factor of (1 – *cr*) to Eq. (1) to approximate the observed linear profile, with *c* = 0.5 and *N* = 20. The simulated profile, shown in fig. 8a, is somewhat more severe than those suggested by the studies of Fischer *et al.* discussed earlier. Although the linear drop-off of tin content over much of the profile has a lower slope than the simulated profiles with *N* < 5 discussed earlier, the reduction of *Jc* at 4.2 K shown in fig. 8b, is quite significant. At 12 T, *Jc* reaches only 1600 A/mm2 , or about 39% of the ideal, no-gradient value of 4090 A/mm2 . The halving of the bulk pinning force is also evident in fig. 8c. However, the shape of the bulk pinningforce curve is almost identical to that of the ideal curve. Since the peak of the *h*(1 – *h*) 2 function lies at *h =* 0.2, not only is the curvature consistent with the expected pinning function but the peak is also close to the expected position near 4/20. A slight tail in *Fp*(*H*) can be seen above the irreversibility field at ~20 T, which is a common observation for real composites. The Kramer curve in fig. 8d is almost perfectly linear below ~14 T, the limit of many modern magnetometers or transport *Jc* evaluations, but above this field *FK*(*H*) curves more strongly, resulting in a large difference between *HK* (20.5 T) and *Hc2* (26.5 T). Taken together, all of these standard evaluations of flux pinning, if made in the laboratory at 4.2 K and below ~15 T, would suggest that this simulated Nb3Sn example is from a strand of high quality. It is rather amazing that this occurs, even though the simulated results are, in fact, far from the ideal curves. Apparently, the strong contribution from magnetization currents flowing in the few stronger shells is very efficient at masking the deleterious contributions from the many weaker shells when all of the shells are averaged. This may be an artifact of the tubular geometry used in the simulation, and might not hold as

![](_page_8_Figure_1.jpeg)

**Caption:** Figure 8 consists of four plots illustrating different aspects of energized PIT strand analysis: (a) depicts the Tin composition profile used in simulation with critical temperature and upper critical field values; (b) shows the critical current density vs field at 4.2 K over relevant technological field ranges; (c) provides the corresponding bulk pinning force; and (d) outlines the 4.2 K Kramer function, complete with extrapolation. Strikingly, all plots benchmark against an 'Ideal' constant 25% Sn profile, revealing advantages and deficiencies in comparison. These plots synthesize a comprehensive view of how Tin gradients within PIT strands govern superconducting properties and operational limits.


Fig. 8. Energy-dispersive x-ray spectroscopy analyses of a PIT strand in ref. 13 are simulated using the tin composition profile shown in plot (a), with corresponding local values of critical temperature and upper critical field indicated on the right axes. In plot (b), the simulated critical current density vs. field curve at 4.2 K is shown over a field range of technological interest. In plot (c), the corresponding bulk pinning force at 4.2 K is presented. Plot (d) shows the 4.2 K Kramer function, along with its extrapolation (dashed line) from data at 12 T and below. In all plots, the solid line represents the ideal profile of a constant 25% Sn.

strongly for the solid cylinders of most bronze or internal Sn conductors, perhaps resulting in more noticeable shifts in the *Fp*(*H*) function and deviations from flux-pinning theories.

The present simulations consider only radial tin gradients, whereas real composition profiles might contain circumferential and longitudinal gradients as well. These might arise due to disconnection of the powder core from the Nb tube, due to the formation of other intermetallic phases, or due to the preferential diffusion of tin along grain boundaries. In fact, one feature of experimental data that could not be reproduced by the simulation was the linearity of Kramer plots all the way down to the field axis at 12 K. We speculate that the cause may be due to circumferential or longitudinal Sn gradients in real wires. If the magnetization current is completely restricted to flow in concentric shells, circumferential variations in the tin composition would obstruct current and effectively switch off a given shell at a lower field than represented by its average composition. This would tend to amplify any effects of the radial compositional gradient. However, if the magnetization current is allowed to transfer between shells or migrate longitudinally, then the effect of circumferential obstructions would be blunted. Moreover, some shells might actually persist in carrying current at fields *above* that representative of their average composition. In the limit that all current shells have some circumferential variations, the effective network of current pathways could well remove the curvature from real Kramer plots altogether for sufficiently high gradient indices.

### **IV. Conclusions**

The variation of the tin composition in powder-in-tube Nb3Sn composites was modeled by analyzing a system of concentric shells. Different profiles of the local tin content were mapped, between the boundaries of Nb25%Sn at the A15/tin core interface and Nb18%Sn at the A15/Nb interface. The profile severity ranged from constant gradients (*N* = 1) to relatively flat plateaus with a steep gradient near the A15/Nb interface (large *N*).

By summing the calculated magnetic moment for each shell, simulations which yield experimentally accessible quantities were estimated. These simulations reproduced the experimental trends rather well, under the experimentally observed assumption that increasing heat treatment time for PIT strands enhances the value of the gradient index *N*. The experimental linear Kramer functions were reproduced well by the simulations for larger gradient indices. Extrapolations of the Kramer plots gave irreversibility fields *HK* that were very close to the weighted average of local upper critical field values corresponding to the input tin composition profiles. This suggests that Kramer plot extrapolations effectively *average* the superconducting properties across the A15 layer. An important result of the simulations is that this averaging property is passed on to the flux-pinning behavior, which explains why the global scaling field is *HK* even though flux pinning behavior on the elementary level is tied to the upper critical field *Hc2*. From analyses of Sn gradient-free, ideal profiles, for which Kramer plots extrapolate to *Hc2*, it was concluded that the difference between *Hc2* and *HK* can be entirely attributed to the *tin composition profile*. The deleterious properties of severe tin gradients are compounded in quantities related to flux pinning, because both the elementary pinning force and the scaling field for summation are reduced. Thus, while *HK* is generally only 80% of *Hc2* in modern Nb3Sn composites, the critical current density at technologically important fields then reaches only ~60% of the available limit imposed by flux pinning for the stoichiometric A15 composition.

### **Acknowledgments**

This work was supported by the US Department of Energy, Office of High Energy Physics. The facilities used for the work were partially supported by the NSF MRSEC for Nanostructured Materials and Interfaces at the University of Wisconsin.

### **References**

- 1 H. Devantay, J. L. Jorda, M. Decroux, J. Muller, and R. Flükiger, J. Mater. Sci. **16**, 2145 (1981). 2
- R. Flükiger, Annales de Chimie, Science des Materiaux **9**, 841 (1984).
- 3 M. C. Jewell, A. Godeke, P. J. Lee and D. C. Larbalestier, "The upper critical field of stoichiometric and off-stoichiometric bulk, binary Nb3Sn ", Adv. Cryo. Engr. (Materials), to appear 2004. 4
- R. Flükiger, Adv. Cryo. Eng. **28**, 399 (1982). 5
- I. W. Wu, D. R. Dietderich, J. T. Holthuis, M. Hong, W. V.
- Hassenzahl, and J. W. Morris, Jr., J. Appl. Phys. **54**, 7139 (1983). 6
- D. R. Dietderich, J. Glazer, C. Lea, W. V. Hassenzahl, and J.

W. Morris, Jr., IEEE Trans. Magn. **21**, 297 (1985). 7

 D. R. Dietderich, W. V. Hassenzahl, and J. W. Morris, Jr., Adv. Cryo. Engr. (Materials) **32**, 881 (1986). 8

 D. B. Smathers and D. C. Larbalestier, in *Filamentary A15 Superconductors*, edited by Masaki Suenaga and Alan F. Clark (Plenum, New York, 1980), p.143.

9 D. B. Smathers, K. R. Marken, D. C. Larbalestier, and J.

- 
- Evans, IEEE Trans. Magn. **19**, 1421 (1983). 10 K. R. Marken, S. J. Kwon, P. J. Lee, D. C. Larbalestier, Adv. Cryo. Engr. (Materials) , **32**, 967 (1986). 11 W. Schauer and W. Schelb, IEEE Trans. Magn. **17**, 374
- (1981).
- 12 E. M. Hornsveld, J. D. Elen, C. A. M. van Beijnen and P.
- Hoogendam, Adv. Cryo. Eng. **34**, 493 (1987). 13 J. L. H. Lindenhovius, E. M. Hornsveld, A. den Ouden, W. A. J. Wessel, and H. H. J. ten Kate, IEEE Trans. Appl. Supercond.
- **<sup>10</sup>**, 975 (2000). 14 C. D. Hawes, P. J. Lee, and D. C. Larbalestier, IEEE Trans. Appl. Supercond. 10, 988 (2000).
- 15 C. D. Hawes, M.S. Thesis, University of Wisconsin, 2000. (Available at www.asc.wisc.edu.)
- 16 C. M. Fischer, P. J. Lee, and D. C. Larbalestier, Adv. Cryo. Eng. (Materials) 48, 1008 (2002).

17 C. M. Fischer, M. S. Thesis, University of Wisconsin, 2002. (Available at www.asc.wisc.edu.)

- 18 A. Godeke, B. Ten Haken, and H. H. J. Ten Kate, IEEE Trans. Appl. Supercond. **12**, 1029 (2002).
- 19 A. Godeke, M. C. Jewell, A. A. Golubov, B. Ten Haken, and
- D. C. Larbalestier, Supercond. Sci. Technol. **16**, 1019 (2003). 20 M. Suenaga, A. K. Ghosh, Y. Xu, and D. O. Welch, Phys.
- Rev. Lett. **66**, 1777 (1991). 21 W. A. Fietz and W. W. Webb, Phys. Rev. **178**, 657 (1969). 22 T. P. Orlando, E. J. McNiff, Jr., S. Foner, and M. R. Beasley,
- 
- Phys. Rev. B 19, 4545 (1979).
- 23 H. Wiesmann, M. Gurvitch, A. K. Ghosh, H. Lutz, O. F.
- Kammerer, and M. Strongin, Phys. Rev. B **17**, 122 (1978). 24 A. Godeke, M.C. Jewell, A.A. Squitieri, P.J. Lee and D.C. Larbalestier, submitted to J. Appl. Phys.
- 25 N. Cheggour and D.P. Hampshire, Cryogenics **42**, 299 (2002). 26 L. D. Cooley, P. J. Lee, and D. C. Larbalestier, Adv. Cryo. Eng. 48, 925 (2002).
- 27 See for instance D Dew-Hughes, Philos. Mag. B **55**, 459 (1987), or A. M Campbell and J. E. Evetts, Adv. Phys. **21**, 199 (1972).
- 28 R. M. Scanlan, IEEE Trans. Appl. Supercond. **11**, 2150 (2001).
- 29 See W. V. Pogosov, K. I. Kugel, A. L. Rakhmanov, and E. H. Brandt, Phys. Rev. B **64**, 064517 (2001), and references therein.