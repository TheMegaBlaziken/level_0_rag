# arXiv:1401.3939

**Paper ID:** f150488516cfa1750e670a6991e82700

**PDF Path:** apl/Superconductors/arXiv:1401.3939.pdf

**Processing Status:** complete

**Captions Added:** 13

**Generated:** 2025-06-24T13:44:26.989650

---

# **MODELING HEAT TRANSFER FROM QUENCH PROTECTION HEATERS TO SUPERCONDUCTING CABLES IN NB3SN MAGNETS**

T. Salmi\* , D. Arbelaez, S. Caspi, H. Felice, S. Prestemon, LBNL, Berkeley, CA 94720, USA G. Chlachidze, Fermi National Laboratory, IL, USA H. H. J. ten Kate, University of Twente, Enschede, the Netherlands

#### *Abstract*

We use a recently developed quench protection heater modeling tool for an analysis of heater delays in superconducting high-field Nb3Sn accelerator magnets. The results suggest that the calculated delays are consistent with experimental data, and show how the heater delay depends on the main heater design parameters.

#### **INTRODUCTION**

The quench protection of the present-day high-field Nb3Sn accelerator magnets is based on resistive protection heaters – typically stainless steel–polyimide laminates on the coil surfaces [1]. They bring large segments of the winding to a resistive state during quench, accelerating the magnet current decay and consequently reducing the hotspot temperature. The goal of the heater design is to provide a short heater delay, i.e. the time delay between heater activation and the heater induced quench, and quench a large fraction of the winding. Physical limitations come from the maximum heater voltage and temperature (typically 400 V and 350 K, respectively). The heater insulation thickness (typically between 0.025 and 0.100 mm) required for electrical integrity has a significant effect on the delay time.

In the magnets under development for the LHC HiLumi upgrade, whose length is of the order of 10 m, the heater delay should be in the order of 10 ms, and the heaters should cover at least 60-100% of the coil surface [2][3]. This has been obtained in shorter and/or lower energy R&D magnets (LARP LQ and HQ) [3], but now the increased coil surface area and also requirements for thicker heater insulation to guarantee electrical integrity (increase from 0.025 mm to 0.050-0.075 mm) bring new challenges. Also, LQ and HQ, which had heaters on both the coils inner and outer surfaces, showed that only the outer surface heaters are mechanically reliable. Therefore, significant optimization of the present technology is needed. An additional complexity comes from the need of heating stations for long magnets, making the geometry of the heater non-uniform along the magnet length and adding an additional degree of freedom to the heater design problem.

This paper summarizes a recently developed numerical modeling tool for simulating heat transfer between the heater and coil. The model accounts for the heater geometry and powering, the cable properties, magnetic field and the various insulation materials allowing the evaluation of the heater delay in different conditions. The model is first applied to the LARP HQ magnet [4]. First, the real heater geometry is simulated and the delays are compared with experimental data from [5]. Second, a parametric analysis is used to examine the impact of main heater design parameters on the quench delay. The model is then applied to simulate the protection heaters in the socalled 11 T dipole prototype, built within a CERN and FNAL joint R&D program [6], and the simulated delays are compared with experimental data. Understanding of the impact of the heater design on the quench delay is important for designing the protection for future magnets.

#### **COMPUTATIONAL MODEL**

#### *Thermal model*

The heat transfer between the heater and the cable is simulated using a numerical two-dimensional heat conduction model, with joule heat generation in the stainless steel component to simulate heater powering. In this approximation the heat propagation between neighboring turns is neglected. At the present stage of development, current sharing between the strands and quench propagation due to Joule heating in the cable is also not simulated.

The two-dimensional heat equation describing the thermal propagation is

$$\gamma\_m c\_{p,m} \frac{\partial T}{\partial t} = \frac{\partial}{\partial \mathbf{y}} \left( k\_m \frac{\partial T}{\partial \mathbf{y}} \right) + \frac{\partial}{\partial \mathbf{z}} \left( k\_m \frac{\partial T}{\partial \mathbf{z}} \right) + \mathbf{f}\_{\text{gen,ss}} \quad , \qquad (\text{l})$$

where *T = T(z,y,t)* is temperature (K), *cp,m = cp,m(T,B)* is specific heat (J/K/kg), *γ<sup>m</sup>* is mass density (kg/m3 ), *km = km(T,B)* is thermal conductivity (W/K/m) of the material *m* at the location (*z, y)* at time *t* (s) and *fgen,ss* = *fgen,ss(t,T)* is the internal volumetric heat source applied only in stainless steel component. (W/m3 ). It is defined using

$$f\_{gen,ss} \text{ (t,T)} = \rho\_{ss}(T) f\_{ss}^2(\mathbf{t}),\tag{2}$$

where *Jss(t)* is the heater current density (A/m2 ) and *ρss(T)* is the stainless steel electrical resistivity (Ωm), or using

$$f\_{gen,ss} \text{ (t)} = P\_{PH} \text{(0)} / d\_{ss} \, e^{-\frac{2\Gamma}{\pi}} \,, \tag{3}$$

where *PPH(0)* (W/m2 ) is the heater adiabatic peak power defined by dividing the heater power by the heating surface area [1], *dss* (m) is the stainless steel thickness, and τ is a time constant of an exponential heater current

decay. \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ \*T. Salmi is now with Tampere University of Technology, Tampere, Finland.

A heater on the coil straight section typically has a periodical geometry (see Fig. 1). Due to the symmetry, each turn can be represented by modeling only half of the heater period, when adiabatic boundary conditions are assumed at the center and at the end of the period (*z* = 0, and at *z* = *PH period/2*). Figure 2 shows a case with one heating segment at the center of the heater period. The boundaries at the top and bottom of the system, i.e. at *y* = 0, or at *y* = *H*, are at fixed temperature, *Tbath*.

#### *Material properties and magnetic field*

The various insulation layers as well as the cable and heater dimensions are taken into account using regions of different material properties. The different layers are assumed in perfect thermal contact. The layers dimensions and materials are an input parameter.

![](_page_1_Figure_3.jpeg)

**Caption:** Shown is a schematic illustrating a generic heater geometry applied to superconducting magnet coils. The diagram highlights periodical coverage of the heater along different turns, indicating areas of heater placement and expected heat influence zones (PH period, PH coverage). This layout underpinning the model reflects the periodic nature of heat application in quench protection systems, important for designing effective heater configurations in terms of both thermal management and electrical integrity.


Figure 1: A schematic view showing how generic heater geometry can be expressed in terms of periodical heater coverage at different turns.

![](_page_1_Figure_5.jpeg)

**Caption:** The thermal model visualizes the thermal transport mechanisms in a superconducting coil turn, capturing the longitudinal and radial heat flow paths through the cable's wide side. The model bears significant implications for understanding thermal dynamics under quench protection scenarios, offering insights into how heat spreads from heaters within the coil structure and affects quench delays.


Figure 2: Thermal model for half period of the protection heater geometry, representing the longitudinal and radial (through wide side of the cable) thermal transport in one coil turn.

The material thermal properties are functions of local temperature and magnetic field. The copper properties and epoxy specific heat are from [7] (with linear extrapolation to 0 J/kg/K at 0 K for epoxy specific heat below 4.4 K), Nb3Sn specific heat is a fit from [8], G10, polyimide (Kapton) and stainless steel properties are from [9] (with extrapolation for Kapton thermal conductivity below 4.3 K [8], and stainless steel specific heat below 5 K [10]). The stainless steel resistivity is based on [11].

The cable is a homogeneous block with properties averaged over its constituents (copper, Nb3Sn, epoxy and/or G10) volume. Thermal conductivities of Nb3Sn and epoxy are assumed negligible relative to that of copper. By default, the magnetic field in the cable crosssection is uniform, and it is an input parameter. The model allows also simulating variable field profile across the cable. In that case the current sharing temperature, *Tcs*, varies at different cable location, and the material thermal properties are based on the average field.

#### *Quench delay determination*

The simulation begins with the powering of the heaters and the quench delay is defined once the cable temperature exceeds *Tcs(B,I),* i.e. the temperature at which the current in the cable is equal to the (temperature and magnetic field dependent) critical current. The model offers two possibilities for fitting the critical surface, Godeke [12] [13] and Summers [14].

#### *Numerical solution*

The numerical solution is based on the thermal network method [15] with explicit finite difference discretization scheme [16] and adaptive time stepping. Several elements in each layer are needed to guarantee numerical stability and accuracy. The segments size is an input parameter. The correct implementation of the equations was verified by comparison with analytical solution of a case in 1-D heat conduction in an insulated slab with steady surface heat flux and constant and uniform material properties.

## **SIMULATION OF THE HQ HEATER IN THE HQ01 QUADRUPOLE**

As the first study case, the model is applied to the LARP HQ magnet, which is a 1-m-long 120-mm-aperture quadrupole based on cosθ geometry with two layers [4]. The outer layer heater implemented in the coils is modeled, and the simulated heater delays are compared with experimental data from the HQ01e tests [5]. Then, the impact of individual heater design parameters on the quench delay is examined using a parametric analysis. The used coil parameters are shown in Table 1, and the field map in Fig. 3. In the next sections the used parameters for both studies are detailed.

#### *Simulation of the HQ heater geometry*

The HQ outer layer heater has a wavy shape, providing partial coverage at several turns. One period of the geometry is shown in Fig. 4. It shows that the heater coverage increases from about 2 cm to 7 cm in approximately 1 cm steps in turns 2nd to 7th (counted from the outer layer (OL) pole).

![](_page_2_Figure_0.jpeg)

**Caption:** This image shows the HQ01 superconducting magnet's outer layer coil's magnetic field distribution map at a current of 14.7 kA, emphasizing the variation in magnetic fields across different turns from #2 to #14. The color scale denotes the magnetic field strength in Tesla. This field map is critical for visualizing where the magnetic forces peak, affecting the quenching behavior and the prioritization of heater application in these regions. Such information is pivotal in developing quench protection strategies tailored to specific regions in the coil where magnetic interactions are most pronounced【8:10†temp_paper】.


Figure 3: HQ01 field map and notation of the turn count from outer layer pole.

Table 1: Simulation parameters for the LARP HQ simulation

| Parameter            | HQ01 (Coil 9)       |
|----------------------|---------------------|
| SSL@ 1.9 K (kA)      | 19.31               |
| SSL @ 4.4 K (kA)     | 17.52               |
| Bpeak, at I [18]     | 0.9505<br>0.00127×I |
| #strands             | 35                  |
| Copper RRR           | 190                 |
| Strand Cu/SC         | 1.05                |
| Cable voids          | 12% epoxy           |
| Cable width (mm)     | 15.00               |
| Cable ins. (mm)      | 0.090 (G10)         |
| Bottom ins. (mm)     | 0.708 (G10)         |
| Top ins. (mm)        | 0.30 (G10)          |
| Stainless steel (mm) | 0.025               |
| PH ins. Kapton (mm)  | 0.0254              |
| Strip path (mm)      | 2220.0              |
| Strip width (mm)     | 11.0                |

After the 7th turn, the continuous heater coverage is smaller than 7 cm. As the heater coverage increases while moving away from the pole, the magnetic field decreases (Fig. 3). Higher field and longer coverage are assumed to compete in reducing the delay, so the location of the first heater-induced quench is not obvious. In the experiment the first quench can be located between turns 2 and 14 based on voltage tap signals, but it is not known in more detail. Here it is assumed to occur in one of the turns from 2nd to 7th and the heater delay is simulated at each of these turns. The shortest quench delay among the modelled turns is chosen for the comparison with experimental data.

![](_page_2_Figure_5.jpeg)

**Caption:** This figure portrays one period of the HQ01 heater configuration on the coil's outer surface. It details the segmental heater coverage along with periodic geometry between turns 2 and 7, essential for calibrating the parametric impact of heater design on thermal delays. The representation aids in identifying optimal heater placement strategies for efficient thermal disruption in Nb3Sn superconducting magnets during quench events.


Figure 4: One period of the HQ01 heater on the coil outer surface. The *PH coverage* (the length of the cable continuously covered by the heater at each turn) in one *PH period* (periodical heater geometry) is shown for the 2nd and 7th turn.

The magnetic field strength is calculated at the coil outer surface (using Cobham Vector field Opera-2D [17]), which is the location closest to the protection heater and this value is used for the whole turn. The fields in turns from 2nd to 7th normalized to the magnet peak field at a given current are respectively 0.75, 0.74, 0.72, 0.70, 0.69 and 0.66.

The Nb3Sn critical surface is calculated using Godeke fit with parameters from HQ coil 9 extracted strand measurements [18]. The calculated *Tcs* varies from 14.2 to 14.4 K at 5 kA and from 9.6 to 10.4 K at 14 kA. The heater power is defined by 230 V over the 2220 mm long strip, giving *Jss* = 210 A/mm2 , which gives a heater power *PPH*(0) about 50 W/cm2 The current decays according to a time constant of 40 ms (defined from the measured current decay profile).

#### *Parametric study*

In the parametric study, we modeled the outer layer 2nd turn at 1.9 K, and magnet current 80% of the short sample limit (15400 A). The computed conductor field is 9.1 T, and *Tcs* is 8.9 K.

The varied parameters are the heater power, the Kapton thickness, and the heater coverage. If not otherwise mentioned, in the parametric analysis the heater power *PPH* is 50 W/cm2 and constant (step function), the heater covers the whole turn, and the Kapton thickness is 0.025 mm.

#### **HQ01 SIMULATION RESULTS**

#### *Comparison with experimental data*

The HQ heater simulation at different turns shows that the delays increase from about 5 to 40 ms when decreasing the magnet current from 80% of short sample limit to 20% (see Fig. 5). The case with infinite heater coverage (1D), at magnet peak field (*B*/*Bpeak* = 1.0) is also shown, and as expected, the delays converge to that when increasing heater coverage or field fraction. The variation between the turns is larger at lower current and the turn that quenches first depends on the current.

![](_page_3_Figure_1.jpeg)

**Caption:** This is a plot of delay time versus magnet current/Short sample limit (%) at different turns and temperatures (4.4 K and 1.9 K) for a superconducting magnet. The x-axis shows the magnet current as a percentage of the short sample limit, and the y-axis represents the delay time in milliseconds. Different lines indicate heater delay behaviors at various coil turns and temperatures, illustrating how reductions in current increase the heater delay, particularly at lower temperatures and different coil turns. These results suggest that the delay is influenced by current, temperature, and coil position, which are critical factors for optimizing heater performance in superconductor quench protection setups.


Figure 5: HQ heater delays simulated at several outer layer coil turns. The solid lines represent operation at 1.9 K and dashed lines at 4.4 K.

The simulation agrees with experimental data within 20%, as shown in Fig. 6, where the shortest delays at each current are plotted together with the experimental data. The impact of the operation temperature on the delays is only a few percent in both the simulation and experiment. Excluding the longest simulated delay times (where the heat diffusion away from the hotspot plays a larger role), this difference is approximately proportional to the difference in the energy margins to quench (i.e., integration of the cable heat capacity from *Tbath* to *Tcs*) at each current.

![](_page_3_Figure_4.jpeg)

**Caption:** This graph illustrates the modeled and experimental HQ heater delays at 1.9 and 4.4 K against normalized magnet current, revealing a correlation between delay times and operational conditions. The x-axis denotes magnet current as a percentage of the short sample limit, while the y-axis indicates delay time in milliseconds. The close alignment of modeled results with experimental data underscores the precision of simulations in predicting real-world behaviors of quench protection systems, facilitating effective design optimizations.


Figure 6: Modeled and experimental (Exp.) HQ heater delays at 1.9 and 4.4 K versus normalized magnet current.

### *Heater delay vs. heater power*

As expected, larger heater power reduces the simulated delays, as shown in Fig. 7. Saturation is visible around 30 W/cm2 . Increasing the power further has only a small effect on the delay. The curve shape is consistent with experiments [1].

![](_page_3_Figure_8.jpeg)

**Caption:** This figure presents the simulated delay time as a function of peak heater power (PH power) in watts per square centimeter, showing a decrease in delay time with increasing heater power, plotted here at a simulation I/Iss of 80%. The x-axis indicates the heater power, while the y-axis shows delay time in milliseconds. As heater power increases to about 30 W/cm², the delay time significantly decreases before reaching a plateau, indicating diminishing returns past this point. This saturation suggests that further increases in power do not substantially enhance quenching efficacy. This result is critical for optimizing heater power settings to achieve efficient quench protection without unnecessary energy expenditure【8:3†temp_paper】.


Figure 7: Heater delay time vs. heater peak power. The heater power is a step function in time.

#### *Delay time vs. insulation thickness*

The increase in the simulated delay when increasing the polyimide thickness is shown in Fig. 8. The delay approximately doubles when the thickness is increased from 0.025 mm to 0.076 mm. Comparison of experimental data from HQ01e (0.025 mm Kapton), and from HQ coil 15 (0.076 mm Kapton), which was tested in the HQM04 mirror structure, shows an increase in the experimental delay approximately 130%, in agreement with the simulated value.

#### *Delay time vs. heater geometry*

The simulation shows that longer heater coverage leads to shorter delays – up to saturation around 20 mm, when the delay approaches 7 ms indicating a local 1-D heat transfer (fully covered cable) (see Fig. 9). At coverage of 5 mm, the delay is more than doubled. Variation of the period between 50 and 180 mm changed the result less than 5% with respect to the reference case with 120 mm long period.

Longer delay for the same short sample fraction was also found in the LARP LQ magnet, which had shorter heater coverage than HQ [3].

![](_page_4_Figure_0.jpeg)

**Caption:** This figure illustrates the relationship between heater delay time and Kapton thickness ranging from 0 to 0.25 mm, using a simulation model with an I/Iss of 80%. The x-axis shows the Kapton thickness in millimeters, and the y-axis represents the delay time in milliseconds. The plot demonstrates a clear trend of increasing delay time as the Kapton thickness rises, highlighting the impact of thermal insulation on quenching delays in superconducting magnet protection heaters. This finding underscores the need to optimize insulation thickness to minimize delay time while maintaining electrical insulation integrity in high-field Nb3Sn accelerator magnets .


Figure 8: Heater delay vs. Kapton thickness.

![](_page_4_Figure_2.jpeg)

**Caption:** This figure plots the delay time against the length of the heater coverage (in mm) with a simulation setting of I/Iss = 80%. The x-axis represents the length of the covered cable segment by the heater, while the y-axis shows the delay time in milliseconds. As the length increases, the delay time decreases, approaching saturation at around 20 mm coverage. This behavior indicates a transition to a local one-dimensional heat transfer when coverage is extensive, pointing to the need for optimal heater coverage to balance efficiency and material costs. Understanding this relationship is essential for designing heater geometry with a maximized protection area while minimizing delay times【8:7†temp_paper】.


Figure 9: Heater delay time vs. length of the covered cable segment.

## **SIMULATION OF THE HEATER IN THE 11 T DIPOLE**

Contrary to HQ, the heater strips in the 2-m long socalled 11 T dipole model (MBHSP01) are straight strips parallel to the magnet axis. Therefore the problem is essentially in 1-D, when the heat transfer between the coil turns is neglected.

The heaters have been tested with Kapton thickness of 0.076 mm and 0.203 mm, i.e. much larger than HQ. Therefore the simulation of the 11 T dipole heaters allows applying the model to a quite different regime. The adhesive (0.038 mm) that is used to glue the stainless steel heater to the Kapton has been neglected in the simulation. The 11 T dipole magnet development and test are described in [6] and the protection heater experiments in [19].

#### *Heater power*

On the outer surface of each coil half, two straight heater strips form a U-shape and are connected in series (see Fig. 10 [19]). The strip closer to the central pole piece (high field region) is 26 mm wide, and the strip closer to the magnetic midplane (low field region) is 21 mm wide. Two U-shapes are connected in parallel and powered by a capacitor discharge in one Heater Firing Unit (HFU). The capacitance is 9.6 mF and the measured cold resistance of the circuit is 2.6 Ω, giving RC-time constant of 25 ms.

The calculated heater current is 77 A for a voltage of 400 V. Using equation (1) and multiplying by the stainless steel thickness we get a peak power of about 17 W/cm2 in the high field heater and 27 W/cm2 in the low field heater, using the stainless steel 304 resistivity (490 nΩm @ 4.2 K) . The calculated resistance of both heaters together is 3.4 Ω. However, the measured resistance of the U-shaped heater is 20% larger, 4.2 Ω. Partial explanation is that the heaters in the 11 T dipole are based on stainless steel 316 L, which has about 5% higher electrical resistivity. Assuming that the measurement gives the correct resistivity (and for example the connection in between the strips or irregularities in the heater shape do not impact), the heater power is 20 W/cm2 in the high field heater, and 31 W/cm2 in the low field heater. In the simulation we use the average of these: 18.5 W/cm2 in the high field and 29 W/cm2 in the low field region.

![](_page_4_Figure_11.jpeg)

**Caption:** This schematic depicts the heater connection scheme for the 11 T dipole, featuring PH-1L and PH-2L configurations which indicate insulation thickness due to Kapton layers. The two-dimensional representation outlines the U-shaped heater strips connected in series, which are crucial for distributing heat across the magnet to prevent domain-specific heat build-up. Such insights are vital for designing effective and safe quench protection measures in large-scale superconducting devices.


Figure 10: 11 T dipole heater connection scheme. PH-1L and PH-2L refer to heater inslualtion thickness with 1 or 2 layers of Kapton [19].

#### *Magnetic field and cable properties*

Under each heater, the first quench is expected to initiate at the coil turn that has the highest magnetic field. The high field and low field heater were considered separately, and the delay was simulated in the turns #2 and #19 from the outer layer pole (see Fig. 11 [20]).

The choice of field value for the turn #2 is not straight forward because the field on the coil outer diameter (OD) is only 78% of the maximum field of that conductor (see Fig. 10). We therefore considered three cases. In Case 1, field was taken at the coil outer surface (65% of the magnet peak field). In Case 2, field was taken as the maximum field in the conductor (82% of the magnet peak field). And, in Case 3, the field profile varies across the conductor (1-D projection of the 2-D field map in the cable cross-section). In the turn #19, the field at the coil surface is the same as the cable maximum field (42% of the magnet peak field), so simulations were done only for this field value.

The HQ simulation corresponds to the Case 2. The field location in the 11 T simulation is more critical for two reasons: First, in HQ the cable outermost field was 87- 95% of the maximum field. Second, in 11 T the expected delays are longer due to smaller heater power and thicker insulation between the heater and cable. The longer delays increase the impact of all factors, including the field. One should keep in mind that while tuning the field location may be useful for finding the best expectation for the experimental results, it may give a false sense of accuracy because the anisotropic cable internal structure (strands' paths) is still not modelled.

The critical surface is based on the Summers fit, using Bc20 = 24.8 T, Tc0 = 16.5 K, C = 9.08×10<sup>3</sup> . Other simulation parameters are summarized in Table 2.

![](_page_5_Figure_3.jpeg)

**Caption:** Illustrated here is the field map of an 11 T dipole system, showing variations of magnetic field strengths across different regions of the coil. It provides critical insights into the locations of peak magnetic fields, influencing the operational dynamics of protection heaters. Understanding these field distributions is crucial for assessing heater efficacy and ensuring effective quench induction within superconducting systems.


Figure 11: 11 T dipole field map [20].

Table 2. Simulation parameters for CERN-FNAL 11 T dipole simulation

| Parameter            | 11 T               |
|----------------------|--------------------|
| SSL@ 1.9 K (kA)      | 15.4               |
| SSL @ 4.4 K (kA)     | 13.8               |
| Bpeak, at I [18]     | 0.9062<br>0.0023×I |
| #strands             | 40                 |
| Copper RRR           | 100                |
| Strand Cu/SC         | 1.13               |
| Cable voids          | 6.5% epoxy,        |
|                      | 6.5% G10           |
| Cable width (mm)     | 14.88              |
| Cable ins. (mm)      | 0.200* (G10)       |
| Bottom ins. (mm)     | 0.706 (G10)        |
| Top ins. (mm)        | 0.64 (Kapton)      |
| Stainless steel (mm) | 0.025              |
| PH ins. Kapton (mm)  | 0.076 / 0.203      |
| Strip path (mm)      | 2100.0             |
| Strip width (mm)     | 26.1               |
|                      |                    |

\* This refers to the insulation between the bare cable and the polyimide of the PH insulation. In the 11 T magnet 0.2 mm includes a 0.1 mm glass sheet that is impregnated on the coil surface.

#### **11 T DIPOLE SIMULATION RESULTS**

The heater delays were measured between 40 and 60% of SSL. Simulations in general show a good agreement with results, giving (i) much longer delays for thicker polyimide and (ii) the correct slope of delay increase at lower currents. At 80% of SSL at 1.9 K the heater delay is expected to be about 55 ms with 0.203 mm Kapton, and 25 ms with 0.076 mm Kapton. The 20% predicted increase in the simulated delays from 1.9 to 4.5 K is not seen in the experiment.

The simulated delays at 1.9 K agree the best with the experimental data for the high field heater when the utilized field was the maximum in the cable (Case 2). The agreement is within 20% for both thicknesses when above 50% of SSL at 1.9 K. The delays using the realistic field profile (Case 3) are about 10-30% longer than the delays with the maximum field. When the field is taken at the coil OD (Case 1), the delay is at least 60% longer than with the maximum field. The delays under the low field heater were about 50-150% longer than the shortest delays under the high field heater. Figures 12 and 13 show the results in the Cases 2 (Bmax) and 3 (Bprof) of the high field heater.

![](_page_6_Figure_0.jpeg)

**Caption:** This graph depicts the experimental and modeled delay times for heater-induced quenches in a 11 T Nb3Sn dipole, characterized by 0.076 mm Kapton insulation, over varying magnet currents (as a percentage of short sample limit, SSL). It compares the behavior under different field configurations (Bmax and Bprof) at temperatures of 1.9 K and 4.6 K. The x-axis is the magnet current over SSL percentage, while the y-axis is the delay time in milliseconds. The consistency between experimental results and modeled predictions, particularly in high magnetic fields, illustrates the model's capacity to reliably simulate delays, which is important for enhancing future quench protection measures【8:4†temp_paper】.


Figure 12: Heater delays in the 11 T dipole, simulated and measured, for the 0.076 mm Kapton thickness.

![](_page_6_Figure_2.jpeg)

**Caption:** This chart shows the simulation and measurement results of heater delays in an 11 T dipole with a Kapton thickness of 0.203 mm. The x-axis represents the magnet current normalized to the short sample limit (SSL), while the y-axis measures the delay time in milliseconds. The two lines highlight the effect of different field conditions (Bmax and Bprof) and temperatures (1.9 K and 4.5 K) on the quench delay—simulations and experimental data align closely within a 20% difference for higher SSL percentages, emphasizing the reliability of the computational model. This alignment underlines the model's usefulness in predicting quench delays under varying operating conditions.


Figure 13: Heater delays in the 11 T dipole, simulated and measured, for the 0.203 mm Kapton thickness.

#### **CONCLUSION**

A computational tool based on a two-dimensional heat conduction model is developed to calculate the protection heater delay time to induce a quench as a function of a large amount of parameters, which include cable properties, magnet operation conditions, and heater geometry, powering and insulation scheme.

The modeling tool is applied to simulate heater delays in the LARP Nb3Sn quadrupole magnet called HQ01e and in the FNAL-CERN Nb3Sn dipole magnet called 11 T. The agreement between the simulation, which does not use any free parameters, and experimental data is within 20% in most cases. A parametric analysis using the HQ01e data showed the heater delay dependence on heater power, polyimide thickness and heater geometry.

This relatively simple modeling approach can be useful in understanding the effect of various parameters on the quench delay time, which is important for optimizing the heater design for future high-field Nb3Sn accelerator magnets.

#### **ACKNOWLEDGMENT**

This work has been supported by the U. S. Department of Energy, under Contract No. DE-AC02-05CH11231. Finalizing this paper was supported by Stability Analysis of Superconducting Hybrid magnets (Academy of Finland, #250652). The authors wish to thank Hugo Bajas and Jerome Feuvrier (CERN) for the heater delay measurements in HQ01e. We thank Fred Nobrega (FNAL), Bernhard Auchmann (CERN) and Emanuela Barzi (FNAL) for providing information for the 11 T simulation. I also gratefully acknowledge the support and useful advice on the work from Ezio Todesco (CERN) and Antti Stenvall (Tampere University of Technology), who also provided Figure 1. I have also received helpful input from Maxim Marchevsky, and Matthijs Mentink (LBNL). I also thank Ray Hafalia (LBNL) for grammar corrections in an earlier draft of the paper (the new mistakes in the present version are not his fault).

#### **REFERENCES**

- [1] H. Felice et al., Instrumentation and Quench Protection for LARP Nb3Sn Magnets, IEEE Trans Appl Supercond vol.19, no.3, pp. 2458-2462, June 2009.
- [2] L. Imbasciati, Studies of Quench Protection in Nb3Sn Superconducting Magnets for Future Particle Accelerators, PhD thesis and Fermilab TD-03-028, 2003.
- [3] T. Salmi et al., Quench Protection Challenges in Long Nb3Sn Accelerator Magnets, Trans. of the Cryogenic Engineering Conference, Vol 57, pp. 656- 663, June 2011.
- [4] H. Felice et al., Design of HQ A High Field Large Bore Nb3Sn Quadrupole Magnet for LARP, IEEE Trans Appl Supercond vol.19, no.3, pp. 1235–1239, 2009.
- [5] H. Bajas et al., Test Results of the LARP. HQ01 Nb3Sn Quadrupole Magnet at 1.9 K, presented at ASC2012.
- [6] A. V. Zlobin et al., Development and Test of a Single Aperture 11 T Nb3Sn Demonstrator Dipole for LHC Upgrades, IEEE Trans Appl Supercond vol. 23, no 3, p 4000904, 2013.
- [7] CRYOCOMP(©copyright [Eckels](http://www.users.qwest.net/%7Evarp/authors.htm%23PEckels) Engineering Inc.).
- [8] G. Manfreda, Review of ROXIE's Material Database for Quench Simulations, CERN internal note 2011- 24, EDMS Nr: 1178007.
- [9] E.D. Marquardt et al., Cryogenic Material Properties Database, in Proc. 11th Int. Cryocooler Conf., Keystone, 2000, available at: http://cryogenics.nist.gov/ MPropsMAY/material%20properties.htm.
- [10] A. Davies, Material properties data for heat transfer modeling in Nb3Sn magnets. Available at:

http://www.illinoisacceleratorinstitute.org/2011%20P rogram/student\_papers/Andrew\_Davies.pdf.

- [11] S. Prestemon (LBNL), private communication.
- [12] A. Godeke et al., General Scaling Relation for the Critical Current Density in Nb3Sn, Supercond. Sci. Technol 2006, 19.
- [13] D. Arbelaez et al., An Improved Model for the Strain Dependence of the Superconducting Properties of Nb3Sn, Supercond. Sci. Technol 2009, 22.
- [14] L. T. Summers, et al. A model for the prediction of Nb3Sn critical current as a function of field, temperature, strain and radiation damage, IEEE Trans. Magnetics, 27,1.
- [15] Y. Ҫengel, Heat Transfer A Practical Approach, McGraw-Hill Education, 2003.
- [16] T. Blomberg, Heat Conduction in Two and Three Dimensions: Computer Modelling of Building

Physics Applications, PhD Thesis, Lund University, Department of Building Physics, Sweden, 1996.

- [17] Opera 2D, website: http://www.cobham.com/aboutcobham/aerospace-and-security/about-us/antennasystems/specialist-technical-services-andsoftware/products-and-services/design-simulationsoftware/opera/opera-2d/staticelectromagnetics.aspx.
- [18] A. Godeke et al., A Review of Conductor Performance for the LARP High-Gradient Quadrupole Magnets, to be published in SUST.
- [19] G. Chlachidze et al., Quench Protection Study of Single-Aperture Nb3Sn Demonstrator Dipole for LHC Upgrades, IEEE Trans on Appl Supercond vol. 23 no 3, 2013.
- [20] B. Auchmann (CERN), private communication.