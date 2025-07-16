# arXiv:2104.01267

**Paper ID:** bbea9463c6ffc7210262ec4b72d7e3a8

**PDF Path:** apl/Superconductors/arXiv:2104.01267.pdf

**Processing Status:** complete

**Captions Added:** 8

**Generated:** 2025-06-24T13:44:27.519133

---

## **Simulations of Nb3Sn Layer RF Field Limits Due to Thermal Impedance**

Paulina Kulyavtsev, Grigory Eremeev, Sam Posen

### **Abstract**

Nb3Sn performance in RF fields is limited to fields far below its superheating critical field. Poor thermal conductivity of Nb3Sn has been speculated to be the reason behind this limit. In order to better understand the contribution of Nb3Sn thermal conductivity to its RF performance, we simulated numerically with Matlab program\*(based on SRIMP and HEAT codes) the limiting fields under different realistic conditions. Our simulations indicate that limiting fields observed presently in the experiments with RF fields cannot be explained by the thermal impedance of Nb3Sn alone. The results change significantly in the presence of higher losses due to extrinsic mechanisms.

## **Background:**

Nb3Sn is a material of interest due to potential in promising performance and cost reduction. It is implemented as a coating of a thin film inside existing Nb cavities, followed by an annealing in vacuum [1]. The goal is to operate at higher temperatures and higher surface magnetic fields.

Operating temperatures of interest are 2K and 4K. 2K is the current typical temperature for operation of Nb cavities in accelerators, and 4K is a potential operating condition for Nb3Sn which would significantly reduce operating costs due to cooling [1].

Although superconductors are capable of conducting high DC currents without any losses, RF currents in SRF cavities dissipate some heat, typically on the scale of watts, which has to be removed by liquid helium. Heat must go through the wall of the cavity to be dissipated by the helium bath.

The goal of this project was to simulate the process of heat flow from the vacuum side to the He side, and show how it is affected by the thermal impedance of the Nb3Sn layer. By varying the thermal impedance, e.g., via the thickness of the Nb3Sn layer, we estimate the RF field at which thermal runaway occurs under different circumstances. For this project, incremental steps are taken to use the existing HEAT model of Nb and modify it to represent Nb3Sn, and then again for a Nb3Sn-Nb bilayer.

Nb3Sn is known to have significantly poorer thermal conductivity than Nb [2]. Our simulations were designed to improve understanding of the field limit in Nb3Sn cavities to see whether it may be due to this poor thermal conductivity causing thermal instability. Primarily, two cases could be considered in this simulation; one dimensional or Global Thermal Instability (GTI), and two dimensional overheating which can be used to study the effect of defects. In this paper, we focus on GTI, assuming a relatively defect-free surface. While thermal instability is reasonably well understood for niobium, it is not as well addressed for Nb3Sn.

A critical aspect of cavity performance is the ability to remove the heat generated on the inside surface of cavity by cooling the other side. Many factors contribute to the ability to remove this heat: the field on the surface, the surface resistance, the conduction of the heat through the cavity wall, Kapitza conduction at the outer wall-bath interface, as well as the helium bath temperature and helium convection. Thermal instability could result from the collective contribution of these factors in hindering the removal of heat from the RF surface. This is important for the simulation as the impact of these contributions may be indirectly compared to how extreme values of each factor impacts the projected performance and to elucidate which factors in which conditions dominate the contribution to thermal instability. Effects of factors such as Kapitza conductance and thickness of Nb3Sn layer were evaluated through the simulation. In our simulation, the onedimensional analysis was done by assuming the Nb3Sn layer is an infinite plane in a lateral direction, limiting heat transfer to only the z direction.

*Figure 1: Increasing thicknesses of Nb3Sn in a specific field. At some point, there is a failing thickness; global thermal runaway occurs as heat is unable to be removed. This is the thermal runaway critical thickness.*

![](_page_1_Figure_3.jpeg)

**Caption:** The schematic demonstrates the conceptual model used to simulate thermal runaway in Nb3Sn. It emphasizes the progressive increase in thickness of the Nb3Sn RF layer and illustrates the point at which global thermal runaway occurs—termed as the 'critical thickness.' Denoted in a hypothetical superconductor with an inner RF surface experiencing 50 mT, 1.3 GHz RF field, this setup uses a liquid helium bath for cooling. This model links the concept of thermal management with material thickness, highlighting critical parameters for avoiding thermal instability in superconductors.


Our simulation is of an ideal situation, but defects can further complicate the story. The contribution of defects would be simulated by a two-dimensional model looking for the defectinduced thermal instability. A defect, which is more lossy than the surrounding material, becomes an area of excess heating, and this additional heat contribution greatly increases the power dissipation driving thermal runaway; therefore, defects are of interest to simulate to better understand their impact on thermal conductivity and the successful removal of heat from the material.

![](_page_1_Figure_5.jpeg)

**Caption:** This illustration details the HEAT simulation program's mesh used to model defect influence on thermal instability. It places emphasis on a two-dimensional mesh representation, owing to the symmetry of the problem with a focus on the defect—a lossy area causing increased heating—on Nb3Sn surfaces. The model is fundamentally utilized to predict thermal breakdown onset, helping advance our understanding of defect impact on superconductor thermal conductance.


*Figure 2: The model used by the HEAT program to simulate defects using a twodimensional mesh, relying on the symmetry of the problem to simulate the onset of thermal breakdown [*3*].*

**Simulation Method**

The following simulations built on the work of Xie and Meng [4]. First, the program was used to evaluate Nb case. The results were found consistent with published results.

### *Exploration of a Nb3Sn Case*

With the program having been benchmarked against the niobium case, separate cases were added to the existing Niobium-based functions to model Nb3Sn. Specifically, Nb3Sn surface resistance and thermal conductivity were added to reflect the new material under study.

For Nb3Sn thermal conductivity, a polynomial fit was derived from published data [2]. The fit as well as the data from [2] are shown in Figure 3. Using the polynomial fit instead of inline approximation has helped to considerably speed up the calculation time.

![](_page_2_Figure_5.jpeg)

**Caption:** The plot illustrates the thermal conductivity of Nb3Sn as a function of temperature, based on data from previous studies. The x-axis denotes the temperature in Kelvin (T), and the y-axis measures the thermal conductivity in W/m·K. An associated polynomial fit is used in simulations to fast-track resistance calculations in superconducting RF applications. This analysis is crucial as it characterizes the Nb3Sn’s heat conduction properties, essential for improving RF cavity performance by addressing limits arising from thermal impedance.


*Figure 3: Thermal Conductivity vs. Temperature for Nb3Sn material from [*2*]. Polynomial fits to the thermal conductivity from [*2*] were used in the program.*

For BCS resistance, SRIMP was first used to calculate RF surface resistance of Nb3Sn in the relevant temperature range [5]. SRIMP-calculated Nb3Sn BCS resistance was fit with an explicit function that was then used in the numerical simulations for surface resistance calculations to significantly reduce the calculation time, Fig. 4.

![](_page_3_Figure_1.jpeg)

**Caption:** This figure illustrates the surface resistance of Nb3Sn as a function of temperature (T). The black squares represent the BCS surface resistance calculated using the SRIMP model, while the red line is a fit using the equation Rs(T) = 0.001 * e^(49.452/T). The x-axis shows the temperature (T) in Kelvin, ranging from 2 to 20 K, and the y-axis represents the surface resistance in nano-ohms (nΩ), presented on a logarithmic scale from 10^0 to 10^5 nΩ. This graphical representation helps validate the SRIMP calculations with an explicit fitting function used for simulations. The findings emphasize how closely the model fits the calculated data, providing insights into the thermal management capabilities of Nb3Sn, particularly crucial in superconducting applications where precise predictions of surface resistance are necessary for optimizing performance.


*Figure 4: BCS resistance case was written based on the SRIMP calculations, using polynomial fit to the calculated resistance.* 

In Figure 4 we show both Nb3Sn surface resistance, calculated using SRIMP, and the fitting function. The fitting function is:

$$R\_s(T) = A \frac{e^{-\frac{B}{T}}}{T}$$

, where the fitting parameters are A = 1.0∙10-3 Ω∙K, B = 49.452 K.

Calculations that we report here were setup in the following way. For each parameter set we simulated the temperature rise as a function of material thickness. For each applied field, we eventually arrived at the thickness above which no stable solution could be found in our simulations. The maximum thickness at which solution still existed was called the "critical" thickness for the given parameter set. We then find the "critical" thickness for varying parameter sets.

#### **Results**

#### *Kapitza conductance*

One of our first assumptions was that Kapitza conductance is not the main factor limiting Nb3Sn performance, and so we simplified our simulation by excluding it from our simulation. Before removing thermal impedance due to Kapitza boundary resistance from our simulation, we first evaluated Kaptiza contribution to field limits by simulating critical film thickness versus field strength for several different Kapitza resistances. Several different Nb cases were used for this simulation: annealed Niobium, unannealed niobium, and annealed niobium\*1000, which means that the Kapitza resistance for annealed niobium was further increased by a factor of 1000.

![](_page_4_Figure_4.jpeg)

**Caption:** The graph examines the critical thickness as a function of magnetic field for varying Kapitza resistances of annealed and unannealed niobium at 2 K with a consistent residual resistance of 10 nΩ. The x-axis reflects the magnetic field (B) in mT, while the y-axis displays the critical thickness (d) in μm, shown in logarithmic scale. Different markers indicate varying Kapitza resistances: annealed (blue) and unannealed (red). It establishes that the Kapitza resistance minimally impacts the thermal breakdown threshold, illustrating that interfaces between the Nb3Sn surface and liquid He barely impede the heat removal process. This informs efforts to fine-tune Nb3Sn films for achieving optimal superconductor performance.


*Figure 5: Critical thickness as a function of field for different Kapitza resistances. Simulations were done for 2K with residual resistance of 10 nOhm.* 

The results of Kapitza resistance variation is that it has negligible effect. High Kapitza conductance means a very low Kapitza resistance and, if Kapitza boundary played a role in the overall heat transport, we would expect different results for different Kapitza resistances. As shown in Fig.5., varying Kapitza conductance did not affect the results, even when it was increased by three orders or magnitude. This shows that removal of heat from the RF surface is not significantly impeded by the cavity surface - liquid He interface for a typical range of values.

#### *Thermal Conductivity*

To check the impact of thermal conductivity, we simulated the critical thickness for different thermal conductivities at two different helium bath temperatures, 2 K and 4 K. As a baseline thermal conductivity, we used the thermal conductivity from [2], also shown in Fig 3. The critical thickness simulations were then done with thermal conductivity reduced by a factor of 2, increase by a factor of two, and a factor of three. The results are shown in Fig 6.

![](_page_5_Figure_4.jpeg)

**Caption:** This figure presents the simulated critical thickness for varying thermal conductivities at two helium bath temperatures—2 K and 4 K. The x-axis shows the applied magnetic field (B) in mT, and the y-axis displays the critical thickness (d) in μm on a logarithmic scale. Conductivities are varied by a factor of 0.5, 1, 2, and 3, represented in different marker colors: + (red), ∘ (green), ✕ (blue). The results establish that thermal conductivity significantly influences the critical thickness; enhanced thermal conductivity increases the thickness tolerance up to which thermal stability is maintained. This highlights the direct correlation between thermal conductivity and cooling efficiency in superconductor applications.


*Figure 6: Critical thickness simulation results for varying thermal conductivities and two different bath temperatures. Kapitza conductance for annealed niobium increased by a factor of 1000 and the residual surface resistance of 10 nOhm were used.*

As expected, thermal conductivity has significant impact on the thermal impedance. The critical thickness scales approximately linearly with the thermal conductivity at both simulated helium bath temperatures.

To check how the helium bath temperature affects the critical thickness, we simulated the critical thickness as a function of applied field for three different helium bath temperatures, 2 K, 4 K, and 6 K. As was discussed earlier, thermal simulation for varying Kapitza resistances indicated that Kapitza resistance contribution to overall thermal impendence is negligible, so the same Kapitza

resistance of unannealed niobium was used for different outer wall temperatures. The results of this simulation are shown in Fig 7.

![](_page_6_Figure_2.jpeg)

**Caption:** This figure shows simulation results for critical thickness as a function of the applied magnetic field for different helium bath temperatures (2 K, 4 K, 6 K) with a residual surface resistance of 10 nΩ. The x-axis represents the magnetic field (B) in millitesla (mT) and the y-axis shows the critical thickness (d) in micrometers (μm) on a logarithmic scale. The data points for each temperature are highlighted in different colors: 2K (cyan), 4K (red), and 6K (blue). The results indicate that lowering the helium bath temperature can significantly improve the critical thickness, especially at higher fields, suggesting that optimizing cooling conditions can enhance the thermal stability of Nb3Sn films in RF cavities.


*Figure 7: Critical thickness simulation results for different helium bath temperatures. Kapitza conductance of unannealed Niobium and the residual surface resistance of 10 nOhm were used.*

The lower outer wall temperature improves the stability of the material. And the effect of the outer wall temperature increases with the applied field. Decreasing the outer wall temperature from 6 K to 2 K improves the critical thickness by almost a factor of three at higher fields.

Finally, we simulated the effect of the residual resistance on the critical thickness, Fig 8. For this simulation four different surface resistances, 1 nOhm, 10 nOhm, 100 nOhm, and 1000 nOhm, were used. For this simulation Kapitza resistance of annealed niobium increased by a factor of 1000 was used.

For the highest simulated surface resistance of 1000 nOhm, the critical thickness drops to about 60 μm at the highest fields, which is comparable to the thickest Nb3Sn grown on SRF cavities. At low fields of about 100 mT, the critical thickness is still at or above 1 mm.

![](_page_7_Figure_1.jpeg)

**Caption:** The graph depicts the critical thickness (d) as a function of the applied magnetic field (B) for varying residual surface resistances. The x-axis represents the magnetic field (B) in millitesla (mT) ranging from 50 to 350 mT, while the y-axis shows the critical thickness (d) in micrometers (μm) on a logarithmic scale from 10^0 to 10^4 μm. Different colored markers indicate assumptions of residual resistance values: 1000 nΩ (blue), 100 nΩ (green), 10 nΩ (red), and 1 nΩ (purple). The findings demonstrate that an increase in residual surface resistance leads to a substantial decrease in critical thickness. This phenomenon points to the significant impact of surface resistance on the onset of thermal instabilities in Nb3Sn films, underlining the necessity to manage surface resistance to prevent thermal runaway in superconductor applications.


*Figure 8: Critical thickness simulation results for different residual surface resistance. Residual resistance has significant impact on the thermal breakdown.* 

# **Discussion**

With the goal to explore how different material properties affect the maximum material thickness, which still allows for the RF surface to be adequately cooled by liquid helium at the outer wall at a given field, we first check the influence of Kapitza resistance. Under the assumption of the thermal resistance value reported in [2] and the residual surface resistance of 10 nOhm, we simulated the critical thickness vs field for three different values of the Kapitza resistance: annealed, unannealed, and also Kapitza resistance of annealed niobium reduced by a factor of 1000, which represented the case of extremely low Kapitza resistance. The results of the simulation shown in Fig. 5. demonstrated that there is little difference in the critical thickness for all simulated fields between the Kapitza resistance reduced by a factor of 1000 and the other two cases. The simulation results show that Kapitza resistance has little impact on the overall thermal impedance and can be ignored in the simulations.

These simulations also showed that the critical thickness for which thermal runaway occurs is, at least, several millimeter even for a the fields as high as 200 mT, which is significantly higher than the highest fields reached in the best Nb3Sn-coated SRF cavities[6,7]. The typical thickness of Nb3Sn coating on RF surface cavities is 1-3 μm, which in the light of present results significantly thinner than the thickness, which will impact heat transfer to the helium bath.

As we expected the critical thickness scales approximately linearly with thermal conductivity, Fig. 6. For a typical thermal conductivity value [2] the critical thickness far exceeds the typical coating thickness in modern SRF cavities, which is consistent with simulation reported earlier [8]. The thermal conductivity must be suppressed by a factor of 10000 to become the key limiting factor for applied RF fields below 100 mT.

The other factor that has significant impact to the critical thickness is the surface resistance, Fig. 8. The critical thickness drops by about two orders of magnitude when the residual surface resistance is increased from nOhm to 1000 nOhm. Three orders of magnitude may seem as a significant change in the surface resistance, but, in fact, there are many well-known physical mechanisms that can cause such degradation. For example, a normal-conducting inclusion will have a surface resistance of several mOhm. Such small localized defects seem to be one of the likely candidates behind present limitations.

These simulations, where we varied potential contributors to GTI, suggest there is an external limitation, not intrinsic to the material. Perhaps defects are the limitation; the next avenue of exploration would be to look into thermal runaway due to defects varying factors such as defect resistance and defect size. A further advancement to be made on these simulations would be to simulate two materials, Nb3Sn film on Nb, and again evaluate where the thermal runaway occurs.

# **Conclusion**

We numerically simulated different thicknesses of Nb3Sn films in varying realistic conditions using a temperature dependent thermal conductivity. Our simulations indicate that limiting fields observed presently in the experiments with RF fields cannot be explained by the thermal impedance of Nb3Sn alone. We found that the primary contribution to the thermal resistance was the Nb3Sn thermal conductivity. The results change significantly in the presence of higher losses due to extrinsic mechanisms. At the magnitude of microns thickness, GTI is not a concern for state-of-the-art Nb3Sn films. From this, there is interest in further simulation of defects as a twodimensional case and Nb-Nb3Sn bilayers to gain a better understanding of what that external limitation may be.

#### **Acknowledgements**

\*The present simulation is based on the existing simulation, Main Thermal Feedback (MTF), created by Yi Xie (Euclid) and Fanbo Meng (IHEP)[4].

This manuscript has been authored by Fermi Research Alliance, LLC under Contract No. DE-AC02-07CH11359 with the U.S. Department of Energy, Office of Science, Office of High Energy Physics.

# **References**

[1] Structural and superconducting properties of Nb3Sn films grown by multilayer sequential magnetron sputtering, Sayeed et al, Journal of Alloys and Compounds, Vol 800, 2020

[2] Thermal Conductivity of Nb3Sn, GD Cody and RW Cohen, RCA Laboratories, Princeton, New Jersey

[3] RF Superconductivity for Accelerators 2nd Edition, H Padamsee, J Knobloch, T Hayes. 2008 [4] Thermal Simulations for the Multi-Layer Coating Model, Meng et al., Proceedings of SRF2013, 674-677, (2013)

[5] J. Halbritter, Zeitschrift fuer Physik 238 (1970) 466

[6] Analysis of RF losses and material characterization of samples removed from a Nb3Sn-coated superconducting RF cavity, Pudansaini et al, Superconductor Science and Technology, Vol 33 Num 4, 2020

[7] Advances in Nb3Sn superconducting radiofrequency cavities towards first practical accelerator applications, Posen et al, 2020

[8] Simulations of RF Field-Induced Thermal Feedback in Niobium and Nb3Sn Cavities, J Ding, D Hall, M Liepe, SRF2017 doi:10.18429/JACoW-SRF2017-THPB079