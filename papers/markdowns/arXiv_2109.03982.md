# arXiv:2109.03982

**Paper ID:** fd668c0279f6905b3cbb239017493b44

**PDF Path:** apl/Superconductors/arXiv:2109.03982.pdf

**Processing Status:** complete

**Captions Added:** 6

**Generated:** 2025-06-24T13:44:27.534336

---

# **Detecting quench in HTS magnets with LTS wires — a theoretical and numerical analysis**

Rui Kang, Juan Wang, Qingjin Xu

Key Laboratory of Particle Acceleration Physics & Technology, Institute of High Energy Physics, Chinese Academy of Sciences, Beijing 100049, China E-mail: xuqj@ihep.ac.cn; kangrui@ihep.ac.cn

# **Abstract**

Protecting a high temperature superconducting (HTS) magnet from a quench event is a challenging task. Because of the slow normal zone propagation velocity, the detection method by directly monitoring coil voltage may not be timely for HTS anymore, leaving HTS magnets under danger of overheating. Using a NbTi low temperature superconducting (LTS) wire to detect quench in coils wound with ReBCO HTS tapes have recently been experimentally proved, yet a theoretical study is still needed to further develop this technique and make it prepared to be applied more generally in high field magnets. In this manuscript, we have demonstrated that it is the significant difference in the temperature dependence of critical current between LTS and HTS but not the normal zone propagation velocity (NZPV), that makes LTSs good quench detectors. Simulations show that LTS quench detectors should have low matrix fraction or high matrix resistivity. At last, at field up to 15 T or 20 T, Nb3Sn is proven to be a good quench detector.

**Key words:** Quench, LTS quench detector, High temperature superconductors, normal zone

# **I. Introduction**

HTS materials, represented by the ReBCO coated conductor, appear advantages over LTS to be used in magnets generating a magnetic field beyond 15 T, like in controlled nuclear fusion reactors and high energy accelerators [1,2]. However, protecting an HTS magnet from a quench event, which might happen due to a local disturbance or defect, is still a challenging task. As the name indicates, HTS has higher critical temperature (Tc) than LTS, and thus the critical current (Ic) of HTS decreases with temperature much slower than that of LTS, as is shown in Figure 1. The scaling relations for NbTi, Nb3Sn, Nb3Al, MgB<sup>2</sup> and ReBCO to plot figure 1 are respectively taken from [3–7]. The absolute critical current of ReBCO, NbTi and Nb3Sn are scaled to the state-of-art commercial products [8,9]. These relations are also used in the later analysis. Once a normal zone is established, it propagates much slower in HTS than in LTS. Consequently, the long reliable quench detection method by coil voltage may not be timely for HTS anymore. A later quench detection is however fatal to a superconducting magnet, as the large current will keep generating heat at the normal zone. This issue is especially critical in accelerator magnets where conductor current density is generally very high.

![](_page_0_Figure_8.jpeg)

**Caption:** Figure 1 depicts the variation of critical current as a function of temperature and magnetic field for ReBCO and several low temperature superconductors (LTS), including NbTi, Nb3Sn, and MgB2. The visualization shows how the critical current (Ic) of HTS materials decreases with temperature much slower than that of LTS materials. The figure highlights that despite the slower normal zone propagation velocity in HTS like ReBCO, the temperature dependency differences between LTS and HTS make LTS an effective quench detector for HTS systems. Such temperature-dependent diversity allows for reliable quench detection using LTS without over-reliance on fast normal zone propagation.


Figure 1. The variation of critical current as function of temperature and magnetic field for ReBCO and several LTSs. The lower field limit is corresponding to the right boundary. For ReBCO and three candidates for high field quench detectors, the curves at 15 T are drawn with dash lines.

Non-Insulation winding technology is a promising approach to solve the quench protection problem for HTS, but it doesn't suit all magnets. As another route, many different quench detection methods are being developed for HTS, such as by optic fibers [10–12], acoustic signals [13], radio-frequency wave [14], stray-capacitance [15] or even fluorescent thermal imaging[16]. Despite that these methods are not ready to replace quench detection by voltage yet, some of them are only applicable to special magnets because of the physical quantities they measure. Apart from the above-mentioned technologies, a brilliant idea to detect quench in HTS magnets was proposed by Hasegawa et al [17–19], by using another superconductor. In one example, an insulated NbTi wire was attached on a ReBCO pancake coil, and a small monitoring current (~3 A) was applied to the NbTi wire. When the ReBCO pancake coil went to quench, a sharp voltage transition from neglectable to more than 1 V was observed in the NbTi wire, at which time voltage in the ReBCO coil was still below 10 mV. Such an approach doesn't reply on any special medium and is in principle feasible to any kind of superconducting magnet. Similar technique was also suggested more recently by Gao et al [20], for faster detecting quench in a Nb3Sn CCT dipole magnet, with a co-wound Nb3Sn wire.

In this manuscript, a more detailed explanation of why and how a LTS wire can work as a quench detector for HTS is studied. The purpose is generally to answer these questions:

- 1) What is the principle that allows LTS wires to detect quench in HTS magnets?
- 2) What kind of LTS wire can best work as a quench detector?
- 3) Can LTS quench detector be more efficient than detecting quench in HTS directly by its voltage?

As a start, a theoretical discussion is presented in section II by analyzing the characteristics of several superconductors. Then section III presents the simulated results on a stacked ReBCO cable, together with a discussion about what kind of LTS wire can best work as a quench detector for HTS. At last, in Section IV, the application of this technique on the high field, where HTS is destined to play a role, is discussed. Several candidates of high field LTS quench detectors are compared.

# **II. A theoretical explanation**

Concerning quench performance, a general idea about the difference between HTS and LTS is their normal zone propagation velocity (NZPV). However, here it will be demonstrated it is not the NZPV, but the resistance-temperature characteristic that makes LTS a possible sensor for a quench event in HTS.

For practical application, the voltage-current characterization of a superconductor is usually described by the E-J power law [21]. In each superconductor, the superconducting composition can be regarded equipotential with the normal metal matrix [22], and the total current in the two kinds of components should equal to the charged current, which gives:

$$E\_c(\frac{I\_{sc}}{I\_c})^n = \eta\_n \frac{I\_s - I\_{sc}}{A\_n} \\ \text{(1)}$$

In the above equation, E<sup>c</sup> stands for a critical electrical field, regarded as the threshold whether the conductor is in superconducting state. n is the index value. Is, Isc and I<sup>c</sup> are respectively the total current charged to the conductor, current in the superconducting component and critical current of the conductor. η<sup>n</sup> is the resistivity of the metal component, and A<sup>n</sup> is the cross section area of normal metals. The proper Isc that meets the above equation can be find by an iteration method [22], which then gives the right value of electrical field and joule heating. A MATLAB code is developed to do the numerical calculation. Although for HTS and LTS, different E<sup>c</sup> may be selected (1 or 0.1 μV/cm, the former is selected here) and their n index may vary (10-40 for HTS and 30-80 for LTS, see page 370 in [23]. Here 40 is chosen in all cases.), the most important difference comes from their Ic. As is already presented in Figure 1, the temperature dependence of I<sup>c</sup> for HTS and LTS can be very different. As a result, the voltagetemperature behavior can also be completely different.

Figure 2 shows the Resistance (per unit length) versus temperature curves for three typical superconductors. The specifications of these wires are listed in Table I. For NbTi, a very sharp voltage transition happens only few kelvins above the operating temperature, depending on the background magnetic field. This sharp transition comes from losing of superconductivity. Then its resistance continues to grow because of increase of resistivity of copper with temperature. Note the magneto resistance of copper also plays a role. For Nb3Sn wires, the transition happens at higher temperature due to its high Tc. As for ReBCO, voltage doesn't show a sudden transition, but grows up gradually when temperature exceeds the current sharing temperature (Tcs). Such disparity of Resistance-temperature characteristic between HTS and LTS gives an opportunity to detect the quench of HTS by the latter, which doesn't reply on propagation of normal zone, as will be further demonstrated by a numerical analysis in the next section.

![](_page_2_Figure_0.jpeg)

**Caption:** Figure 2 demonstrates the resistance (mΩ/m) variation per unit length as a function of temperature (K), across different magnetic fields, for ReBCO, Nb3Sn, and NbTi. The figure indicates a sharp voltage transition in NbTi a few kelvin above the operating temperature, which is due to the loss of superconductivity. This makes NbTi, with its high magneto-resistive properties, an effective quench detector for ReBCO. Resistance is observed to increase due to the rising resistivity of copper and further illustrates the disparity in resistance-temperature characteristics between HTS and LTS materials, emphasizing how LTS can serve as quench detectors without relying on normal zone propagation.


Figure 2. The variation of resistance (per unit length) as function of temperature and magnetic field for ReBCO, Nb3Sn and NbTi. The operating current (Iop) of ReBCO takes example of the case in section IV. A small monitoring current of 1 A is assumed for both LTSs. The lower field limit is corresponding to the right boundary.

| Table I. Wire specifications corresponding to Figure 2. |                                |                 |               |  |  |  |
|---------------------------------------------------------|--------------------------------|-----------------|---------------|--|--|--|
| Wire type                                               | Materials ratio                | Geometry        | RRR of copper |  |  |  |
| ReBCO                                                   | ReBCO: Cu: Hastelloy = 1:40:50 | 4 mm × 0.091 mm | 100           |  |  |  |
| NbTi                                                    | NbTi: Cu = 1:1                 | Φ 0.8 mm        | 100           |  |  |  |
| Nb3Sn                                                   | Nb3Sn: Cu = 1:1                | Φ 0.8 mm        | 150           |  |  |  |

# **III. Quench simulation of a ReBCO tape stack attached with a NbTi wire**

#### **1. Simulation model**

As an example, this section numerically analyzes the thermal and electrical behavior of a ReBCO tape stack and an attached NbTi wire initially operated at 4.2 K and a background field of 5 T. A thicker copper is assumed in the ReBCO tape to keep a moderate copper current density of 500 A/mm<sup>2</sup> . The related parameters are listed in Table II.

| Wire Type        | Materials ratio      | Geometry        | Copper | Ic at 4.2 K and 5 T | Calculated Tcs with Iop |
|------------------|----------------------|-----------------|--------|---------------------|-------------------------|
|                  |                      |                 | RRR    | [A]                 | [K]                     |
| ReBCO Cable      | ReBCO: Cu: Hastelloy | 4 mm × 0.231 mm | 100    | 4518                | ~10.9                   |
|                  | = 1:180:50           | ×10             |        |                     |                         |
| NbTi detector #1 | NbTi: Cu = 1:1       | Φ 0.8 mm        | 100    | 616                 | ~7.40                   |
| NbTi detector #2 | NbTi: Cu = 1.78:1    | Φ 0.25 mm       | 100    | 77                  | ~7.36                   |

Numerical analysis of an adiabatic superconducting system with two kinds of superconductors insulated with each other can be achieved by solving the below 1-D heat balance equations with finite difference method in MATLAB when the longitudinal length is much longer than its sizes in cross section [22]:

$$A\_l \rho\_l \mathcal{C} p\_l \frac{\partial T\_l}{\partial t} - \frac{\partial}{\partial \mathbf{x}} \left( A\_l k\_l \frac{\partial T\_l}{\partial \mathbf{x}} \right) + \frac{\{T\_l - T\_l\}}{H\_m} = \phi'\_{\text{ext}} + \phi'\_{Ioule} \text{(2)}$$

The subscript i and j indicate ReBCO cable or the LTS detector. For each conductor, A is the cross-section area, ρ is the average density, C<sup>p</sup> is the average specific heat capacity, k is the average thermal conductivity. The average thermal properties of each wire are calculated according to the law of mixture. The third term indicates the heat transfer between the two components, where H<sup>m</sup> is their thermal contact resistance (TCR). The two terms in right are respectively applied heating (to initiate quench in a simulation) and joule heating, which is calculated by solving eq. (1). For two insulated superconductors at DC condition, there is no electrical coupling between them.

A conductor length of 2 m is imagined, which is long enough to avoid boundary effect and requires moderate meshes. Them mesh size is 4 mm and time step is 10 μs. The TCR between the ReBCO stack and the NbTi is initially assumed 0.1 K·m/W, which stands for a good heat conduction at least at low temperatures. This crucial parameter will be discussed later. Heat pulses with different power, distributing over 0.01 m and lasting for 0.01 s, are applied on the center of the HTS stack. The thermal properties of the materials refer the data base of the THEA code [24]. Results show the minimum quench energy (MQE, expressed here in power per unit length) is about 600 W/m.

#### **2. Results and discussion**

![](_page_3_Figure_2.jpeg)

**Caption:** Figure 3 presents the temperature profiles of the ReBCO stack (solid lines) and the attached NbTi wire (dashed lines) at varying times. This time evolution of temperature profiles suggests that there is no faster normal zone propagation in the LTS detector compared to the ReBCO cable, asserting that it is the ReBCO's role as a heat sink coupled with the NbTi's sensitivity to temperature rise that allows it to detect quenches effectively.


Figure 3. Temperature profiles of the ReBCO stack (solid lines) and attached NbTi wire (dashed lines) at different times. The frontier of temperature profiles for both conductor is totally the same, indicating no faster normal zone propagation in the NbTi wire.

Figure 3 shows the evolution of temperature profile over the two conductors with 600 W/m power applied. At a first glance, it is clear that normal zone, or more precisely the frontier of high temperature profile, is not going to propagate faster in the LTS quench detector than in the HTS cable. It is because that the current charged to the LTS detector is so small, that it generates so little heat and can hardly drive the normal zone to propagate. In addition, the ReBCO cable is working as a huge heat sink. Temperature in the LTS detectors increase only because of the heat transfer from ReBCO Cable. Therefore, if a LTS can work as quench detector for HTS, it is definitely not because of its higher NZPV.

Because of the rather high copper current density of the ReBCO stack, the hot spot temperature increases dramatically to above 100 K in 500 ms, while the normal zone, if defined by temperature above the Tcs temperature, only propagates for about 1 m. Note such NZPV is already a rather high value for ReBCO, because of the high current, low operating temperature and large copper content imagined for this example [25]. Figure 4 compares the voltage, hot spot temperature and normal zone length of the two conductors, all shown in dual log axis to see details at early times. Another recovery case is also compared. The voltages of ReBCO stack with the two deposited energy respectively show the typical thermal run away and recovery behavior after the external heat pulse switched off at 10 ms, at which time the voltage is still quite low. In practice, a voltage threshold around 100 mV is usually used to determine a magnet is quenching because of the inevitable noises. According to this value, the quench of the HTS stack can be recognized after 300 ms, despite that a delay time is still needed for validation and switching on the protection system.

The voltage over the LTS wire presented in figure 4(a), on one hand, do start to increase in less than 1 ms when the temperature reaches its Tcs, much earlier than HTS. From figure 4(c), the normal zone length is still at the level of centimeters. This further helps to conclude that it is not fast NPZV make LTS a good quench detector for HTS. Note that the normal zone length in NbTi detector is still longer than that in the ReBCO cable, which is only because of the lower Tcs of the former. On the other hand, in this example the voltage in the LTS wire is quite small (at the level of 10<sup>2</sup> μV). Besides, it is difficult to judge if the HTS stack is going to quench or recovery only by voltage of the LTS wire until after 100 ms. This example seems not demonstrate very well the idea to detection HTS quench with a LTS wire.

![](_page_4_Figure_0.jpeg)

**Caption:** Figure 4 presents the variation of voltage (V), hot spot temperature (K), and normal zone length (m) in the ReBCO conductor and NbTi detector as a function of time (s). The subfigures compared the quench and recovery cases, revealing that the voltage in the LTS sensor, despite having a small current in the LTS wire and large copper cross section, increases in less than 1 ms once the LTS reaches its critical temperature surface (Tcs). This figure illustrates that LTS quench detectors do not primarily rely on fast normal zone propagation velocities to be effective. The significant observation is that the LTS detector's voltage can serve as an early indicator for quench detection faster than traditional HTS monitoring, making them superior for early detection, thus preventing damage due to quenching events.


Figure 4. Variation of voltage, hot spot temperature and normal zone in ReBCO conductor and NbTi detector as function of time. Quench and recovery cases are compared.

The small voltage in LTS sensor is due to the small current in the LTS wire and the rather large copper cross section. In a LTS quench detector, while the small current is obligatory, the copper cross section is redundant. The 0.8 mm NbTi wire imagined in this case contains half fraction of copper, which is the usual case for winding magnet. As a result, even all current goes to copper, the electrical field is still only about 13 μV/cm at 4.2 K and 5 T. It should be noted that for the usual LTS wire, a considerable amount of pure copper is mandatory for stability or quench protection, because the operating current is high. However, when it works as a detector with a very small working current, both of them would not be problems. As a consequence, if a LTS wire is designed to be a HTS quench detector, it should have a small diameter, with a small fraction of copper or with other matrix having higher resistivity, or even get the matrix removed, as was done by Hasegawa. The ideal one would be the superconducting filament itself. Figure 5 shows the voltage in the LTS detector if the NbTi is replaced by a 0.25 mm one (marked as NbTi detector #2) with thickness of copper layer 0.02 mm (so the ratio between copper and superconductor is about 0.56). The voltage in this thin LTS wire significantly increased.

As is mentioned above, the voltage threshold for validating a quench in a magnet is usually set around 100 mV because of the noise signals, which may come from the inductive contribution or the ripple of power supply itself during charging the magnet. While the inductive noises can be much released by techniques like monitoring the voltage difference of two similar coil sections, the one from power supply is not easy to get rid of. On the other hand, the LTS quench detector should be powered a constant small current, so the noises from power supply is also expected to be very small. Even if the magnet generates some inductive voltages in LTS detector, it can be easily cancelled. Therefore, the voltage threshold for detecting quench with LTS wires can in principle be set at a rather low value, as long as it is high enough to be distinguished from a stability event.

#### **3. Effect of TCR between HTS cable and LTS quench detector**

In the early simulations, the TCR is assumed a rather low value. However, this critical parameter may vary over a wide range due to different insulation layers, contact conditions and impregnation or not. As a reference, as Hasegawa et al reported, when a Φ0.3 mm NbTi is attached to ReBCO tape with epoxy, the thermal resistance is around 0.02 Km<sup>2</sup> /W. When accounting the wire diameter, it is equivalent to a TCR around 67 K·m/W. A poor thermal contact condition could be deathful for such quench detection method. In figure 5, the voltage evolution in the NbTi detector #2 with different TCR are compared. It can be noted that, as TCR increases, the presence of sharp voltage transition does delay, but with modest values. Even if the TCR is increased to a large value of 100 Km/W, NbTi detector can still present a sharp voltage at around 15 ms when its temperature exceeds the Tcs, much earlier than the HTS cable reaching 100 mV (about 300 ms). Note that with the high copper current density (500 A/mm<sup>2</sup> ), the hot spot temperature in HTS cable is dramatically increasing. Therefore, LTS quench detector still shows superiority even if it has a relatively poor heat transfer with the ReBCO cable.

![](_page_5_Figure_0.jpeg)

**Caption:** Figure 5 explores the variation of voltage, hot spot temperature, and normal zone in the ReBCO conductor and NbTi detector as a function of time, with different thermal contact resistances (TCRs). This analysis shows that even with high TCR rates, an LTS detector like NbTi can still detect a quench event in a significantly shorter time window compared to the threshold voltage detection in HTS, highlighting the superiority of LTS detectors in detecting quenching due to their sensitivity to temperature changes.


Figure 5. Variation of voltage, hot spot temperature and normal zone in ReBCO conductor and NbTi detector as function of time, with different TCRs.

# **4. High field application**

In practice, ReBCO is most likely to be applied at magnetic field higher than 15 T, which is beyond the capability of NbTi wires. Apart from it, other LTS materials like Nb3Sn, Nb3Al and MgB<sup>2</sup> are also not commonly considered to build magnets at this field region, because of their low Ic. However, they may eventually contribute themselves well as quench detectors. In this section, the feasibility of these materials working as quench detectors for a ReBCO cable operating at 4.2 K and 15 T is studied by simulation. The parameters of ReBCO cable refers a prototype of a novel Roebel structure cable under developing at our lab for high field accelerator magnets, as shown in Table III. Based on the scaling relation mentioned above, the critical current of the cable at 4.2 K and 15 T is about 15.9 kA. Taking Iop/I<sup>c</sup> as 80%, the Tcs of the cable is then about 10 K. As a start point, the quench detector should have some critical current at 4.2 K, and loss superconductivity at around 10 K. Coincidently, the above mentioned three LTS all show such behavior. Table III lists the configuration of LTS wires for this analysis. Nb3Sn refers a standard product from [4,8]. Nb3Al and MgB<sup>2</sup> are much less commercialized than Nb3Sn or NbTi. As a result, two wires are taken for examples from publications [5,6]. Although at 4.2 K and 15 T, the critical current of these three kinds LTS can be significantly different, all of them loss their superconductivity at temperature around 10 K with a slight difference, as earlier shown in figure 1. As discussed above, the resistivity of matrix is another crucial condition for the detection capability. Despite the three kinds of wires may be produced with quiet different matrix, here they are all assumed the same matrix as Nb3Sn wire (pure copper), but the original filling factors are kept. In such a way, the different detection capability would mainly represent the superconducting properties of these superconductors themselves. A good TCR of 0.1 Km/W is assumed for the simulations in this section. Due to a lack of thermal properties of MgB<sup>2</sup> and Nb3Al at different temperatures, the specific heat capacity and thermal conductivity of these two superconductors are also assumed the same as Nb3Sn. Since the cross-section area of ReBCO cable is much larger than these LTS detectors, and a low TCR is assumed, such assumption shall not change the result.

| Table III. Specifications of ReBCO cable and LTS quench detectors relevant to 15 T application. |
|-------------------------------------------------------------------------------------------------|
|-------------------------------------------------------------------------------------------------|

| Wire Type        | Materials ratio      | Geometry        | RRR of copper | Ic at 4.2 K and 15 T | Calculated Tcs with |
|------------------|----------------------|-----------------|---------------|----------------------|---------------------|
|                  |                      |                 |               | [A]                  | Iop [K]             |
| ReBCO Cable      | ReBCO: Cu: Hastelloy | 4 mm × 0.091 mm | 100           | 15900                | ~10.2               |
|                  | = 1:40:50            | ×72             |               |                      |                     |
| Nb3Sn detector   | Nb3Sn: Cu = 1:1      | Φ 0.818 mm      | 150           | 122.6                | ~10.6               |
| Nb3Al detector   | Nb3Al: Cu = 1:1.5    | Φ 0.507 mm      | 150           | 133.4                | ~9.9                |
| MgB2<br>detector | MgB2: Cu = 1:6.63    | Φ 0.83 mm       | 150           | 4.4                  | ~10.2               |

Figure 6 shows the simulated results. All three LTS detectors show sharp voltage transition when hot spot temperature of ReBCO cable is around 10 K, but with a slight difference. Sharp voltage transition first shows in Nb3Al wire at about 4 ms when hot spot temperature at ReBCO cable reaches 10 K. Then after few milliseconds voltage transition occurs in both Nb3Sn and MgB<sup>2</sup> wires, but in MgB<sup>2</sup> wire the transition is a little bit relaxing. These behaviors also agree with their Iop/Ic-T relations. Later, when temperature further increases, three curves show complete the same trend, which comes from the temperature dependent resistivity of copper. It should be noted that although sharp voltage occurs in all three LTS detectors, the absolute values are all at the level of 1e-4 V, even less than the voltage signals in ReBCO cable. This is because of the high fraction of high purity copper matrix are assumed for all of them. Once the matrix is replaced with metals or alloys with higher resistivity or the amount is reduced (like using acid), the voltage signals in LTS detectors can be dramatically increased. As is discussed above, loaded with a small current, these LTS quench detectors don't need large amount of low resistive matrix to ensure stability or quench protection.

![](_page_6_Figure_1.jpeg)

**Caption:** Figure 6 depicts the voltage and hot spot temperature variations in a ReBCO conductor and attached LTS candidates for high field application under assumed good thermal contact. The result shows significant voltage transition in the Nb3Sn wire at 15 T, stressing its potential as a quench detector for high-field ReBCO magnets. Despite a reduction in critical current due to strain, Nb3Sn remains effective, providing evidence for its utility in high-field environments.


Figure 6. Variation of voltage (a) and hot spot temperature (b) in ReBCO conductor and the attached LTS candidates for high field application. Good thermal contact is assumed.

This example takes an operating field at 15 T, and the Tcs of ReBCO cable is happened to be around 10 K. From figure 1, we can see that even at 20 T, these three LTS wires can still carry some superconducting but loss it at a little bit higher temperature. They don't make big differences when working as quench detector. In this sense, Nb3Sn would be the best option as quench detectors for high field ReBCO magnets, since it is commercialized.

Unfortunately, there is a last hurdle Nb3Sn must step over, which is its strain sensitivity. When applied in ReBCO magnets, the Nb3Sn quench detector must be already been through the heat treatment, which means the brittle Nb3Sn wire could be easily degraded during the magnet assembly. Therefore, it is important to figure out if a Nb3Sn can still work if its critical current is greatly reduced. Figure 6 also shows the voltage signal in the Nb3Sn wire with the same condition as the above case, but the critical current is reduced to 20%, which is corresponding to a tension of about 0.65% or a compression of about -0.7% at 4.2 K and 15 T according to the scaling law reported in [4]. Quench of the ReBCO cable could still be recognized, even earlier. The absolute value of I<sup>c</sup> of the LTS quench detectors is not as crucial as when it is used to build a magnet. Actually, the MgB<sup>2</sup> case also sets such an example: the MgB<sup>2</sup> wire chosen for the above discussion has a small critical current of 4.4 A at the imagined condition, but it also work well as quench detector.

# **5. Summary**

The significant difference in temperature dependence of critical current between LTS and HTS, makes LTSs good quench detectors for HTS, and the quench detectivity does not rely much on the normal zone propagation.

The LTS quench detector will be loaded with a small monitoring current, thus is free from stability problem or quench protection. High purity copper matrix should be removed as much as possible or be replaced with high resistive materials to enhance the detectivity of LTS wires.

As such quench detection strongly relies on the heat transfer from HTS cable to LTS detector, their thermal contact resistance (TCR) would have important influence on the quench detectivity. Even though, simulation shows that with a rather poor TCR of about 100 K·m/W, a NbTi wire can still detector a quench event in 20 ms, much earlier than voltage in HTS cable reaching a reasonable detectable threshold (about 300 ms in the simulated case to reach 100 mV).

When ReBCO is applied to high field up to 20 T, Nb3Sn, Nb3Al and MgB<sup>2</sup> could all work as quench detector. As the most commercialized candidate, Nb3Sn seems to be the most promising LTS quench detector at high field. It has also been proven by simulation that the absolute critical current of the LTS quench detector is rarely relevant, so some degradation due to strain could be allowed during assembling, so long the wire is not totally damaged. Note this doesn't mean Nb3Al or MgB<sup>2</sup> have no chances. Comparing to Nb3Sn, Nb3Al can tolerate much higher strain, and MgB<sup>2</sup> has high critical temperature, so they may become irreplaceable quench detection in some cases.

Bases on these results, we'll later test the feasibility of Nb3Sn quench detector on ReBCO insert coils. Besides, a proper voltage threshold or a strategy that can efficiently recognize a quench event will also be studied by both experiment and simulation.

#### **Reference**

[1] Mitchell N, Zheng J, Vorpahl C, Corato V, Sanabria C, Segal M, Sorbom B N, Slade R A, Brittles G, Bateman R, Miyoshi Y, Banno N, Saito K, Kario A, Ten Kate H H J, Bruzzone P, Wesche R, Schild T, Bykovskiy N, Dudarev A, Mentink M G, Mangiarotti F J, Sedlak K, Evans D, van der Laan D C, Weiss J D, Liao M and Liu G 2021 Superconductors for Fusion: a Roadmap *Supercond. Sci. Technol.*

[2] Wang X, Gourlay S A and Prestemon S O 2019 Dipole Magnets Above 20 Tesla: Research Needs for a Path via High-Temperature Superconducting REBCO Conductors *Instruments* **3** 62

[3] Bottura L 2000 A practical fit for the critical surface of NbTi *IEEE Trans. Appl. Supercond.* **10** 1054–7

[4] Ilyin Y, Nijhuis A and Krooshoop E 2007 Scaling law for the strain dependence of the critical current in an advanced ITER Nb <sup>3</sup> Sn strand *Supercond. Sci. Technol.* **20** 186–91

[5] Li G Z, Yang Y, Susner M A, Sumption M D and Collings E W 2012 Critical current densities and *n* -values of MgB <sup>2</sup> strands over a wide range of temperatures and fields *Supercond. Sci. Technol.* **25** 025001

[6] Banno N, Takeuchi T, Tagawa K, Itoh K, Wada H and Nakagawa K 2000 Field and temperature dependences of critical current density in rapid quenched and transformed Nb/sub 3/Al multifilamentary conductors *IEEE Transactions on Applied Superconductivity* **10** 1026–9

[7] Kang R, Uglietti D, Wesche R, Sedlak K, Bruzzone P and Song Y 2020 Quench Simulation of REBCO Cable-in-Conduit Conductor With Twisted Stacked-Tape Cable *IEEE Trans. Appl. Supercond.* **30** 1–7

[8] Western Superconducting Technologies Co,Ltd. http://www.wstitanium.com/

[9] Shanghai Superconducting Technology Co., Ltd. http://www.shsctec.com/index.php?m=list&a=index&classid=51

[10] Salazar E E, Badcock R A, Bajko M, Castaldo B, Davies M, Estrada J, Fry V, Gonzales J T, Michael P C, Segal M, Vieira R F and Hartwig Z S 2021 Fiber optic quench detection for large-scale HTS magnets demonstrated on VIPER cable during high-fidelity testing at the SULTAN facility *Supercond. Sci. Technol.* **34** 035027

[11] Chen B, Li J, Hu Y, Ma H, Wang T, Zhou C and Liu H 2020 Distributed optical fiber sensor for investigation of normal zone propagation and hot spot location in REBCO cables *Fusion Engineering and Design* **156** 111569

[12] Zhou K, Ren L, Shi J, Xu Y, Pu D, Chen G and Tang Y 2020 Feasibility study of optical fiber sensor applied on HTS conductors *Physica C: Superconductivity and its Applications* **575** 1353693

[13] Takayasu M 2019 An Acoustic Quench Detection Method for CICC Conductor Operating in Gas or Liquid *IEEE Trans. Appl. Supercond.* **29** 1–5

[14] Chen B, Hu Y, Li J, Yu B and Fu P 2020 Research on Quench Detection Method Using Radio Frequency Wave Technology *IEEE Trans. Appl. Supercond.* **30** 1–5

[15] Ravaioli E, Davis D, Marchevsky M, Sabbi G, Shen T, Verweij A and Zhang K 2020 A new quench detection method for HTS magnets: stray-capacitance change monitoring *Phys. Scr.* **95** 015002

[16] Gyuráki R, Schreiner F, Benkel T, Sirois F and Grilli F 2020 Fluorescent thermal imaging of a quench in insulated and noninsulated REBCO-wound pancake coils following a heater pulse at 77 K *Supercond. Sci. Technol.* **33** 035006

[17] Hasegawa S, Ito S and Hashizume H 2018 Numerical and Experimental Evaluations of the Quench Detection Performance of an YBCO/Nb–Ti d Tape *IEEE Transactions on Applied Superconductivity* **28** 1–5

[18] Hasegawa S, Ito S, Nishijima G and Hashizume H 2019 Fundamental Evaluations of Applicability of LTS Quench Detectors to REBCO Pancake Coil *IEEE Transactions on Applied Superconductivity* **29** 1–5

[19] Hasegawa S, Ito S, Nishijima G and Hashizume H 2021 Quench Detection Performance of Low-Temperature Superconducting

Quench Detectors for REBCO Tape in Magnetic Fields *IEEE Trans. Appl. Supercond.* **31** 1–5

[20] Gao J, Auchmann B, Hug C, Pautz A and Sanfilippo S 2021 Study of a current-based quench detection method for CCT magnets via a co-wound superconducting sensing wire *IEEE Transactions on Applied Superconductivity* 1–1

[21] Bruzzone P 2004 The index n of the voltage–current curve, in the characterization and specification of technical superconductors *Physica C: Superconductivity* **401** 7–14

[22] Bottura L, Rosso C and Breschi M 2000 A general model for thermal, hydraulic and electric analysis of superconducting cables q 10

[23] Iwasa Y 1995 *Case Studies in Superconducting Magnets: Design and Operational Issues* vol 48

[24] CryoSoft Software - Solids. https://supermagnet.sourceforge.io/solids.html

[25] van Nugteren J, Dhallé M, Wessel S, Krooshoop E, Nijhuis A and ten Kate H 2015 Measurement and Analysis of Normal Zone Propagation in a ReBCO Coated Conductor at Temperatures Below 50K *Physics Procedia* **67** 945–51