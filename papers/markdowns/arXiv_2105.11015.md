# arXiv:2105.11015

**Paper ID:** 877e31cc87cf3a6933d3375a8abb3667

**PDF Path:** apl/Superconductors/arXiv:2105.11015.pdf

**Processing Status:** complete

**Captions Added:** 4

**Generated:** 2025-06-24T13:44:27.523348

---

#### **A high-luminosity superconducting twin** + <sup>−</sup> **linear collider with energy recovery**

# **V. I. Telnov**,

*Budker Institute of Nuclear Physics, Novosibirsk, Russia*

*Novosibirsk State University,*

*Novosibirsk, Russia*

*E-mail:* [telnov@inp.nsk.su](mailto:telnov@inp.nsk.su)

Abstract: Superconducting technology makes it possible to build a high energy + − linear collider with energy recovery (ERLC) and reusable beams. To avoid parasitic collisions inside the linacs, a twin (dual) LC is proposed. In this article, I consider the principle scheme of the collider and estimate the achievable luminosity, which is limited by collision effects and available power. Such a collider can operate in a duty cycle (DC) and in a continuous (CW) modes, if sufficient power. With current SC Nb technology ( = 1.8 K, RF = 1.3 GHz, used for ILC) and with power = 100 MW, a luminosity ∼ 0.33 × 10<sup>36</sup> cm−<sup>2</sup> s −1 is possible at the Higgs factory with 2<sup>0</sup> = 250 GeV. Using superconductors operating at 4.5 K with high <sup>0</sup> values, such as Nb3Sn, and RF = 0.65 GHz, the luminosity can reach ∼ 1.4 × 10<sup>36</sup> cm−<sup>2</sup> s −1 at 2<sup>0</sup> = 250 GeV (with P=100 MW) and ∼ 0.8 × 10<sup>36</sup> cm−<sup>2</sup> s −1 at 2<sup>0</sup> = 500 GeV (with P=150 MW), which is almost two orders of magnitude greater than at the ILC, where the beams are used only once. This technology requires additional efforts to obtain the required parameters and reliably operation. Such a collider would be the best machine for precision Higgs studies, including the measurement of Higgs self-coupling.

Keywords: Accelerator modelling and simulations (multi-particle dynamics; single-particle dynamics), Beam dynamics, Instrumentation for particle accelerators and storage rings - high energy (linear accelerators), Lasers.

ArXiv ePrint: [2105.11015](https://arxiv.org/abs/2105.11015)

## **Contents**

| 1 | Introduction                                                            | 1  |
|---|-------------------------------------------------------------------------|----|
| 2 | Superconducting twin linear collider with energy recovery               | 3  |
| 3 | Collision effects limiting the luminosity                               | 5  |
| 4 | Luminosity restrictions due to collision effects                        | 7  |
| 5 | RF losses in cavities                                                   | 10 |
| 6 | High-order mode losses (HOM)                                            | 11 |
| 7 | Duration of pulse in DC mode.                                           | 13 |
| 8 | Optimum values of 𝑁, distance between bunches<br>𝑑<br>and duty cycle 𝐷𝐶 | 13 |
| 9 | Ways to reduce power consumption                                        | 15 |
|   | 10 From 250 to 500 GeV                                                  | 16 |
|   | 11 Summary tables                                                       | 16 |
|   | 12 Conclusion                                                           | 16 |

## <span id="page-1-0"></span>**1 Introduction**

Linear + − colliders (LC) have been actively developed since the 1970s as a way to reach higher energies. Their main advantage over storage rings is the absence of synchrotron radiation during acceleration, which makes it possible to achieve much higher energies. Their main weak point is the one-pass use of beams. At storage rings, the same beams are used many millions of times, whereas in LC they are sent to beam dumps after a single collision. This inefficient use of electricity results in a low collision rate and therefore a lower luminosity.

There were many LC projects in the 1990s (VLEPP, NLC, JLC, CLIC, TESLA, etc.) [\[1\]](#page-19-0); since 2004 only two remain: ILC [\[2,](#page-19-1) [3\]](#page-19-2) and CLIC [\[4\]](#page-19-3). The ILC is based on superconducting (SC) Nb technology (in the footsteps of the TESLA), while the CLIC uses Cu cavities and operates at room temperature. Both colliders work in pulse mode; their beam structures are given in Table 1. The difference is only in the length of bunch trains: for the ILC it is 4150 times longer. The luminosities and wall plug powers are very similar. In fact, the use of superconducting technology does not provide significant benefits to the ILC in luminosity. Moreover, the accelerating gradient in the

|                                  | ILC     | CLIC        |
|----------------------------------|---------|-------------|
| 2𝐸0, GeV                         | 250     | 250         |
| bunches/train, 𝑛𝑏                | 1312    | 354         |
| bunch spacing, ns/m              | 554/166 | 0.5/0.15    |
| train length, 𝜇s/km              | 727/220 | 0.177/0.053 |
| rep. rate, Hz                    | 5       | 50          |
| collision rate, kHz              | 6.56    | 17.7        |
| power (wall plug), MW            | 129     | 225         |
| luminosity, 1034 cm−2<br>−1<br>s | 1.35    | 1.37        |

**Table 1**. Pulse structure of the ILC [\[3\]](#page-19-2) and CLIC [\[4\]](#page-19-3).

ILC is 2–3 times lower. Although there are some technical advantages - greater spacing between bunches (which is good for detectors), lower peak RF power and lower tolerance requirements.

The main advantage of the superconducting technology is the feasibility of energy recovery, where the beam, after passing the interaction point (IP), is decelerated in the opposing linac and thus returns its energy to the accelerator. This opportunity was noticed originally and discussed in the very first publications on linear colliders by M. Tigner [\[5\]](#page-19-4), A. Skrinsky [\[6\]](#page-19-5), U. Amaldi [\[7\]](#page-20-0). The LC scheme with energy recovery considered by H. Gerke and K. Steffen in 1979 [\[8\]](#page-20-1) is shown in Fig. [1.](#page-2-0) This scheme provides not only energy recovery but also multiple use of the same electron and positron beams. One of the problems with multiple use of beams is a large energy spread that appears due to beamstrahlung at the IP. After beam deceleration, the relative energy spread becomes too large for injection to the damping rings. In order to reduce it, authors have foreseen "de(bunchers)" (bunch compressors and decompressors) that change the energy spread and the bunch length while keeping constant.

![](_page_2_Figure_4.jpeg)

**Caption:** Figure: Schematic layout of the experimental setup for the linear collider with energy recovery. The diagram shows the configuration of linear accelerators (Linac) ranging from 1—100 GeV and 100—1 GeV, interconnected by damping rings and (De)bunchers, culminating in an experimental hall. The system is designed to optimize beam path and stability, crucial for reducing beam emittance and improving energy efficiency during the collision process. This arrangement is integral to minimizing parasitic collisions while maintaining high luminosity in superconducting twin linear colliders .


<span id="page-2-0"></span>**Figure 1**. Gerke–Steffan's scheme of a linear collider with energy recovery [\[8\]](#page-20-1)

However, the Gerke-Steffen scheme has as few deficiencies:

- the quality factor of SC cavities at that time was <sup>0</sup> ∼ 2 × 10<sup>9</sup> , which was not enough for the continuous operation mode. Removal of the heat from cryogenic structures requires a lot of energy due to Carnot efficiency; therefore, a duty cycle of 1/30 was adopted.
- in order to exclude parasitic bunch collisions inside the linac, only one bunch is present at any one moment at each half linac, which limits the collision rate to of = 30 kHz (for a total

LC length of 10 km). With a duty cycle of 1/30, the average rate would be a mere 1 kHz.

- electron and positron bunches cannot be focused by the same final focusing systems (no one had noticed this obstacle), so this scheme could work (which is not obvious) only in one direction of the beams.
- the estimated luminosity was = 3.6 × 10<sup>31</sup> cm−<sup>2</sup> s −1 , which is too low to be of interest.

Since the 1980s, LC energy recovery schemes have no longer been considered. This is because the collision rate at a single-pass LC is similar to that at an ERL collider (as discussed above), and the luminosity per collision can be much higher at a single-pass LC due to the larger permissible disruption of the beams.

For many years, linear colliders have been considered as the obvious next large HEP project after the LHC. People expected very rich new physics to emerge in the energy range covered by LCs (2<sup>0</sup> =100–3000 GeV). Unfortunately, the LHC has found only the Higgs boson. Physicists are therefore in doubt about linear colliders. It is only absolutely clear that we need an + <sup>−</sup> Higgs factory at the energy 2<sup>0</sup> = 250 GeV. But after the discovery of the light Higgs boson, the FCC-ee and CEPC circular 100 km + − colliders came into play, promising an order of magnitude higher luminosity at this energy, followed by the 2 × 100 TeV proton collider in the same tunnel. This is one of reasons (beside the cost) that a decision on the ILC has not yet been made, although it was expected shortly after the publication of the TDR in 2013.

Below, we revisit the concept of an energy-recovery LC and show that the above problems can be overcome. In addition, significant progress has been made on SC cavities over the past three decades. The quality factor has been increased by more than an order of magnitude. The emphasis will be on the Higgs boson energy, the case of 2<sup>0</sup> = 500 GeV will be considered as well. The result is intriguing and can change the course of the game.

The outline of the article is the following. I first explain why parasitic collisions should be avoided and suggest a way to solve this problem. In the Section 3 and 4, the collision effects limiting the luminosity are analyzed and expressions for the luminosities are obtained. Sections 5 and 6 are devoted to the main sources of energy consumptions associated with RF losses in the SC cavities and beam losses to HOM (high order modes), where the main power is spent on heat removal. Clause 7 estimates the active running time with a duty cycle. Section 8 discusses the optimal value of the duty cycles and the number of particles in the bunch. Methods to reduce power consumptions are discussed in Section 9. Section 10 briefly discusses the problem of increasing energy from 250 to 500 GeV. Then, in Section 11, summary tables and figures are presented with the dependencies of the luminosities on the total power for 2<sup>0</sup> = 250 and 500 GeV for several variants of SC cavities.

## <span id="page-3-0"></span>**2 Superconducting twin linear collider with energy recovery**

The first question is why it is necessary to exclude parasitic beam collisions inside the linacs. At first glance, the transverse beam sizes in linacs are much larger than at the IP, so the loss of particles due to such collisions appears to be insignificant. The reason lies in the instability of the beams. If we want to use beams multiple times, the instability criteria are the same as in storage rings and are determined by the vertical tune shift (or the beam-beam parameter) [\[9\]](#page-20-2)

$$\xi\_{\mathcal{Y}} = \frac{Nr\_e \beta\_{\mathcal{Y}}}{2\pi\gamma\sigma\_{\mathcal{X}}\sigma\_{\mathcal{Y}}} \lesssim 0.1, \qquad \sigma\_{\mathcal{i}} = \sqrt{\epsilon\_{n,\mathcal{i}}\beta\_{\mathcal{i}}/\gamma}.\tag{2.1}$$

The ratio of the beam-beam parameters in the linac and at the IP is

$$\frac{\xi}{\xi^\*} = \frac{\sqrt{\beta\_\text{y}/\beta\_\text{x}}}{\sqrt{\beta\_\text{y}^\*/\beta\_\text{x}^\*}} \gg 1,\tag{2.2}$$

because in the linac ∼ , while at the IP ∗ ≫ ∗ . Note that this result is independent of the energy at which parasitic collisions occur.

To solve this problem I propose a twin linear collider in which the beams are accelerated and then decelerated down to ≈ 5 GeV in separate parallel linacs with coupled RF systems, see Fig. [2.](#page-4-0) RF power is always divided equally among the linacs. RF energy comes to the beams both from an external RF source and from the decelerating beam. These can be either two separate SC linacs connected by RF couplers at the ends of multi-cell cavities (9-cell TESLA cavity), or one linac consisting of twin (dual) cavities with axes for two beams. Such cavities have been designed and tested for XFELs [\[10–](#page-20-3)[13\]](#page-20-4).

![](_page_4_Figure_7.jpeg)

**Caption:** Figure: Detailed view of beam dynamics in the collision scheme of a high-luminosity energy-recovery linear collider. It showcases the head-on collision mechanism and illustrates the paths for electron (e-) and positron (e+) beams undergoing acceleration and deceleration cycles, along with associated compressors and decompressors. Central to the design is achieving energy efficiency while keeping synchrotron radiation within allowable limits. Key features include energy recovery circuits and beam dump systems, essential for minimizing energy loss and maintaining sustainability of the collider design. This scheme aims to enhance beam stability and accuracy during experimental runs .


<span id="page-4-0"></span>**Figure 2**. The layout of the SC twin linear collider.

It is assumed that the collider will operate at an energy 2<sup>0</sup> ≈ 250 GeV (with possible increasing up to 500 GeV) in a semi-continuous mode with a duty cycle (DC): collisions for about 10 seconds, then a break to cool the cavities. In one cycle, the beams make about 50 thousand revolutions. Continuous (CW) operation is more attractive, it is also possible after additional R&D (discussed below).

Beams are prepared in damping rings, as usual. In continuous mode, you only need to add the lost particles. In the DC mode, the bunches are prepared anew each cycle. The number of bunches in the ERLC is large, but there is enough time (injection time up to 1-2 s). The required average production rate is an order of magnitude lower than at the ILC.

During collisions, beams get an additional energy spread that is damped by wigglers installed in the return pass at the energy ≈ 5 GeV. The relative energy loss in wigglers is about / ∼ 1/200. We require that the steady-state equilibrium energy spread at the IP due to beamstrahlung is better than /<sup>0</sup> ∼ 0.2%, the same as at the ILC and CLIC before the beam collision. Such a spread would be sufficient for beam focusing. The beam parameters are determined not only by the energy spread at the IP, but also by beam lifetime which is caused by single beamstrahlung. We will consider both effects.

When the beam is decelerated down to 5 GeV, its relative energy spread increases by 0/ ∼ 25 times to / ∼ 5%. To make it acceptable for travel without losses in the arcs, its energy spread is reduced by 10–15 times with the help of the bunch (de)compressor; then, the relative energy spread in the arcs will be less than 0.5%. The beam lifetime will be determined by the tails in beamstrahlung radiation. This loss should not exceed about 1% after 10000 revolutions. The IP energy spread, the beam instability and beam particle losses (due to single beamstrahlung) determine the IP beam parameters, and hence the luminosity.

An important question is the injection and extraction of the beams. When the collider is full, the distance between bunches is small, optimally equal to the linac's RF-wavelength ( = 23 cm for TESLA/ILC 1.3 GHz cavities); they are accelerated and decelerated due to the exchange of energy between the beams. External RF power is required only for energy stabilization and compensation for radiation and high order mode (HOM) losses. During the injection/extraction of the beams, normal energy exchange does not occur until the bunches fill the entire orbit, so the external RF system must work at full power. However, at the ILC, the power of the RF system is only sufficient to accelerate beams with a bunch distance of 100–150 m. In our case, with energy recovery, we need a much shorter inter bunch distance (and smaller bunch charges). This is so necessary because the energy lost by a bunch into high-order modes (HOM) is proportional to 2 ( is the number of particles in the bunch). To solve this problem, one must first inject bunches (better short trains) with a large interval and then (at subsequent revolutions) add new trains between the trains already circulating. The distance of 23 cm between the bunches is too small for working with individual bunches, while the use of trains with gaps of 1–2 m makes it possible to manipulate trains using impulse deflectors. Removal of beams is done in reverse order.

## <span id="page-5-0"></span>**3 Collision effects limiting the luminosity**

#### **Energy spread in beam collisions**

During the beam collisions, the particles emit synchrotron radiation (beamstrahlung), which contributes to the energy spread of the beam. The increase of the beam energy spread in a single collision ( < 1) [\[14](#page-20-5)[–16\]](#page-20-6)

$$
\Delta \sigma\_E^2 = n\_\gamma \langle \epsilon\_\gamma^2 \rangle = \frac{\langle \epsilon\_\gamma^2 \rangle}{\langle \epsilon\_\gamma \rangle^2} \frac{\left(n\_\gamma \langle \epsilon\_\gamma \rangle\right)^2}{n\_\gamma} \approx \frac{5.5 \left(\Delta E\right)^2}{n\_\gamma},\tag{3.1}
$$

$$\frac{\Delta E}{E\_0} \approx \frac{0.84r\_e^3 N^2 \gamma}{\sigma\_z \sigma\_x^2}, \ n\_\gamma \approx 2.16 \frac{\alpha r\_e N}{\sigma\_x},\tag{3.2}$$

where Δ is the average energy loss, is the average number of photons per collision ( ≪ 1 under our conditions), = 2 /ℏ ≈ 1/137, = 2 /<sup>2</sup> . Here, we neglect the energy spread due to the inhomogeneity of the Gaussian beam ( ≈ 0.54Δ), which is much smaller. As a result, we get

<span id="page-5-1"></span>
$$\frac{\Delta\sigma\_E^2}{E\_0^2} \approx 1.8 \frac{N^3 r\_e^5 \chi^2}{\alpha \sigma\_\chi^3 \sigma\_z^2}. \tag{3.3}$$

After the collision the bunch decelerates and then stretches during the bunch decompression where its Δ and decrease proportionally. Due to SR radiation in damping wigglers at the energy ∼ 5 GeV, where particles lose the energy in one pass, an equilibrium energy spread is achieved [\[9\]](#page-20-2):

<span id="page-6-0"></span>
$$\frac{\Delta\sigma\_E^2}{\sigma\_E^2} = 2\frac{\delta E}{E}.\tag{3.4}$$

Substitution of [\(3.3\)](#page-5-1) into [\(3.4\)](#page-6-0) gives the equilibrium energy spread at the IP

$$
\left(\frac{\sigma\_E}{E\_0}\right)^2 \approx 0.9 \frac{N^3 r\_e^5 \chi^2}{\alpha \sigma\_\chi^3 \sigma\_\natural^2 (\delta E/E)}. \tag{3.5}
$$

For the desired damping rate and energy spread, we obtain the requirements for the beam parameters at the IP

<span id="page-6-1"></span>
$$\frac{N^3}{\sigma\_X^3 \sigma\_z^2} < \frac{8 \times 10^{-3}}{r\_e^5 \gamma^2} \left(\frac{\sigma\_E}{E\_0}\right)^2 \frac{\delta E}{E} \,. \tag{3.6}$$

### **Beam instability**

Our linear collider behaves as a cyclic storage ring, so there is a second limitation on the beam parameters at the IP, due to the tune shift. For flat beams and head-on collisions, it is

<span id="page-6-2"></span>
$$
\xi = \xi\_{\mathcal{Y}} = \frac{N r\_e \sigma\_{\mathcal{Z}}}{2 \pi \chi \sigma\_{\mathcal{X}} \sigma\_{\mathcal{Y}}} \lesssim 0.1 \quad (\text{for} \quad \beta\_{\mathcal{Y}} \approx \sigma\_{\mathcal{Z}}). \tag{3.7}
$$

At the ILC, the beams collide at a crossing angle of ≈ 15 mrad, which makes it easier to remove highly disrupted beams. In the ILC case, you should use the crab-crossing scheme (tilt of the bunches by /2) to preserve the luminosity. In the case of considered cyclic LC, beam disruption is small and beams can be removed through the aperture of the opposing final quadrupole; therefore we assume nearly head-on collisions with a small crossing angle to facilitate the separation of the beams.

In this article we do not consider collisions at a large crossing angle ("crab-waist" scheme) because it would provide no benefit when beamstrahlung is important or the beams are short. Some gain is possible at low energies (at Z-boson); this will be discussed in subsequent articles.

#### **Beam lifetime**

The beam lifetime at high-energy + − storage rings is determined by the emission of high-energy beamstrahlung photons [\[17\]](#page-20-7). An electron (positron) is lost when its energy loss is greater than 0, where is the energy acceptance. In our case, the bunches are decelerated by a factor of 125/5=25 and then expanded by a factor of ∼ 15; therefore, the energy acceptance in the 5 GeV arc should be approximately 25/15=1.67 greater than the maximum acceptable relative energy loss at the IP. If we take the energy acceptance in arcs at 5 GeV equal to 3%, then at the IP it should be = 0.03/1.67 = 0.018 for <sup>0</sup> = 125 GeV and 0.018(125/E) for other energies.

The formulas for calculating the beam lifetime are given in the Ref. [\[18\]](#page-20-8). For the lifetime of the beam to correspond to col collisions in the collider with energy acceptance , it is necessary to have

$$\frac{N}{\sigma\_{\lambda}\sigma\_{z}} < \frac{3.6 \times 10^{-3}\eta}{\gamma r\_{e}^{-2} \ln\left(7 \times 10^{-7}\eta \sigma\_{z} n\_{\rm col}/\gamma r\_{e}\right)}\tag{3.8}$$

Below, we will show that in the case of operation with a duty cycle, the duration of the active phase can be around 10 s. During this time, each bunch collides about 50 thousand times. To have a beam loss at the level of <3%, the beam lifetime must correspond to col ∼ 1.5 × 10<sup>6</sup> (the lifetime of about 5 minutes). Since the dependance on col is logarithmic, suppose for further consideration col = 1.5 × 10<sup>6</sup> , it will be good also for continuous (CW) operation. As explained above, we assume = 0.018(125/) = 4400/. As for the value of the in the logarithm, the range of acceptable bunch lengths is rather narrow, 0.3–2 mm, we put to the logarithm = 0.5 mm. With these assumptions, we obtained the third constraint on beam parameters associated with the beam lifetime due to single beamstrahlung

<span id="page-7-1"></span>
$$\frac{N}{\sigma\_{\chi}\sigma\_{\varepsilon}} < \frac{7.9}{\gamma^2 r\_e^2 \Lambda}, \quad \Lambda \approx \ln \frac{120}{E\_0/125}, \quad E\_0 \text{ in GeV.} \tag{3.9}$$

The inaccuracy due to fixed values of the quantities in the logarithm does not exceed 20%.

#### **Bunch length**

Below we will consider limitations on luminosity due to the three effects described above. There is an optimal bunch length for each beam energy. For low energies it is shorter, for higher energy it is longer, but the preferred range of bunch lengths for the collider based on 1.3 GHz SC cavities is =(0.2–0.3)–(1–2) mm. This will be the fourth constraint which also significantly affects the optimization of the parameters

<span id="page-7-2"></span>
$$
\sigma\_{\mathbb{Z},\text{min}} < \sigma\_{\mathbb{Z}} < \sigma\_{\mathbb{Z},\text{max}}.\tag{3.10}
$$

## <span id="page-7-0"></span>**4 Luminosity restrictions due to collision effects**

For four unknown beam parameters at the IP: , , , we have four restrictions, [\(3.6\)](#page-6-1), [\(3.7\)](#page-6-2), [\(3.9\)](#page-7-1), [\(3.10\)](#page-7-2) and one relationship: ≈ √︁ /. Collision effects for flat beams depend on the combination /; therefore, as a free parameter, its optimum value is discussed in the following sections.

Now we have to understand the complex problem of finding the luminosity under four constraints on the parameters. Let's enumerate and give short names to these constraints:

- 1. (BI) Beam Instability [\(3.6\)](#page-6-1).
- 2. (MBS) Multiple Beamstrahlung ( at the IP) [\(3.7\)](#page-6-2).
- 3. (SBS) Single Beamstrahlung (beam life time) [\(3.9\)](#page-7-1).
- 4. (SZ) Bunch length, a) > ,min, b) < ,max [\(3.10\)](#page-7-2).

With increasing energy, the luminosity is limited by the following combinations of constraints: BI-SZ(a) ⇒ BI-MBS ⇒ BI-SBS ⇒ SBS-SZ(b). The combination MBS-SBS is omitted because MBS and SBS restrict almost the same combinations: /() and /( 3/2 ); MBS-SZ(b) is omitted because SBS-SZ(b) is stronger at high energies.

The results for these four cases are the following.
**BI-SZ**

$$
\sigma\_X = \frac{N r\_e \sigma\_{z, \text{min}}^{1/2}}{2 \pi \gamma^{1/2} \epsilon\_{\text{ny}}^{1/2} \xi}, \quad \sigma\_\text{y} = \left(\frac{\sigma\_{z, \text{min}} \epsilon\_{\text{ny}}}{\gamma}\right)^{1/2}, \quad \sigma\_\text{z} = \sigma\_{z, \text{min}}.\tag{4.1}
$$

$$L \approx \frac{N^2 f}{4\pi \sigma\_\lambda \sigma\_\chi} = \frac{N f \xi \gamma}{2r\_e \sigma\_{z,\text{min}}}.\tag{4.2}$$

**BI-MBS**

$$
\sigma\_z \approx 19.2 \frac{\xi^{6/7} \epsilon\_{\rm ny}^{3/7} r\_e^{4/7} \gamma}{(\sigma\_E / E\_0)^{4/7} (\delta E / E)^{2/7}},\tag{4.3}
$$

$$
\sigma\_{\rm x} \approx 0.7 \frac{N r\_e^{9/7}}{\xi^{4/7} \epsilon\_{\rm ny}^{2/7} (\sigma\_E/E\_0)^{2/7} (\delta E/E)^{1/7}}, \qquad \sigma\_{\rm y} = 4.4 \frac{\xi^{3/7} \epsilon\_{\rm ny}^{5/7} r\_e^{2/7}}{(\sigma\_E/E\_0)^{2/7} (\delta E/E)^{1/7}}. \tag{4.4}
$$

$$L \approx \frac{N^2 f}{4\pi \sigma\_x \sigma\_y} = 2.6 \times 10^{-2} \frac{N f \xi^{1/7}}{\epsilon\_{ny}^{3/7} r\_e^{11/7}} \left(\frac{\sigma\_E}{E\_0}\right)^{4/7} \left(\frac{\delta E}{E}\right)^{2/7}. \tag{4.5}$$

For /<sup>0</sup> = 2 × 10−<sup>3</sup> , / = 0.5 × 10−<sup>2</sup> , = 0.1, = 3 · 10−<sup>8</sup> m (as at the ILC), we have

$$
\sigma\_x \approx 0.9 \left( \frac{N}{10^9} \right) \text{ } \mu\text{m}, \quad \sigma\_z \approx 0.3 \frac{E[\text{GeV}]}{125} \text{ mm}, \quad \sigma\_y = 6.1 \text{ nm}. \tag{4.6}
$$

$$L \approx 4.35 \times 10^{35} \frac{(N/10^9)}{d \text{[m]}} \approx 9 \times 10^{36} I \text{[A]} \text{ cm}^{-2} \text{s}^{-1},\tag{4.7}$$

where = / is the distance between the bunches. Please note, this luminosity is for continuous operation (100% duty cycle). For example (the choice of and will be discussed later),

$$N = 10^9, d = \lambda\_{RF} = 0.23 \text{ m} \ (I = 0.21 \text{A}) \Rightarrow L \approx 1.9 \times 10^{36} \text{ cm}^{-2} \text{s}^{-1}. \tag{4.8}$$

In this example, the power radiated in damping wigglers by both beams is SR = 10.4 MW.

#### **BI-SBS**

<span id="page-8-0"></span>
$$
\sigma\_z = 0.86 \Lambda^{2/3} \epsilon\_{\rm ny}^{1/3} (\xi r\_e)^{2/3} \gamma^{5/3}, \qquad \Lambda = \ln \frac{120}{E\_0/12S}, \tag{4.9}
$$

<span id="page-8-1"></span>
$$
\sigma\_X = \frac{0.15 \Lambda^{1/3} N r\_e^{4/3} \gamma^{1/3}}{\epsilon\_{\rm ny}^{1/3} \xi^{2/3}}, \quad \sigma\_\text{y} = 0.93 \Lambda^{1/3} \epsilon\_{\rm ny}^{2/3} (\xi r\_e \gamma)^{1/3}, \tag{4.10}
$$

<span id="page-8-2"></span>
$$L = \frac{0.58Nf\xi^{1/3}}{\epsilon\_{ny}^{1/3}r\_e^{5/3}\gamma^{2/3}\Lambda^{2/3}}.\tag{4.11}$$

For = 0.1 and = 3 · 10−<sup>8</sup> m

$$L \approx 4.15 \times 10^{35} \frac{(N/10^9)}{d \, [\text{m}]} \left( \frac{\ln 120}{\ln \left( 120 (125/E\_0 \text{[GeV]}) \right)} \right)^{2/3} \left( \frac{125}{E\_0 \text{[GeV]}} \right)^{2/3} . \tag{4.12}$$

**SBS-SZ**

$$
\sigma\_X = \frac{0.125 N \Lambda \gamma^2 r\_e^2}{\sigma\_{z,\text{max}}}, \quad \sigma\_\text{y} = (\sigma\_{z,\text{max}} \epsilon\_{\text{ny}} / \gamma)^{1/2}, \quad \sigma\_\text{z} = \sigma\_{z,\text{max}}, \tag{4.13}
$$

$$L = \frac{0.63 N f \sigma\_z^{1/2}}{\Lambda \epsilon\_{ny}^{1/2} \gamma^{3/2} r\_e^2}. \tag{4.14}$$

#### **Summary of collision effects**

We have considered various limitations on the luminosity due to collision effects. The maximum luminosity can be written as

$$L = \left( Nf \right) F(E) \times DC, \quad f = c/d, \quad F(E) = \min F\_i(E), \tag{4.15}$$

where () are shown in Fig. [3.](#page-9-0) One can see which effect is most important for a given energy. For = RF = 23 cm

$$L = 1.3 \times 10^{18} \left(\frac{N}{10^9}\right) F(E) \times DC. \tag{4.16}$$

![](0__page_9_Figure_8.jpeg)

**Caption:** Figure 3: Graphical representation of the impact of collision effects on luminosity, specifically focusing on limitations caused by Beam Instability (BI) and Single Beamstrahlung (SBS) effects for 2E0 in the 250–500 GeV range. The y-axis denotes F_lum/10^18, representing a multiplicative factor of influence, while the x-axis is 2E0 in GeV, indicating beam energy. It highlights constraints where SBS-SZ(b) effects are predominant, thus limiting luminosity gains at higher energies. The data sheds light on how beam dynamics and electron interactions restrict achievable luminosities and informs modifications needed in collider designs to overcome these challenges .


<span id="page-9-0"></span>**Figure 3**. Luminosity restrictions due to various collision effects (see the text). The maximum value of = × lum, where the limit is given by lowest lum at each energy.

For 2<sup>0</sup> =250–500 GeV the luminosity is limited by BI-SBS (beam instability and single beamstrahlung). Using Eqs. [\(4.9\)](#page-8-0), [\(4.10\)](#page-8-1), [\(4.11\)](#page-8-2) we get for = 23 cm the following luminosity and beam sizes for <sup>0</sup> = 250 and 500 GeV in continuous (CW) mode:

$$2E\_0 = 250: L = 1.81 \times 10^{36} (N/10^9), \sigma\_\lambda = 0.94 (N/10^9) \,\mu\text{m}, \sigma\_\circ = 6.2 \,\text{nm}, \sigma\_\sharp = 0.31 \,\text{nm}. \quad (4.17)$$

$$1.2E\_0 = 500 : L = 1.27 \times 10^{36} (N/10^9), \sigma\_X = 1.12 (N/10^9) \,\mu\text{m}, \sigma\_\text{y} = 7.4 \,\text{nm}, \sigma\_\text{z} = 0.89 \,\text{nm}. \quad (4.18)$$

For = 23 cm the average current = 0.21(/10<sup>9</sup> ) A. The optimum value of and is discussed in the Section [8.](#page-13-0) It will be shown below that the optimum value of /10<sup>9</sup> ∼ 0.5–2 for various types of SC cavities.

## **5 RF losses in cavities**

One of the main problems of SC linear accelerators operating in a continuous mode is heat removal from the low-temperature SC cavities. Energy dissipation in one (multi-cell) cavity

$$P\_{\rm RF,dis} = \frac{V\_{\rm acc}^2}{(R/Q)Q\_0},\tag{5.1}$$

where acc is the operating voltage and <sup>0</sup> is the cavity quality factor. The 1.3 GHz TESLA–ILC cavity has / = 1036 Ohm and the length = 1.04 m. For the accelerating gradient = 20 MeV/m and <sup>0</sup> = 3 × 1010, the thermal power diss = 13.5 W/m, or 680 W/GeV. Also, some heat will add the absorption in SC cavities of the High Order Mode (HOM) losses, we will take them into account separately in the next section. Such a continuous-mode SC linac is currently being developed for the XFEL LCLS-II at SLAC [\[19,](#page-20-0) [20\]](#page-20-1), they plan RF,dis ∼ 1 kW/GeV.

The overall heat transfer efficiency from temperature <sup>2</sup> ≈ 1.8 K to room temperature <sup>1</sup> ∼ 300 K is = 2/(<sup>1</sup> − 2) ≈ 0.18 × 1.8/300 = 1/900 [\[21\]](#page-20-2). The required refrigeration power for the twin 250 GeV collider is

$$P\_{\rm RF-refr,2K} \approx 2 \times 250 \,\text{GeV} \times 900 \times 0.68 \,\text{kW/GeV} = \Im 06 \,\text{MW}.\tag{5.2}$$

It is only from RF losses. As we will see below, under optimal conditions, approximately similar power is required for removal of HOM losses.

In recent years, great progress has been made both in increasing the maximum accelerating voltage and in increasing the quality factor 0. In the ILC project, it is assumed <sup>0</sup> = 10<sup>10</sup> and = 31.5 MeV/m. For continuous operation, it is advantageous to work at ≈ 20 MeV/m, where <sup>0</sup> ∼ 3 × 10<sup>10</sup> is within reach now. Moreover, N-doping and other surface treatment technologies have already resulted in <sup>0</sup> ∼ 5 × 10<sup>10</sup> at = 2 K and <sup>0</sup> ∼ (3 − 4) × 10<sup>11</sup> at < 1.5 K [\[22,](#page-20-3) [23\]](#page-20-4). According to a leading expert [\[24\]](#page-20-5), one can hope for a reliable <sup>0</sup> = 8 × 10<sup>10</sup> at = 1.8 K. Currently, we can take <sup>0</sup> ∼ 3 × 10<sup>10</sup> and work with duty cycle (DC) in order to keep the total power consumption of the collider below 150 MW, as at the ILC.

A even more promising way to reduce the cooling power is to use superconductors with a higher operating temperature, such as Nb3Sn. Thus, at <sup>2</sup> = 4.5 K the technical efficiency of heat removal is about 30% [\[21\]](#page-20-2), and the total one (with Carnot) about 1/220, that is about 4 times higher than that at 1.8 K. This question is discussed also in the Section [8.](#page-13-0)

## **6 High-order mode losses (HOM)**

When particles are accelerated in linear accelerators, they take energy Δ = E0Δ from the cavity due to the destructive interference of the RF field in the cavity E<sup>0</sup> and the wave E radiated by the bunch into the cavity. When the particles are decelerated (Δ = −E0Δ), they return their energy back to the cavity due to constructive interference between the RF field and the radiated field. However, such a picture with an ideal energy exchange is valid only for the fundamental cavity mode. High radiation modes (longitudinal wake fields ∝ bunch charge) lead to energy losses during both acceleration and deceleration. For this reason, the energy recovery efficiency is not 100%.

When the beam passes through a single diaphragm with a radius of , the energy loss can be easily estimated as the energy of the beam field at > . However, for a long linear accelerator with multiple apertures, the picture is more complex. In this case, according to R. Palmer [\[25\]](#page-20-6), the energy loss by one electron per unit length

<span id="page-11-0"></span>
$$-\frac{dE}{dz} \approx \frac{2e^2N}{r\_a^2}.\tag{6.1}$$

It is noteworthy than the energy losses do not depend on the distance between the diaphragms and on the bunch length. This simple formula is supported by detailed numerical calculations. There is a dependence on the bunch length, but very weak. For TESLA–ILC accelerating structures ( = 3.5 cm), a numerical calculation [\[26\]](#page-20-7) gives the energy losses in the wakefield for = 400 m (the energy loss in the cryomodule is divided by the active length of accelerating cavities)

$$-\frac{dE}{dz} \approx 2.2 \left(\frac{N}{10^9}\right) \frac{\text{keV}}{\text{m}}.\tag{6.2}$$

For bunch lengths = 0.25–1 mm, the coefficient varies within the range 2.42–1.76. The formula [\(6.1\)](#page-11-0) gives 2.35. So, for TESLA cavities and = 10<sup>9</sup> , these HOM losses are ∼0.01% of the accelerating gradient ∼ 20–30 MeV/m.

For 2 = 250 GeV, = 20 MeV/m, the active collider length is = 12.5 km. The total power of HOM energy losses (twin collider, both beams)

$$P\_{\rm HOMO} = \frac{2.65}{d \, \text{[m]}} \left( \frac{N}{10^9} \right)^2 \text{ MW.} \tag{6.3}$$

For = 10<sup>9</sup> and = 0.23 m HOM = 11.5 MW, which is close to the power of synchrotron radiation in damping wigglers, equal to 10.4 MW. Keep in mind that these numbers are for continuous operation.

What happens with the HOM power generated by the beam inside the linear accelerator? For = 10<sup>9</sup> , this power is about 35 times greater than the RF power dissipation in cavities. Fortunately, most of this energy can be extracted from the SC cavities in two ways: a) using HOM couplers which dissipate the energy at room temperature; b) with the help of special HOM absorbers located between the cavities. The latter are maintained at an intermediate temperature around 80 K where refrigeration systems operate at much higher efficiencies. However, some small part of the HOM energy is dissipated in the walls of SC cavities at 1.8 K. The ratio of the total HOM power to the RF dissipation power

$$\frac{P\_{\text{HOM}}}{P\_{\text{RF,dis}}} \propto \frac{N^2 \mathcal{Q}\_0}{E\_{\text{acc}}^2 d}. \tag{6.4}$$

Comparing ILC ( = 2 × 1010, <sup>0</sup> = 1010, = 100 m) and ERLC ( ≈ 10<sup>9</sup> , <sup>0</sup> = 3 × 1010, = 0.23 m) we see that this ratio is 7.9 times higher at the ERLC. At the ILC HOM/RF,dis ≈ 4.3 and about 7% of HOM power is dissipated at 2 K, so HOM losses adds about 30%. For the ERLC, in order for the HOM absorption in the SC cavities to be less than 50% of the dissipated RF power, it is necessary that the fraction of HOM power absorbed in the SC resonators does not exceed 1.5%. High frequency HOMs are not a problem, they propagate like photons, the reflection coefficient from the walls of SC cavities is very high, 1 − < 10−<sup>7</sup> , so that freely flying photons are very efficiently absorbed by the HOM absorbers located between cavities. The main difficulty present some trapped low frequency HOM modes with low amplitude at the ends of the cavity where they are damped by HOM couplers.

Here I want to make an important point. Above, we assumed that the HOM loss from the bunch train is the sum of the losses of each bunch separately. However, this is obviously not the case for sufficiently low HOM frequencies. The bunches run equidistantly and the HOM fields emitted into cavities are added with different phases. If the phases were random, then the radiation powers would add up, but in our case there is a strict periodicity. If the HOM frequency is not in resonance with the RF frequency, then the amplitude of the resulting HOM wave from many bunches can only be several times higher than the amplitude from one bunch, which is proportional to and it does not depend on the repetition rate of the bunches. This means that bunches not only emit, but also take HOM energy from the cavity. For high frequencies, the phase is lost after many reflections due to geometry imperfections, and then the addition of powers is justified, but for low frequencies, which include the captured modes, this is not true. In this case, the HOM power at low frequencies at the ERLC is lower than in the ILC as 2 , that is (2 · 1010/10<sup>9</sup> ) <sup>2</sup> = 400 times! This issue is very important and requires further detailed study. Perhaps we significantly overestimate the HOM power from the bunch train. In the ILC, 7% of HOM energy is absorbed in cavities, which is not critical for the ILC. Proceeding from the above considerations, we take for the ERLC the fraction of HOM losses in SC cavities lower, equal to 1 %.

This HOM problem in high-current ERL linacs is well-known. The HOM power can be reduced by increasing the iris radius (be decreasing the RF frequency) since HOM ∝ 1/ 2 or by decreasing the number of particles in the bunch: the HOM power is proportional to 2 , while ∝ .

Let us find the refrigeration power needed for removal of HOM losses assuming that all HOM energy is dissipated in HOM absorbers at a liquid nitrogen temperature of 77 K. Assuming that refrigeration efficiency is = 0.32/(<sup>1</sup> − 2) = 0.3 · 77/(300 − 77) ∼ 0.1 we find that the refrigeration power for removal of HOM losses is ( = 23 cm)

$$P\_{\text{HOM,refr},77\text{K}} = P\_{\text{HOM}}/\eta = 10 \, P\_{\text{HOM}} \approx (11.5/0.1)(N/10^9)^2 = 115(N/10^9)^2 \text{ MW}.\tag{6.5}$$

One should add to this number the electric power needed for compensation of beam energy losses. Assuming 50% efficiency it is about 2 HOM.

Less definite are HOM losses in the walls of the SC cavities at 2 K. Taking the heat transfer efficiency from 1.8 K to 300 K equal 1/900, as in the previous section, and the fraction of HOM losses in cavities equal to 1% (as discussed above) we find the refrigeration power

$$P\_{\text{HOM,ref,2K}} = P\_{\text{HOM}} \times 900 \times 0.01 \approx 9 \, P\_{\text{HOM}} \approx 103 (N/10^9)^2 \text{ MW}.\tag{6.6}$$

The total electric power consumption from the wall plug (w.p.) due to HOM losses for 2<sup>0</sup> = 250 GeV collider in the continuous mode of operation

$$P\_{\text{HOM,w.p.}} \approx 21 \, P\_{\text{HOM}} \approx 240 (N/10^9)^2 \, \text{MW.} \tag{6.7}$$

## **7 Duration of pulse in DC mode.**

If the refrigeration power needed for continuous operation is too high one can work with a duty cycle ∼ 0.1–1 (see optimization below). Duration of continuous operation is determined by the heat capacity of the liquid He that surrounds the cavity and can be estimated as

$$
\Delta t = \frac{c\_p m \Delta T}{P\_{diss}} \sim 20 \,\text{s},\tag{7.1}
$$

where (He) = 2.8 J/(g·K) at T=1.8 K, is the mass of liquid He per one TESLA cavity (we take 0.02 m<sup>3</sup> or 2.9 kg), diss ∼ 20 W, Δ ∼ 0.05 K. So, we can safely choose the work duration Δ = 10 s.

More promising for ERLC are superconductors working at ≈ 4.5 (temperature of boiling He). The product for this temperature is 1.6 times larger, beside Δ can be taken larger by a factor of 3, then Δ = 20×1.6×3 = 96 s. In this case, the active cycle time can last about 1 minute.

Duration of the break is described by the duty cycle , which depends of the available power, the optimization is given in the next section.

# <span id="page-13-0"></span>**8 Optimum values of , distance between bunches and duty cycle**

Let us find the optimum number of particles in one bunch when the luminosity is maximum for a given power consumption. There are three main energy consumers (numbers correspond to 2<sup>0</sup> = 250 GeV)

- Radiation in the damping wigglers. SR/ = 20.8(/10<sup>9</sup> ) × MW at = 50 %. This assumes that new beams are generated for each cycle.
- Electric power for cooling of the RF losses in cavities (at 2 K) is RF−refr,2K = 305 × MW, it does not depend on .
- Electric power due to HOM losses, it is HOM,w.p. ≈ 240(/10<sup>9</sup> ) <sup>2</sup> × , MW.

The total power (only main contributions)

$$P\_{tot} = \left(20.8\frac{N}{10^9} + 305 + 240\left(\frac{N}{10^9}\right)^2\right) \times DC.\tag{8.1}$$

This number are for = 23 cm. Most of power is spent on removal of RF and HOM losses.

Let us first discuss the dependence of the maximum luminosity of the bunch distance and find the optimum and .

Neglecting power losses in wigglers, the power in operation with a duty cycle

<span id="page-14-1"></span>
$$P = \left(a + b\frac{N^2}{d}\right)DC,\tag{8.2}$$

where coefficients and describe RF and HOM losses, respectively, they are both proportional to the collider length (or 0). The luminosity

$$L \propto \frac{N}{d} DC = \frac{N}{d} \left( \frac{P}{a + bN^2/d} \right). \tag{8.3}$$

The maximum luminosity

<span id="page-14-0"></span>
$$L \propto \frac{P}{\sqrt{abd}} \quad \text{at} \quad N = \sqrt{\frac{ad}{b}}, \quad DC = \frac{P}{2a}. \tag{8.4}$$

The luminosity reaches the maximum when the energy spent for removal of and losses are equal (that is only for < 1). We see that ∝ 1/ √ , so the distance between bunches should be as small as possible, that is why we assumed this in all above considerations. Second, the coefficient ∝ 1/(0), where T is the temperature of SC cavities, is the technical efficiency (0.18 at T=1.8 K, 0.3 at T > 4 K), therefore ∝ √ 0. The optimum number of particles in the bunch does not depends on P and beam energy (because both and are proportional to ).

According to the above power estimates for 2<sup>0</sup> = 250 GeV, = 305(0/125) MW, / = 240(0/125)/(10<sup>9</sup> ) <sup>2</sup> MW, therefore the optimum number of particles in the bunch

$$N/10^9 \approx \sqrt{305/240} = 1.13.\tag{8.5}$$

Let's consider now the same for the CW operation. Now = 1 and

$$P = (a + b\frac{N^2}{d}), \quad L \propto \frac{N}{d},\tag{8.6}$$

that gives

<span id="page-14-2"></span>
$$N = \sqrt{\frac{(P-a)d}{b}}, \quad L \propto \sqrt{\frac{(P-a)}{bd}}.\tag{8.7}$$

Again ∝ 1/ √ . The minimum power for CW operation = ∝ 1/(0). The number of particle in the bunch in the CW mode depends on the available power:

$$N/10^9 \approx \sqrt{\frac{(P-a)d}{b}} \sim \sqrt{\frac{(P(125/E\_0) - \Im 05}{240}}\tag{8.8}$$

The luminosity in the CW mode is proportional to <sup>√</sup> − , at = 2 it becomes equal to the maximum luminosity with a duty cycle ( = 1 in [\(8.4\)](#page-14-0)). It has sense to work at exceeding the threshold power only by about 35% when CW/DC,max = 0.85. In this case /10<sup>9</sup> ∼ 0.67 and the power required for the Higgs factory is = 410 MW, it's too much.

Now all parameters are known, the resulting luminosities, calculated numerically with account of the neglected SR term are shown in the Fig. [4.](#page-17-0) For the Higgs factory with 2 = 250 GeV there are, for example, two options:

- a duty cycle mode, = 120 MW, = 0.39 × 10<sup>36</sup> cm−<sup>2</sup> s −1 , ≈ 0.19;
- a continuous mode, = 410 MW, = 1.13 × 10<sup>36</sup> cm−<sup>2</sup> s −1 .

The continuous mode is very attractive, but the required power is too high. Below we will discuss how it can be reduced.

The above numbers are for <sup>0</sup> = 3×10<sup>10</sup> and = 1.8 K. If = 4.5 K (Nb3Sn or other) and <sup>0</sup> is the same, then is 4 times larger, one can have = 0.6 × 10<sup>36</sup> in CW mode already at = 100 MW, instead of 410 MW, that would be great.

## **9 Ways to reduce power consumption**

The main power is spent on heat removal from RF and HOM losses. Two ways are seen: a) the use of a superconductor with a higher operating temperature, and) a decrease in the RF frequency ≡ RF (in this section). A promising material is Nb3Sn, which operates at a temperature of 4.5 K, where the efficiency of heat removal is about 4 times higher than for Nb at T = 1.8 K (apart from the Carnot efficiency, the technical efficiency is also about 1.6 times higher) [\[27\]](#page-21-0). Its thermal conductivity is about 1000 times lower than that of niobium, so it is used in the form of a thin film on material with high thermal conductivity, such as niobium or copper. Cavities with Nb3Sn reach the same high <sup>0</sup> values as with Nb, although the technology is not yet reliable enough. As for Nb, the value of <sup>0</sup> ∝ 1/ is limited by the BCS surface conductivity BCS ∝ 2 ; therefore, it is advantageous to lower the RF frequency.

So, the transition from T = 1.8 K to T = 4.5 K increases the efficiency of heat removal by a factor of 4. Let us denote this factor by the letters . The RF power loss in cavities per unit length RF ∝ <sup>−</sup><sup>1</sup> ∝ (if = BCS). In addition, a decrease of leads to a decrease of HOM losses (per unit length): HOM ∝ 1/ 2 ∝ 2 . The minimum distance between bunches ∝ 1/ . As a result, the parameters in the Eq. [8.2](#page-14-1) have the following dependence: ∝ /(), ∝ 2 , ∝ 1/ . The luminosity for operation with DC, Eq. [8.4,](#page-14-0)

$$L \propto \frac{P}{\sqrt{abd}} \propto P \frac{\sqrt{\epsilon T}}{f}, \qquad \frac{L}{P} \propto \frac{\sqrt{\epsilon T}}{f}. \tag{9.1}$$

With = 4 and a 2-fold decrease of , we obtain a 4-fold increase of the luminosity at the same power.

For continuous operation (Eq. [8.7\)](#page-14-2) ∝ √︁ ( − )/(), the threshold power ∝ ∝ /. At = 4 and 2-fold decrease of it decreases 8 times! Earlier we obtained the required power for CW operation at 250 GeV equal to 410 MW, it will decrease to 50 MW, which is already acceptable.

In CW mode ∝ √︁ / ∝ 1/ √ at ∼ ∝ /, that gives / = √ / , the same gain as when working with DC.

Thus, a transition from = 1.8 K to = 4.5 K and a halving of the RF-frequency increases the luminosity by a factor of 4 and decreases the threshold power for continuous operation by a factor of 8. This is in the ideal case when the surface conductivity ≈ BSC. Such collider variants are also presented in the summary table.
# **10 From 250 to 500 GeV**

In this article, the main focus was on the high luminosity 250 GeV Higgs factory. An increase in luminosity gives a sensitivity to masses of new particles responsible for deviations of the Higgs coupling constants from the SM. However, there is also a great interest to higher energies: 2<sup>0</sup> =360 GeV (the top-quark threshold), 2<sup>0</sup> =500 GeV (the Higgs self-coupling).

Beside collision effects there is also a problem of emittance dilution in the beam delivery where horizontal dispersion is required for chromatic correction at the final focus. In the ERLC, the beam pass this region about 400 times during the damping time. Some increase of the length will be needed to solve this problem.

The continuous operation required twice more threshold power than at 2<sup>0</sup> = 250 GeV. The CW mode will be realistic in case of success with Nb3Sn cavities.

The expected luminosities at 2<sup>0</sup> =500 GeV are given in the Table [3](#page-18-0) and Fig. [4,](#page-17-0) they are about 3 times lower than at 2<sup>0</sup> =250 GeV for the same total powers.

## **11 Summary tables**

In this article on the new linear collider scheme, only the most important problems affecting luminosity are considered, there are many issues that require careful consideration by accelerator experts. The preliminary parameters of the collider with the energy 2<sup>0</sup> = 250 and 500 GeV are presented in Tables [2](#page-18-1) and [3.](#page-18-0) Each table contains 4 ERLC options and ILC.[1](#page-16-0) The dependence of the luminosities on the total power for various options is shown in the Fig. [4.](#page-17-0) Beam emittances are chosen similar to the ILC (just because 5 GeV arcs with wigglers looks like the ILC DR). The ERLC and ILC consist of the same elements: linear accelerators, arcs, compressors, but in the ILC the bunch passes this way once, while in the ERLC the damping time corresponds to about 400 revolution. Bunch compressors are of greatest concern, a decrease of the beam energy in the return loop may be required to reduce the emittance dilution. There are questions on transverse beam dynamics due to the HOMs. Hope the problem is not serious because the bunch charge is about 20 times lower than at the ILC. The contributions of the main energy consumers for two ERLC options at 2<sup>0</sup> = 250 GeV are shown in Table [4.](#page-19-0)

## **12 Conclusion**

Currently, the design of superconducting ILC is very similar to that of any room-temperature LC: the beams are used only once and superconductivity does not add much (a slight increase in efficiency, larger distances between bunches, looser tolerances and a lower peak klystron power). The luminosities are about the same. This scheme was laid down 40 years ago. Even earlier, there were proposals to use energy recovery in SC linear colliders, but they were considered unattractive, since the expected luminosity was much lower than in single-pass colliders.

In this article, I propose a way to overcome the main obstacle faced by SC linear colliders with energy recovery: parasitic collisions in linacs. The proposed scheme of a *twin* linear collider opens the way to an energy-recovery LC with multiple use of beams. At 2<sup>0</sup> =250–500 GeV, the possible luminosity is up to two order of magnitude higher than at the ILC and much higher than at the FCC.

<span id="page-16-0"></span><sup>1</sup>Please note, for the ILC the total power is given in the tables. Linac itself consumes about 1/3 at 2<sup>0</sup> = 250 GeV (large fraction add sources and damping rings) and 2/3 at 2<sup>0</sup> = 500 GeV.

![](1__page_17_Figure_0.jpeg)

**Caption:** Figure 4: This figure illustrates the dependence of luminosity (L) on total power (P_tot) for two different scenarios at 2E0 = 250 GeV (top) and 2E0 = 500 GeV (bottom). The y-axis denotes the factor L/10^36 cm^2 s^-1, indicating the luminosity, while the x-axis shows P_tot in MW, representing total power consumption. Solid blue lines represent operation at the optimum duty cycle, whereas red dashed lines indicate continuous (CW) operation. This visualization highlights the challenges of power consumption in achieving high luminosities with superconducting technologies like Nb3Sn and Nb. Achieving a continuous operational mode requires significantly higher power, which raises concerns regarding feasibility and economic viability for particle collider designs, especially in achieving the desired luminosity levels within current technological constraints .


<span id="page-17-0"></span>**Figure 4**. Dependence of the luminosity on the total power for 2<sup>0</sup> = 250 GeV (upper) and 2<sup>0</sup> = 500 GeV (bottom); blue (solid) line — optimum duty cycle operation, red (dashed) curves — continuous (CW) operation, see the text.

|                             | unit                 | ERLC<br>pulsed   | ERLC<br>pulsed   | ERLC<br>contin.       | ERLC<br>contin. | ILC       |
|-----------------------------|----------------------|------------------|------------------|-----------------------|-----------------|-----------|
|                             |                      | Nb               | Nb               | Nb3Sn                 | Nb3Sn           | Nb        |
|                             |                      | 1.8 K            | 1.8 K            | 4.5 K                 | 4.5 K           | 1.8 K     |
|                             |                      | 1.3 GHz          | 0.65 GHz         | 1.3 GHz               | 0.65 GHz        | 1.3 GHz   |
| Energy 2𝐸0                  | GeV                  | 250              | 250              | 250                   | 250             | 250       |
| Luminosity Ltot             | 1036 cm−2<br>−1<br>s | 0.39             | 0.75             | 0.83                  | 1.6             | 0.0135    |
| 𝑃<br>(wall) (collider)      | MW                   | 120              | 120              | 120                   | 120             | 129(tot.) |
| Duty cycle, 𝐷𝐶              |                      | 0.19             | 0.37             | 1                     | 1               | n/a       |
| Accel. gradient, 𝐺          | MV/m                 | 20               | 20               | 20                    | 20              | 31.5      |
| Cavity quality, 𝑄           | 1010                 | 3                | 12               | 3                     | 12              | 1         |
| Length 𝐿act/𝐿tot            | km                   | 12.5/30          | 12.5/30          | 12.5/30               | 12.5/30         | 8/20      |
| 𝑁<br>per bunch              | 109                  | 1.13             | 2.26             | 0.46                  | 1.77            | 20        |
| Bunch distance              | m                    | 0.23             | 0.46             | 0.23                  | 0.46            | 166       |
| Rep. rate, 𝑓                | Hz                   | 2.47<br>108<br>· | 2.37<br>108<br>· | 1.3<br>109<br>·       | 6.5<br>108<br>· | 6560      |
| 𝜖𝑥, 𝑛/𝜖𝑦, 𝑛                 | 10−6 m               | 10/0.035         | 10/0.035         | 10/0.035              | 10/0.035        | 5/0.035   |
| 𝛽<br>∗<br>/𝛽𝑦<br>at IP<br>𝑥 | cm                   | 2.7/0.031        |                  | 10.8/0.031 0.46/0.031 | 6.8/0.031       | 1.3/0.04  |
| 𝜎𝑥<br>at IP                 | 𝜇m                   | 1.05             | 2.1              | 0.43                  | 1.66            | 0.52      |
| 𝜎𝑦<br>at IP                 | nm                   | 6.2              | 6.2              | 6.2                   | 6.2             | 7.7       |
| 𝜎𝑧<br>at IP                 | cm                   | 0.03             | 0.03             | 0.03                  | 0.03            | 0.03      |
| (𝜎𝐸/𝐸0)BS<br>at IP          | %                    | 0.2              | 0.2              | 0.2                   | 0.2             | ∼<br>1    |

<span id="page-18-1"></span>**Table 2**. Parameters of + − linear colliders ERLC and ILC, 2<sup>0</sup> = 250 GeV.

<span id="page-18-0"></span>**Table 3**. Parameters of + − linear colliders ERLC and ILC, 2<sup>0</sup> = 500 GeV.

|                             | unit                 | ERLC<br>pulsed   | ERLC<br>pulsed   | ERLC<br>pulsed  | ERLC<br>contin. | ILC      |
|-----------------------------|----------------------|------------------|------------------|-----------------|-----------------|----------|
|                             |                      | Nb               | Nb               | Nb3Sn           | Nb3Sn           | Nb       |
|                             |                      | 1.8 K            | 1.8 K            | 4.5 K           | 4.5 K           | 1.8 K    |
|                             |                      | 1.3 GHz          | 0.65 GHz         | 1.3 GHz         | 0.65 GHz        | 1.3 GHz  |
| Energy 2𝐸0                  | GeV                  | 500              | 500              | 500             | 500             | 500      |
| Luminosity Ltot             | 1036 cm−2<br>−1<br>s | 0.174            | 0.342            | 0.412           | 0.78            | 0.018    |
| 𝑃<br>(wall) (collider)      | MW                   | 150              | 150              | 150             | 150             | 163(tot) |
| Duty cycle, 𝐷𝐶              |                      | 0.121            | 0.237            | 0.47            | 1               | n/a      |
| Accel. gradient, 𝐺          | MV/m                 | 20               | 20               | 20              | 20              | 31.5     |
| Cavity quality, 𝑄           | 1010                 | 3                | 12               | 3               | 12              | 1        |
| Length 𝐿act/𝐿tot            | km                   | 25/50            | 25/50            | 25/50           | 25/50           | 16/31    |
| 𝑁<br>per bunch              | 109                  | 1.13             | 2.26             | 0.685           | 1.23            | 20       |
| Bunch distance              | m                    | 0.23             | 0.46             | 0.23            | 0.46            | 166      |
| Rep. rate, 𝑓                | Hz                   | 1.57<br>108<br>· | 1.54<br>108<br>· | 6.1<br>108<br>· | 6.5<br>108<br>· | 6560     |
| 𝜖𝑥, 𝑛/𝜖𝑦, 𝑛                 | 10−6 m               | 10/0.035         | 10/0.035         | 10/0.035        | 10/0.035        | 10/0.035 |
| 𝛽<br>/𝛽𝑦<br>∗<br>at IP<br>𝑥 | cm                   | 7.7/0.089        | 31/0.089         | 2.85/0.089      | 9.4/0.089       | 1.1/0.04 |
| 𝜎𝑥<br>at IP                 | 𝜇m                   | 1.26             | 2.5              | 0.76            | 1.38            | 0.47     |
| 𝜎𝑦<br>at IP                 | nm                   | 7.4              | 7.4              | 7.4             | 7.4             | 5.9      |
| 𝜎𝑧<br>at IP                 | cm                   | 0.089            | 0.089            | 0.089           | 0.089           | 0.03     |
| (𝜎𝐸/𝐸0)BS<br>at IP          | %                    | 0.1              | 0.1              | 0.1             | 0.1             | ∼<br>1   |

|                           | Nb, 1.3 GHz, T=1.8 K                            | Nb3Sn, 0.65 GHz, T=4.5 K                     |  |
|---------------------------|-------------------------------------------------|----------------------------------------------|--|
|                           | 𝐿<br>0.39<br>1036 cm−2<br>−1<br>×<br>=<br>s     | 𝐿<br>1.6<br>1036 cm−2<br>−1<br>×<br>=<br>s   |  |
|                           | 𝑁<br>1.13<br>, 𝐷𝐶<br>0.19<br>109<br>=<br>×<br>= | 𝑁<br>1.77<br>, 𝐷𝐶<br>109<br>=<br>×<br>=<br>1 |  |
| Beam generation           | small                                           | small                                        |  |
| Radiation in wigglers     | 4.45                                            | 18.4                                         |  |
| HOMs, beam energy         | 5.5                                             | 9                                            |  |
| HOMs cool., 1.8(4.5) K    | 24.8                                            | 10                                           |  |
| HOMs cool., 77 K          | 27.6                                            | 44.7                                         |  |
| RF diss.cool., 1.8(4.5) K | 57.6                                            | 38                                           |  |
| 𝑃𝑡𝑜𝑡, MW                  | 120                                             | 120                                          |  |

<span id="page-19-0"></span>**Table 4**. Power consumption of two ERLC options at 2<sup>0</sup> = 250 GeV, presented in the first and fourth columns of the Table [2.](#page-18-1)

To achieve the best performance, additional R&D efforts are required to develope superconducting cavities that reliably operate at 4.5 K with high <sup>0</sup> values.

## **Acknowledgment**

This work was supported by the Russian Foundation for Basic Research grant RFBR 20-52-12056. I would like to thank U. Amaldi, A. Bondar, A. Hutton, N. Dikansky, M. Klein, E. Levichev, P. Logatchev, I. Meshkov, V. Parkhomchuk, F. Richard, B. Sharkov, A. Skrinsky, S. Stapnes and N. Vinokurov for their interest to this proposal and Akira Yamamoto and Kaoru Yokoya for useful remarks. Following my talk at the LCWS-2021, a dedicated HEP sub-panel was formed to evaluate this proposal. It is very important to quickly understand the feasibility of the project and, if everything is realistic, move towards the goal.

## **References**

- [1] G. A. Loew et al., Intern. Linear Collider Technical Review Committee, 2003. [ILC-TRC-2003](https://www.slac.stanford.edu/xorg/ilc-trc/2002/2002/report/PAPERS/TRC03SC.PDF)
- [2] T. Behnke et al., "The Intern. Linear Collider Technical Design Report Volume 1: Executive Summary," [arXiv:1306.6327 \[physics.acc-ph\].](https://arxiv.org/ftp/arxiv/papers/1306/1306.6327.pdf)
- [3] P. Bambade et al., "The Intern. Lin. Collider: A Global Project", [arXiv:1903.01629 \[hep-ex\] \(2019\).](https://arxiv.org/pdf/1903.01629.pdf)
- [4] M. Aicheler et al., "A Multi-TeV Linear Collider Based on CLIC Technology: CLIC Conceptual Design Report," [CERN-2012-007.](https://inspirehep.net/files/d3a7fa3945ffcab02a8d62eba876627b) [doi:10.5170/CERN-2012-007.](http://cds.cern.ch/record/1500095)
- [5] M. Tigner, "A possible apparatus for electron clashing-beam experiments," Nuovo Cim. **37** (1965), 1228-1231 [doi:10.1007/BF02773204](https://link.springer.com/article/10.1007/BF02773204) .
- [6] see U. Amaldi, "Evolution of the concepts and the goals of linear colliders," [CERN-PPE-92-020.](https://lib-extopc.kek.jp/preprints/PDF/1992/9206/9206285.pdf)
- [7] U. Amaldi, "A Possible Scheme to Obtain − − and + <sup>−</sup> Collisions at Energies of Hundreds of GeV," Phys. Lett. B **61** [\(1976\), 313-315.](https://www.sciencedirect.com/science/article/abs/pii/037026937690157X?via%3Dihub)
- [8] H. Gerke and K. Steffen, "Note on a 45-GeV 100-GeV 'Electron Swing' Colliding Beam Accelerator," [DESY-PET-79-04.](https://lib-extopc.kek.jp/preprints/PDF/1979/7911/7911031.pdf)
- [9] H. Wiedemann, "Particle Accelerator Physics" (Springer–Verlag, 2007).
- [10] S. Noguchi and E. Kako, "Multi-beam accelerating structures," [KEK-PREPRINT-2003-130.](https://lib-extopc.kek.jp/preprints/PDF/2003/0327/0327130.pdf)
- [11] C. Wang, J. Noonan, J. Lewellen, "Dual-axis energy-recovery linac", [in Proceedings of ERL-07,](https://accelconf.web.cern.ch/erl07/papers/18.pdf) [Daresbury, 2007](https://accelconf.web.cern.ch/erl07/papers/18.pdf)
- [12] H. Park, S. De Silva, J. Delayen, A. Hutton and F. Marhauser, "Development of a Superconducting Twin Axis Cavity, [doi:10.18429/JACoW-LINAC2016-THPLR037.](https://accelconf.web.cern.ch/linac2016/papers/thplr037.pdf)
- [13] I. V. Konoplev, K. Metodiev, A. J. Lancaster, G. Burt, R. Ainsworth and A. Seryi, "Experimental studies of 7-cell dual axis asymmetric cavity for energy recovery linac", [Phys. Rev. Accel. Beams](https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.20.103501) **20** [\(2017\) no.10, 103501.](https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.20.103501)
- [14] K. Yokoya, "Quantum Correction to Beamstrahlung Due to the Finite Number of Photons," [Nucl.](https://www.sciencedirect.com/science/article/abs/pii/0168900286911447?via%3Dihub) [Instrum. Meth. A](https://www.sciencedirect.com/science/article/abs/pii/0168900286911447?via%3Dihub) **251** (1986), 1.
- [15] R. J. Noble, "Bremsstrahlung From Colliding Electron-Positron Beams With Negligible Disruption," [Nucl. Instrum. Meth. A](https://www.sciencedirect.com/science/article/abs/pii/0168900287902841?via%3Dihub) **256** (1987), 427.
- [16] K. Yokoya and P. Chen, "Beam-beam phenomena in linear colliders," [Lect. Notes Phys.](https://www.researchgate.net/publication/2278298_Beam-Beam_Phenomena_In_Linear_Colliders) **400** (1992), [415-445.](https://www.researchgate.net/publication/2278298_Beam-Beam_Phenomena_In_Linear_Colliders)
- [17] V. I. Telnov, "Restriction on the energy and luminosity of + − storage rings due to beamstrahlung," Phys. Rev. Lett. **110** [\(2013\), 114801.,](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.114801) [arXiv:1203.6563.](https://arxiv.org/pdf/1203.6563.pdf)
- [18] V. I. Telnov, "Issues with current designs for + − and gamma-gamma colliders," [PoS Photon2013](https://inspirehep.net/files/318ba516bccc34a8c2670f7b460fff34) [\(2013\), 070.](https://inspirehep.net/files/318ba516bccc34a8c2670f7b460fff34)
- [19] LCLS-II Final Design Report, SLAC, Menlo Park, CA, USA, Nov. 2015 [LCLSII-1.1-DR-0251-R0.](https://portal.slac.stanford.edu/sites/ad_public/people/galayda/Shared_Documents/FDR.pdf)
- [20] T. Raubenheimer, "The LCLS-II-HE, A High Energy Upgrade of the LCLS-II," [doi:10.18429/JACoW-FLS2018-MOP1WA02.](https://accelconf.web.cern.ch/fls2018/papers/mop1wa02.pdf)
- [21] W. J. Schneider, P. Kneisel and C. H. Rode, "Gradient Optimization for SC CW Accelerators," [Conf.](https://accelconf.web.cern.ch/p03/PAPERS/RPAB063.pdf) Proc. C **030512** [\(2003\), 2863 PAC03-RPAB063.](https://accelconf.web.cern.ch/p03/PAPERS/RPAB063.pdf)
- [22] D. Gonnella, S. Aderhold, D. Bafia, A. Burrill, M. Checchin, M. Ge, A. Grassellino, G. Hays, M. Liepe and M. Martinello, *et al.,* "The LCLS-II HE High Q and Gradient R&D Program," [doi:10.18429/JACoW-SRF2019-MOP045.](https://inspirehep.net/files/a70f7615b53db40181e6be4f231dea4d)
- [23] S. Posen, A. Romanenko, A. Grassellino, O. S. Melnychuk and D. A. Sergatskov, "Ultralow Surface Resistance via Vacuum Heat Treatment of Superconducting Radio-Frequency Cavities," [Phys. Rev.](https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.13.014024) Applied **13** [\(2020\) no.1, 014024.](https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.13.014024)
- [24] H. Padamsee, "50 years of success for SRF accelerators—a review," [Supercond. Sci. Technol.](https://iopscience.iop.org/article/10.1088/1361-6668/aa6376/pdf) **30** [\(2017\) no.5, 053003.](https://iopscience.iop.org/article/10.1088/1361-6668/aa6376/pdf)
- [25] R. B. Palmer, "A Qualitative Study of Wake Fields for Very Short Bunches," Part. Accel. **25** (1990), 97-106. [SLAC-PUB-4433.](https://slac.stanford.edu/pubs/slacpubs/4250/slac-pub-4433.pdf)
- [26] A. Novokhatsky, M. Timm and T. Weiland, "Single bunch energy spread in the TESLA cryomodule," [DESY-TESLA-99-16.](https://flash.desy.de/sites2009/site_vuvfel/content/e403/e1644/e1418/e1420/infoboxContent1438/tesla1999-16.pdf)

[27] S. Posen and D. L. Hall, "Nb3Sn superconducting radiofrequency cavities: fabrication, results, properties, and prospects," [Supercond. Sci. Technol.](https://iopscience.iop.org/article/10.1088/1361-6668/30/3/033004/pdf) **30** (2017) no.3, 033004