# arXiv:2306.13071

**Paper ID:** 18bd575f0c5d45106b6f6eb388e1b47f

**PDF Path:** apl/Superconductors/arXiv:2306.13071.pdf

**Processing Status:** complete

**Captions Added:** 12

**Generated:** 2025-06-24T13:44:27.801181

---

#### **Characteristic length for pinning force density in Nb3Sn**

E. F. Talantsev1,2 , E. G. Valova-Zaharevskaya<sup>1</sup> , I. L. Deryagina<sup>1</sup> , E. N. Popova<sup>1</sup>

<sup>1</sup> M.N. Miheev Institute of Metal Physics, Ural Branch, Russian Academy of Sciences, 18, S. Kovalevskaya St., Ekaterinburg, 620108, Russia

<sup>2</sup>NANOTECH Centre, Ural Federal University, 19 Mira St., Ekaterinburg, 620002, Russia

# **Abstract**

The pinning force density ⃗⃗⃗ ( , ) = ⃗⃗ × ⃗ (where *J*<sup>c</sup> is the critical current density and *B* is the applied magnetic field) is one of the main parameters that characterize the resilience of a superconductor to carry a dissipative-free transport current in an applied magnetic field. Kramer (1973 *J. Appl. Phys.* **44** 1360), and Dew-Hughes (1974 *Phil. Mag.* **30** 293) proposed a widely used scaling law for the pinning force density amplitude: | ⃗⃗⃗ ()| = , × (+) + × ( 2 ) × (1 − 2 ) , where ,, 2, *p*, and *q* are free-fitting parameters. Since the late 1970-s till now, several research groups have reported experimental data on the dependence of , on the average grain size, , in Nb3Sn-based conductors. Godeke (2006 *Supercond. Sci. Techn.* **19** R68) proposed that the dependence obeys the law |,()| = × (1⁄) + . However, this scaling law has several problems, for instance, the logarithm is taken from a non-dimensionless variable, and |,()| < 0 for large grain sizes, and |,()| → ∞ for → 0. Here, we reanalysed the full inventory of publicly available ,() data for Nb3Sn conductors and found that the dependence can be described by ,() = ,(0) × (− ⁄) law, where the characteristic length, , varies within a remarkably narrow range, that is, = 175 ± 13 , for samples fabricated by different technologies. The interpretation of the result is based on the idea that the in-field supercurrent flows within a thin surface layer (thickness of ) near the grain boundary surfaces (similar to London's law, where the self-field supercurrent flows within a thin surface layer with a thickness of the London penetration depth, , and the surface is a superconductor-vacuum surface). An alternative interpretation is that represents the characteristic length of the exponential decay flux pinning potential from the dominant defects in Nb3Sn superconductors, which are grain boundaries.

#### **Characteristic length for pinning force density in Nb3Sn**

## **1. Introduction**

Multifilamentary superconducting Nb3Sn-based wires are used in many high-energy physics and fusion energy projects, including international mega-science projects such as the Large Hadron Collider (LHC) [1] and International Thermonuclear Experimental Reactor (ITER) [2]. The Nb3Sn-based superconductors should, first of all, have high current-carrying capacity in high magnetic fields. In particular, the modernization of LHC [3] involves the replacement of a part of NbTi conductors with Nb3Sn-based conductors. Particularly, to create high-field large-aperture quadrupole MQXF [4] and high-field 11-T dipoles [5] for the high-luminosity LHC Upgrade Project, the development of a new generation of high-field Nb3Sn-based superconductors is required for the effective use of the advantages of Nb3Sn wires over previously used NbTi, to provide a minimum critical current of approximately 360 A and higher in a field of 15 T at 4.2 K [3]. The critical current density of the modern designed Nb3Sn strand has achieved record values of non-Cu ( = 12 T, = 4.2 K) = 3000 A mm<sup>2</sup> ⁄ and ( = 15 T, = 4.2 K) = 1700 A mm<sup>2</sup> ⁄ [6]. However, to create a Future Circular Collider (FCC) at CERN, Nb3Sn-based wires with ( = 16 T, = 4.2 K) = 1500 A mm<sup>2</sup> ⁄ or ( = 12 T, = 4.2 K) = 3500 A mm<sup>2</sup> ⁄ are required [7].

For thermonuclear power engineering, bronze-processed Nb3Sn-based wires were developed for superconducting magnets of the ITER project, providing a J<sup>c</sup> of approximately 750 A/mm<sup>2</sup> and higher in a field of 12 T [2]. The next mega-science project after the ITER should be the DEMO experimental facility, the primary goal of which is to demonstrate the possibility of obtaining a positive power balance from a thermonuclear reactor as the whole system. This goal requires the development of superconducting Nb3Sn-based conductors with even better characteristics [8].

Extensive (nearly five decades) R&D studies of Nb3Sn-based conductors have shown that the key factors affecting the in-field critical current in these wires are the local composition, structure, and morphology of the superconducting A-15 phase [9–17].

These studies also showed that at high magnetic fields, the main pinning centers in Nb3Snbased composites are grain boundaries, and the conventional approach to increasing (, ), in Nb3Sn is to increase the density of grain boundaries, that is, to ensure grain refinement. To achieve this, various manufacturing methods and designs of multifilamentary wires have been proposed [7], targeting the creation of small average grain sizes in the superconducting phase [18–21].

Superconducting wires based on Nb3Sn are produced by one of the following methods: bronze route, internal tin (IT), and power in tube (PIT) [22–24]. In the bronze route, an initial billet formed of Nb, Nb-Ti or Nb-Ta rods assembled in a bronze Cu-Sn matrix and external copper tube is extruded and drawn to a small diameter. The Nb3Sn phase is formed by Sn diffusion from the matrix to Nb filaments under heat treatment (HT), which is usually referred to as diffusion annealing. The solid-state diffusion of Sn at relatively low temperatures of HT prevents excessive grain growth and increases the pinning efficiency. The main disadvantage of the bronze method is the limited solubility of Sn in the bronze matrix when the Sn concentration increases to more than 8 mass. %, brittle phases are precipitated, which impedes plastic deformation and leads to cracking of the composite wire at the manufacturing stage. Therefore, to ensure a sufficient amount of Sn for the formation of the Nb3Sn phase, the ratio of the volume fractions of bronze and niobium should not be less than 3:1. Owing to these restrictions, bronze-processed wires have lower *J*cvalues than those potentially possible for the Nb3Sn phase. An important step in the development of bronze technology was the development of the Osprey method for producing high-tin bronze, which retains its plasticity up to 15–17 mas. % Sn. Using such bronze makes it possible to increase the number of Nb filaments in the strand, provide a complete transformation of Nb filaments into

the superconducting phase, and increase the Sn concentration in the Nb3Sn layers, which results in an increase of *J*<sup>c</sup> [25]. However, even in the Nb3Sn strands fabricated using a high-Sn bronze matrix, it is not possible to avoid large Nb3Sn composition gradients across the superconducting layer. These gradients, in turn, produce large gradients in the superconducting properties that limit the overall current density, particularly in high fields [9]. The deficiency of tin leads to the formation of a relatively large fraction of non-stoichiometric Nb3Sn compounds [26], which are stable from 18 to 25 at. % Sn, and the low-tin part of superconducting layers loses its superconductivity in high fields [27].

The IT process was developed to avoid frequent in-process annealing during wire drawing and to enhance the available Sn concentration with respect to the bronze process using separate Sn, Cu, and Nb billet stacking elements rather than specially melted high-Sn bronze matrix alloys [28]. The modified design of modern IT strands (e.g. strands with distributed diffusion barriers) makes it possible to obtain *J*<sup>c</sup> beyond 2200 A/mm<sup>2</sup> and achieve a record-braking value of 3000 A/mm<sup>2</sup> (non-copper, l2 T, 4.2 K) [12,29]. The highest critical current density strands have Nb3Sn layers with minimal chemical and microstructural inhomogeneity and a high fraction of the close to stoichiometric phase.

To increase the *J*<sup>c</sup> of superconductors designed to operate in high magnetic fields (15-16 T and higher), new designs of superconducting strands are created based on the IT technology, which are referred to as high-*J*<sup>c</sup> strands. According to Ref. [30], the OST company produces high-*J*<sup>c</sup> strands using the so called Restacked Rod Process (RRP) design of the wires. In the RRP strands, the many Cu-clad niobium filaments surrounding the tin source inside the subelement grow through the inter-filamentary Cu and formed a single Nb3Sn tube in the volume of the strand. Each subelement was surrounded by a Nb-Ta diffusion barrier, which was designed to partially react, and *J<sup>c</sup>* values of these strands were approximately 3000 A/mm<sup>2</sup> . The compositional analysis of the high-current wires indicated that the Sn content was relatively uniform at approximately 24 ± 1 at. % Sn in the A15 volume [31].

The PIT process [32] combines an abundant Sn source with a relatively high current density (over 2500 A/mm<sup>2</sup> ) and fine filaments (approximately 35 μm). The abundant Sn source results in a relatively high Sn content in the A15 phase. This indicates that the PIT wires contained a relatively large A15 fraction rich in Sn. The maximum non-Cu *J*<sup>c</sup> is from 2600 A/mm<sup>2</sup> (at 12 T, 4.2 K) in 1.25 mm wires, for superconducting wires, which were developed for the Next European Dipole (NED) program. The main advantages of the PIT process are shorter heat treatments because of the close location of the Sn source to the niobium, no pre-heating treatment is required compared to other methods, and relatively small filaments (30–50 µm) can be obtained, which leads to low hysteresis losses. The main disadvantage of the PIT manufacturing routine is its higher cost compared with other fabrication technologies [33,34].

The resilience of any superconducting wire to carry a dissipative-free transport current at an applied magnetic field can be quantified by the pinning force density, ⃗⃗⃗ , (defined as a vector product of the transport critical current density, ⃗⃗ , and the applied magnetic field, ⃗ ):

$$
\overrightarrow{F\_p}(f\_c, B) = \overrightarrow{f\_c} \otimes \overrightarrow{B}.\tag{1}
$$

For an isotropic superconductor and maximal Lorentz force geometry, i.e. when ⃗⃗ ⊥ ⃗ , Kramer [35] and Dew-Hughes [36] proposed a widely used scaling expression for the amplitude of the pining force density [37]:

$$\left|\overrightarrow{F\_p}(B)\right| = F\_{p,max} \times \frac{(p+q)^{p+q}}{p^p q^q} \times \left(\frac{B}{B\_{c2}}\right)^p \times \left(1 - \frac{B}{B\_{c2}}\right)^q,\tag{2}$$

where ,, 2, *p*, and *q* are free-fitting parameters, and 2 is the upper critical field, and , is pinning force density amplitude.

Figure 1 shows a typical | ⃗⃗⃗ (, 4.2 K)| for Nb3Sn superconductors reported by Flükiger *et al* [38], where the data fit to Equation (2) and deduced free-fitting parameters, ,, 2, *p*, and *q* are shown.

While the upper critical field, 2, is one of the fundamental parameters for a given superconducting phase, three other parameters in Equation (2), that is, ,, *p*, and *q*, depend on the superconductor microstructure, presence of secondary phases, and so on. In accordance with the approach proposed by Dew-Hughes [36], the shape of the | ⃗⃗⃗ ()| (defined by *p* and *q*) reflects the primary pinning mechanism in a sample.

![](_page_5_Figure_0.jpeg)

**Caption:** Figure 1 presents the pinning force density Fp as a function of applied magnetic field B for several bronze-route Nb3Sn samples with different average grain sizes. Each subplot (a-d) corresponds to specific average grain sizes: (a) d = 35 nm, (b) d = 74 nm, (c) d = 143 nm, and (d) d = 191 nm. The datasets from Flükiger et al. illustrate how varying the grain size influences superconducting properties like maximum pinning force density (Fp,max) and upper critical field (Bc2). The fits employ the Kramer-Dew-Hughes model, yielding specific parameters for each sample. Significant fit quality (0.9982 - 0.9997) for each suggests model reliability in describing the pinning mechanisms at play.


**Figure 1.** Pinning force density *F*<sup>p</sup> versus *B* for bronze-route wires of different average grain sizes, *d*: **(a)** *d* = 35 nm; deduced *F*p,max = 7.19 ± 0.02 GN/m<sup>3</sup> , *B*c2 = 19.2 ± 0.3 T, *p* = 0.68 ± 0.01, *q* = 2.7 ± 0.1; fit quality is 0.9997; **(b)** *d* = 74 nm; deduced *F*p,max = 4.99 ± 0.03 GN/m<sup>3</sup> , *B*c2 = 23 ± 1 T, *p* = 0.84 ± 0.03, *q* = 3.5 ± 0.3; fit quality is 0.9982; **(c)** *d* = 143 nm; deduced *F*p,max = 3.52 ± 0.02 GN/m<sup>3</sup> , *B*c2 = 21.4 ± 0.4 T, *p* = 0.55 ± 0.02, *q* = 1.8 ± 0.1; fit quality is 0.9987; **(d)** *d* = 191 nm; deduced *F*p,max = 2.45 ± 0.01 GN/m<sup>3</sup> , *B*c2 = 23.5 ± 0.3 T, *p* = 0.42 ± 0.01, *q* = 1.50 ± 0.07; fit quality is 0.9986. The *p* and *q* parameters for the fit were determined using the Kramer-Dew-Hughes equation (Equation (2)). Raw data reported by Flükiger *et al* [38]. 95% confidence bands are shown by pink shadow areas.

While the upper critical field, 2, is one of the fundamental parameters for a given superconducting phase, three other parameters in Equation (2), that is, ,, *p*, and *q*, depend on the superconductor microstructure, presence of secondary phases, and so on. In accordance with the approach proposed by Dew-Hughes [36], the shape of the | ⃗⃗⃗ ()| (defined by *p* and *q*) reflects the primary pinning mechanism in a sample. Dew-Hughes [36] calculated theoretical characteristic values for *p* and *q* for different pinning mechanisms, in particularly for point defect (PD) and grain boundary (GB) pinning.

The evolution of the dominant pinning mechanism from GB- to PD-pinning in Nb3Sn under neutron irradiation was recently reported by Wheatley *et al* [39], who showed that the unirradiated Nb3Sn alloy exhibits | ⃗⃗⃗ (, )| form indicating the dominance of the GB-pinning, and after the neutron irradiation the | ⃗⃗⃗ (, )| form transforms towards the PD-pinning mode.

The fourth parameter in Equation (2), which is the ,, represents the maximal performance of a given superconductor in an applied magnetic field. It is well-established experimental fact [38,40–46] that the , in Nb3Sn depends on the average grain size, , of the material. The traditional approach to representing the , vs. dependence is to use a reciprocal semi-logarithmic plot (Figure 2). Godeke [41] proposed the following form for the , vs. dependence:

$$F\_{p,max}(d) = A \times \ln(1/d) + B,\tag{3}$$

where free-fitting parameter = 22.7 and = −10.

Following traditional methodology [37], Godeke [41] proposed that because grain boundaries are primary pinning centers in Nb3Sn, there is an optimum grain size, , at which the maximum performance for a given wire can be achieved for a given applied magnetic field, *B*. This field [41] is equal to the flux line spacing in the hexagonal vortex lattice, ℎ [47], at the applied field , which can be designated as the matching field, ℎ, at the maximum pinning force density:

$$d\_{opt} = a\_{hexagonal} = \left(\frac{4}{3}\right)^{1/4} \times \left(\frac{\phi\_0}{B\_{match}}\right)^{1/2},\tag{4}$$

where <sup>0</sup> = ℎ 2 is superconducting flux quantum.

![](_page_7_Figure_2.jpeg)

**Caption:** Figure 2 depicts the relationship between maximum pinning force density |Fp,max| and reciprocal average grain size (1/d) for datasets reported by various researchers. The data, plotted in a reciprocal semi-logarithmic format, follow the model described by Godeke, using the linear function Fp,max(d) = A * ln(1/d) + B. The proposed fitting curve characterizes grain boundary pinning mechanisms predominant in Nb3Sn superconductors. Given the fitting parameters A = 22.7 and B = -10, this figure critically evaluates grain size optimization for maximum superconducting performance.


**Figure 2.** Maximum pinning force density, ,, vs. reciprocal average grain size, 1⁄, for datasets reported by Marken [42], West *et al* [43], Fischer [40], Shaw [44], Schauer *et al* [45], and Scanlan *et al* [46]. Fitting curve (Equation (3)) was proposed by Godeke [41], who also presented full dataset in the loglinear plot.

Here, we show that neither Equation (3) nor Equation (4) provides a valuable description of the available experimental ,() data measured over several decades in Nb3Sn conductors. We also propose a new model to describe a full set of publicly available experimental datasets on the maximum pinning force density vs. grain size, ,().

### **2. Problems associated with current models**

Equation (4) implies that if the grain size, , in some Nb3Sn conductors has been determined, then the matching applied magnetic field, ℎ, can be calculated as: 

$$B\_{match} \{ d\_{opt} \} = \left( \frac{4}{3} \right)^{1/2} \times \left( \frac{\phi\_0}{d\_{opt}^{-2}} \right). \tag{5}$$

Following this logic [41], one can expect that the maximal performance in magnetic flux pinning, that is, ,, should be observed at ℎ:

$$B\_{match} \{ d\_{opt} \} = B\_{Fp.max} \{ d\_{opt} \} = \left( \frac{4}{3} \right)^{1/2} \times \left( \frac{\phi\_0}{d\_{opt}^{-2}} \right). \tag{6}$$

In Figure 1, we fitted | ⃗⃗⃗ ()| data [38] to Equation (1) for Nb3Sn conductors with different grain sizes, , from which the .,() were extracted. In Figure 3, we show .,() and calculated .,() (Equation (6)), from which it can be concluded that the traditional understanding of the primary mechanism governing dissipative-free high-field current capacity in Nb3Sn conductors [41] is incorrect.

The validity of the ,() scaling law proposed by Godeke (Equation (3) [41]) was analyzed and it was concluded that there are at least three fundamental problems with the law:

1. The logarithmic function used in Equation (3), as well as all other mathematical functions, can operate only with the dimensionless variable, whereas the variable in Equation (3) has the dimension of inverse length. For instance, the variable in the Kramer-Dew-Hughes scaling law (Equation (2)) has the dimension cancelation term <sup>1</sup> 2 . The same general approach can be found for all equations in Ginzburg-Landau [47], Bardeen-Cooper-Schrieffer [48], and other physical theories [49], all of which implement this general rule.

For instance, the lower critical field, 1, in superconductors has traditional form [50]:

$$B\_{c1}(T) = \frac{\phi\_0}{4\pi\lambda^2(T)} \times \left(\ln\left(\frac{\lambda(T)}{\xi(T)}\right) + \alpha\left(\frac{\lambda(T)}{\xi(T)}\right)\right),\tag{7}$$

where

$$\alpha(\kappa) = \alpha\_{\infty} + e^{\left(-c\_0 - c\_1 \times \ln\left(\frac{\lambda(T)}{\xi(T)}\right) - c\_2 \times \left(\ln\left(\frac{\lambda(T)}{\xi(T)}\right)\right)^2\right)} \pm \varepsilon,\tag{8}$$
where () is the London penetration depth, () is the superconducting coherence length, <sup>∞</sup> = 0.49693, <sup>0</sup> = 0.41477, <sup>1</sup> = 0.775, <sup>2</sup> = 0.1303, and ≤ 0.00076. Equations (7), (8) were recently simplified to the following form [51]:

![](0__page_9_Figure_1.jpeg)

**Caption:** Figure 1 shows the pinning force density Fp versus magnetic field B for bronze-route Nb3Sn wires with varying average grain sizes, d. The figure presents data for four samples, each with different grain sizes and corresponding deduced parameters such as maximum pinning force density, upper critical field, and fitting coefficients. This demonstrates the pinning mechanism and performance variability dependent on grain size .


()

**Figure 3.** ., was calculated using Equation (4) (red) [41] and ., was extracted from experimental data reported by Flükiger [38] for Nb3Sn conductors fabricated by bronze technology.

In Equations (7) and (9) the variable under the logarithm is dimensionless. The same can be found in the equation for the universal self-field critical current density, <sup>c</sup> (, ), in thin film superconductors [52]:

$$f\_{\mathcal{E}}\{\mathcal{sf},T\} = \frac{\phi\_0}{4\pi\mu\_0\lambda^3(T)} \times \left(\ln\left(\frac{\lambda(T)}{\xi(T)}\right) + 0.5\right),\tag{10}$$

where <sup>0</sup> is the permeability of the free space. It should be noted that Equation (10) was recently confirmed by Paturi and Huhtinen [53] for YBa2Cu3O7- thin films that exhibit different mean-free paths for charge carriers. Equation (10) was also recently used by Troyan *et al* [54] to deduce the ground state London penetrations depth, (0), and ground state superconducting energy gap, Δ(0), in highly-compressed hydrogen-rich superconductor 4. 

The same principle of unitless variable is implemented in all general physics laws, for instance, in Planck's law [49]:

$$B\_{\nu}\{\nu, T\} = \frac{2h\nu^3}{c^2} \times \frac{1}{e^{\left(\frac{h\nu}{k\_BT}\right)} - 1},\tag{11}$$

where (, ) is the spectral radiance of a body, ℎ is the Planck constant, is the frequency, is the speed of light in the medium, is the Boltzmann constant, and where the variable under the exponential function, ℎ , is dimensionless.

Based on all above, Equation (3) has a fundamental mistake based on a simple fact that (1⁄) is an absurdum expression.

2. Even if the problem mentioned above (i.e. in #1) is omitted, there are two other problems associated with Equation (3). One problem is the limit of Equation (3) for a large grain size. In Figure 4, we replotted |,()| data from Figure 2 in a linear-linear plot and showed both side extrapolations of Equation (3) within the range of 20 ≤ ≤ 800 , which is the usual range of grain sizes in Nb3Sn conductors. In Figures 2, 4 one can see that:

$$\left| F\_{p,max}(d) \right|\_{d \ge 550 \text{ nm}} = (A \times \ln(1/d) + B)|\_{d \ge 550 \text{ nm}} < 0,\tag{12}$$

which is the absurdum. We also noted that the free-fitting parameters deduced by us ( = 21.9 ± 1.2 , = −9.9 ± 2.7) from the fit of the |,()| dataset to Equation (3), are different from the values reported by Godeke [41], = 22.7, = −10, who analysed the same |, ()| dataset.

3. A similar validity problem of Equation (3) is for small grain sizes:

$$\lim\_{d \to 0} \left| F\_{p,max}(d) \right| = \lim\_{d \to 0} (A \times \ln(1/d) + B) = \infty,\tag{13}$$

which is unphysical because when becomes comparable to the double coherence length (which is the size of a normal vortex core):

$$d\_{\min} \{ 4.2 \text{ K} \} \cong 2 \times \xi \{ T \} = 2 \times \frac{\xi(0)}{\sqrt{1 - \frac{T}{T\_c}}} = \frac{2 \times 3.0 \text{ nm}}{\sqrt{1 - \frac{4.2 \text{ K}}{18 \text{ K}}}} = 6.9 \text{ nm}, \tag{14}$$

11

where (0) = 3.0 nm [55] and = 18 K [55] were used, a further decrease in the grain size should not cause any changes in the magnetic flux pinning, and thus in |,()| amplitude.

![](0__page_11_Figure_1.jpeg)

**Caption:** Figure 4 provides a linear plot of |Fp,max(d)| data against various grain sizes derived from previous datasets. The data fits Equation (3) and explores extrapolations over a typical grain size range. This visualization is critical to evaluate existing models of Nb3Sn pinning mechanisms, exhibiting both physical and unphysical ranges for pinning density .


**Figure 4**. |,()| data from Figure 2 (reported by Fischer [40] and Godeke [41]) in linear-linear plot, and the fitting curve to Equation (3) [41], where we also showed both side extrapolations within the average grain size range of 20 ≤ ≤ 800 of Nb3Sn. Raw data reported by Marken [42], West *et al* [43], Fischer [40], Shaw [44], Schauer *et al* [45], and Scanlan *et al* [46].

#### **3. Results**

By experimenting with many analytical functions that can approximate |,()| dependence shown in Figures 2 and 4, we found a remarkably simple, robust, heuristic, and physically sounded expression:

$$\left| F\_{p,\max}(d) \right| = \left| F\_{p,\max}(0) \right| \times e^{-\frac{d}{\delta}} \tag{15}$$

where |,(0)| and are free fitting parameters. This function exhibits physically sounded limits:

$$\lim\_{d \to \infty} \left| F\_{p,max}(d) \right| = \lim\_{d \to \infty} \left( \left| F\_{p,max}(0) \right| \times e^{-\frac{d}{\delta}} \right) = 0,\tag{16}$$

$$\lim\_{d \to 0} |F\_{p,max}(d)| = \lim\_{d \to 0} \left( |F\_{p,max}(0)| \times e^{-\frac{d}{\delta}} \right) = |F\_{p,max}(0)| < \infty. \tag{17}$$

We proposed interpretations for |,(0)| and of parameters in the *Discussion* Section. Before that, in this Section we show the robustness of Equation (15) to fit publicly available datasets for Nb3Sn conductors. Data fitting was performed in Origin2017 software.

#### *3.1. Bronze technology samples*

Bronze technology for Nb3Sn-based wires has been described in detail elsewhere [1]. For our analysis, we used |,()| dataset reported by Godeke [41]. Godeke [56] pointed out that Fischer [40] collected raw |,()| data (shown in Figures 2 and 4), and these data are "*all pre-2002 results*" and this dataset includes Fischer's [41] "*the non-Cu area*" data.

In Figure 5, we fitted this largest publicly available dataset for Nb3Sn conductors fabricated using bronze technology to Equation (15). The deduced parameters were |,(0)| = 74 ± 3 GN m3 , and = 176 ± 12 nm. The parameters have low dependence (~ 0.87), which indicates that our model (Equation (15)) is not over-parameterized.

![](0__page_12_Figure_4.jpeg)

**Caption:** Figure 5 illustrates the maximum pinning force density, |Fp,max(d)|, as a function of average grain size, d, for non-Cu Nb3Sn wires. The data fit to Equation (15) with deduced parameters |Fp,max(0)| = 74 ± 3 GN/m³ and δ = 176 ± 12 nm. The raw data reported by various researchers indicate an excellent fit quality of 0.9248. This figure, with grey confidence bands, highlights the dependence of pinning force density on grain size, providing insight into Nb3Sn conductor performance under bronze technology fabrication .


**Figure 5.** Maximum pinning force density, |,()|, vs average grain size, , for the non-Cu Nb3Sn wires and data fit to Equation (15). Raw data reported by Marken [42], West *et al* [43], Fischer [40], Shaw

[44], Schauer *et al* [45], and Scanlan *et al* [46]. Nb3Sn conductors were fabricated by bronze technology. Deduced parameters are |,(0)| = 74 ± 3 GN m3 , = 176 ± 12 nm; fit quality is 0.9248. 95% confidence bands are shown by grey shadow areas.

### *3.2. Powder-in-tube technology samples*

Powder-in-tube technology for Nb3Sn-based wires has been described in detail elsewhere

[1]. For our analysis, we used |,()| dataset reported by Fischer [40] and Xu *et al* [57]. In Figure 6, we show the results of the fit of this dataset to Equation (15).

It is interesting to note that the deduced = 175 ± 13 nm is in remarkable agreement with its counterpart deduced for samples fabricated by bronze technology. The deduced parameters also have low dependence (~ 0.87), which is an additional indication that our model (Equation (15)) is not over-parameterized.

![](0__page_13_Figure_5.jpeg)

**Caption:** Figure 6 shows maximum pinning force density, |Fp,max(d)|, versus average grain size, d, for the A15 layer made via powder-in-tube technology. Data from Fischer and Xu fit to Equation (15), where |Fp,max(0)| = 189 ± 11 GN/m³ and δ = 175 ± 13 nm. This figure highlights the consistency of pinning characteristics across fabrication techniques, enhanced by visible pink confidence bands.


**Figure 6.** Maximum pinning force density, |,()|, vs average grain size, , for the A15 layer fabricated by powder-in-tube technology [40,57]. Raw data reported by Fischer [40] and Xu *et al* [57]. Deduced parameters are |,(0)| <sup>=</sup> <sup>189</sup> <sup>±</sup> <sup>11</sup> GN m3 , = 175 ± 13 nm; fit quality is 0.9093. 95% confidence bands are shown by pink shadow areas.

# *3.3. Samples fabricated by Flükiger et al by bronze technology* **[38]**

Flükiger *et al* [38] reported full | ⃗⃗⃗ ()| curves, which we analysed in Figure 1, for four samples fabricated using bronze technology. It should be noted that this research group utilized a different normalization procedure for the absolute value of the pinning force density from that used by other research groups [40,42–46]. Therefore, we analyzed this dataset separately (Figure 7). Although this dataset has only four |,()| data points, we fitted this dataset to Equation (15) to estimate the robustness of our approach for extracting the characteristic length, , from limited |,()| datasets. The deduced = 146 ± 15 nm is in the same ballpark as the values deduced from the fits to Equation (15) for large datasets (Figures 5, 6).

![](0__page_14_Figure_1.jpeg)

**Caption:** Figure 7 illustrates the maximum pinning force density, |Fp,max(d)|, compared to average grain size, d, specific to samples fabricated by bronze technology, fit to Equation (15). The derived parameters include |Fp,max(0)| = 8.9 ± 0.5 GN/m³ and δ = 146 ± 15 nm. This analysis discerns intrinsic length constants crucial for optimizing Nb3Sn conductor performance, delineated by pink confidence bands .


**Figure 7.** Maximum pinning force density, |,()|, vs average grain size, , for samples fabricated by bronze technology and data fit to Equation (15). Raw data reported by Flükiger *et al* [38]. Deduced parameters are |,(0)| = 8.9 ± 0.5 GN m3 , = 146 ± 15 nm. fit quality is 0.9837. 95% confidence bands are shown by pink shadow areas.

# **4. Discussion**

Primary result of our analysis is that Nb3Sn conductors exhibit fundamental length constant, , which is in the range of 146 nm ≤ ≤ 176 nm, and which characterizes maximal intrinsic in-field performance of real world multifilamentary Nb3Sn-based wires.

Our current understanding of this unexpected result can be explained by two hypotheses, both of which are based on the interpretation that one of the two multiplication terms in the formal definition of the pinning force density (Equation (1)), ⃗⃗⃗ ( , ) = ⃗⃗ ⊗ ⃗ , exhibits exponential decay with characteristic length . Thus, there are two possible scenarios/mechanisms:

#### *4.1. Exponential dependence of the* | ⃗ | *vs grain size at* |,|

This interpretation is based on an analog to the exponential decay ~ − (more accurately ~ ℎ( ) ℎ( ) dependence, where is the slab half-thickness and the layer thickness is London penetration depth [55]) of the self-field transport current density from the superconductor-vacuum interface, which is the London's law. Considering that under high-field conditions, the interfaces in polycrystalline Nb3Sn are grain boundaries, we naturally came to Equation (15), where the thickness of the layer (where the dissipative-free transport current flows at the condition of the pinning force maximum) is the characteristic length .

A schematic representation of -layers in the polycrystalline Nb3Sn phase, where we drew the -layer, is shown in Figure 8.

![](0__page_15_Figure_4.jpeg)

**Caption:** Figure 8 exhibits a schematic representation of effective areas, referred to as δ-layers, in the cross-section of an equiaxed Nb3Sn layer. It visualizes the grain boundaries and effective regions for dissipative-free transport current, emphasizing the concept of δ as a characteristic length where the central areas of large grains are less effective for current transport compared to small grains .


**Figure 8.** Schematic representation of the effective areas (-layer) in a cross-section of the eqiaxed Nb3Sn layer.

In this interpretation, large-size grains, ≫ , are less effective areas to carry dissipativefree transport current, because central areas of these large grains do not contribute in transferring the transport current (Figure 8), and the current density is reduced by the exponential law. At the same time, small grains, ≤ , are very effective areas for carrying dissipative-free transport current flow (Figure 8), because the full grain cross-section area works with approximately the same efficiency.

# *4.2. Exponential dependence of the* |⃗⃗ | *vs grain size at* |,|

Alternative interpretation is based on an assumption that the flux pinning potential has exponential dependence ~ − . As a result, the dissipative-free current can flow only within a thin layer (the thickness of ) from both sides of grain boundaries, because the flux pinning is strong there and vortices can be hold by the potential vs the Lorentz force. In this interpretation, central areas of large-size grains, ≫ , also do not contribute to transfer dissipative-free in-field transport current, because vortices are not hold strong enough vs the Lorentz force. While, the small-size grains, ≤ , are very effective to carry dissipative-free transport current flow (Figure 8), because vortices are pinned by pinning potential across full grain area cross-section.

It is interesting to note that the schematic for the effective areas that can carry dissipativefree transport current is the same for both scenarios (Figure 8).

Thus, our current interpretation of the result is that the highest performance of the in-field transport current capacity of Nb3Sn wires is determined by the thin layer with characteristic thickness of ≅ 175 nm which surrounds the grain boundaries from both sides.

# **5. Refined model**

Both our interpretations (Sections 4.1, 4.2, and Figure 8) of the derived characteristic length are based on the idea of exponential spatial decay of the , toward the grain center. Based on this, we came to a need to refine our primary fitting equation (Eq. 15) to account for the fact that the grain boundary surrounds the grain body and the total integrated , associated with the whole grain is a two-dimensional integral from the ,(, ) function within each grain (in this axis arrangement, we assume that the transport current is flowing along the *z*-axis).

In an attempt to compromise between the complexity of mathematics and the accuracy of describing the grain boundary network (observed experimentally [58], Figure 8]) we developed a square lattice model (Figure 9). This model can serve as a model on which the basic properties of other more accurate models can be depicted.

![](0__page_17_Figure_2.jpeg)

**Caption:** Figure 9 displays a schematic of a square grain model with red grain boundaries and green characteristic layers. This illustration supports the London-like model of effective pinning force density within Nb3Sn grains, illuminating the role of δ as a characteristic thickness affecting superconductor performance .


**Figure 9.** Schematic representation of the square grain model, where grain boundaries are indicated in red, and the characteristic thickness is green. The characteristic layer thickness, , and grain size, , are also shown.

In the square lattice model (Figure 9), the global maximum of the pinning force density for the sample, ,,, is the sum of the equal terms, which are the integrated maximum of the pinning force density of each grain:

$$\begin{aligned} F\_{p,max,global} \{ d \} &= \sum\_{n=1}^{N} \left( \int\_{Area} F\_{p,max,grain} \text{(x, y, d)} dx dy \right)\_n = N \times \\\\ \int\_{area} F\_{p,max,grain} \text{(x, y, d)} dx dy &= N \times F\_{p,max,single\,grain} \text{(d)}. \end{aligned} \tag{18}$$

In accordance with Eq. 18, the primary task is to calculate the integrated , over the area of one grain as follows:
$$F\_{p,max,single\ gain}(d) = \int\_{area} F\_{p,max,grain\ area}(\mathbf{x}, \mathbf{y}, \mathbf{d}) d\mathbf{x} d\mathbf{y}.\tag{19}$$

First, consider a one-dimensional problem (for which the grain is within (− 2 , 2 )) for which we can use the London theory solution for exponentially decaying current density or magnetic flux density for rectangular slabs [55,59–61]:

$$F\_{p,max,grain\,width} \left(\text{x}, d\right) = F\_{p,max,grain\,boundary} \times \frac{cosh\left(\frac{\text{x}}{\delta}\right)}{cosh\left(\frac{d}{2\delta}\right)}.\tag{20}$$

The integration of Eq. 20 gives:

 2 − 2

$$\int\_{\frac{d}{2}}^{\frac{d}{2}} F\_{p,max,grain\,width}(\mathbf{x}, d) d\mathbf{x} = F\_{p,max,grain\,boundary} \times \left(\frac{2\delta}{d}\right) \times \tanh\left(\frac{d}{2\delta}\right). \tag{21}$$

The solution for *y*-axis is similar to that for *x*-axis:

$$\int\_{\frac{d}{2}}^{\frac{d}{2}} F\_{p,max,grain\,length}(\mathbf{y}, d\mathbf{j}) d\mathbf{y} = F\_{p,max,grain\,boundary} \times \left(\frac{2\delta}{d}\right) \times \tanh\left(\frac{d}{2\delta}\right). \tag{22}$$

Considering that the London equation is a linear differential equation, the sum of two partial solutions is also the solution of the equation, and the solution for the two-dimensional problem can be written in the form:

$$F\_{p,max,single\ gain}(d) = \int\_{-\frac{d}{2}}^{\frac{d}{2}} F\_{p,max,grain\ width}(\mathbf{x}, d)d\mathbf{x} + \int\_{-\frac{d}{2}}^{\frac{d}{2}} F\_{p,max,grain\ length}(\mathbf{y}, d)d\mathbf{y} = 2 \times F\_{p,max,grain\ boundary} \times \left(\frac{2\delta\_L}{d}\right) \times \tanh\left(\frac{d}{2\delta\_L}\right),$$

where is the designation for the characteristic length of the two-dimensional square model (Figure 9), which is based on the London theory solution for current density and magnetic flux density distributions for a square slab [55,59–61].

Thus, the final fitting equation for the refined model is:

$$F\_{p,max,global}(d) \equiv F\_{p,max}(d) = F\_{p,max}(0) \times \left(\frac{2\delta\_L}{d}\right) \times \tanh\left(\frac{d}{2\delta\_L}\right),\tag{24}$$

where ,(0) = × 2 × ,, , and the other terms are defined above.

,() dataset (which we have already analyzed in Figs. 2,4,5) and data fit to Eq. (24) are shown in Figure 10.

![](1__page_19_Figure_0.jpeg)

**Caption:** Figure 10 presents maximum pinning force density, |Fp,max(d)|, versus average grain size, d, for non-Cu Nb3Sn wires, aligning with Equation (24) parameters: |Fp,max(0)| = 66 ± 3 GN/m³ and δ = 35 ± 3 nm. The dark yellow shadow areas denote the confidence bands. The study underscores the effects of the bronze technology fabrication on Nb3Sn wire performance in applied fields .


**Figure 10.** Maximum pinning force density, |,()|, vs. average grain size, , for the non-Cu Nb3Sn wires and the data fit to Equation (24). Raw data reported by Marken [42], West *et al* [43], Fischer [40], Shaw [44], Schauer *et al* [45], and Scanlan *et al* [46]. Nb3Sn conductors were fabricated using bronze technology. Deduced parameters are |,(0)| = 66 ± 3 GN m3 , = 35 ± 3 nm; fit quality is 0.8100. 95% confidence bands are shown as dark yellow shadow areas.

,() dataset for Nb3Sn powder-in-tube technology samples [40,57] (which we have

already analyzed in Figure 6) and data fit to Eq. (24) are shown in Figure 11.

![](1__page_19_Figure_4.jpeg)

**Caption:** Figure 11 illustrates the maximum pinning force density |Fp,max(d)| in relation to average grain size d for Nb3Sn samples fabricated using powder-in-tube technology. Raw data from Fischer and Xu is fit to Equation (24), and the deduced parameters are |Fp,max(0)| = 162 ± 16 GN/m³ and δL = 38 ± 5 nm with a fit quality of 0.8966. Pink shadow areas denote 95% confidence bands. This visualization emphasizes the consistency in pinning characteristics despite differing fabrication methods and underscores the potential universality of the derived characteristic length δL.


**Figure 11.** Maximum pinning force density, |,()|, vs. average grain size, , for the A15 layer fabricated by the powder-in-tube technology [40,57]. Raw data reported by Fischer [40] and Xu *et al* [57]. Data fit to Equation (24). Deduced parameters are |,(0)| <sup>=</sup> <sup>162</sup> <sup>±</sup> <sup>16</sup> GN m3 , = 38 ± 5 nm; fit quality is 0.8966. 95% confidence bands are shown by pink shadow areas. 

,() dataset for samples reported by Flükiger *et al* [38] (which we have already analyzed in Figure 7) and data fit to Eq. (24) are shown in Figure 12.

![](1__page_20_Figure_1.jpeg)

**Caption:** Figure 12 depicts the maximum pinning force density, |Fp,max(d)|, relative to average grain size, d, for samples fabricated by bronze technology. The data was fitted to Equation (24), yielding parameters |Fp,max(0)| = 7.9 ± 0.5 GN/m³ and δ = 30 ± 4 nm. The fit quality was high, at 0.9812, suggesting a reliable model for understanding Nb3Sn phase properties, independent of manufacturing technique .


**Figure 12.** The maximum pinning force density, |,()|, vs average grain size, , for samples fabricated using bronze technology and data fit to Equation (24). Raw data reported by Flükiger *et al* [38]. Deduced parameters are |,(0)| = 7.9 ± 0.5 GN m3 , = 30 ± 4 nm; fit quality is 0.9812. 95% confidence bands are shown by pink shadow areas.

The derived from the three available ,() datasets (Figs. 10-12) are within a narrow range:

$$30 \pm 4 \text{ nm} \le \delta\_L (T = 4.2 \text{ K}) \le 38 \pm 5 \text{ nm},\tag{25}$$

which is the evidence that perhaps represents a universal constant for the Nb3Sn phase, which is independent of the sample manufacturing technique.

If so, we can make a comparison of the ( = 4.2 ) with two other fundamental lengths of any superconductors, i.e. the London penetration depth, ( = 4.2 ), and the coherence length, ( = 4.2 ):

$$\frac{\delta\_L(T=4.2\,\,K)}{\lambda(T=4.2\,\,K)} = \frac{\delta\_L(T=4.2\,\,K)}{\frac{\lambda(0)}{\lambda(0)}} = 0.53 \cong \frac{1}{2},\tag{26}$$

$$\frac{\delta\_L(T=4.2\ K)}{\xi(T=4.2\ K)} = \frac{\delta\_L(T=4.2\ K)}{\frac{\xi(\text{o})}{\sqrt{1-\left(\frac{T=4.2\ K}{T\_C=18\ K}\right)}}} = 9.9 \cong 10,\tag{27}$$

where we used the two-fluid (Chapter 2 in [55]), and the Ginzburg-Landau theory expressions (Chapter 6 in [55]), and (0) = 65 (Table 12.1, p. 343 in [55]), (0) = 3 (Table 12.1, p. 343 in [55]), = 18 (Table 12.1, p. 343 in [55]), and average value for the derived characteristic length ̅̅̅( = 4.2 ) = 34 (Figures 10-12).

The obtained ratios (Equations 26, 27) demonstrate that the derived is half the London penetration depth, (), and one order of magnitude larger than the coherence length, ().

This implies that might represent a new characteristic length constant for superconductors, which lies between () and () for high- superconductors and is associated with the maximum superconductor performance in an applied magnetic field.

## **6. Conclusions**

Finning force maximum, |,( , )|, represents global maximum of the vector product of the transport critical current density, ⃗⃗ , and the applied magnetic field, ⃗ , and it can be derived as from | ()| [35–37], as from | ( )| [62] projections of the ( , ) function (Equation (1)).

In this report we re-analysed experimental data on the dependence of the maximum pinning force density, |,(, = 4.2 K)| (deduced from the | ()| [35–37] projection) from the average grain size in practical low-*T*<sup>c</sup> multifilamentary Nb3Sn conductors [1–34,38–46,55,56,62] fabricated by bronze and power-in-tube technologies.

The primary result of our analysis is that Nb3Sn conductors at their maximum in-field performance exhibit characteristic length = 175 nm, which is the same for samples fabricated by bronze and powder-in-tube technologies.

Following an interpretation that there is a characteristic thickness of the layer surrounding the grain boundary network, where a dissipative-free transport current flows, we developed London-like model in assumption of square Nb3Sn grains, for which the characteristic length, , was deduced in the range:

$$30 \pm 4 \text{ nm} \le \delta\_L (T = 4.2 \text{ K}) \le 38 \pm 5 \text{ nm}.\tag{28}$$

The comparison of the derived with two fundamental characteristic lengths of Nb3Sn, i.e. the London penetration depth, , and the coherence length, , shows that:

$$\frac{\delta\_L}{\lambda} = \frac{1}{2},\tag{30}$$

$$\frac{\delta\_L}{\delta} = 10,\tag{31}$$

which is more likely implied that represents a new characteristic length constant for superconductors, which lies between () and () for high- superconductors, and which is associated with the maximum superconductor performance in applied magnetic field.

**Author Contributions:** E.F.T. conceived the work and proposed exponential and hyperbolic tangent dependence for ,(), E.F.T. and E.G.V.-Z. searched publicly available experimental data and performed data fit and calculations, E.F.T. proposed to interpret as the characteristic thickness for transport current flow, E.G.V.-Z. proposed to interpret as the characteristic length for flux pinning potential. All authors discussed results. E.G.V.-Z. prepared final figures. E.F.T. wrote the manuscript, which was revised by E.G.V.-Z., I.L.D. and E.N.P.

**Funding:** The research was carried out within the state assignment of Ministry of Science and Higher Education of the Russian Federation (theme "Pressure" No. 122021000032-5). E.F.T. thanks the research funding from the Ministry of Science and Higher Education of the Russian Federation (Ural Federal University Program of Development within the Priority-2030 Program).

**Conflicts of Interest:** The authors declare no conflict of interest. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.

## **References**

- 1. Rossi, L.; Bottura, L. Superconducting Magnets for Particle Accelerators. *Rev. Accel. Sci. Technol.* **2012**, *05*, 51–89, doi:10.1142/S1793626812300034.
- 2. Tronza, V.I.; Lelekhov, S.A.; Stepanov, B.; Bruzzone, P.; Kaverin, D.S.; Shutov, K.A.; Vysotsky, V.S. Test Results of RF ITER TF Conductors in the SULTAN Test Facility.

*IEEE Trans. Appl. Supercond.* **2014**, *24*, 1–5, doi:10.1109/TASC.2013.2289361.

- 3. Ambrosio, G. Nb3Sn High Field Magnets for the High Luminosity LHC Upgrade Project. *IEEE Trans. Appl. Supercond.* **2015**, *25*, 1–7, doi:10.1109/TASC.2014.2367024.
- 4. Ferracin, P.; Ambrosio, G.; Anerella, M.; Borgnolutti, F.; Bossert, R.; Cheng, D.; Dietderich, D.R.; Felice, H.; Ghosh, A.; Godeke, A.; et al. Magnet Design of the 150 Mm Aperture Low-β Quadrupoles for the High Luminosity LHC. *IEEE Trans. Appl. Supercond.* **2014**, *24*, 1–6, doi:10.1109/TASC.2013.2284970.
- 5. Karppinen, M.; Andreev, N.; Apollinari, G.; Auchmann, B.; Barzi, E.; Bossert, R.; Kashikhin, V. V.; Nobrega, A.; Novitski, I.; Rossi, L.; et al. Design of 11 T Twin-Aperture Nb3Sn Dipole Demonstrator Magnet for LHC Upgrades. *IEEE Trans. Appl. Supercond.* **2012**, *22*, 4901504, doi:10.1109/TASC.2011.2177625.
- 6. Parrell, J.A.; Zhang, Y.; Field, M.B.; Meinesz, M.; Huang, Y.; Miao, H.; Hong, S.; Cheggour, N.; Goodrich, L. Internal Tin Nb3Sn Conductors Engineered for Fusion and Particle Accelerator Applications. *IEEE Trans. Appl. Supercond.* **2009**, *19*, 2573–2579, doi:10.1109/TASC.2009.2018074.
- 7. Ballarino, A.; Bottura, L. Targets for R&D on Nb3Sn Conductor for High Energy Physics. *IEEE Trans. Appl. Supercond.* **2015**, *25*, 1–6, doi:10.1109/TASC.2015.2390149.
- 8. Lelekhov, S.A.; Krasil'nikov, A. V.; Kuteev, B. V.; Kovalev, I.A.; Ivanov, D.P.; Ryazanov, A.I.; Surin, M.I.; Shavkin, S. V.; Vysotsky, V.S.; Potanina, L. V.; et al. Further Developments of Fusion-Enabling System in Russia: Suggestions on Superconductors and Current Leads for DEMO-FNS Reactor. *IEEE Trans. Appl. Supercond.* **2018**, *28*, 1–5, doi:10.1109/TASC.2017.2768186.
- 9. Lee, P.J.; Larbalestier, D.C. Microstructural Factors Important for the Development of High Critical Current Density Nb3Sn Strand. *Cryogenics (Guildf).* **2008**, *48*, 283–292, doi:10.1016/j.cryogenics.2008.04.005.
- 10. Sanabria, C.; Field, M.; Lee, P.J.; Miao, H.; Parrell, J.; Larbalestier, D.C. Controlling Cu– Sn Mixing so as to Enable Higher Critical Current Densities in RRP® Nb3Sn Wires. *Supercond. Sci. Technol.* **2018**, *31*, 064001, doi:10.1088/1361-6668/aab8dd.
- 11. Segal, C.; Tarantini, C.; Sung, Z.H.; Lee, P.J.; Sailer, B.; Thoener, M.; Schlenga, K.; Ballarino, A.; Bottura, L.; Bordini, B.; et al. Evaluation of Critical Current Density and Residual Resistance Ratio Limits in Powder in Tube Nb3Sn Conductors. *Supercond. Sci. Technol.* **2016**, *29*, 1–10, doi:10.1088/0953-2048/29/8/085003.
- 12. Pong, I.; Hopkins, S.C.; Fu, X.; Glowacki, B.A.; Elliott, J.A.; Baldini, A. Microstructure Development in Nb3Sn(Ti) Internal Tin Superconducting Wire. *J. Mater. Sci.* **2008**, *43*, 3522–3530, doi:10.1007/s10853-008-2522-4.
- 13. Xu, X.; Sumption, M.; Wan, F.; Peng, X.; Rochester, J.; Choi, E.S. Significant Reduction in the Low-Field Magnetization of Nb3Sn Superconducting Strands Using the Internal Oxidation APC Approach. *Supercond. Sci. Technol.* **2023**, *36*, 085008, doi:10.1088/1361- 6668/acdf8c.
- 14. Xu, X.; Peng, X.; Wan, F.; Rochester, J.; Bradford, G.; Jaroszynski, J.; Sumption, M. APC Nb3Sn Superconductors Based on Internal Oxidation of Nb–Ta–Hf Alloys. *Supercond. Sci. Technol.* **2023**, *36*, 035012, doi:10.1088/1361-6668/ACB17A.
- 15. Pfeiffer, S.; Baumgartner, T.; Löffler, S.; Stöger-Pollach, M.; Hopkins, S.C.; Ballarino, A.; Eisterer, M.; Bernardi, J. Analysis of Inhomogeneities in Nb3Sn Wires by Combined SEM and SHPM and Their Impact on Jc and Tc. *Supercond. Sci. Technol.* **2023**, *36*, 045008, doi:10.1088/1361-6668/acb857.
- 16. Senatore, C.; Bagni, T.; Ferradas-Troitino, J.; Bordini, B.; Ballarino, A. Degradation of Ic Due to Residual Stress in High-Performance Nb3Sn Wires Submitted to Compressive Transverse Force. *Supercond. Sci. Technol.* **2023**, *36*, 075001, doi:10.1088/1361- 6668/acca50.
- 17. Rochester, J.; Ortino, M.; Xu, X.; Peng, X.; Sumption, M. The Roles of Grain Boundary Refinement and Nano-Precipitates in Flux Pinning of APC Nb3Sn. *IEEE Trans. Appl.*

*Supercond.* **2021**, *31*, 1–5, doi:10.1109/TASC.2021.3057560.

- 18. Deryagina, I.; Popova, E.; Patrakov, E.; Valova-Zaharevskaya, E. Structure of Superconducting Layers in Bronze-Processed and Internal-Tin Nb3Sn-Based Wires of Various Designs. *J. Appl. Phys.* **2017**, *121*, 233901, doi:10.1063/1.4986232.
- 19. Deryagina, I.L.; Popova, E.N.; Patrakov, E.I.; Valova-Zaharevskaya, E.G. Effect of Nb3Sn Layer Structure and Morphology on Critical Current Density of Multifilamentary Superconductors. *J. Magn. Magn. Mater.* **2017**, *440*, 119–122, doi:10.1016/j.jmmm.2016.12.091.
- 20. Popova, E.N.; Deryagina, I.L. Optimization of the Microstructure of Nb3Sn Layers in Superconducting Composites. *Phys. Met. Metallogr.* **2018**, *119*, 1229–1235, doi:10.1134/S0031918X18120153.
- 21. Deryagina, I.; Popova, E.; Patrakov, E. Effect of Diameter of Nb3Sn-Based Internal-Tin Wires on the Structure of Superconducting Layers. *IEEE Trans. Appl. Supercond.* **2022**, *32*, 1–5, doi:10.1109/TASC.2022.3157577.
- 22. Bottura, L.; Godeke, A. Superconducting Materials and Conductors: Fabrication and Limiting Parameters. *Rev. Accel. Sci. Technol.* **2012**, *05*, 25–50, doi:10.1142/S1793626812300022.
- 23. Uglietti, D.; Abacherli, V.; Cantoni, M.; Flukiger, R. Grain Growth, Morphology, and Composition Profiles in Industrial Nb3Sn Wires. *IEEE Trans. Appl. Supercond.* **2007**, *17*, 2615–2618, doi:10.1109/TASC.2007.898226.
- 24. Banno, N. Low-Temperature Superconductors: Nb3Sn, Nb3Al, and NbTi. *Superconductivity* **2023**, *6*, 100047, doi:10.1016/j.supcon.2023.100047.
- 25. Abächerli, V.; Uglietti, D.; Seeber, B.; Flükiger, R. (Nb,Ta,Ti)3Sn Multifilamentary Wires Using Osprey Bronze with High Tin Content and NbTa/NbTi Composite Filaments. *Phys. C Supercond.* **2002**, *372*–*376*, 1325–1328, doi:10.1016/S0921-4534(02)01020-1.
- 26. Abächerli, V.; Uglietti, D.; Lezza, P.; Seeber, B.; Flükiger, R.; Cantoni, M.; Buffat, P.A. The Influence of Ti Doping Methods on the High Field Performance of (Nb, Ta, Ti)3Sn Multifilamentary Wires Using Osprey Bronze. *IEEE Trans. Appl. Supercond.* **2005**, *15*, 3482–3485, doi:10.1109/TASC.2005.849070.
- 27. Godeke, A.; Haken, B. ten; Kate, H.H.J. ten; Larbalestier, D.C. A General Scaling Relation for the Critical Current Density in Nb3Sn. *Supercond. Sci. Technol.* **2006**, *19*, R100–R116, doi:10.1088/0953-2048/19/10/R02.
- 28. Lee, P.J.; Squitieri, A.A.; Larbalestier, D.C. Nb3Sn: Macrostructure, Microstructure, and Property Comparisons for Bronze and Internal Sn Process Strands. *IEEE Trans. Appl. Supercond.* **2000**, *10*, 979–982, doi:10.1109/77.828395.
- 29. Pong, I.; Oberli, L.-R.; Bottura, L. Cu Diffusion in Nb3Sn Internal Tin Superconductors during Heat Treatment. *Supercond. Sci. Technol.* **2013**, *26*, 105002, doi:10.1088/0953- 2048/26/10/105002.
- 30. Godeke, A. Advances in Nb3Sn Performance. In Proceedings of the Workshop Accelerator Magnet Superconductors, Design and Optimization; CERN: Geneva, Switzerland, 19–23 May 2008; pp. 24–27.
- 31. Barzi, E.; Bossert, R.; Caspi, S.; Dietderich, D.R.; Ferracin, P.; Ghosh, A.; Turrioni, D. RRP Nb3Sn Strand Studies for LARP. *IEEE Trans. Appl. Supercond.* **2007**, *17*, 2607–2610, doi:10.1109/TASC.2007.899579.
- 32. Godeke, A.; den Ouden, A.; Nijhuis, A.; ten Kate, H.H.J. State of the Art Powder-in-Tube Niobium–Tin Superconductors. *Cryogenics (Guildf).* **2008**, *48*, 308–316, doi:10.1016/j.cryogenics.2008.04.003.
- 33. Hawes, C.D.; Lee, P.J.; Larbalestier, D.C. Measurements of the Microstructural, Microchemical and Transition Temperature Gradients of A15 Layers in a High-Performance Nb3Sn Powder-in-Tube Superconducting Strand. *Supercond. Sci. Technol.* **2006**, *19*, S27–S37, doi:10.1088/0953-2048/19/3/004.
- 34. Cantoni, M.; Scheuerlein, C.; Pfirter, P.-Y.; Borman, F. de; Rossen, J.; Arnau, G.; Oberli,

L.; Lee, P. Sn Concentration Gradients in Powder-in-Tube Superconductors. *J. Phys. Conf. Ser.* **2010**, *234*, 022005, doi:10.1088/1742-6596/234/2/022005.

- 35. Kramer, E.J. Scaling Laws for Flux Pinning in Hard Superconductors. *J. Appl. Phys.* **1973**, *44*, 1360–1370, doi:10.1063/1.1662353.
- 36. Dew-Hughes, D. Flux Pinning Mechanisms in Type II Superconductors. *Philos. Mag.* **1974**, *30*, 293–305, doi:10.1080/14786439808206556.
- 37. Ekin, J.W. *Experimental Techniques for Low-Temperature Measurements*; Oxford University Press: Oxford, UK, 2006;
- 38. Flükiger, R.; Senatore, C.; Cesaretti, M.; Buta, F.; Uglietti, D.; Seeber, B. Optimization of Nb3Sn and MgB2 Wires. *Supercond. Sci. Technol.* **2008**, *21*, 054015, doi:10.1088/0953- 2048/21/5/054015.
- 39. Wheatley, L.E.; Baumgartner, T.; Eisterer, M.; Speller, S.C.; Moody, M.P.; Grovenor, C.R.M. Understanding the Nanoscale Chemistry of As-Received and Fast Neutron Irradiated Nb3Sn RRP® Wires Using Atom Probe Tomography. *Supercond. Sci. Technol.* **2023**, *36*, 085006, doi:10.1088/1361-6668/acdbed.
- 40. Fischer, C.M. Investigation of the Relationships between Superconducting Properties and Nb3Sn Reaction Conditions in Powder-in-Tube Nb3Sn Conductors, Master Thesis, University of Winsconsin-Madison, 2002.
- 41. Godeke, A. A Review of the Properties of Nb3Sn and Their Variation with A15 Composition, Morphology and Strain State. *Supercond. Sci. Technol.* **2006**, *19*, R68–R80, doi:10.1088/0953-2048/19/8/R02.
- 42. Marken, K.R. Characterization Studies of Bronze-Process Filamentary Nb3Sn Composites, PhD Thesis, Wisconsin Univ., Madison, USA, 1986.
- 43. West, A.W.; Rawlings, R.D. A Transmission Electron Microscopy Investigation of Filamentary Superconducting Composites. *J. Mater. Sci.* **1977**, *12*, 1862–1868, doi:10.1007/BF00566248.
- 44. Shaw, B.J. Grain Size and Film Thickness of Nb3Sn Formed by Solid-State Diffusion in the Range 650–800 °C. *J. Appl. Phys.* **1976**, *47*, 2143–2145, doi:10.1063/1.322861.
- 45. Schauer, W.; Schelb, W. Improvement of Nb3Sn High Field Critical Current by a Two-Stage Reaction. *IEEE Trans. Magn.* **1981**, *17*, 374–377, doi:10.1109/TMAG.1981.1060900.
- 46. Scanlan, R.M.; Fietz, W.A.; Koch, E.F. Flux Pinning Centers in Superconducting Nb3Sn. *J. Appl. Phys.* **1975**, *46*, 2244–2249, doi:10.1063/1.321816.
- 47. Tinkham, M. *Introduction to Superconductivity*; 2nd ed.; Dover Publications: Mineola, New York, USA, 2004;
- 48. Bardeen, J.; Cooper, L.N.; Schrieffer, J.R. Theory of Superconductivity. *Phys. Rev.* **1957**, *108*, 1175–1204, doi:10.1103/PhysRev.108.1175.
- 49. Kittel, C. *Introduction to Solid State Physics*; 8th ed.; Wiley: Hoboken, New Jersey, USA, 2004;
- 50. Brandt, E.H. The Vortex Lattice in Type-II Superconductors: Ideal or Distorted, in Bulk and Films. *Phys. Status Solidi B* **2011**, *248*, 2305–2316, doi:10.1002/pssb.201147095.
- 51. Talantsev, E.F. The Electron–Phonon Coupling Constant and the Debye Temperature in Polyhydrides of Thorium, Hexadeuteride of Yttrium, and Metallic Hydrogen Phase III. *J. Appl. Phys.* **2021**, *130*, 195901, doi:10.1063/5.0065003.
- 52. Talantsev, E.F.; Tallon, J.L. Universal Self-Field Critical Current for Thin-Film Superconductors. *Nat. Commun.* **2015**, *6*, 7820, doi:10.1038/ncomms8820.
- 53. Paturi, P.; Huhtinen, H. Roles of Electron Mean Free Path and Flux Pinning in Optimizing the Critical Current in YBCO Superconductors. *Supercond. Sci. Technol.* **2022**, *35*, 065007, doi:10.1088/1361-6668/ac68a5.
- 54. Troyan, I.A.; Semenok, D. V; Ivanova, A.G.; Sadakov, A. V; Zhou, D.; Kvashnin, A.G.; Kruglov, I.A.; Sobolevskiy, O.A.; Lyubutina, M. V; Perekalin, D.S.; et al. Non-Fermi-Liquid Behavior of Superconducting SnH\$\_4\$. *arXiv* **2023**,

doi:10.48550/arXiv.2303.06339.

- 55. Poole, C.P.; Farach, H.; Creswick, R.; Prozorov, R. *Superconductivity*; 2nd ed.; Academic Press: London, UK, 2007;
- 56. Godeke, A. *Performance Boundaries in Nb3Sn*; PhD Thesis, University of Twente, Enschede, Netherlands, 2005; Vol. PhD; ISBN 90 365 2224 2.
- 57. Xu, X.; Sumption, M.D.; Peng, X. Internally Oxidized Nb3Sn Strands with Fine Grain Size and High Critical Current Density. *Adv. Mater.* **2015**, *27*, 1346–1350, doi:10.1002/adma.201404335.
- 58. Valova-Zaharevskaya, E.; Popova, E.; Deryagina, I.; Abdyukhanov, I.; Tsapleva, A.; Alekseev, M. Growth Rate and Morphology of Nb3Sn Layers in ITER-Type Bronze-Processed Wires Under Different Diffusion Annealing Regimes. *IEEE Trans. Appl. Supercond.* **2018**, *28*, 1–5, doi:10.1109/TASC.2018.2801324.
- 59. London, H. The Electromagnetic Equations of the Supraconductor. *Proc. R. Soc. London. Ser. A - Math. Phys. Sci.* **1935**, *149*, 71–88, doi:10.1098/rspa.1935.0048.
- 60. Talantsev, E.F.; Crump, W.P.; Tallon, J.L. Universal Scaling of the Self-Field Critical Current in Superconductors: From Sub-Nanometre to Millimetre Size. *Sci. Rep.* **2017**, *7*, 10010, doi:10.1038/s41598-017-10226-z.
- 61. Talantsev, E.; Crump, W.P.; Tallon, J.L. Thermodynamic Parameters of Single- or Multi-Band Superconductors Derived from Self-Field Critical Currents. *Ann. Phys.* **2017**, *529*, 1– 18, doi:10.1002/andp.201700197.
- 62. Talantsev, E.F. New Scaling Laws for Pinning Force Density in Superconductors. *Condens. Matter* **2022**, *7*, 74, doi:10.3390/condmat7040074.
- 63. Tarantini, C.; Kametani, F.; Balachandran, S.; Heald, S.M.; Wheatley, L.; Grovenor, C.R.M.; Moody, M.P.; Su, Y.-F.; Lee, P.J.; Larbalestier, D.C. Origin of the Enhanced Nb3Sn Performance by Combined Hf and Ta Doping. *Sci. Rep.* **2021**, *11*, 17845, doi:10.1038/s41598-021-97353-w.