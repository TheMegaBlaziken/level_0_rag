# arXiv:1112.3483

**Paper ID:** 90492220dbeea000236d170a19a67409

**PDF Path:** apl/Superconductors/arXiv:1112.3483.pdf

**Processing Status:** complete

**Captions Added:** 7

**Generated:** 2025-06-24T13:44:26.901946

---

# Ab initio derivation of electronic low-energy models for C<sup>60</sup> and aromatic compounds

Yusuke Nomura<sup>1</sup> , Kazuma Nakamura<sup>1</sup>,<sup>2</sup> , and Ryotaro Arita<sup>1</sup>,2,<sup>3</sup>

2 JST CREST, 7-3-1 Hongo, Bunkyo-ku, Tokyo, 113-8656, Japan and

3 JST PRESTO, Kawaguchi, Saitama, 332-0012, Japan

(Dated: February 25, 2024)

We present a systematic study for understanding the relation between electronic correlation and superconductivity in C<sup>60</sup> and aromatic compounds. We derived, from first principles, extended Hubbard models for twelve compounds; fcc K3C60, Rb3C60, Cs3C<sup>60</sup> (with three different lattice constants), A15 Cs3C<sup>60</sup> (with four different lattice constants), doped solid picene, coronene, and phenanthrene. We show that these compounds are strongly correlated and have a similar energy scale of the bandwidth and interaction parameters. However, they have a different trend in the relation between the strength of electronic correlation and superconducting transition temperature; while the C<sup>60</sup> compounds have a positive correlation, the aromatic compounds exhibit negative correlation.

PACS numbers: 74.20.Pq, 74.70.Kn, 74.70.Wz

#### I. INTRODUCTION

Superconductivity in π-electron systems, whose history dates back to studies on graphite intercalation compounds in 60's,[1](#page-11-0) has attracted broad interest in condensed matter physics. Recently, two seminal discoveries for carbon-based superconductors have been reported. One is A15/fcc Cs3C60, with a new method to synthesize highly crystalline samples[.](#page-11-1)2[–5](#page-11-2) The observed superconducting transition temperatures (Tc) for the highpressure samples are found to be as high as 38 K for A15 and 35 K for fcc. The other is potassium-doped solid picene,[6](#page-11-3) which opened a new avenue to various 'aromatic superconductors', for which the maximum T<sup>c</sup> has reached 33 K.[7](#page-11-4)[–10](#page-11-5)

The mechanism of superconductivity of these new superconductors has not been fully understood. For alkalidoped fullerides, while there are several experimental reports which seemingly support the conventional BCS mechanism, various indications for unconventional superconductivity have been also observed. For example, although the positive correlation between T<sup>c</sup> and the lattice constant found in K- and Rb-doped fullerides has been understood in terms of the standard BCS theory,[11](#page-11-6) more recent experiments for larger cations have revealed that T<sup>c</sup> does not necessarily behave monotonically as a function of the lattice parameters.[2](#page-11-1)[–5](#page-11-2) The fact that the superconducting phase has a dome-like shape in the phase diagram is indeed reminiscent of cuprates[12](#page-11-7) and unconventional organic superconductors such as BEDT-TTF.[13](#page-11-8) In addition, these new C<sup>60</sup> superconductors are insulators at ambient pressure,[2](#page-11-1)[–5](#page-11-2) indicating that the superconducting phase resides in the vicinity of an insulating phase. In fact, considering these characteristic features, it has been proposed that interplay between orbital, spin, and lattice degrees of freedom is the origin of the high T<sup>c</sup> superconductivity.[14](#page-11-9)

For aromatic superconductors, following the discovery

of K-doped picene (C22H14),[6](#page-11-3) it has been found that various hydrocarbon compounds such as coronene (C24H12), phenanthrene (C14H10), and 1,2:8,9-dibenzopentacene (C30H18) also exhibit superconductivity.[7](#page-11-4)[–10](#page-11-5) Diversity of hydrocarbon molecules suggests the possibility of new and higher T<sup>c</sup> aromatic superconductors. So far, electronic structure,[15](#page-11-10)[–19](#page-11-11) electronic correlations,[20](#page-11-12)[,21](#page-11-13) electron-phonon interactions,[22](#page-11-14)[,23](#page-11-15) and exciton/plasmon properties of the normal state[24](#page-11-16)[–27](#page-11-17) have been studied, while studies on superconducting properties are still limited. The pairing mechanism is totally an open question.

There are several similarities between C<sup>60</sup> and aromatic superconductors; they are both molecular solids having narrow bands around the Fermi level, whose energy scale competes with that of electron-phonon and electron-electron interactions. The competition among these three factors is a characteristic aspect of carbonbased materials. Ab initio derivations of effective lowenergy models for these compounds are important to make the situation transparent and to clarify the origin of their high T<sup>c</sup> superconductivity. By comparing the parameters in the effective models for the C<sup>60</sup> and aromatic superconductors, the differences and similarities are quantitatively identified and analyzed.

Recent methodology for construction of the electronic model, based on the combination use of the maximally localized Wannier orbital (MLWO) (Ref. [28\)](#page-11-18) and the constrained random phase approximation (cRPA),[29](#page-11-19) extends its applicability range. It does not place limitations on the character of basis orbitals of the effective model, whether atomic or molecular. Indeed, it has already been applied to various complex systems such as BEDT-TTF (Refs. [30](#page-11-20) and [31\)](#page-11-21) or zeolites.[32](#page-11-22) While electronic interaction parameters of the C<sup>60</sup> and aromatic superconductors have been estimated by various methods,[25](#page-11-23)[,33](#page-11-24)[–36](#page-11-25) explicit and direct comparison of these systems by the same method has yet to be done. Thus it is imperative to evaluate the interaction parameters of C<sup>60</sup> and aro-

<sup>1</sup>Department of Applied Physics, University of Tokyo,

<sup>7-3-1</sup> Hongo, Bunkyo-ku, Tokyo, 113-8656, Japan

matic superconductors by exploiting the state-of-the-art technique and perform a systematic comparison. In the present study, we constructed ab initio extended Hubbard models which describe the low-energy electronic structure of twelve examples of C<sup>60</sup> and aromatic compounds. The transfer integrals were given as matrix elements of the Kohn-Sham Hamiltonian in the Wannier basis. The interaction parameters were evaluated by calculating the Wannier matrix elements of the screened Coulomb interaction, which is obtained by cRPA. The estimated correlation strength as the ratio of the interaction energy to the kinetic one is nearly or beyond unity for the studied materials, indicating that both the C<sup>60</sup> and aromatic systems are classified into a strongly correlated electron systems. On the other hand, we observed a notable difference between the two systems; for the C<sup>60</sup> system, there exist positive correlation regime in the correlation strength and the experimental Tc. In contrast, the aromatic system exhibits negative correlation between these two quantities.

This paper is organized as follows. In Sec[.II,](#page-1-0) we show how to construct the low-energy models from ab initio calculations. In Sec[.III,](#page-1-1) we show the calculated band structure, MLWOs, transfer integrals, and effective interaction parameters. We discuss the material dependence of the derived parameters and relation between the strength of electronic correlation and superconductivity in Sec[.IV.](#page-8-0) Finally we give a summary in Sec[.V.](#page-10-0)

#### <span id="page-1-0"></span>II. METHODS

We derive electronic low-energy models with the combination of MLWO and cRPA. This method has widely been applied to the derivation of effective models for 3d transition metals,[37](#page-11-26)[,38](#page-11-27) their oxides,[37](#page-11-26) organic conductors,[30](#page-11-20)[,31](#page-11-21) zeolites,[32](#page-11-22) iron-based superconductors,[39](#page-11-28)[,40](#page-11-29) and 5d transition metal oxides.[41](#page-11-30) We first perform band calculations based on density functional theory (DFT),[42](#page-11-31)[,43](#page-11-32) and choose 'target bands' of the effective model. By constructing MLWOs for the target bands, we calculate transfer integrals and effective interactions in the effective model. In the calculation of the effective interaction, the screening by electrons besides target-band electrons is considered within cRPA (see below).

We apply this scheme to the derivation of effective models, i.e., extended multi-orbital Hubbard models, for the C<sup>60</sup> and aromatic compounds. The Hamiltonian consists of the transfer part Ht, the Coulomb repulsion part H<sup>U</sup> , and the exchange interactions and pair hopping part H<sup>J</sup> defined as

$$\mathcal{H} = \mathcal{H}\_t + \mathcal{H}\_U + \mathcal{H}\_J,\tag{1}$$

where

<span id="page-1-2"></span>
$$\mathcal{H}\_t = \sum\_{\sigma} \sum\_{ij} \sum\_{nm} t\_{nm}(\mathbf{R}\_{ij}) a\_{\dot{m}}^{\sigma \dagger} a\_{\dot{m}n}^{\sigma},\tag{2}$$

$$\mathcal{H}\_{U} = \frac{1}{2} \sum\_{\sigma \rho} \sum\_{ij} \sum\_{nm} U\_{nm}(\mathbf{R}\_{ij}) a\_{\dot{m}}^{\sigma \dagger} a\_{\dot{m}}^{\rho \dagger} a\_{\dot{m} n}^{\rho} a\_{\dot{m}}^{\sigma},\tag{3}$$

$$\mathcal{H}\_J = \frac{1}{2} \sum\_{\sigma \rho} \sum\_{ij} \sum\_{nm} J\_{mn}(\mathbf{R}\_{ij}) (a\_{\dot{m}}^{\sigma \dagger} a\_{\dot{j}m}^{\rho \dagger} a\_{\dot{m}}^{\rho} a\_{\dot{m}}^{\sigma} + a\_{\dot{m}}^{\sigma \dagger} a\_{\dot{m}}^{\rho \dagger} a\_{\dot{j}m}^{\rho} a\_{\dot{j}m}^{\sigma}) (4)$$

with a σ† in (a σ in) being a creation (annihilation) operator of an electron with spin σ in the n-th MLWO localized at a C<sup>60</sup> or aromatic hydrocarbon molecule located at R<sup>i</sup> and Rij=Ri−Rj. The parameters tnm(Rij ) represent an onsite energy (Rij=0) and hopping integrals (Rij6=0), which are described with the translational symmetry as

$$t\_{nm}(\mathbf{R}) = \left\langle \phi\_{n\mathbf{R}} \middle| \mathcal{H}\_{KS} \middle| \phi\_{m\mathbf{0}} \right\rangle,\tag{5}$$

where <sup>φ</sup>nR<sup>i</sup> =a † in 0 and HKS is the Kohn-Sham Hamiltonian.

To evaluate effective interaction parameters Unm(R) and Jnm(R), we calculate the screened Coulomb interaction W(r, r ′ ) at the low-frequency limit. We first calculate non-interacting polarization function χ with excluding polarization processes within the target bands. Note that screening by the target electrons is considered when we solve the effective models, so that we have to avoid double counting of it when we derive the effective models. With the resulting χ, the W interaction is calculated as W= 1−vχ−<sup>1</sup> v, where v is the bare Coulomb interaction v(r, r ′ )= <sup>1</sup> |r−r ′ | .

Once screened Coulomb interaction W(r, r ′ ) is calculated, the matrix elements of W are obtained as

$$\begin{aligned} U\_{rm}(\mathbf{R}) &= \left\langle \phi\_{n\mathbf{R}} \phi\_{m\mathbf{0}} \middle| W \middle| \phi\_{n\mathbf{R}} \phi\_{m\mathbf{0}} \right\rangle \\ &= \int \int d\mathbf{r} d\mathbf{r}' \phi\_{n\mathbf{R}}^{\*}(\mathbf{r}) \phi\_{n\mathbf{R}}(\mathbf{r}) W(\mathbf{r}; \mathbf{r}') \phi\_{m\mathbf{0}}^{\*}(\mathbf{r}') \phi\_{m\mathbf{0}}(\mathbf{r}') \end{aligned} (6)$$

and

$$\begin{split} J\_{\rm rnr}(\mathbf{R}) &= \langle \phi\_{r\mathbf{R}} \phi\_{m\mathbf{0}} \left| W \right| \phi\_{m\mathbf{0}} \phi\_{r\mathbf{R}} \rangle \\ &= \int \int d\mathbf{r} \, d\mathbf{r}' \phi\_{n\mathbf{R}}^{\*}(\mathbf{r}) \phi\_{m\mathbf{0}}(\mathbf{r}) W(\mathbf{r}, \mathbf{r}') \phi\_{m\mathbf{0}}^{\*}(\mathbf{r}') \phi\_{r\mathbf{R}}(\mathbf{r}') . \end{split} (7)$$

For comparison with the cRPA results, we calculate interaction parameters with different levels of screening. One is the unscreened one, i.e., the bare Coulomb interaction, and the other is the fully-screened one where we calculate χ including the target-band screening. To distinguish these from cRPA, we denote them as 'bare' and 'fRPA'.

### <span id="page-1-1"></span>III. RESULTS

#### A. Calculation conditions

We performed DFT band calculation with Tokyo Ab initio Program Package, [44](#page-11-33) based on the pseudopo-

<span id="page-2-0"></span>TABLE I: Basic property of fcc and A15 alkali-doped C<sup>60</sup> compounds; the lattice parameter a, corresponding C<sup>60</sup> <sup>3</sup><sup>−</sup> volume in solid, and measured superconducting transition temperature T<sup>c</sup> or the N´eel temperature T<sup>N</sup> . For fcc Cs3C60, the three samples are specified with the C<sup>3</sup><sup>−</sup> <sup>60</sup> volume and corresponds to those in the superconducting phase with maximum T<sup>c</sup> (V opt.P SC ), in the vicinity of the metal-insulator transition (VMIT), and in the anti-ferromagnetic insulating phase (VAFI), respectively. For A15 structure, we also list another sample with a higher pressure, for which T<sup>c</sup> is lowered than that of V opt.P SC , and is abbreviated to V highP SC .

|               |                     | a        | Volume/C60    | 3− Pressure | Tc(TN ) |      |
|---------------|---------------------|----------|---------------|-------------|---------|------|
|               |                     | (˚A)     | 3<br>(˚A<br>) | (kbar)      | (K)     | Ref. |
|               | K                   | 14.240   | 722           | 0           | 19      | 49   |
|               | Rb                  | 14.420   | 750           | 0           | 29      | 49   |
| fcc<br>A3C60  | opt.P<br>Cs(V<br>SC | ) 14.500 | 762           | 7           | 35      | 4    |
|               | Cs(VMIT)            | 14.640   | 784           | 2           | 26      | 4    |
|               | Cs(VAFI)            | 14.762   | 804           | 0           | (2.2)   | 4    |
|               | highP<br>V<br>SC    | 11.450   | 751           | 15          | 35      | 3    |
| A15<br>Cs3C60 | opt.P<br>V<br>SC    | 11.570   | 774           | 7           | 38      | 3    |
|               | VMIT                | 11.650   | 791           | 3           | 32      | 3    |
|               | VAFI                | 11.783   | 818           | 0           | (46)    | 3    |

<span id="page-2-1"></span>TABLE II: Lattice parameters for pristine solid picene, coronene, and phenanthrene and superconducting transition temperature T<sup>c</sup> observed for doped systems.

|              | a      | b     | c      | β        | Tc   |      |
|--------------|--------|-------|--------|----------|------|------|
|              | (˚A)   | (˚A)  | (˚A)   | ( ◦<br>) | (K)  | Ref. |
| picene       | 8.480  | 6.154 | 13.515 | 90.46    | 18,7 | 6,51 |
| coronene     | 16.094 | 4.690 | 10.049 | 110.79   | 15   | 9,52 |
| phenanthrene | 8.453  | 6.175 | 9.477  | 98.28    | 5-6  | 7,8  |

tential plus plane-wave framework. We used the generalized-gradient approximation (GGA) exchangecorrelation functional with the parameterization of Perdew-Burke-Ernzerhof[45](#page-11-38) and the Troulliar-Martins norm-conserving pseudopotentials[46](#page-11-39) in the Kleinman-Bylander representation.[47](#page-11-40) The pseudopotentials for alkali metals, K, Rb, and Cs were supplemented with partial core correction.[48](#page-12-3) The cutoff energies for wavefunctions and charge densities were set to 36 Ry and 144 Ry, respectively, and we employed 5×5×5 k-point sampling. We confirmed that this condition ensures well converged results.

The DFT calculations were performed for the following twelve materials: fcc K3C60, fcc Rb3C60, fcc Cs3C<sup>60</sup> with three different lattice parameters, A15 Cs3C<sup>60</sup> with four different lattice parameters, doped solid picene, coronene, and phenanthrene. The lattice parameters were taken from the experiments and internal coordinates were optimized.[50](#page-12-4) In fcc A3C60, the disorder of the orientation of C<sup>60</sup> molecules was ignored, so the crystal symmetry is lowered from Fm¯3m to Fm¯3.

Before presenting the computational results, we summarize the basic properties of the compounds studied in the present paper. Table [I](#page-2-0) lists experimental values for the C<sup>60</sup> compounds, including the lattice constant a, the volume per C<sup>60</sup> <sup>3</sup><sup>−</sup> in solid,[54](#page-12-5) applied pressure, and measured superconducting transition temperature T<sup>c</sup> or the N´eel temperature T<sup>N</sup> . The a value and/or C<sup>60</sup> <sup>3</sup><sup>−</sup> volume can be controlled by the chemical and external pressures. In this table, the samples are arranged in the order of the increase of the lattice constant. For fcc A3C60, T<sup>c</sup> first increases and reaches the maximum (35 K) around a=14.500 ˚A. Then, it decreases down to T<sup>c</sup> ∼ 25 K where the system experiences the metal-insulator transition (MIT) and becomes an antiferromagnetic insulator (AFI), for which the N´eel temperature T<sup>N</sup> is around 2.2 K. A similar behavior is observed in the A15 system, while T<sup>N</sup> is significantly higher (46 K). This is because the A15 structure is bipartite and therefore less frustrated.[55](#page-12-6) Hereafter, we label the nine C<sup>60</sup> compounds as fcc-K, fcc-Rb, fcc-Cs(V opt.P SC ), fcc-Cs(VMIT), fcc-Cs(VAFI), A15- Cs(V highP SC ), A15-Cs(V opt.P SC ), A15-Cs(VMIT), and A15- Cs(VAFI).

Table [II](#page-2-1) shows the experimental lattice parameters for undoped solid picene, coronene, and phenanthrene and T<sup>c</sup> observed for doped systems. For doped solid picene, two different T<sup>c</sup> (18 K or 7 K) have been observed depending on the preparation conditions.[6](#page-11-3) The superconductivity appears when the system is doped, but the details of the crystal structures in the superconducting phases have not been determined. Thus in the present study, the band calculations for aromatic compounds were performed for the artificially charged system where three negative charges per one hydrocarbon molecule were doped with a uniform compensating positive background charge. Hereafter, doped solid picene, coronene, and phenanthrene are referred to as solid picene<sup>3</sup><sup>−</sup>, coronene<sup>3</sup><sup>−</sup>, and phenanthrene<sup>3</sup><sup>−</sup>, respectively.

#### B. Band structure and density of states

Figure [1](#page-3-0) shows our calculated GGA band structures for the fcc A3C<sup>60</sup> (upper 5 panels), A15 Cs3C<sup>60</sup> (middle 4 panels), and aromatic compounds (lower 3 panels). These compounds have common features in their band structure; i.e., we see narrow bands near the Fermi level separated from other bands, being preferable when we choose the target bands to construct an effective model. In the C<sup>60</sup> compounds, there are threefold degenerated states, which form the so-called '⁀1u band' near the Fermi level, and we construct effective models for these bands. For aromatic compounds, the target bands are made from the lowest two unoccupied molecular orbitals (LUMO and LUMO+1) in an isolated molecule.[15](#page-11-10)[,18](#page-11-41)[,19](#page-11-11) It should be noted that unoccupied bands lie above the target

![](_page_3_Figure_0.jpeg)

**Caption:** Figure 1 depicts the ab initio band structures for different phases of alkali-doped C60 (fcc and A15 structures) and aromatic compounds. The band structures show narrow bands around the Fermi level, essential for low-energy electronic properties modeling. In C60 systems, the threefold degenerate '⁀1u' band closely associates with the Fermi level, crucial for superconductivity studies. Aromatic systems, while structurally different, exhibit narrow target bands derived from molecular orbitals, indicating potential for electronic cooperative effects. This comparative look into band structures derives significant information about density of states, helping to understand the electronic basis of superconductivity .


<span id="page-3-0"></span>FIG. 1: (Color online) Calculated ab intio electronic band structure of fcc-K, fcc-Rb, fcc-Cs(V opt.P SC ), fcc-Cs(VMIT), fcc-Cs(VAFI), A15-Cs(V opt.P SC ), A15-Cs(V highP SC ), A15-Cs(VMIT), A15-Cs(VAFI), solid picene<sup>3</sup><sup>−</sup>, solid coronene<sup>3</sup><sup>−</sup>, and solid phenanthrene<sup>3</sup><sup>−</sup>. In the case of aromatic compounds with monoclinic structure, the horizontal axis is labeled by the special points in the Brillouin zone with Γ, X, A, Y, Z, D, E, and C, respectively, corresponding to (0, 0, 0), (1/2, 0, 0), (1/2, 1/2, 0), (0, 1/2, 0), (0, 0, 1/2), (1/2, 0, 1/2), (1/2, 1/2, 1/2) and (1/2, 1/2, 1/2) in units of (a ∗ , b ∗ , c ∗ ). The interpolated band dispersion with the derived tight binding Hamiltonian is depicted as blue dashed lines.

bands more densely in the order of solid picene<sup>3</sup>−, solid coronene<sup>3</sup>−, and solid phenanthrene<sup>3</sup>−. Since conduction bands can generate stronger screening when they reside closer to the target bands, we expect a weak repulsive interaction in solid picene<sup>3</sup><sup>−</sup> compared to the other two.

We show in Fig. [2](#page-4-0) the calculated density of states (DOS) of the t1<sup>u</sup> band for fcc A3C<sup>60</sup> (a) and A15 Cs3C<sup>60</sup> (b). For both fcc or A15, the bandwidth W monotonically increases as decreasing the lattice constant, but the DOS profile does not change drastically. We list the values of W in table [III.](#page-5-0) The bandwidth of A15 (∼0.6 eV) tends to be larger than that of fcc (∼0.4 eV), which is due to the difference in the inter-C<sup>60</sup> contact, i.e., 'hexagon-to-hexagon' configuration for A15 and 'bondto-bond' one for fcc.[56](#page-12-7) It was found that the bandwidths of the aromatic compounds are nearly 0.5 eV (see table [III\)](#page-5-0).

![](_page_4_Figure_2.jpeg)

**Caption:** Figure 2 displays the density of states (DOS) for the 1u band in various alkali-doped C60 phases, fcc and A15 structures. DOS plots reveal sharp states near Fermi level, indicative of electron localization facilitating superconducting states. Different lattice constants and alkali doping levels alter electronic configurations, highlighting the adjustment of electronic character vital for material optimization. Observations on differences among DOS profiles suggest tunable superconducting properties by altering lattice and doping parameters .


<span id="page-4-0"></span>FIG. 2: (Color online) (a) Our calculated density of states (DOS) for 1u band of fcc-K (red), fcc-Rb (green), fcc- ⁀ Cs(V opt.P SC ) (blue), fcc-Cs(VMIT) (purple), and fcc-Cs(VAFI) (light blue). (b) DOS for 1u band of A15-Cs( ⁀ V highP SC ) (red), A15-Cs(V opt.P SC ) (green), A15-Cs(VMIT) (blue), and A15-Cs(VAFI) (purple).

![](_page_4_Figure_4.jpeg)

**Caption:** Figure 3 presents the isosurfaces of a px-like maximally localized Wannier orbital from the t1u bands of A15-Cs(VAFI), visualized along the x, y, and z axes. Positive and negative isosurface lobes characteristic of px-like geometry reveal localization within a single C60 molecule. These orbitals directly influence electronic band dispersion, essential for constructing accurate low-energy models of the electronic structure in fullerite systems, impacting theoretical superconductivity predictions .


<span id="page-4-1"></span>FIG. 3: (Color online) Isosurface of our calculated px-like maximally localized Wannier orbital of A15-Cs(VAFI) viewed along the (a) x axis, (b) y axis, (c) z axes, drawn by VESTA.[57](#page-12-8) The red surfaces indicate positive isosurface and the blue surfaces indicate negative isosurface.

## C. Maximally localized Wannier orbitals

Figure [3](#page-4-1) shows a contour plot of one of MLWOs for the t1<sup>u</sup> bands of A15-Cs(VAFI). The results of other C<sup>60</sup> compounds are almost the same. From this figure we see that the resulting Wannier orbital is well localized at the single C<sup>60</sup> molecule. In this plot, we displayed the same orbital along the three directions; panels (a), (b), and (c) correspond to the view along the x, y, and z axis, respectively. We see a node in the center of this orbital for (b) and (c), thus this orbital has px-like symmetry. Note that the view along the y axis is not identical to the view along the z axis, which is in contrast with the case of atomic p orbitals. We note that the other two py- and pzlike Wannier orbitals are symmetrically equivalent to the presented px-like orbital. We also note that the weight of the Wannier orbitals concentrates in the vicinity of the cage of the C<sup>60</sup> molecule and there is little weight inside

TABLE III: Calculated bandwidth W of the target band and spatial Wannier spread Ω for twelve materials: fcc-K, fcc-Rb, fcc-Cs(V opt.P SC ), fcc-Cs(VMIT), fcc-Cs(VAFI), A15-Cs(V highP SC ), A15-Cs(V opt.P SC ), A15-Cs(VMIT), A15-Cs(VAFI), solid picene<sup>3</sup><sup>−</sup>, solid coronene<sup>3</sup><sup>−</sup>, and solid phenanthrene<sup>3</sup><sup>−</sup>. For the aromatic compounds, the two value of Ω are listed; the left is the 'lower-level' orbital and the right is 'higher-level' one. Units are given in meV for W and ˚A for Ω.

<span id="page-5-0"></span>

|   | fcc A3C60 |      |                     |                     | A15 Cs3C60 |                  |                  |           | aromatic compounds |            |            |                           |
|---|-----------|------|---------------------|---------------------|------------|------------------|------------------|-----------|--------------------|------------|------------|---------------------------|
|   | K         | Rb   | opt.P<br>Cs(V<br>SC | ) Cs(VMIT) Cs(VAFI) |            | highP<br>V<br>SC | opt.P<br>V<br>SC | VMIT VAFI |                    | picene3−   |            | coronene3− phenanthrene3− |
| W | 502       | 454  | 427                 | 379                 | 341        | 740              | 659              | 614       | 535                | 477        | 447        | 505                       |
| Ω | 4.28      | 4.21 | 4.19                | 4.14                | 4.10       | 4.27             | 4.20             | 4.16      | 4.12               | 4.08, 4.13 | 3.64, 3.67 | 3.20, 3.08                |

![](_page_5_Figure_3.jpeg)

**Caption:** Figure 8 presents the electronic density of states (DOS) for fcc and A15 Cs3C60 systems, following constrained random phase approximation (cRPA) calculations. Panel (a) displays DOS for different doping levels of the fcc phase, while panel (b) addresses the A15 phase. The DOS reflects the stability of the metallic state and potential for superconductivity. K and Rb have narrow states indicating less metallicity, whereas Cs surrounds superconductive potentials near the Fermi level under various lattice configurations. These comparisons of DOS profiles highlight the interplay of electronic correlations and lattice geometry in influencing superconducting transitions in alkali-doped fullerene compounds .


<span id="page-5-1"></span>FIG. 4: (Color online) Isosurface of maximally localized Wannier orbitals of solid phenanthrene<sup>3</sup><sup>−</sup> with (a) lower onsite energy and (b) higher onsite energy, drawn by VESTA.[57](#page-12-8) The red surfaces indicate positive isosurface and the blue surfaces indicate negative isosurface.

it.

We next show in Fig. [4](#page-5-1) a contour plot of two MLWOs of solid phenanthrene<sup>3</sup>−. In the aromatic compounds, the two basis orbitals of the effective model are not symmetrically equivalent and therefore we specify these orbitals as 'lower' and 'higher' orbitals in terms of the onsite level of MLWOs. The lower and higher orbitals are shown in the panels (a) and (b), respectively. We again see the resulting orbitals are well localized at the single molecules. The MLWOs of solid picene<sup>3</sup><sup>−</sup> and coronene<sup>3</sup><sup>−</sup> are similar to those of undoped systems calculated in Refs. [15](#page-11-10) and [19](#page-11-11).

We list in table [III](#page-5-0) our calculated spatial spread Ω<sup>n</sup> of MLWO for the twelve materials, where Ω<sup>n</sup> is defined as

$$
\Omega\_n = \sqrt{\langle \phi\_{n\mathbf{0}} | r^2 | \phi\_{n\mathbf{0}} \rangle - \left| \langle \phi\_{n\mathbf{0}} | \mathbf{r} | \phi\_{n\mathbf{0}} \rangle \right|^2}. \tag{8}
$$

In the C<sup>60</sup> compounds, the calculated Wannier spread is roughly 4 ˚A and thus the estimated effective volume

4 3 πΩ 3 is ∼268 ˚A<sup>3</sup> . The value is compared with the C<sup>60</sup> 3− volume listed in table [I](#page-2-0) (∼720-820 ˚A<sup>3</sup> ), clearly indicating well-localized nature of MLWO on the single molecule. We see that the Wannier spread has a weak positive correlation with the bandwidth W. In the aromatic compounds, the molecular size itself is different from each other, which result in the appreciable difference in Ω. Note that, Ω has no clear correlation with W.

#### D. Transfer integrals

Let us move on to transfer integrals. For the C<sup>60</sup> compounds, the band dispersion of the target band was found to be well reproduced only with nearest neighbor (NN) and next nearest neighbor (NNN) transfers. The orbital index, 1, 2, and 3 denote px-, py-, and pz-like orbitals, respectively. Onsite energies for three MLWOs are set to zero. From now on, 'site' means one molecule and the coordinate of site R is defined as the center of the molecule. The transfer integrals tnm(R) are represented as 3 × 3 matrix. In fcc (A15) structure, there are 12 (8) NN sites and 6 (6) NNN sites per site. From transfers to the specific site, other transfers to the equivalent sites are reproduced by proper symmetry operations. As a representative NN site, we choose R=(Rx, Ry, Rz)=(0.5, 0.5, 0.0) and (0.5, 0.5, 0.5) for fcc and A15 structure, respectively, where the coordinate is based on conventional cell. The transfer matrix to this site is

<span id="page-5-2"></span>
$$
\begin{pmatrix} F\_1 F\_2 \ 0 \\ F\_2 F\_3 \ 0 \\ 0 \ 0 \ F\_4 \end{pmatrix} \text{and} \begin{pmatrix} A\_1 & A\_{2(3)} \ A\_{3(2)} \\ A\_{3(2)} & A\_1 & A\_{2(3)} \\ A\_{2(3)} & A\_{3(2)} & A\_1 \end{pmatrix} \tag{9}
$$

for the fcc and A15 structure, respectively. In A15, the two C<sup>60</sup> molecules in the unit cell (denoted as A- and B-site) are not equivalent in terms of their orientations. So, in the matrix [\(9\)](#page-5-2), we show both the transfers from Asite to B-site and from B-site to A-site (in parentheses). We choose R=(1, 0, 0) for a representative NNN site for both structure, then the transfer matrix is written as

<span id="page-5-3"></span>
$$
\begin{pmatrix} F\_5 \ 0 \ 0 \\ 0 \ F\_6 \ 0 \\ 0 \ 0 \ F\_7 \end{pmatrix} \text{and} \begin{pmatrix} A\_4 & 0 & 0 \\ 0 & A\_{5(6)} & 0 \\ 0 & 0 & A\_{6(5)} \end{pmatrix} \tag{10}
$$

for the fcc and A15 structure, respectively. The transfer matrix for A15 represents the A-A transfer and the B-B

<span id="page-6-0"></span>TABLE IV: Hopping parameters for fcc A3C<sup>60</sup> in Eqs. [\(9\)](#page-5-2) and [\(10\)](#page-5-3). Units are given in 10<sup>−</sup><sup>4</sup> eV.

|                              | F1  | F2   | F3  | F4   | F5  | F6  | F7 |
|------------------------------|-----|------|-----|------|-----|-----|----|
| fcc-K                        | −40 | −339 | 421 | −187 | −94 | −14 | −2 |
| fcc-Rb                       | −16 | −306 | 392 | −159 | −75 | −8  | 15 |
| opt.P<br>fcc-Cs(V<br>)<br>SC | 26  | −299 | 372 | −120 | −60 | −3  | 36 |
| fcc-Cs(VMIT)                 | 15  | −267 | 332 | −104 | −40 | 1   | 30 |
| fcc-Cs(VAFI)                 | 13  | −241 | 302 | −94  | −33 | 1   | 24 |

TABLE V: Hopping parameters for A15 Cs3C<sup>60</sup> in Eqs. [\(9\)](#page-5-2) and [\(10\)](#page-5-3). Units are given in 10<sup>−</sup><sup>4</sup> eV.

<span id="page-6-1"></span>

|                              | A1   | A2  | A3 | A4 | A5   | A6   |
|------------------------------|------|-----|----|----|------|------|
| highP<br>A15-Cs(V<br>)<br>SC | −297 | 448 | 67 | 74 | −105 | −289 |
| opt.P<br>A15-Cs(V<br>)<br>SC | −262 | 400 | 61 | 74 | −97  | −239 |
| A15-Cs(VMIT)                 | −239 | 371 | 57 | 76 | −89  | −212 |
| A15-Cs(VAFI)                 | −206 | 329 | 53 | 73 | −79  | −180 |

one (in parentheses). Table [IV](#page-6-0) shows the value of the parameters from F<sup>1</sup> to F<sup>7</sup> for fcc C<sup>60</sup> compounds. We note that F66=F<sup>7</sup> is due to the lowering the symmetry from Fm¯3m to Fm¯3. Table [V](#page-6-1) shows the value of parameters from A<sup>1</sup> to A<sup>6</sup> for A15 Cs3C60. For both systems, the values of hopping parameters decrease as the lattice parameters increases.

We next describe the procedure for obtaining transfers to the other NN sites or NNN sites. First we consider the NN case. For fcc structure, the transfer matrices to other five NN sites are written with F1-F<sup>4</sup> as follows:

$$
\begin{pmatrix} F\_4 & 0 & 0 \\ 0 & F\_1 & F\_2 \\ 0 & F\_2 & F\_3 \end{pmatrix} \text{ for } \mathbf{R} = (0.0, 0.5, 0.5)
$$

$$
\begin{pmatrix} F\_4 & 0 & 0 \\ 0 & F\_1 & -F\_2 \\ 0 & -F\_2 & F\_3 \end{pmatrix} \text{ for } \mathbf{R} = (0.0, 0.5, -0.5)
$$

$$
\begin{pmatrix} F\_3 & 0 & F\_2 \\ 0 & F\_4 & 0 \\ F\_2 & 0 & F\_1 \end{pmatrix} \text{ for } \mathbf{R} = (0.5, 0.0, 0.5)
$$

$$
\begin{pmatrix} F\_3 & 0 & -F\_2 \\ 0 & F\_4 & 0 \\ -F\_2 & 0 & F\_1 \end{pmatrix} \text{ for } \mathbf{R} = (-0.5, 0.0, 0.5)
$$

$$
\begin{pmatrix} F\_1 & -F\_2 & 0 \\ -F\_2 & F\_3 & 0 \\ 0 & 0 & F\_4 \end{pmatrix} \text{ for } \mathbf{R} = (0.5, -0.5, 0.0).
$$

For A15 structure, we have

$$
\begin{pmatrix}
A\_1 & A\_{2(3)} - A\_{3(2)} \\
A\_{3(2)} & A\_1 & -A\_{2(3)} \\
-A\_{2(3)} - A\_{3(2)} & A\_1 \\
\end{pmatrix} \text{ for } \mathbf{R} = \text{(} 0.5, 0.5, -0.5 )
$$

$$
\begin{pmatrix}
A\_1 & -A\_{2(3)} & A\_{3(2)} \\
-A\_{3(2)} & A\_1 & -A\_{2(3)} \\
A\_{2(3)} - A\_{3(2)} & A\_1 \\
\end{pmatrix} \text{ for } \mathbf{R} = \text{(} 0.5, -0.5, 0.5 )
$$

$$
\begin{pmatrix}
A\_1 & -A\_{2(3)} - A\_{3(2)} \\
-A\_{3(2)} & A\_1 & A\_{2(3)} \\
-A\_{2(3)} & A\_{3(2)} & A\_1 \\
\end{pmatrix} \text{ for } \mathbf{R} = \text{(} 0.5, -0.5, -0.5 ).
$$

The remaining transfers to the NN sites are reproduced by the relation tnm(R) = tnm(−R).

Similarly, the transfers to the other NNN sites are described as

$$
\begin{pmatrix} F\_7(A\_{6(5)}) & 0 & 0 \\ 0 & F\_5(A\_4) & 0 \\ 0 & 0 & F\_6(A\_{5(6)}) \end{pmatrix} \text{ for } \mathbf{R} = (0, 1, 0)
$$
 
$$
\begin{pmatrix} F\_6(A\_{5(6)}) & 0 & 0 \\ 0 & F\_7(A\_{6(5)}) & 0 \\ 0 & 0 & F\_5(A\_4) \end{pmatrix} \text{ for } \mathbf{R} = (0, 0, 1)
$$

for fcc (A15) structure and the remaining NNN transfers are generated according to tnm(R) = tnm(−R).

Using these NN and NNN transfer parameters, we construct the transfer part H<sup>t</sup> in Eq. [\(2\)](#page-1-2) of the effective model. The band dispersion for the C<sup>60</sup> compounds calculated from the resulting H<sup>t</sup> is depicted as blue dashed lines in Fig. [1,](#page-3-0) from which we see that the original GGA band dispersion is satisfactorily reproduced.

For the aromatic compounds, since there is no simple symmetry operation, their transfers are difficult to show concisely.[15](#page-11-10)[,19](#page-11-11) For the aromatic compounds, we describe only some characteristic features of the transfers. The aromatic compounds are regarded as the stacking layered systems, so we expect 2-dimensional hopping structure. However, in the present transfer analysis, we found that the anisotropy of the transfers is not so simple. Table [VI](#page-7-0) compares the maximum absolute value of the intralayer transfers (tk) with that of the interlayer transfers (t⊥), where the intralayer is defined as the ab plane. The intralayer transfers are further decomposed in the three directions and compared with each other (see Fig. [5](#page-7-1) for the definition of the three directions). The anisotropy (t max <sup>⊥</sup> /tmax k ) is not so appreciable for solid picene<sup>3</sup>−, estimated as 20/59∼0.34, and phenanthrene<sup>3</sup>−, as ∼0.49. In contrast, the anisotropy of coronene<sup>3</sup><sup>−</sup> is significant and is ∼0.16. In the case of coronene<sup>3</sup>−, the intralayer anisotropy is even strong t max k1 /tmax <sup>k</sup><sup>2</sup> =7/87∼0.08; this system is almost quasi-one-dimensional chain along the b axis. We note that the original GGA band dispersion is well reproduced by short-range transfer hoppings (Fig. [1\)](#page-3-0).
![](0__page_7_Figure_0.jpeg)

**Caption:** Figure 5 visualizes the intralayer transfer interactions among aromatic compounds, designated by transfer integrals t_|| for three primary directions due to anisotropy in these molecular systems. The figure schematically shows transfer paths (t_||1, t_||2, t_||3), critical for understanding electron hopping between molecular sites. This configuration emphasizes aromatic molecular packing that can lead to direction-dependent electronic responses, essential for tailoring material properties such as conductivity or superconductivity. The anisotropic nature of these transfers plays a significant role in mesoscopic material behavior .


<span id="page-7-0"></span>FIG. 5: (Color online) Schematic picture of intralayer transfer t<sup>k</sup> along the three directions in aromatic compounds. The ellipses indicate molecules.

TABLE VI: Comparison between the maximum absolute values of intralayer transfer t<sup>k</sup> and interlayer transfer t<sup>⊥</sup> for aromatic compounds. For the three directions in tk, see Fig. [5.](#page-7-0) Units are given in meV.

|                      | max<br>t<br>k1 | max<br>t<br>k2 | max<br>t<br>k3 | max<br>t<br>⊥ |
|----------------------|----------------|----------------|----------------|---------------|
| solid picene3−       | 48             | 39             | 59             | 20            |
| solid coronene3−     | 7              | 87             | 7              | 14            |
| solid phenanthrene3− | 49             | 32             | 73             | 36            |

### <span id="page-7-2"></span>E. Effective interaction parameters

We performed RPA calculations to evaluate the screened Coulomb interaction W(r, r ′ ) in Eqs. [\(6\)](#page-1-0) and [\(7\)](#page-1-1), where the dielectric function was expanded in plane waves with an energy cutoff 7.5 Ry for fcc A3C<sup>60</sup> and aromatic compounds and 5.0 Ry for A15 Cs3C60. The total number of bands considered in the polarization function was set to 335 (120 occupied+3 target+212 unoccupied) for fcc A3C60, 670 (240 occupied+6 target+424 unoccupied) for A15 Cs3C60, 310 (102 occupied+4 target+204 unoccupied) for solid picene<sup>3</sup>−, 315 (108 occupied+4 target+203 unoccupied) for solid coronene<sup>3</sup>−, and 270 (66 occupied+4 target+200 unoccupied) for solid phenanthrene<sup>3</sup>−. The Brillouin-zone integral on wavevector was evaluated by generalized tetrahedron method.[58](#page-12-0) A problem due to the singularity of long-wavelengthlimit Coulomb interaction in the evaluation of the Wannier matrix elements, Unm(R) in Eq. [\(6\)](#page-1-0) and Jnm(R) in Eq. [\(7\)](#page-1-1), was treated in the manner described in Ref. [59](#page-12-1).

The onsite interactions are specified by U=Unn(0) and U ′=Unm(0) and J=Jnm(0) for n6=m. In the case of C<sup>60</sup> compounds, U, U ′ , and J take only one value according to the symmetry. For aromatic compounds, U is different for two orbitals, so we present two values. We also denote the Coulomb repulsion between the neighboring sites as V .

Table [VII](#page-8-0) shows our calculated interaction parameters U, U ′ , J, and V with three screening levels ('bare', 'cRPA', and 'fRPA'). We see that the value of the Coulomb repulsion decreases as the screening processes increases. In the C<sup>60</sup> compounds, the bare value is ∼3.4 eV and after considering the screening by cRPA, the value is reduced to ∼1 eV. By taking account of the intra-target-band screening by fRPA, the value is further reduced to ∼0.1 eV. It should be noted here that the material dependence of the bare values in fcc or A15 is small; for example, 3.27 eV for fcc-K and 3.37 for fcc-Cs(VAFI). The difference is nearly 3 %. This difference is ascribed to the difference in the spatial spread of MLWOs (see Table [III\)](#page-5-0). On the other hand, the material difference in the cRPA values is beyond 20 %; 0.82 eV for fcc-K and 1.07 for fcc-Cs(VAFI). Indeed, this appreciable difference originates from the difference in the macroscopic dielectric constant defined as

<span id="page-7-1"></span>
$$\epsilon\_M^{\text{cRPA}} = \lim\_{\mathbf{Q} \to 0} \lim\_{\omega \to 0} \frac{1}{\epsilon\_{\mathbf{G}\mathbf{G}'}^{\text{cRPA}^{-1}}(\mathbf{q}, \omega)} \tag{11}$$

with ω being frequency and Q=q+G, where q is a wave vector in the first Brillouin zone and G is a reciprocal lattice vector. We list the value in the bottom of the table. We see that the material dependence of ǫ cRPA M is appreciable as 5.6 for fcc-K and 4.4 for fcc-Cs(VAFI), clearly indicating the importance of the screening effect in addition to the spatial Wannier spread.

For the aromatic compounds, differences in both the bare interaction and the screening effect contribute to the material dependence of the cRPA values; the bare interactions are U picene3<sup>−</sup> bare ∼ U coronene3<sup>−</sup> bare < Uphenanthrene3<sup>−</sup> bare and after consideration of the cRPA screening, we obtain U picene3<sup>−</sup> cRPA < Ucoronene3<sup>−</sup> cRPA ∼ U phenanthrene3<sup>−</sup> cRPA . Especially, in picene<sup>3</sup><sup>−</sup>, the dielectric constant is markedly high as ∼12,[60](#page-12-2) making the cRPA U value small appreciably.

We finally remark some points: As for the C<sup>60</sup> compounds, the equality U ′∼U−2J holds among effective parameters. This relationship also roughly holds for the aromatic compounds. The present U value of cRPA for the C<sup>60</sup> compounds are small compared to the previous estimates of U (∼1-1.5eV).[33](#page-11-0)[–36](#page-11-1) For all materials, the cRPA Coulomb interaction decays as 1/(ǫ cRPA <sup>M</sup> r) with r being the distance between the centers of MLWOs, while the fRPA Coulomb interaction is limited to be short ranged due to the metallic screening (see table [VII\)](#page-8-0). We note that the fRPA U gives an opposite trend to the bare and cRPA U; for example, in the fcc C<sup>60</sup> compounds, the fRPA value slightly decreases as proceeding from fcc-K to fcc-Cs(VAFI). This is due to the fact that the Coulomb interaction is efficiently screened due to the increase in the density of states accompanied by the decrease of bandwidth. We also found that, in these systems, the exchange interaction J are also efficiently screened; i.e., JcRPA/Jbare∼0.3. This makes a clear contrast to the case of the inorganic materials as JcRPA/Jbare∼0.8 such as 3d transition metals,[37](#page-11-2) its oxides SrVO3, [37](#page-11-2) and iron-based superconductors.[39](#page-11-3)[,40](#page-11-4)

<span id="page-8-0"></span>TABLE VII: U, U ′ , J, and V with three different screening levels [unscreened (bare), constrained RPA (cRPA), and fullyscreened RPA (fRPA)] for the twelve compounds: fcc-K, fcc-Rb, fcc-Cs(V opt.P SC ), fcc-Cs(VMIT), fcc-Cs(VAFI), A15-Cs(V highP SC ), A15-Cs(V opt.P SC ), A15-Cs(VMIT), A15-Cs(VAFI), solid picene<sup>3</sup><sup>−</sup>, solid coronene<sup>3</sup><sup>−</sup>, and solid phenanthrene<sup>3</sup><sup>−</sup>. For the aromatic compounds, the two value of U are presented; the left is the 'lower-level' orbital and the right is 'higher-level' one. For 'bare' and 'cRPA' U, U ′ and V values, the unit is given in eV and J is given by meV. For 'fRPA', the unit is given in meV. In the bottom, we present our calculated cRPA macroscopic dielectric constant ǫ cRPA <sup>M</sup> in Eq. [\(11\)](#page-7-1).

|                | fcc A3C60                 |                     |                     |                     |                     | A15 Cs3C60       |                                         |      | aromatic compounds |           |           |                                    |
|----------------|---------------------------|---------------------|---------------------|---------------------|---------------------|------------------|-----------------------------------------|------|--------------------|-----------|-----------|------------------------------------|
|                | K                         | Rb                  | opt.P<br>Cs(V<br>SC | ) Cs(VMIT) Cs(VAFI) |                     | highP<br>V<br>SC | opt.P<br>V<br>SC                        | VMIT | VAFI               |           |           | picene3− coronene3− phenanthrene3− |
| Ubare          | 3.27                      | 3.31                | 3.32                | 3.35                | 3.37                | 3.36             | 3.39                                    | 3.40 | 3.42               | 4.43,4.41 | 4.64,4.59 | 5.05,5.17                          |
| ′<br>U<br>bare | 3.08                      | 3.11                | 3.12                | 3.15                | 3.17                | 3.16             | 3.18                                    | 3.20 | 3.22               | 3.55      | 4.33      | 4.55                               |
| Jbare          | 96                        | 99                  | 100                 | 101                 | 102                 | 97               | 99                                      | 100  | 101                | 166       | 129       | 275                                |
| Vbare          |                           | 1.31-1.37 1.30-1.35 | 1.29-1.34           |                     | 1.28-1.33 1.27-1.32 |                  | 1.37-1.38 1.36-1.37 1.35-1.36 1.34-1.34 |      |                    | 2.08-2.32 | 2.79-2.84 | 2.29-2.43                          |
| UcRPA          | 0.82                      | 0.92                | 0.94                | 1.02                | 1.07                | 0.93             | 1.02                                    | 1.07 | 1.14               | 0.73,0.74 | 1.29,1.26 | 1.33,1.37                          |
| ′<br>U<br>cRPA | 0.76                      | 0.85                | 0.87                | 0.94                | 1.00                | 0.87             | 0.95                                    | 0.99 | 1.06               | 0.58      | 1.15      | 1.17                               |
| JcRPA          | 31                        | 34                  | 35                  | 35                  | 36                  | 30               | 36                                      | 36   | 37                 | 53        | 58        | 101                                |
|                | VcRPA 0.24-0.25 0.26-0.27 |                     | 0.27-0.28           | 0.28-0.29           | 0.30                | 0.30             | 0.31                                    | 0.32 | 0.34               | 0.26      | 0.59-0.60 | 0.47-0.48                          |
| UfRPA          | 93                        | 91                  | 91                  | 86                  | 83                  | 107              | 102                                     | 99   | 93                 | 155,151   | 149,120   | 166,172                            |
| ′<br>U<br>fRPA | 41                        | 39                  | 39                  | 35                  | 32                  | 50               | 45                                      | 42   | 37                 | 51        | 53        | 60                                 |
| JfRPA          | 25                        | 26                  | 26                  | 26                  | 25                  | 28               | 28                                      | 28   | 28                 | 38        | 39        | 57                                 |
| VfRPA          | 1-3                       | 1-3                 | 1-3                 | 1-3                 | 1-3                 | 2-3              | 2                                       | 2    | 1-2                | 1-4       | 1-4       | 2                                  |
| cRPA<br>ǫ<br>M | 5.6                       | 5.1                 | 4.9                 | 4.6                 | 4.4                 | 4.7              | 4.4                                     | 4.3  | 4.1                | 12.0      | 5.5       | 6.3                                |

### IV. DISCUSSIONS

### A. Material dependence of effective parameters

Let us move on to the comparison of the effective interaction parameters among the twelve compounds. Figure [6](#page-9-0) summarizes the results of the cRPA calculation; the onsite Coulomb repulsion U¯ averaged over MLWOs derived from the target band, the onsite exchange interaction J, the offsite interaction V¯ averaged over nearest-neighbor sites, and the ratio (U¯ − V¯ )/W which measures the correlation strength in the system. Note that the net interaction is estimated as U¯ −V¯ , based on the analysis of the extended Hubbard model.

As for the C<sup>60</sup> systems, we see that U¯ has appreciable material dependence, ranging from 0.8 eV to 1.1 eV [Fig. [6\(](#page-9-0)a)]. This is ascribed to the differences in the size of Wannier orbitals and dielectric screening (see Sec. [III E\)](#page-7-2). On the other hand, the material dependence of J is weak and the value itself is negligibly small as ∼0.03 eV [Fig. [6\(](#page-9-0)b)]. In general, small J favors low-spin states, as is observed in experiments.[3](#page-11-5)[–5](#page-11-6)[,61](#page-12-3) It is interesting to note that there is a proposal that the Jahn-Teller coupling dominates over the Hund's rule coupling J, and induces superconductivity with the help of sufficiently large U. [14](#page-11-7) Compared to J, we found that V¯ is substantially large as ∼0.3 eV, being as large as ∼25% of U¯ [Fig. [6\(](#page-9-0)c)]. As for W, which measures kinetic energy, we observed a decreasing trend as the lattice constant increases [Fig. [6\(](#page-9-0)d)]. We also note that the energy scale for A15 Cs3C<sup>60</sup> is larger than that of fcc A3C<sup>60</sup> as mentioned in Sec. [III B.](#page-2-0) The derived correlation strength of (U¯ −V¯ )/W exhibits a rather simple monotonic increasing behavior [Fig. [6\(](#page-9-0)e)], with the lattice constant increase. The presented (U¯ −V¯ )/W∼1 indicates that C<sup>60</sup> superconductors are categorized as strongly correlated electron systems.

For the aromatic superconductors, we found that the energy scale of U¯ is similar to that of the C<sup>60</sup> superconductors [Fig. [6\(](#page-9-0)a)]. On the other hand, it is interesting to note that the aromatic superconductors tend to have larger J and V¯ [Figs. [6\(](#page-9-0)b) and (c)]. We can also see that the material dependence of the interaction parameters among the aromatic superconductors is also more significant, since the size and shape of the aromatic molecules are quite different from each other. As for W, they are similar for the aromatic and C<sup>60</sup> superconductors [Fig. [6\(](#page-9-0)d)]. We found that aromatic superconductors are also in strongly correlated regime as C<sup>60</sup> ones, based on the analysis of the correlation strength [Fig. [6\(](#page-9-0)e)].

## B. Relation between electronic correlation and superconductivity

Next, let us discuss the relation between electronic correlation and superconductivity for the C<sup>60</sup> and aromatic superconductors. In Figs. [7](#page-10-0) (a) and (b), we plot the superconducting transition temperature T<sup>c</sup> and the N´eel temperature T<sup>N</sup> as a function of volume occupied per fulleride anion (V0) for fcc A3C<sup>60</sup> and A15 Cs3C60, respectively (see also table [I\)](#page-2-1). To see the relation between the electron correlation and the superconductivity we superpose a plot of (U¯ − V¯ )/W on the phase diagram. We see that while (U¯ −V¯ )/W and T<sup>c</sup> have a positive correlation up to V0∼760-770 ˚A<sup>3</sup> , for larger V0, electron correlation becomes fatal for superconductivity and the system eventually becomes an insulator. We note that the critical value of (U¯ − V¯ )/W for the MIT sample is larger for fcc A3C<sup>60</sup> (∼1.9) than A15 Cs3C<sup>60</sup> (∼1.2). As discussed in Ref. [56,](#page-12-4) it is important to consider the influence of lattice and orbital structure on MIT.[14](#page-11-7)[,62](#page-12-5)

In Fig. [7\(](#page-10-0)c), we plot T<sup>c</sup> and (U¯ − V¯ )/W for the three aromatic superconductors, which shows a negative correlation. Therefore, it seems that electronic correlation does not favor superconductivity in these aromatic superconductors. Recently, doped 1,2:8,9-

![](0__page_9_Figure_0.jpeg)

**Caption:** Figure 6 shows the material dependence of effective electronic interaction parameters in C60 and aromatic compounds. (a) The onsite Coulomb repulsion U¯, an indicator of electron localization, varies between 0.8 eV to 1.1 eV for C60 systems, illustrating significant material dependence due to variations in Wannier orbital sizes and dielectric screening. (b) Onsite exchange interaction J remains consistent and small at ~0.03 eV, suggesting a preference for low-spin states. (c) Offsite Coulomb repulsion V¯ is substantial, approximately 0.3 eV, indicating significant electron-electron interaction at adjacent sites. (d) Bandwidth W decreases with lattice constant increase, suggesting reduced electronic delocalization. (e) Correlation strength (U¯ − V¯ )/W increases monotonically, categorizing these materials as strongly correlated electron systems. The findings highlight both the shared and distinct electronic behaviors across these compound classes, emphasizing the intricacy of electronic correlations and their critical role in the superconductivity observed in fullerene and aromatic systems.


<span id="page-9-0"></span>FIG. 6: (Color online) Material dependence of (a) the average of the onsite effective Coulomb repulsion U¯, (b) the onsite effective exchange interaction J, (c) the average of the offsite effective Coulomb repulsion between neighboring sites V¯ , (d) the bandwidth of the target band W, and (e) the correlation strength (U¯ − V¯ )/W.

![](0__page_10_Figure_1.jpeg)

**Caption:** Figure 7 illustrates the correlation between superconducting (Tc) or magnetic transition (TN) temperatures and the correlation strength (U¯ − V¯ )/W across different volumes for fcc A3C60, A15 Cs3C60, and aromatic compounds. Panels (a) and (b) show correlations for the fcc and A15 structures, where positive correlation exists at lower volumes but inhibits superconductivity as correlation increases further. Panel (c) for aromatic compounds, however, displays an inverse trend, where stronger correlations do not support, and even suppress, superconductivity. This highlights critical material design factors when engineering new superconducting materials based on these compounds .


<span id="page-10-0"></span>FIG. 7: (Color online) Relation between the experimental curve of superconducting or magnetic transition temperature (Tc, T<sup>N</sup> ) as a function of the C<sup>3</sup><sup>−</sup> <sup>60</sup> volume and the estimated correlation strength (U¯ − V¯ )/W (vertical bar): (a) fcc A3C<sup>60</sup> and (b) A15 Cs3C60. For aromatic compounds (c), the measured T<sup>c</sup> in table [II](#page-2-2) and the calculated correlation strength are compared, where picene<sup>3</sup><sup>−</sup>=(C22H14) <sup>3</sup><sup>−</sup>, coronene<sup>3</sup><sup>−</sup>=(C24H12) <sup>3</sup><sup>−</sup>, and phenanthrene<sup>3</sup><sup>−</sup>=(C14H10) <sup>3</sup><sup>−</sup>. For the panels (a) and (b), the experimental phase diagram were taken from Ref. [4](#page-11-8) for fcc and Ref. [3](#page-11-5) for A15.

dibenzopentacene was found to have a quite high Tc∼33 K. Since 1,2:8,9-dibenzopentacene is a bigger molecule than picene, coronene, and phenanthrene, the former interaction is expected to be small compared to the latter ones, reflecting the large Wannier spread of the 1,2:8,9-dibenzopentacene molecule. If there is no drastic change in the bandwidth W, which is probable in terms of the tendency shown in Fig. [6\(](#page-9-0)d), the weakest electronic correlation will be realized in doped 1,2:8,9 dibenzopentacene. This trend is consistent with Fig. [7\(](#page-10-0)c).

Regarding the role of electronic correlation in the C<sup>60</sup> and aromatic superconductors, there are two possibilities: The pairing mechanism in these compounds has a common root or these superconductors have completely different pairing glues. If we assume that the aromatic superconductors reside in the vicinity of the border between the superconducting and insulating phases, the first scenario is (at least partially) explicable to the behavior in Fig. [7.](#page-10-0) On the other hand, in the second scenario, the electronic correlation enhances superconductivity for the C<sup>60</sup> compounds and inversely suppresses that for the aromatic compounds. In order to clarify this issue, experimental studies to determine the phase diagram for aromatic superconductors against temperature and volume occupied per anion are highly desired.[63](#page-12-6) Theoretically, microscopic calculations considering both electronic correlation and electron-lattice coupling are needed, which will be an interesting future problem.

# V. SUMMARY

To provide insight into the role of electronic correlation in C<sup>60</sup> and aromatic superconductors, we derived effective models for wide range of the examples; fcc-K, fcc-Rb, fcc-Cs(V opt.P SC ), fcc-Cs(VMIT), fcc-Cs(VAFI), A15-Cs(V highP SC ), A15-Cs(V opt.P SC ), A15-Cs(VAFI), solid picene<sup>3</sup><sup>−</sup>, coronene<sup>3</sup><sup>−</sup>, and phenanthrene<sup>3</sup><sup>−</sup>. To define the basis orbital of the effective model, we constructed MLWOs of isolated bands around the Fermi level. Transfer parameters are derived by evaluating the matrix elements of the Kohn-Sham Hamiltonian between the ML-WOs. The low-energy electronic structures of the C<sup>60</sup> compounds are highly symmetric and isotropic, so that the original GGA band is reproduced with only 6 or 7 parameters. On the other hand, the aromatic compounds have quite anisotropic electronic structure.

To quantify the strength of electronic correlation in these compounds, we estimated the effective interaction parameters such as U, V and J, by means of the cRPA method. It was found that, in addition to the appreciable reduction of the diagonal part of the Coulomb interaction (U and V ), the off-diagonal part J is also efficiently screened. Interestingly, all the C<sup>60</sup> and aromatic superconductors studied in the present work have a similar energy scale for the bandwidth and interaction parameters: W∼0.5 eV, U∼1 eV, J∼0.05 eV, V ∼0.3 eV. This parameter range suggests that these compounds are a strongly correlated electron system. However, after examination of the material dependence, we found that a clear difference between the C<sup>60</sup> and aromatic compounds in the relation between electronic correlation strength and Tc; i.e., a positive correlation in the C<sup>60</sup> system and a negative correlation in the aromatic system.

In the present study, we focused on the derivation for the electronic part of the effective model. For deep understanding of the low-energy physics for the carbon-based materials, however, the derivation of the electron-phonon interaction part is also imperative. The derivation for this part includes subtle problems on the definition of the basis for the phonon mode (Refs. [64](#page-12-7) and [65](#page-12-8)) and/or the exclusion of the double counting of the screening of the low-energy degree of freedoms, which needs the future studies.

### Acknowledgments

We thank Taichi Kosugi for providing us with the optimized structure data of undoped coronene and also for stimulating discussions. This work was supported by Grants-in-Aid for Scientific Research (No. 22740215, 22104010, 23110708, 23340095, 19051016), Funding Pro-

- <sup>1</sup> N. B. Hannay, T. H. Geballe, B. T. Matthias, K. Andres, P. Schmidt and D. MacNair, Phys. Rev. Lett. 14, 225 (1965).
- <sup>2</sup> A. Y. Ganin, Y. Takabayashi, Y. Z. Khimyak, S. Margadonna, A. Tamai, M. J. Rosseinsky, and K. Prassides, Nature Mater. 7, 367 (2008).
- <span id="page-11-5"></span><sup>3</sup> Y. Takabayashi, A. Y. Ganin, P. Jegliˇc, D. Arˇcon, T. Takano, Y. Iwasa, Y. Ohishi, M. Takata, N. Takeshita, K. Prassides, and M. J. Rosseinsky, Science 323, 1585 (2009).
- <span id="page-11-8"></span><sup>4</sup> A. Y. Ganin, Y. Takabayashi, P. Jegliˇc, D. Arˇcon, A. Potoˇcnik, P. J. Baker, Y. Ohishi, M. T. McDonald, M. D. Tzirakis, A. McLennan, G. R. Darling, M. Takata, M. J. Rosseinsky, and K. Prassides, Nature 466, 221 (2010).
- <span id="page-11-6"></span><sup>5</sup> Y. Ihara, H. Alloul, P. Wzietek, D. Pontiroli, M. Mazzani, and M. Ricc´o, Phys. Rev. Lett. 104, 256402 (2010).
- <sup>6</sup> R. Mitsuhashi, Y. Suzuki, Y. Yamanari, H. Mitamura, T. Kambe, N. Ikeda, H. Okamoto, A. Fujiwara, M. Yamaji, N. Kawasaki, Y. Maniwa, and Y. Kubozono, Nature 464, 76 (2010).
- <span id="page-11-10"></span><sup>7</sup> X. F. Wang, R. H. Liu, Z. Gui, Y. L. Xie, Y. J . Yan, J. J. Ying, X. G. Luo, and X. H. Chen, Nat. Commun. 2, 507 (2011).
- <span id="page-11-11"></span><sup>8</sup> X. F. Wang, Y. J. Yan, Z. Gui, R. H. Liu, J. J. Ying, X. G. Luo, and X. H. Chen, Arxive:1110.5458.
- <sup>9</sup> Y. Kubozono, H. Mitamura, X. Lee, X. He, Y. Yamanari, Y. Takahashi, Y. Suzuki, Y. Kaji, R. Eguchi, K. Akaike, T. Kambe, H. Okamoto, A. Fujiwara, T. Kato, T. Kosugi, and H. Aoki, Phys. Chem. Chem. Phys. 13, 16476 (2011).
- <sup>10</sup> M. Xue, T. Cao, D. Wang, Y. Wu, H. Yang, X. Dong, J. He, F. Li, and G. F. Chen, Arxive:1111.0820.
- <sup>11</sup> For a review, see e.g., O. Gunnarsson, Rev. Mod. Phys. 69, 575 (1997).
- <sup>12</sup> Handbook of High-Temperature Superconductivity, J. R. Shrieffer, ed., Springer (2007).
- <sup>13</sup> Orgnacic Superconductors, T. Ishiguro, K. Yamaji, and G. Saito, Springer (1997).
- <span id="page-11-7"></span><sup>14</sup> See M. Capone, M. Fabrizio, C. Castellani, and E. Tosatti, Rev. Mod. Phys. 81, 943 (2009), and references therein.
- <sup>15</sup> T. Kosugi, T. Miyake, S. Ishibashi, R. Arita, and H. Aoki, J. Phys. Soc. Jpn, 78, 113704 (2009).
- <sup>16</sup> T. Kosugi, T. Miyake, S. Ishibashi, R. Arita, and H. Aoki, Phys. Rev. B, 84, 214506 (2011).
- <sup>17</sup> P. L. de Andres, A. Guijarro, and J. A. Verg´es, Phys. Rev. B 83, 245113 (2011).
- <sup>18</sup> P. L. de Andres, A. Guijarro, and J. A. Verg´es, Phys. Rev. B 84, 144501 (2011).
- <sup>19</sup> T. Kosugi, T. Miyake, S. Ishibashi, R. Arita, and H. Aoki, Phys. Rev. B 84, 020507(R) (2011).
- <sup>20</sup> G. Giovannetti and M. Capone, Phys. Rev. B 83, 134508 (2011).
- <sup>21</sup> M. Kim, B. I. Min, G. Lee, H. J. Kwon, Y. M. Rhee, and J. H. Shim, Phys. Rev. B 83, 214510 (2011).
- <sup>22</sup> A. Subedi and L. Boeri, Phys. Rev. B 84, 020508(R)

gram for World-Leading Innovative R&D on Science and Technology (FIRST program) on "Quantum Science on Strong Correlation", JST-PRESTO, the Strategic Programs for Innovative Research (SPIRE), MEXT, and the Computational Materials Science Initiative (CMSI), Japan.

(2011).

- <sup>23</sup> M. Casula, M. Calandra, G. Profeta, and F. Mauri, Phys. Rev. Lett. 107, 137006 (2011).
- <sup>24</sup> F. Roth, M. Gatti, P. Cudazzo, M. Grobosch, B. Mahns, B. B¨uchner, A. Rubio, and M. Knupfer, New. J. Phys. 12, 103036 (2010).
- <sup>25</sup> F. Roth, B. Mahns, B. B¨uchner, and M. Knupfer, Phys. Rev. B 83, 165436 (2011).
- <sup>26</sup> F. Roth, B. Mahns, B. B¨uchner, and M. Knupfer, Phys. Rev. B 83, 144501 (2011).
- <span id="page-11-9"></span><sup>27</sup> P. Cudazzo, M. Gatti, F. Roth, B. Mahns, M. Knupfer, and A. Rubio, Phys. Rev. B 84, 155118 (2011).
- <sup>28</sup> N. Marzari and D. Vanderbilt, Phys. Rev. B 56, 12847 (1997); I. Souza, N. Marzari, and D. Vanderbilt, ibid. 65, 035109 (2001).
- <sup>29</sup> F. Aryasetiawan, M. Imada, A. Georges, G. Kotliar, S. Biermann, and A. I. Lichtenstein, Phys. Rev. B 70, 195104 (2004).
- <sup>30</sup> K. Nakamura, Y. Yoshimoto, T. Kosugi, R. Arita, and M. Imada, J. Phys. Soc. Jpn. 78, 083710 (2009).
- <sup>31</sup> H. Shinaoka, T. Misawa, K. Nakamura, and M. Imada, [arXiv:1110.6299.](http://arxiv.org/abs/1110.6299)
- <sup>32</sup> K. Nakamura, T. Koretsune, and R. Arita, Phys. Rev. B 80, 174420 (2009).
- <span id="page-11-0"></span><sup>33</sup> R. W. Lof, M. A. van Veenendaal, B. Koopmans, H. T. Jonkman, and G. A. Sawatzky, Phys. Rev. Lett. 68, 3924 (1992).
- <sup>34</sup> M. R. Pederson and A. A. Quong, Phys. Rev. B 46, 13584 (1992).
- <sup>35</sup> V. P. Antropov, O. Gunnarsson, and O. Jepsen, Phys. Rev. B 46, 13647 (1992).
- <span id="page-11-1"></span><sup>36</sup> P. A. Br¨uhwiler, A. J. Maxwell, A. Nilsson, N. M˚artensson, and O. Gunnarsson, Phys. Rev. B 48, 18296 (1993).
- <span id="page-11-2"></span><sup>37</sup> T. Miyake and F. Aryasetiawan, Phys. Rev. B 77, 085122 (2008).
- <sup>38</sup> T. Miyake and F. Aryasetiawan, and M. Imada, Phys. Rev. B 80, 155134 (2009).
- <span id="page-11-3"></span><sup>39</sup> K. Nakamura, R. Arita, and M. Imada, J. Phys. Soc. Jpn. 77, 093711 (2008).
- <span id="page-11-4"></span><sup>40</sup> T. Miyake, K. Nakamura, R. Arita, and M.Imada, J. Phys. Soc. Jpn. 79, 044705 (2010).
- <sup>41</sup> R. Arita, J. Kuneˇs, A. Kozhevnikov, A. G. Eguiluz, and M. Imada, [arXiv:1107.0835.](http://arxiv.org/abs/1107.0835)
- <sup>42</sup> P. Hohenberg and W. Kohn, Phys. Rev. 136, B864 (1964).
- <sup>43</sup> W. Kohn and L. J. Sham, Phys. Rev. 140, A1133 (1965).
- <sup>44</sup> J. Yamauchi, M. Tsukada, S. Watanabe, and O. Sugino, Phys. Rev. B 54, 5586 (1996).
- <sup>45</sup> J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).
- <sup>46</sup> N. Troullier and J. L. Martins, Phys. Rev. B 43, 1993 (1991).
- <sup>47</sup> L. Kleinman and D. M. Bylander, Phys. Rev. Lett. 48,
- <sup>48</sup> S. G. Louie, S. Froyen, and M. L. Cohen, Phys. Rev. B 26, 1738 (1982).
- <sup>49</sup> O. Zhou and D. E. Cox, J. Phys. Chem. Solids 53, 1373 (1992).
- <sup>50</sup> Initial internal coordinates for optimization are made after Ref[.51](#page-12-9) for solid picene and Ref[.53](#page-12-10) for solid phenenthrene.
- <span id="page-12-9"></span><sup>51</sup> A. De, R. Ghosh, S. Roychowdhury, and P. Roychowdhury, Acta. Crystallogr., Sect. C 41, 907 (1985).
- <sup>52</sup> T. Echigo, M. Kimata, and T. Maruoka, Am. Mineral. 92, 1262 (2007).
- <span id="page-12-10"></span><sup>53</sup> J. Trotter, Acta. Cryst. 16, 605 (1963).
- <sup>54</sup> The volume per C<sup>60</sup> <sup>3</sup><sup>−</sup> is defined as Vconv/4 for fcc structure and Vunit/2 for A15 structure with Vconv (Vunit) being the volume of conventinal (unit) cell.
- <sup>55</sup> Y. Iwasa, Nature 466, 191 (2010).
- <span id="page-12-4"></span><sup>56</sup> G. R. Darling, A. Y. Ganin, M. J. Rosseinsky, Y. Takabayashi, and K. Prassides, Phys. Rev. Lett. 101, 136404 (2008).
- <sup>57</sup> K. Momma and F. Izumi, J. Appl. Crystallogr. 41, 653 (2008).
- <span id="page-12-0"></span><sup>58</sup> T. Fujiwara, S. Yamamoto, and Y. Ishii, J. Phys. Soc. Jpn.

72, 777 (2003); Y. Nohara, S. Yamamoto, and T. Fujiwara, Phys. Rev. B 79, 195110 (2009).

- <span id="page-12-1"></span><sup>59</sup> M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986); M. S. Hybertsen and S. G. Louie, ibid. 35, 5585 (1987).
- <span id="page-12-2"></span><sup>60</sup> We note that the observed large ǫ cRPA <sup>M</sup> of picene<sup>3</sup><sup>−</sup> (∼12) originates from the polarization along the interlayer axis. The value of ǫ cRPA <sup>M</sup> along the interlayer axis is 21.2 and those along the intralayer axis are 6.7 for the a ∗ axis and 8.2 for the b ∗ axis. The observed anisotropy is consistent with the result of Ref. [27](#page-11-9).
- <span id="page-12-3"></span><sup>61</sup> P. Jegliˇc, D. Arˇcon, A. Potoˇcnik, A. Y. Ganin, Y. Takabayashi, M. J. Rosseinsky, and K. Prassides, Phys. Rev. B 80, 195424 (2009).
- <span id="page-12-5"></span><sup>62</sup> J. E. Han, E. Koch, and O. Gunnarsson, Phys. Rev. Lett. 84, 1276 (2000).
- <span id="page-12-6"></span><sup>63</sup> Indeed, the measurement of T<sup>c</sup> under pressure for alkali or alkali-earth metal doped phenanthrene was performed.[7](#page-11-10)[,8](#page-11-11)
- <span id="page-12-7"></span><sup>64</sup> W. Kohn, Phys. Rev. B 7, 2285 (1973).
- <span id="page-12-8"></span><sup>65</sup> F. Giustino, M. L. Cohen, and S. G. Louie, Phys. Rev. B 76, 165108 (2007).