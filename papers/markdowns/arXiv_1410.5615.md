# arXiv:1410.5615

**Paper ID:** a1d28de1bfaa4c66f6d7f3966394dd4b

**PDF Path:** apl/Superconductors/arXiv:1410.5615.pdf

**Processing Status:** complete

**Captions Added:** 1

**Generated:** 2025-06-24T13:44:27.006197

---

# **Vortex-penetration field at a groove with a depth smaller than the penetration depth**<sup>∗</sup>

Takayuki Kubo†

KEK, High Energy Accelerator Research Organization, Tsukuba, Ibaraki 305-0801 Japan

### *Abstract*

Analytical formula to evaluate the vortex-penetration field at a groove with a depth smaller than penetration depth is derived, which can be applied to surfaces of cavities or test pieces made from extreme type II superconductors such as nitrogen-doped Nb or alternative materials like Nb3Sn or NbN.

### **INTRODUCTION**

The vortex-penetration field B<sup>v</sup> is the field at which a vortex overcome the Bean-Livingston barrier [\[1\]](#page-2-0) and start to penetrate into the superconductor (SC). B<sup>v</sup> of extreme type II SC, where the penetration depth λ is much larger than the coherence length ξ, can be evaluated in the framework of the London theory. Materials that attract much attentions in the field of SC accelerating cavity such as dirty Nb like nitrogen-doped Nb and alternative materials like Nb3Sn or NbN are all categorized into this class. For an SC with an ideal flat surface, B<sup>v</sup> is given by B<sup>v</sup> = φ0/(4πλξ) ≃ 0.7Bc, where φ<sup>0</sup> is the flux quantum and B<sup>c</sup> is the thermodynamic critical magnetic field. Actually, experiments shows fields can not reach such a level. More realistic assumption, such as surface irregularities, should be incorporated.

In this paper we consider a groove with a depth δ smaller than λ as a simple example of a surface irregularity, which assume irregularities on cavity surfaces or test pieces made from extreme type II SC such as a nitrogen-doped Nb or alternative materials. B<sup>v</sup> at this type of irregularity has not been obtained so far, in spite of the fact that there are many studies on B<sup>v</sup> at a surface irregularity [\[2,](#page-2-1) [3,](#page-2-2) [4,](#page-2-3) [5\]](#page-2-4).

### **MODEL**

Let us consider a groove shown in Fig. [1\(](#page-0-0)a). Gray and white regions represent the SC and the vacuum, respectively. Surface, groove and applied magnetic-field are perpendicular to the x-y plane. The half width of the groove and the slope angle are given by R and π(α − 1)/2, respectively, and thus the depth is given by δ = R tan[π(α− 1)/2], where 1 < α < 2. The depth is assumed to satisfy ξ ≪ δ ≪ λ.

![](_page_0_Figure_12.jpeg)

**Caption:** Figure 12 presents a schematic diagram divided into two parts: (a) the z-plane and (b) the w-plane, representing the contour transformation pertinent in the analysis of superconductor groove dynamics. In (a), the diagram highlights the geometry of a groove, with angles and regions clearly marked to showcase the distinction between magnetic field application areas and superconductive surfaces. The direction of the applied current J₀ is indicated along the positive x-axis. The transformation to the w-plane in (b) reveals the similar layout, key to understanding the conformal mapping whereby points A, B, and C in the z-plane are transformed to points A', B', and C' in the w-plane. The figure elucidates the translation of vortices within superconductors when experiencing external magnetic fields. The scientific significance lies in its application to evaluating vortex-penetration fields at grooves with depth shallower than the penetration depth in extreme type II superconductors, a crucial component for enhancing the performance of superconducting cavities in high-energy accelerators.


<span id="page-0-0"></span>Figure 1: (a) Groove with a depth that is smaller than the penetration depth of the material and (b) its map on the wplane.

## **FORCES ACTING ON A VORTEX AND THE VORTEX-PENETRATION FIELD**

Suppose a vortex is at the position (x, y) = (0, δ + ξ), inside the bottom of groove. This vortex feels two distinct forces: (i) F<sup>M</sup> a force from a Meissner current due to an external field and (ii) F<sup>I</sup> a force due to an image antivortex that is introduced to satisfy the boundary condition of zero current normal to the surface. The former and the latter draw the vortex to the inside and the outside of the SC, respectively. The vortex-penetration field is a field at which these competing forces are balanced.[1](#page-0-1)

### *Force due to an external field*

An external magnetic-field pushes a vortex into the superconductor by a force F<sup>M</sup> = J<sup>M</sup> × φ0ˆz, where J<sup>M</sup> is a Meissner screening-current, <sup>φ</sup><sup>0</sup> = 2.<sup>07</sup> <sup>×</sup> <sup>10</sup>−<sup>15</sup> Wb is the flux quantum and ˆz is the unit vector parallel to the z-axis. To evaluate FM, we evaluate J<sup>M</sup> as follows. J<sup>M</sup> satisfies div J<sup>M</sup> = 0 and one of the Maxwell equations, J<sup>M</sup> = rot H, where the magnetic field H plays the role of the vector potential of JM. For our two-dimensional problem, H can be written as H = (0, 0, −ψ(x, y)), and J<sup>M</sup>

<sup>∗</sup> The work is supported by JSPS Grant-in-Aid for Young Scientists (B), Number 26800157

<sup>†</sup> kubotaka@post.kek.jp

<span id="page-0-1"></span><sup>1</sup>Detailed reviews are given in Ref. [\[6,](#page-2-5) [7\]](#page-2-6).

is given by J<sup>M</sup> = rot H = (−∂ψ/∂y, ∂ψ/∂x, 0). On the other hand, since λ is assumed to be much larger than the typical scale of the model, the London equation is reduced to rot J<sup>M</sup> = −△H = 0, which allows us to introduce a scalar potential of JM. For our two-dimensional problem the scalar potential can be written as φ(x, y), and J<sup>M</sup> is given by J<sup>M</sup> = −gradφ = (−∂φ/∂x, −∂φ/∂y, 0). Since both the two approaches should lead the same JM, we find

$$J\_{\rm Mx} = -\frac{\partial \phi}{\partial x} = -\frac{\partial \psi}{\partial y}, \qquad J\_{\rm My} = -\frac{\partial \phi}{\partial y} = \frac{\partial \psi}{\partial x}, \tag{1}$$

which are the Cauchy-Riemann conditions. Thus a function defined by

$$\Phi\_{\mathcal{M}}(z) \equiv \phi(x, y) + i\psi(x, y) \,, \tag{2}$$

is an holomorphic function of a complex variable z = x + iy, which is called the complex potential. If ΦM(z) is given, components of J<sup>M</sup> are derived from

<span id="page-1-1"></span>
$$J\_{\mathbf{M}x} - iJ\_{\mathbf{M}y} = -\frac{\partial \phi}{\partial x} + i\frac{\partial \phi}{\partial y} = -\frac{\partial \phi}{\partial x} - i\frac{\partial \psi}{\partial x} = -\frac{d\Phi\_{\mathbf{M}}(z)}{dz},\tag{3}$$

where the property of the holomorphic function, Φ ′ <sup>M</sup>(z) = ∂φ/∂x+i∂ψ/∂x, is used. Thus our two-dimensional problem is reduced to a problem of finding ΦM(z).

The complex potential ΦM(z) is derived from a complex potential ΦeM(w) on a complex w-plane shown in Fig. [1\(](#page-0-0)b) through a conformal mapping z = F(w), by which orthogonal sets of field lines in the w-plane are transformed into those in the z-plane. The map is given by the Schwarz-Christoffel transformation,

<span id="page-1-2"></span>
$$z = F(w) = K\_1 \int\_0^w f(w) dw + K\_2 \,. \tag{4}$$

The function f(w) is given by

$$f(w) = w^{\alpha - 1} (w^2 - 1)^{-\frac{\alpha - 1}{2}},\tag{5}$$

and the constants K<sup>1</sup> and K<sup>2</sup> are given by

$$K\_1 = \frac{\sqrt{\pi}R}{\Gamma(\frac{\alpha}{2})\Gamma(\frac{3-\alpha}{2})\cos\frac{\pi(\alpha-1)}{2}},\qquad(6)$$

$$K\_2 \quad = \quad i\delta = iR\tan\frac{\pi(\alpha - 1)}{2},\tag{7}$$

which are determined by conditions that A′ and C ′ on the w-plane are mapped into A and C on the z-plane, respectively. The complex potential on the w-plane is given by ΦeM(w) = Je0w (Je<sup>0</sup> ≡ K1J0), which reproduces the current distribution on the <sup>w</sup>-plane: <sup>−</sup>Φe′ <sup>M</sup>(w) = −Je0. Thus the complex potential on the z-plane is given by

<span id="page-1-0"></span>
$$\Phi\_{\mathcal{M}}(z) = \tilde{\Phi}\_{\mathcal{M}}(F^{-1}(z)) = F^{-1}(z)\tilde{J}\_0 \,, \tag{8}$$

where F −1 is an inverse function of F.

All that is left is to substitute Eq. [\(8\)](#page-1-0) into Eq. [\(3\)](#page-1-1). The we obtain

<span id="page-1-4"></span>
$$J\_{\rm Mx} - iJ\_{\rm My} = -\frac{J\_0}{f(w)},\tag{9}$$

where dF <sup>−</sup><sup>1</sup>/dz = dw/dz = (dz/dw) <sup>−</sup><sup>1</sup> = (dF/dw) −1 is used. In order to evaluate J<sup>M</sup> at the vortex position z = z<sup>v</sup> ≡ i(δ+ξ), w corresponding to z<sup>v</sup> is necessary. While no closed form of w = F −1 (z) exist, that of an approximate expression can be derived. Suppose w = iǫ (0 < ǫ ≪ 1) is mapped into z = z<sup>v</sup> on the z-plane by Eq. [\(4\)](#page-1-2). Then we obtain i(δ + ξ) ≃ iδ + iK1ǫ <sup>α</sup>/α, and find a relation

$$
\epsilon = \left(\frac{\alpha \xi}{K\_1}\right)^{\frac{1}{\alpha}},\tag{10}
$$

which immediately leads

<span id="page-1-3"></span>
$$f(i\epsilon) \simeq \epsilon^{\alpha - 1} = \left(\frac{\alpha \xi}{K\_1}\right)^{\frac{\alpha - 1}{\alpha}}.\tag{11}$$

Substituing Eq. [\(11\)](#page-1-3) into Eq. [\(9\)](#page-1-4), we find

$$J\_{\mathbf{M}x}(z\_v) = -\left(\frac{K\_1}{\alpha \xi}\right)^{\frac{\alpha - 1}{\alpha}} J\_0, \qquad \qquad J\_{\mathbf{M}y}(z\_v) = 0. \tag{12}$$

Then the force due to the external field can be evaluated as

<span id="page-1-7"></span>
$$\begin{aligned} \mathbf{F}\_{\mathbf{M}} &= \mathbf{J}\_{\mathbf{M}} \times \phi\_0 \mathbf{\hat{z}} \\ &= \left( \frac{\sqrt{\pi}}{\Gamma(\frac{\alpha}{2}) \Gamma(\frac{3-\alpha}{2}) \alpha \cos \frac{\pi(\alpha-1)}{2}} \frac{R}{\xi} \right)^{\frac{\alpha-1}{\alpha}} \phi\_0 J\_0 \mathbf{\hat{y}}(\mathbf{l}\mathbf{3}) \end{aligned}$$

where yˆ is the unit vector parallel to the y-axis.

### *Force due to the image antivortex*

A current associated with a vortex near the surface satisfies the boundary condition of zero current normal to the surface. This boundary condition can be satisified by removing the surface and introducing appropriate image antivortex (antivotices). Then the current can be expressed as JV+I = J<sup>V</sup> + J<sup>I</sup> , where J<sup>V</sup> and J<sup>I</sup> represent currents due to the vortex and image antivortex (antivortices), respectively. The force due to the image antivortex (antivortices) FI is given by F<sup>I</sup> = J<sup>I</sup> × φ0zˆ. Thus our next task is to evaluate J<sup>I</sup> at the vortex position z = z<sup>v</sup> ≡ i(δ + ξ).

A scalar and a vector potentials of JV+I, and the complex potential ΦV+I can be introduced in much the same way as the above. Then components of JV+I are given by

<span id="page-1-6"></span>
$$J\_{\rm V+Ix} - iJ\_{\rm V+Iy} = -\frac{d\Phi\_{\rm V+I}(z)}{dz},\tag{14}$$

where ΦV+I(z) can be derived from the complex potential ΦeV+I(w) on the w-plane. Since the vortex and the image antivortex on the w-plane are located at w = +iǫ and −iǫ, respectively, ΦeV+I(w) is given by

$$\tilde{\Phi}\_{\rm V+I}(w) = \frac{i\phi\_0}{2\pi\mu\_0\lambda^2} \left[ \log(w - i\epsilon) - \log(w + i\epsilon) \right], \tag{15}$$

and thus the complex potential on the z-plane is given by

<span id="page-1-5"></span>
$$\Phi\_{\rm V+I}(z) = \widetilde{\Phi}\_{\rm V+I}(F^{-1}(z))\,. \tag{16}$$

F is the Schwarz-Christoffel transformation given by Eq. [\(4\)](#page-1-2). Substituting Eq. [\(16\)](#page-1-5) into Eq. [\(14\)](#page-1-6), we find

$$J\_{\rm V+Ix} - iJ\_{\rm V+Iy} = \frac{1}{K\_1 f(w)} \frac{-i\phi\_0}{2\pi\mu\_0\lambda^2} \left(\frac{1}{w - i\epsilon} - \frac{1}{w + i\epsilon}\right). \tag{17}$$

At the vortex position z = z<sup>v</sup> or w = iǫ, the first term of the square bracket diverges, which is contribution from the current due to the vortex and should be abandoned for the computation of J<sup>I</sup> . Then J<sup>I</sup> at the vortex position is give by

$$iJ\_{\rm Ix} - iJ\_{\rm Iy} = \frac{1}{K\_1 f(i\epsilon)} \frac{i\phi\_0}{2\pi\mu\_0\lambda^2} \left(\frac{1}{2i\epsilon}\right) = \frac{\phi\_0}{4\pi\mu\_0\lambda^2\xi\alpha},\tag{18}$$

or,

$$J\mathbf{l}\_x(z\_v) = \frac{\phi\_0}{4\pi\mu\_0\lambda^2\xi\alpha}, \qquad \qquad J\mathbf{l}\_y(z\_v) = 0,\tag{19}$$

where a relation ǫf(iǫ) = ǫ <sup>α</sup> = αξ/K<sup>1</sup> is used. Then the force due to the image anti-vortex is given by

<span id="page-2-7"></span>
$$\mathbf{F}\_{\rm I} = \mathbf{J}\_{\rm M} \times \phi\_0 \hat{\mathbf{z}} = -\frac{\phi\_0^2}{4\pi\mu\_0\lambda^2\xi\alpha} \hat{\mathbf{y}}\,. \tag{20}$$

Note that Eq. [\(20\)](#page-2-7) is reduced to the force from the flat surface when α = 1, and is maximized when the groove is a crack with α ≃ 2. Eq. [\(20\)](#page-2-7) is identical with that given in Ref. [\[3\]](#page-2-2).

### *Vortex-penetration field*

The vortex-penetration field B<sup>v</sup> can be evaluated by balancing the two competing forces given by Eq. [\(13\)](#page-1-7) and [\(20\)](#page-2-7):

$$\left(\frac{\sqrt{\pi}}{\Gamma(\frac{\alpha}{2})\Gamma(\frac{3-\alpha}{2})\alpha\cos\frac{\pi(\alpha-1)}{2}}\frac{R}{\xi}\right)^{\frac{\alpha-1}{\alpha}}\phi\_0 J\_0 = \frac{\phi\_0^2}{4\pi\mu\_0\lambda^2\xi\alpha} . \tag{21}$$

The surface current J<sup>0</sup> is given by J<sup>0</sup> = −µ −1 0 dB/dx|x=0 = B0/µ0Λ, where B<sup>0</sup> is the surface magnetic-field and Λ is a quantity with the dimension of length. For examples,

$$\Lambda = \begin{cases} \lambda & \text{(semi-infinite SC)},\\ \lambda \frac{\cosh\frac{d\_S}{\lambda} + (\frac{\lambda'}{\lambda} + \frac{d\_T}{\lambda})\sinh\frac{d\_S}{\lambda}}{\sinh\frac{d\_S}{\lambda} + (\frac{\lambda'}{\lambda} + \frac{d\_T}{\lambda})\cosh\frac{d\_S}{\lambda}} & \text{(multilayer SC)}. \end{cases} (22)$$

where d<sup>S</sup> , dI, and λ ′ are an SC layer thickness, insulator layer thickness and penetration depth of SC substrate material, respectively.[2](#page-2-8) The finally we obtain

<span id="page-2-9"></span>
$$B\_v = \frac{\phi\_0}{4\pi\lambda\xi} \frac{\Lambda}{\lambda} \frac{1}{\alpha} \left( \frac{\Gamma(\frac{\alpha}{2})\Gamma(\frac{3-\alpha}{2})\alpha\cos\frac{\pi(\alpha-1)}{2}}{\sqrt{\pi}} \frac{\xi}{R} \right)^{\frac{\alpha-1}{\alpha}}, (23)$$

Note that Eq. [\(23\)](#page-2-9) is reduced to B<sup>v</sup> of semi-infinite SC or multilayer SC with ideal flat-surface when α = 1.

### **SUMMARY**

Analytical formula to evaluate the vortex-penetration field at a groove with a depth smaller than penetration depth was derived. The formula would be useful to analyze relation between surfaces and performance-test results of cavities or test pieces made from extreme type II SC such as a dirty Nb like nitrogen-doped Nb or alternative materials like Nb3Sn or NbN.

### **REFERENCES**

- <span id="page-2-1"></span><span id="page-2-0"></span>[1] C. P. Bean and J. D. Livingston, Phys. Rev. Lett. **12**, 14 (1964).
- <span id="page-2-2"></span>[2] F. Bass, V. D. Freilikher, B. Ya. Shapiro, and M. Shvartser, Physica C **260**, 231 (1996).
- <span id="page-2-3"></span>[3] A. Buzdin and M. Daumens, Physica C **294**, 257 (1998).
- <span id="page-2-4"></span>[4] A. Buzdin and M. Daumens, Physica C **332**, 108 (2000).
- <span id="page-2-5"></span>[5] A. Yu. Aladyshkin, A. S. Mel'nikov, I. A. Shereshevsky, and I. D. Tokman, Physica C **361**, 67 (2001).
- <span id="page-2-6"></span>[6] T. Kubo, Y. Iwashita, and T. Saeki, in *Proceedings of SRF2013, Paris, France* (2013), p. 427, TUP007.
- <span id="page-2-10"></span>[7] T. Kubo, Y. Iwashita, and T. Saeki, in *Proceedings of IPAC'14, Doresden, Germany* (2014), p. 2522, WEPRI023.
- <span id="page-2-11"></span>[8] A. Gurevich, Appl. Phys. Lett. **88**, 012511 (2006).
- [9] T. Kubo, Y. Iwashita, and T. Saeki, Appl. Phys. Lett. **104**, 032603 (2014).
- <span id="page-2-12"></span>[10] T. Kubo, Y. Iwashita, and T. Saeki, in *Proceedings of IPAC'13, Shanghai, China* (2013), p. 2343, WEPWO014.

<span id="page-2-8"></span><sup>2</sup>See Ref. [\[8,](#page-2-10) [9\]](#page-2-11). Detailed reviews are given in Ref. [\[6,](#page-2-5) [7,](#page-2-6) [10\]](#page-2-12).