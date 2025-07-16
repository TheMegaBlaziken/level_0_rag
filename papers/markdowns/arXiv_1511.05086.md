# arXiv:1511.05086

**Paper ID:** 46f8d87aa03cbe42778c571819ba3266

**PDF Path:** apl/Superconductors/arXiv:1511.05086.pdf

**Processing Status:** complete

**Captions Added:** 7

**Generated:** 2025-06-24T13:44:27.091134

---

# **A model for the compositions of non-stoichiometric intermediate phases formed by diffusion reactions, and its application to Nb3Sn**

## **superconductors**

X. Xu\*, M. D. Sumption

Department of Materials Science and Engineering, the Ohio State University, Columbus, OH 43210 USA \*Electronic mail: xu.452@osu.edu

*In this work we explore the compositions of non-stoichiometric intermediate phases formed by diffusion reactions: a mathematical framework is developed and tested against the specific case of Nb3Sn superconductors. In the first part, the governing equations for the bulk diffusion and inter-phase interface reactions during the growth of a compound are derived, numerical solutions to which give both the composition profile and growth rate of the compound layer. The analytic solutions are obtained with certain approximations made. In the second part, we explain an effect that the composition characteristics of compounds can be quite different depending on whether it is the bulk diffusion or grain boundary diffusion that dominates in the compounds, and that "frozen" bulk diffusion leads to unique composition characteristics quite distinct from equilibrium expectations; then the model is modified for the case of grain boundary diffusion. Finally, we apply this model to the Nb3Sn superconductors and propose the approaches to control their compositions.*

Keywords: Non-stoichiometric compounds, diffusion reaction, composition, model, Nb3Sn

### **Introduction**

Intermediate phases with finite composition ranges represent a large class of materials, and their compositions may influence their performance in application, as demonstrated in a variety of materials, such as electrical conductivity of oxides (e.g., TiO2-y 1 ), electromagnetic properties of superconductors (e.g., Nb3Sn and YBa2Cu3O7-y 2 ), and mechanical properties of some

intermetallics (e.g., Ni-Al0.4-0.55 3 ), etc. For instance, the superconducting Nb3Sn phase, which finds significant applications in the construction of 12-20 T magnets4,5 , has a composition range of ~17-26 Sn at.%6,7 , and its superconducting transition temperature and magnetic field decrease dramatically as Sn content drops4,7-8 . The Nb3Sn phase, which is generally fabricated from Cu-Sn and Nb precursors through reactive diffusion processes, is always found to be Sn-poor (e.g., 21- 24 at.%4,8-9 ), making composition control one of the primary concerns in Nb3Sn development since 1970s<sup>10</sup> . Although a large number of previous experiments (e.g., 4,8-11) have uncovered some factors that influence the Sn content, it is still a puzzle what fundamentally determines the Nb3Sn composition. This work aims to fill that gap. Here it is worth mentioning that the composition interval of a compound layer does not necessarily coincide with its equilibrium phase field ranges – the former can be narrower (e.g., the Nb3Sn example above) if the interphase interface reaction rates are slow relative to the diffusion rate across the compound, which results in discontinuities in chemical potentials at the interfaces.

There have been numerous studies regarding diffusion reaction processes, most of which have focused on layer growth kinetics (e.g.,12-16 ), compound formation and instability (e.g., 14-16 ), phase diagram determination (e.g.,<sup>17</sup> ), and interdiffusion coefficient measurement (e.g.,<sup>18</sup> ), while a systematic model exploring how to control compound composition is still lacking. We find it indeed possible to modify the model developed by Gosele and Tu<sup>13</sup> for deriving the layer growth kinetics of compounds to calculate their compositions; however, certain assumptions (e.g., steady-state diffusion and first-order interface reaction rates) that the model was based on may limit the accuracy of the composition results. In this work, we aim to develop a rigorous, systematic mathematical framework for the compositions of intermediate phases.

#### **Results**

Let us consider that an AnB compound with a B content range of *X<sup>l</sup>* - *X<sup>u</sup>* (*X<sup>l</sup>* < *X<sup>u</sup>* <1) is formed in a system of M-B/A, where M is a third element that does not dissolve in AnB lattice<sup>19</sup> , and the solubility of B in A is negligible<sup>6</sup> . This is the case we see for the Nb3Sn example above (for which A stands for Nb, B for Sn, and M for Cu), but the work below can be modified for other cases. An isothermal cross section of such an M-A-B phase diagram at a certain temperature is shown in Fig. 1. The use of the third element M is to decrease the chemical potential of B, so that unwanted high-B A-B compounds (e.g., NbSn<sup>2</sup> and Nb6Sn<sup>5</sup> 6 ) that would form in the B/A binary system can be avoided. Similar to the Cu-Nb-Sn system, let us assume B is the primary diffusing species in the AnB layer<sup>20</sup> and that the diffusivity of B in M-B is high<sup>10</sup> . A schematic of the M-B/AnB/A system for a planar geometry is shown in Fig. 2, but the model can be modified for cylindrical and spherical geometries. Let us denote the M-B/AnB and AnB/A inter-phase interfaces as I and II, respectively, and the concentrations, chemical potentials, activities, and diffusion fluxes of B in the AnB layers near interfaces I and II as *X<sup>I</sup>* , *μ<sup>I</sup>* , *a<sup>I</sup>* , *J<sup>I</sup>* , and *XII*, *μII*, *aII*, *JII*, respectively. Let us also denote the *XB*s, *μB*s, *aB*s of M-B source and A-*X<sup>l</sup>* B as *Xs*, *μs*, *as*, and *X<sup>l</sup>* , *μ<sup>l</sup>* , *a<sup>l</sup>* , respectively.

In this work let us assume the diffusivity of B in AnB, *D*, and the molar volume of AnB, *Vm*, do not vary with *XB*, in which case the continuity equation in the AnB layer is given by:

$$\frac{\partial X\_B}{\partial t} = D \frac{\partial^2 X\_B}{\partial \mathbf{x}^2} \tag{1}$$

According to mass conservation, in a unit time the amount of B transferring across the interface I should equal to that diffusing into AnB from the interface I, and the amount arriving at the interface II should equal to that transferring across it, i.e., *dn*/*dt*|<sup>I</sup> = *J<sup>I</sup>* ∙*A<sup>I</sup>* , and *dn*/*dt*|II = *JII*∙*AII*, where *A<sup>I</sup>* and *AII* are the areas of the interfaces I and II, respectively. The molar transport rate *dn*/*dt* across an interface equals to *r*∙*Aint*∙exp(-*Q*/*RT*)∙[1-exp(-Δ*μ*/*RT*)], where *r* is the transfer rate constant for this interface with the unit of mol/(m<sup>2</sup> ∙s), *Aint* is the interface area, *Q* is the energy barrier, *R* is the gas constant, *T* is the temperature in K, and Δ*μ* is the driving force for atom transfer. For the interface I, Δ*μ*|<sup>I</sup> =*μs*-*μ<sup>I</sup>* . For the interface II, Δ*μ*|II =*μII*–*μ<sup>l</sup>* , because *μB*(A)=*μB*(A-*X<sup>l</sup>* B). With *J<sup>B</sup>* = -(*D*/*Vm*)∙(∂*X*B/∂*x*), we have:

$$\begin{aligned} \text{(\infty), we have.}\\ \left[r\_I \exp\left(-\frac{\mathcal{Q}\_I}{RT}\right)\right] \left[1 - \exp\left(-\frac{\mu\_s - \mu\_I}{RT}\right)\right] = -\frac{D}{V\_m} \frac{\partial X\_B}{\partial \mathbf{x}}\Big|\_{I} \end{aligned} \tag{2}$$

$$r\_{\rm II} \exp\left(-\frac{\mathcal{Q}\_{\rm II}}{RT}\right) \left[1 - \exp\left(-\frac{\mu\_{\rm II} - \mu\_{\rm I}}{RT}\right)\right] = -\frac{D}{V\_{\rm in}} \frac{\partial \mathcal{X}\_{\rm B}}{\partial \mathbf{x}}\Big|\_{\rm II} \tag{3}$$

Eqs. (2) and (3) are the boundary conditions for Eq. (1). Note that *X<sup>s</sup>* drops with annealing time as B in M-B is used for AnB growth, so *μ<sup>s</sup>* drops with *t*:

$$\begin{aligned} \text{for } & \mathbf{A}\_{\mathbf{n}} \mathbf{B} \text{ growth, so } \mu\_s \text{ drops with } t; \\\\ \frac{d\,\mu\_s}{dt} &= \frac{d\,\mu\_s}{d\mathbf{X}\_s} \frac{d\mathbf{X}\_s}{dt} = -\frac{d\,\mu\_s}{d\mathbf{X}\_s} \frac{n\_M}{\left(n\_M + n\_{B0} - \int\_t d\mathbf{n} \wedge \, dt \Big|\_I\right)^2} \frac{dn}{dt} \Big|\_I \end{aligned} \tag{4}$$

where *n<sup>M</sup>* and *nB0* are the moles of M and B in the M-B precursor. For those systems without the third element, *μ<sup>s</sup>* is constant, and Eq. (4) is not needed. In addition, since the B atoms diffusing to the interface II are used to form new AnB layers, we have:

$$\frac{dl}{dt} = \frac{J\_{\text{II}} V\_m}{X\_{\text{II}}} = -\frac{D}{X\_{\text{II}}} \frac{\partial X\_{\text{B}}}{\partial \mathbf{x}} \Big|\_{\text{II}}\tag{5}$$

Eqs. (1)-(5) are the governing equations for the system set up above, solutions to which give both the *XB*(*x*, *t*) and the *l*(*t*) of a growing AnB layer. It should be noted that for the systems with large volume expansion associated with transformation from A to AnB, stress effects need to be considered<sup>21</sup> .

To simplify Eqs. (2) and (3), we notice that 1-exp[-(*μs*-*μI*)/*RT*] = 1-*aI*/*as*, since *μs*-*μ<sup>I</sup>* = *RT*ln(*as*/*aI*); similarly, 1-exp[-(*μII*-*μl*)/*RT*] = 1-*al*/*aII*. Let us also denote *D*/[*Vm*∙*r<sup>I</sup>* ∙exp(-*QI*/*RT*)] as *φI* , and *D*/[*Vm*∙*rII*∙exp(-*QII*/*RT*)] as *φII*: clearly *φ<sup>I</sup>* and *φII* represent the ratios of diffusion rate over

interface reaction rates, and have a unit of meter. Then Eqs. (2) and (3) can be respectively written as:

$$1 - \frac{a\_I}{a\_s} = -\varphi\_I \frac{\left. \partial X\_B}{\left. \right|\_I} \right|\_I \tag{6}$$

$$1 - \frac{a\_l}{a\_{\rm II}} = -\varphi\_{\rm II} \frac{\partial X\_B}{\partial \mathbf{x}}\Big|\_{\rm II} \tag{7}$$

For now, let us consider two extreme cases.

First, for the case that the interface reaction rates are much higher than the diffusion rate across the AnB layer (i.e., diffusion-rate limited), *φ<sup>I</sup>* and *φII* are near zero; according to Eqs. (2)- (3), *μB*s are continuous at both interfaces. Suppose *μ<sup>s</sup>* and the position of interface I, *x<sup>I</sup>* , are both constant with time, then *X<sup>I</sup>* is also constant, and the solutions to Eqs. (1) and (5) are respectively *XB*(*x*, *t*) = *XI*-(*XI*-*Xl*)∙erf{(*x-xI*)/[2√(*Dt*)]}/erf(*k*/2) and *l*=*k*√(*Dt*) for the AnB layer, where *k* can be numerically solved from *k*∙exp(*k* 2 /4)∙erf(*k*/2) = 2/√π∙(*XI*-*Xl*)/*X<sup>l</sup>* . For instance, for *X<sup>I</sup>* = 0.26 and *X<sup>l</sup>* = 0.17, *k*=0.953. On the other hand, if the interface reaction rates are much lower than the diffusion rate across AnB (e.g., as the AnB layer is thin), *φ<sup>I</sup>* and *φII* are large; according to Eqs. (2)-(3), *X*<sup>B</sup> and *J<sup>B</sup>* are nearly constant in the entire AnB layer. Thus, (1-*aB*/*as*)/*φ<sup>I</sup>* = (1-*al*/*aB*)/*φII*, from which *a<sup>B</sup>* can be calculated. Integration of Eq. (5) gives: *l* ∝ *t*, and the pre-factor depends on the interface reaction rates.

For a general case between these two extremes, the equations can only be solved with the *μ*(*X*) relations of M-B and AnB provided. Next, let us consider a compound with a narrow composition range, so that as a Taylor series expansion is performed around *X<sup>l</sup>* for its *a*(*XB*) curve, high-rank terms can be neglected because |*X*-*X<sup>l</sup>* | ≤ (*Xu*-*Xl*) is small; that is, *a<sup>X</sup>* ≈ *a<sup>l</sup>* + *κ*(*X*-*Xl*), where *κ* is the linear coefficient of the *a*(*X*) curve. Given the complex boundary conditions for Eq. (1), to obtain the analytic solutions we introduce a second approximation if the AnB composition range is narrow: the *X*(*x*) profile of the AnB layer is linear so that at a certain time *J* is constant with *x*, such that -(∂*X*B/∂*x*)*|*<sup>I</sup> ≈ -(∂*X*B/∂*x*)*|*II ≈(*XI*-*XII*)/*l*. With these two approximations, we can solve Eqs. (6)-(7) and obtain that: ( ) 4 ( ) ( ) <sup>2</sup> *a l a a a l a l a <sup>a</sup>* 

$$\begin{aligned} \text{We can solve Eqs. (6)-(7) and obtain that:}\\ a\_{\mu} &= \frac{\sqrt{(\wp\_{l}a\_{s} + \kappa l - \wp\_{\mu}a\_{s})^{2} + 4\wp\_{\mu}a\_{l}(\wp\_{l}a\_{s} + \kappa l)} - (\wp\_{l}a\_{s} + \kappa l - \wp\_{\mu}a\_{s})}{2\wp\_{\mu}} = \frac{2a\_{l}}{1 - \eta + \sqrt{(1 - \eta)^{2} + 4\eta^{2}\big/\_{a\_{s}}a\_{s}}} \end{aligned} (8)$$

where *η*=*φIIas*/(*φIas*+*κl*). Then *a<sup>I</sup>* can be calculated from *aII* , and *X<sup>I</sup>* and *XII* can be calculated from *a<sup>I</sup>* and *aII* using *X* =*X<sup>l</sup>* +(*aX*-*al*)/*κ*.

To verify the results, the equations are solved for a hypothetical system analytically and numerically, with and without the assumption that *X*(*x*) is linear, respectively. The obtained composition profiles are shown in Fig. 3 (a). For simplicity, *μ<sup>s</sup>* of the system is set as *μB*(A-*X<sup>u</sup>* B) and is constant (for Nb3Sn systems, this means that Nb6Sn<sup>5</sup> serves as Sn source), and the other parameters are specified in the figure. The difference between the analytic and numerical solutions is <0.1%, showing that the approximation of linear *X*(*x*) is good if the composition range is small (2 at.% in this case). The *l*(*t*) result (where *t* is the annealing time after the incubation period) from the numerical calculations is shown in Fig. 3 (b). While the analytic *l*(*t*) solution is complicated, some *l*(*t*) relations with simple forms can be used as approximations. The most widely used *l*(*t*) relation for the case of constant *μ<sup>s</sup>* is *l*=*bt<sup>m</sup>* , in which *m*=1 for reactionrate limited and *m*=0.5 for diffusion-rate limited; however, a defect with this relation is that as *l* increases from zero, it may shift from reaction-rate limited to diffusion-rate limited, so *m* may vary with *t*. Here a new relation *l*=*q*[√(*t*+*τ*)-√*τ*] (where *q* is a growth constant and *τ* is a characteristic time) is proposed. Such a relation is consistent with *l* 2 /*v1*+*l*/*v2*=*t* (where *v<sup>1</sup>* and *v<sup>2</sup>* are constants related to diffusion rate and interface reaction rates, respectively) proposed by previous studies13,14 . This relation overcomes the above problem because as *t* << *τ*, *l* =[*q*/(2*τ*)]∙*t*

and as *t* >> *τ*, *l*=*q*√*t*. As can be seen from Fig. 3 (b), a better fit to the numerical *l*(*t*) curve in the whole range is achieved by *l*=*q*(√(*t*+*τ*)-√*τ*).

#### **Discussion**

Before discussing the application of this model to a specific material system, it must be pointed out that all of the analysis and calculations above are for the case that B diffuses through AnB bulk. In such a case, for an M-B/AnB/A system, as *μ<sup>s</sup>* drops with the growth of AnB layer, *XB*(*x*) of AnB should decrease with *μs*, because *μ<sup>s</sup>* ≥ *µ<sup>I</sup>* ≥ *µII* ≥ *µ<sup>l</sup>* . Finally, one of two cases will occur: either *µ<sup>s</sup>* drops to *µ<sup>l</sup>* (if A is in excess) so the system ends up with the equilibrium among A, A-*X<sup>l</sup>* B, and M-*X<sup>1</sup>* B (as shown by the shaded region in the isothermal M-A-B phase diagram in Fig. 1), or A is consumed up and AnB gets homogenized with time and finally *µB*(AnB)=*µB*(M-B) (as shown by the dashed line in Fig. 1). In either case, AnB eventually reaches homogeneity.

However, we find that the composition could be different for a compound in which the bulk diffusion is low while grain boundary diffusion dominates. One such example is Nb3Sn, the composition of which displays some extraordinary features. As an illustration, the *XSn*s of a Cu-Sn/Nb3Sn/Nb diffusion reaction couple after various annealing times are shown in Fig. 4. Clearly, as the *XSn* (and *µSn*) of Cu-Sn drop with time, the *XSn*s of Nb3Sn do not drop accordingly; instead, they more or less remain constant with time. In addition, from 320 hours to 600 hours, although Nb has been fully consumed, the *XSn* of Nb3Sn does not homogenize (i.e., the *XSn* gradient does not decrease) with time. In many other studies on Cu-Sn/Nb systems with Nb in excess (e.g., 4,8-9 ), even after extended annealing times after the Nb3Sn layers have finished growing (which indicates that the Sn sources have been depleted, i.e., *µSn*s have dropped to *µl*), *XSn*s of Nb3Sn remain high above *X<sup>l</sup>* , without dropping with annealing time.

The reason for these peculiarities is that grain boundary diffusion in Nb3Sn dominates due to extremely low bulk diffusivity (e.g., lower than 10-23 m 2 /s at 650 °C)20,22,23 and small Nb3Sn grain size (~100 nm). In this case, our model and equilibrium-state analysis apply only to the diffusion zones (i.e., the grain boundaries and the inter-phase interfaces) instead of the bulk. To clarify this point more clearly, a schematic of the diffusion reaction process is shown in Fig. 5. At time *t1*, at the AnB/A interface, high-B AnB (L<sup>2</sup> layer) reacts with A (L<sup>3</sup> layer) to form some new AnB cells, leaving B vacancies (noted as VBs) in L<sup>2</sup> layer (time *t2*). If bulk diffusivity is high, VBs simply diffuse through bulk (e.g., from L<sup>2</sup> to L1, as shown by grey dotted arrows) to the B source. If bulk diffusion is frozen, the VBs diffuse first along AnB/A inter-phase interface (as shown by green solid arrows), and then along AnB grain boundaries to the B source. This process continues until this L<sup>3</sup> layer entirely becomes AnB (time *t3*), so the reaction frontier moves ahead to L3/L4, while the L2/L<sup>3</sup> interface now becomes an inter-plane inside AnB lattice. If bulk diffusion is frozen, the VBs in the L<sup>2</sup> layer that have not diffused to B source will be frozen in this layer forever, and will perhaps transform to A-on-B anti-site defects later (e.g., for Nb3Sn, Nb-on-Sn anti-sites are more stable than Sn vacancies<sup>24</sup>). Since these point defects determine the AnB composition, the *X<sup>B</sup>* in this L<sup>2</sup> layer cannot change anymore regardless of *μ<sup>B</sup>* variations in grain boundaries. That is to say, *X<sup>B</sup>* of any point is just the *XII* of the moment when the reaction frontier sweeps across this point, i.e., the *XB*(*x*) of an AnB layer is simply an accumulation of *XII*s with *l* increase. Returning to Fig. 3 (a), the dashed lines display the evolution of *XB*(*x*) with *l* increase for bulk diffusion, while that for grain boundary diffusion is shown by the solid lines. Since the energy dispersive spectroscopy (EDS) attached to scanning electron microscopes (SEM) that is used to measure the compositions typically has a spatial resolution of 0.5-2 μm, and thus mainly reflects the bulk composition, the composition characteristics of Nb3Sn layers as described above can be explained. It should be noted that knowledge of the difference between
bulk diffusion and grain boundary diffusion is important in controlling the final composition of a compound. For instance, if bulk diffusivity is high, one method to form high-B AnB is increasing the starting B/A ratio so that after long annealing time for homogenization subsequent to the full consumption of A, *µB*(M-B)=*µB*(A-*X<sup>u</sup>* B). However, our experiments demonstrate that for compounds with low bulk diffusivity (e.g., Nb3Sn), such an approach does not work; instead, controlling the *XII*s while the compounds are growing is the only way. For those compounds with low but non-negligible bulk diffusivities, their compositions would be between these two extremes.

Then what determines the bulk composition as grain boundary diffusion dominates? From Fig. 5, it can be clearly seen that there is a competition deciding the V<sup>B</sup> fraction in the frontier AnB layer: at *t<sup>2</sup>* the reaction across the AnB/A interface produces VBs in L<sup>2</sup> layer, while the diffusion of B along AnB GBs and AnB/A interface fills these VBs. Thus, if the diffusion rate is slow relative to the reaction rate at interface II (i.e., *φII* is low), a high fraction of VBs would be left behind as the interface II moves ahead, causing low B content; if, on the other hand, the diffusion rate is high relative to the reaction rate at interface II, the AnB layer has enough time to get homogenized with the B source, causing low *X<sup>B</sup>* gradient. In this case, the *μ<sup>B</sup>* of B source and the reaction rate at interface I together set a upper limit for *μ<sup>B</sup>* of AnB.

Next, we will modify the above model for the case of grain boundary diffusion for quantitative analysis. As pointed out earlier, the chemical potentials of grain boundaries can change with *μ<sup>s</sup>* and *l*, while those of the bulk cannot. In such a case, *µ<sup>I</sup>* and *µII* (suppose the diffusivities along the inter-phase interfaces are large) can still be calculated using our model, provided that the *μ*(*X*) relation and *D* of the AnB grain boundary (instead of the bulk) are used in all of the equations, and that *φ<sup>I</sup>* and *φII* are multiplied by a factor of ∑*AGB*/*Aint* (where ∑*AGB* is the sum of the cross section areas of the grain boundaries projected to the inter-phase interfaces), because B diffuses only along grain boundaries in AnB while reactions occur at the entire interfaces. Approximately, ∑*AGB*/*Aint* ≈ [1-*d* 2 /(*d*+*w*) 2 ] ≈ 2*w*/*d* (where *w* is the AnB grain boundary width, and *d* is the grain size). Apparently, grain growth with annealing time reduces the diffusion rate. According to Eq. (8), *aII* is determined by *η* and *as*, and increases with them, as shown by Fig. 6. Since *η*=*φIIas*/(*φIas*+*κl*)= 1/[*φI*/*φII*+*κl*/(*φIIas*)], clearly *η* decreases as *φI*/*φII* and *l* increase, and the influence of *l* (which reflects the *XII*-*x* gradient) is mitigated as *φIIa<sup>s</sup>* increases. Thus, to improve *XII* of AnB at *l*=0, one should increase *μ<sup>s</sup>* and the reaction rate at interface I, and reduce the reaction rate at interface II; meanwhile, to reduce *XII*(*x*) gradient, one should increase *φII* (which means improving the diffusion rate or reducing the reaction rate at interface II) and *as*. Apparently, these quantitative conclusions are consistent with the above qualitative analysis.

Next let us compare this model with the example of Nb3Sn. It has been well established from experimental work that there are mainly two factors that can significantly influence the Sn content of Nb3Sn in a Cu-Sn/Nb3Sn/Nb diffusion reaction couple: heat treatment temperature and Cu-Sn source. The heat treatment temperature can simultaneously influence multiple factors of Eq. (8), such as *as*, *D*, and reaction rates at both interfaces, etc. Thus, the explanation of the influence of temperature on Sn contents using this theory requires knowledge of the quantitative variations of these factors with temperature. For the other factor, Cu-Sn source, the diffusion reaction couples can be classified into two types based on the Cu-Sn source: the type I uses bronze (with Sn content in Cu-Sn typically below 9 at.%) as Sn source, and the type II uses high-Sn Cu-Sn (e.g., Cu-25 at.% Sn). Previous measurements4,8-10,25 show that both types of samples have Sn contents above 24 at.% for the Nb3Sn layer next to the Cu-Sn source; however, they have distinct Sn content gradients as the Nb3Sn layers grow thicker: the type I generally has Sn content gradients above 3 at.%/μm<sup>25</sup>, while those of the type II are below 0.5 at.%/μm4,8-9 . Such a difference in the Sn content gradients leads to distinct grain morphologies and superconducting properties. The different *XSn* gradients in the two types of samples with different Cu-Sn sources can be easily explained by our theory above: according to Eq. (8), increased *μ<sup>s</sup>* can decrease *XSn* gradients. It may also need further investigation regarding whether Cu-Sn source can also influence diffusion rates in Nb3Sn layer (e.g., via thermodynamic factor), because greater *D* leads to greater *φII*, which helps decreasing *XSn* gradients. As to the phenomenon that different wires have similar *XSn* in the Nb3Sn layer next to the Cu-Sn source, the relation between *μSn*(Cu-Sn) and *μSn*(Nb-*XSn* Sn) is required. The Cu-Sn system has been well studied, and the phase diagram calculated by the CALPHAD technique using the thermodynamic parameters given by Ref. 26 is well consistent with the experimentally measured diagram<sup>27</sup> . Thus, the parameters from Ref. 26 are used to calculate *μSn* of Cu-Sn, which is shown in Fig. 7. On the other hand, although thermodynamic data of Nb-Sn system were proposed by Refs. 26 and 28, in these studies Nb3Sn was treated as a line compound. However, some information about *μSn* of Nb3Sn can be inferred from its relation with *μSn* of Cu-Sn: since Cu-7 at.% Sn leads to the formation of Nb-24 at.% Sn near the Cu-Sn source<sup>25</sup>, we have *μSn*(Cu-7 at.% Sn) ≥ *μSn*(Nb-24 at.% Sn). Thus, the expected approximate *μSn*(Nb-*XSn* Sn) curve in a power function is shown in Fig. 7. Besides, we can also infer that the Sn transfer rate at the Cu-Sn/Nb3Sn interface must be much faster than that at the Nb3Sn/Nb interface, so *μSn* discontinuity across the interface I is small. These explain why low-Sn Cu-Sn can lead to the formation of high-Sn Nb3Sn. It is worth mentioning that from Fig. 7, it is clear that the Taylor series for the true *a*(*X*) relation of Nb3Sn have more high-rank terms than *a*(*X*) ≈ *a<sup>l</sup>* + *κ*(*X*-*Xl*); however, our numerical calculations show that adding high-rank terms to the *a*(*X*) relation does not lead to different conclusions regarding the influences of *as*, *φ<sup>I</sup>* , *φII*, and *l* on *XII*. Thus, the above qualitative and quantitative analysis still applies.

In summary, a mathematical framework is developed to describe the compositions and layer growth rates of non-stoichiometric intermediate phases formed by diffusion reactions. The governing equations are derived and analytic solutions are given for compounds with narrow composition ranges under certain approximations. We also modify our model for compounds in which bulk diffusion is frozen, of which the bulk is not in equilibrium with the rest of the system. Based on this model, the factors that fundamentally determine the compositions of nonstoichiometric compounds formed by diffusion reactions are found and approaches to control the compositions are proposed.

## **Methods**

For the Cu-Sn/Nb3Sn/Nb diffusion reaction couples that were used for Sn content measurements (the results of which are shown in Fig. 4), the initial composition of the precursor Cu-Sn alloy was Cu-12 at.% Sn. The samples were reacted at 650 °C for 65 h, 130 h, 320 h, and 600 h. Then the samples were polished to 0.05 μm and the compositions were measured using an EDS system attached to an SEM. An accelerating voltage of 15 kV was used for the quantitative line scans. A standard Nb-25 at.% Sn bulk sample provided by Dr. Goldacker from Karlsruhe Institute of Technology was used for calibrating the Sn content of the samples. The standard deviation in the measurements was found to be about ± 0.5 at.%.

## **References**

- 1. Kim, M., Baek, S., Paik, U., Nam, S. & Byun, J. Electrical conductivity and oxygen diffusion in nonstoichiometric TiO2-x. *J. Korean Phys. Soc.* **32,** S1127-S1130 (1998).
- 2. Park, S. I., Tsuei, C. C. & Tu, K. N. Effect of oxygen deficiency on the normal and superconducting properties of YBa2Cu3O7-δ. *Phys. Rev. B* **37,** 2305-2308 (1988).
- 3. Noebe, R. D., Bowman, R. R. & Nathal, M. V. Physical and mechanical properties of the B2 compound NiAl. *Int. Mater. Rev.* **38,** 193-232 (1993).
- 4. Godeke, A. Performance boundaries in Nb3Sn superconductors (Ph.D. thesis) 89-104 (University of Twente, 2005).
- 5. Xu, X., Sumption, M. D. & Peng, X. Internally oxidized Nb3Sn strands with fine grain size and high critical current density. *Adv. Mater.* **27,** 1346-1350 (2015).
- 6. Charlesworth, J. P., Macphail, I. & Madsen, P. E. Experimental work on the niobium-tin constitution diagram and related studies. *[J. Mater. Sci.](http://link.springer.com.proxy.lib.ohio-state.edu/journal/10853)* **5,** 580-603 (1970).
- 7. Zhou, J. *et al*. Evidence that the upper critical field of Nb3Sn is independent of whether it is cubic or tetragonal. *Appl. Phys. Lett.* **99,** 122507 (2011).
- 8. Xu, X., Sumption, M. D. & Collings, E. W. Influence of heat treatment temperature and Ti doping on low-field flux jumping and stability in (Nb-Ta)3Sn strands. *Supercond. Sci. Technol.* **27**, 095009 (2014).
- 9. Peng, X. et al. Composition profiles and upper critical field measurement of internal-Sn and tube-type conductors. *IEEE Trans. Appl. Supercon.* **17,** 2668-2671 (2007).
- 10. Suenaga, M. in *Superconductor Materials Science: Metallurgy, Fabrication, and Applications* (eds Foner, S. & Schwartz, B. B.) 201-274 (Plenum, 1981).
- 11. Xu, X., Collings, E. W., Sumption, M. D., Kovacs, C. & Peng, X. The effects of Ti addition and high Cu/Sn ratio on tube type (Nb, Ta)3Sn strands, and a new type of strand designed to reduce unreacted Nb ratio. *IEEE Trans. Appl. Supercond.* **24,** 6000904 (2014).
- 12. Farrell, H. H., Gilmer, G. H. & Suenaga, M. J. Grain boundary diffusion and growth of intermetallic layers: Nb3Sn. *J. Appl. Phys.* **45,** 4025-4035 (1974).
- 13. Gosele, U. & Tu, K. N. Growth kinetics of planar binary diffusion couples: "Thin-film case" versus "Bulk cases". *J. Appl. Phys.* **53,** 3252-3260 (1982).
- 14. Debkov, V. I. in *Reaction Diffusion and Solid State Chemical Kinetics* (ed. Debkov, V. I.) 3-19 (IPMS, 2002).
- 15. Gusak, A. M. *et al*. in *Diffusion-controlled Solid State Reactions: In Alloys, Thin Films and Nanosystems* (eds Gusak, A. M. *et al*.) 99-133 (Wiley-VCH, 2010).
- 16. Tu, K. N. & Gusak, A. M. in *Kinetics in Nanoscale Materials* (eds Tu, K. N. & Gusak, A. M.) 187-193 (Wiley-VCH, 2014).
- 17. Kodentsov, A. A., Bastin, G. F. & van Loo, F. J. J. The diffusion couple technique in phase diagram determination. *J. Alloy Compd.* **320,** 207-217 (2001).
- 18. Gong, W., Zhang, L., Wei, H. & Zhou, C. Phase equilibria, diffusion growth and diffusivities in Ni-Al-Pt system using Pt/β-NiAl diffusion couples. *Prog. Nat. Sci.* **21,** 221-226 (2011).
- 19. Sandim, M. J. R. *et al*. Grain boundary segregation in a bronze-route Nb3Sn superconducting wire studied by atom probe tomography. *Supercond. Sci. Technol.* **26,** 055008 (2013).
- 20. Laurila, T., Vuorinen, V., Kumar, A. K. & Paul, A. Diffusion and growth mechanism of Nb3Sn superconductor grown by bronze technique. *Appl. Phys. Lett.* **96,** 231910 (2010).
- 21. Cui, Z. W., Gao, F. & Qu, J. M. Interface-reaction controlled diffusion in binary solids with applications to lithiation of silicon in lithium-ion batteries. *J. Mech. Phys. Solids.* **61,** 293-310 (2013).
- 22. Bochvar, A. A. *et al*. Diffusion of tin during growth of the Nb3Sn layer. *Met. Sci. Heat. Treat.* **22,** 904-907 (1980).
- 23. Müller, H. & Schneider, Th. Heat treatment of Nb3Sn conductors. *Cryogenics* **48,** 323- 330 (2008).
- 24. Besson, R., Guyot, S. & Legris, A. Atomic-scale study of diffusion in A15 Nb3Sn. *Phys. Rev. B* **75,** 054105 (2007).
- 25. Abächerli, V. *et al*. The influence of Ti doping methods on the high field performance of (Nb, Ta, Ti)3Sn multifilamentary wires using Osprey bronze. *IEEE Trans. Appl. Supercon.* **15,** 3482-3485 (2005).
- 26. Li, M., Du, Z., Guo, C. & Li, C. Thermodynamic optimization of the Cu-Sn and Cu-Nb-Sn systems. *J. Alloys Compd.* **477,** 104-117 (2009).
- 27. Hansen, M. & Anderko, R. P. in *Constitution of binary alloys* (eds Hansen, M. & Anderko, R. P.) 634 (McGraw-Hill, 1958).
- 28. Toffolon, C., Servant, C., Gachon, J. C. & Sundman, B. Reassessment of the Nb-Sn system. *J. Phase Equilib.* **23,** 134-139 (2002).

## **Acknowledgements**

The authors thank S. Dregia and J. Morral for useful discussions, and X. Peng and Hyper Tech Research Inc. for providing Nb3Sn samples for analysis. The work is funded by the US Department of Energy, Division of High Energy Physics, under an SBIR program.

## **Author Contributions**

X.X initiated this study, developed the model, and wrote the manuscript. M.D.S. supported this work, discussed the results, and reviewed the manuscript.

## **Additional information:**

**Competing Financial Interests:** The authors declare no competing financial interests.

# **Figure Captions**

FIG. 1: Schematic of an isothermal cross section of the M-A-B ternary phase diagram. The shaded region shows the equilibria among M-*X<sup>1</sup>* B, A-*X<sup>l</sup>* B, and A phases, and the dashed line shows the equilibrium between M-B and AnB phases.

FIG. 2: (a) Schematic of the M-B/AnB/A diffusion reaction system in the planar geometry, and (b) *X<sup>B</sup>* profiles of the system.

FIG. 3: (a) The calculated *XB*(*x*) profiles of the hypothetical system for the analytic and numerical solutions, with and without the assumption that *XB*(*x*) is linear, respectively. (b) The *l*(*t*) results from the numerical calculations, with the fits of *l*=*q*[√(*t*+*τ*)-√*τ*] and *l*=*bt<sup>m</sup>* .

FIG. 4: The measured *XSn*s of a Cu-Sn/Nb3Sn/Nb system after various annealing times at 650°C. The standard deviation in the Sn content measurements is around ± 0.5 at.%.

FIG. 5: A schematic of the diffusion reaction process for grain boundary diffusion.

FIG. 6: The variation of *aII* with *η* and *as*, according to Eq. (8).

FIG. 7: The variation of *μSn* with *XSn* for Cu-Sn calculated based on thermodynamic data given in Ref. 26, and a rough, speculative *μSn*(*XSn*) relation for Nb3Sn sketched according to the phase formation relation between Cu-Sn and Nb3Sn.
![](1__page_16_Figure_0.jpeg)

**Caption:** An isothermal section of the M-A-B ternary phase diagram, illustrating the equilibria among M-X^1 B, A-X^l B, and A phases, with a focus on the equilibrium between M-B and AnB phases. The shaded region and dashed lines represent specific equilibrium conditions, useful for understanding phase stability and transitions within the system. Such diagrams are critical for predicting phase relationships and tailoring material compositions to achieve desired properties in engineering applications.


Xu *et al.*

![](1__page_17_Figure_0.jpeg)

**Caption:** This figure provides a schematic of an M-B/AnB/A diffusion reaction system in a planar geometry. In part (a), the system is illustrated with two interfaces: the M-B/AnB interface (I) and the AnB/A interface (II). This setup is essential to analyze the diffusion and reaction dynamics for different layers and interfaces. Part (b) presents the XB profiles across the system, indicating the concentration gradients that arise due to the diffusion process. This visualization is critical in understanding how the chemical potentials and reaction kinetics at interfaces contribute to the observed concentration profiles. Scientifically, this figure plays a crucial role in demonstrating the complex interactions and equilibrium conditions necessary for precise control of composition in multi-phase systems, significantly impacting the development and optimization of materials for high-performance applications like superconductors .


Xu *et al.*

![](1__page_18_Figure_0.jpeg)

**Caption:** A composite figure showing (a) the calculated profiles of XB(x) for predicted analytic and numerical results with differing assumptions about linearity. A separate graph (b) depicts layer thickness, l(t), over time from numerical solutions with fitted analytical expressions for growth dynamics. This figure accurately demonstrates the relationship between predicted and actual diffusion kinetics and the implications for layer growth rates, providing critical insights into the application of diffusion models to predict material behaviors under non-steady-state conditions .


Xu *et al.*

FIG 4

![](1__page_19_Figure_1.jpeg)

**Caption:** This graph depicts the Sn concentration profile across a Cu-Sn/Nb3Sn/Nb system after various annealing periods at 650°C. The x-axis represents the distance in micrometers (μm), while the y-axis shows the Sn concentration in atomic percent (at.%). Different symbols and colors correspond to samples annealed for 65 h, 130 h, 320 h, and 600 h. Key findings indicate the evolution of Sn concentration gradients from the Cu-Sn through the Nb3Sn layer to the Nb, highlighting the changes in Sn content with increased annealing time. This figure is significant in understanding the thermal and compositional stability of the Nb3Sn layer, which is critical for its application as a superconductor.


Xu et al.

![](1__page_20_Figure_0.jpeg)

**Caption:** Schematic representation of the diffusion reaction process for grain boundary diffusion. The figure outlines the migration of atoms along grain boundaries and their interaction with inter-phase interfaces, emphasizing the importance of grain boundary diffusion in controlling the layer composition. Time intervals t1, t2, and t3 are depicted, demonstrating progressive changes in atomic distribution and revealing how grain boundaries serve as pathways for diffusion, which is otherwise inhibited in the bulk. This schematic is fundamental to interpreting how microstructural features like grain boundaries can dominate diffusion processes, thereby influencing the properties and performances of materials like Nb3Sn superconductors .


FIG 5

Xu *et al.*

![](1__page_21_Figure_0.jpeg)

**Caption:** 3D plot illustrating the variation of the ratio aII/aI with variables η and as, as derived from equation (8). The axes represent aII/aI, η, and as, highlighting the dependencies and interactions between these parameters under specified thermodynamic conditions. This graphical analysis is pivotal in visualizing the complex interdependencies that control phase equilibrium and stability in multi-component systems, providing valuable insights for tailoring material properties through precise chemical potential management .


Xu *et al.*

![](1__page_22_Figure_0.jpeg)

**Caption:** Graph presenting the chemical potential of Sn, μSn, versus its concentration, XSn, in Cu-Sn alloys at 675°C. The μSn values are plotted for both the Cu-Sn and Nb3Sn phases, using thermodynamic data and speculative relations, respectively. This helps to elucidate the phase stability and energetic considerations essential for phase formation. Understanding these potential-concentration relationships allows for the optimization of alloy performance, which is crucial for the development of stable superconducting materials .


Xu et al.