# arXiv:2302.07825

**Paper ID:** 2c0c2f2a9557364b27e1f47f52226cd5

**PDF Path:** apl/Superconductors/arXiv:2302.07825.pdf

**Processing Status:** complete

**Captions Added:** 7

**Generated:** 2025-06-24T13:44:27.751367

---

# Verification Testing of MQXFA Nb3Sn Wires Procured Under LARP

Jeremy Levitan, Jun Lu, Vito Lombardo, and Lance Cooley *Senior Member, IEEE* 

*Abstract***— The High-Luminosity LHC Accelerator Upgrade Project (AUP) in the U.S. will construct quadrupole magnets to be delivered to CERN. An initial 3 tons, over 600 km total length, of conductor was procured under the LHC Accelerator R&D Program (LARP) for this project. Programs for quality control (QC) at the supplier and quality verification (QV) at the laboratories were solidified into components of the overall quality plan for strand procurement under AUP. Measurements of the critical current (***Ic***) and residual resistance ratio (RRR), and related probes and techniques, are central to the quality plan. Described below is the verification testing that has taken place at the National High Magnetic Field Laboratory (NHMFL). Testing challenges are presented by the high sensitivity of these wires. In addition, new RRR test software was developed to accommodate challenges presented by meeting international standards with existing configurations of test strands.** 

*Index Terms***—critical current, Large Hadron Collider, Nb3Sn, residual resistance ratio** 

## I. INTRODUCTION

HE High-Luminosity LHC Accelerator Upgrade Project (AUP) in the U.S. will construct quadrupole magnets to be delivered to CERN [1]. An initial 3 tons, over 600 km total length, of conductor for this project was procured under the LHC Accelerator R&D Program (LARP). These wires must meet challenging critical current (*Ic*) and residual resistivity ratio (RRR) specifications, which are shown in Table I. In addition to quality control tests performed by the manufacturer, verification tests are performed by the National High Magnetic Field Laboratory (NHMFL). The process and results of these tests are presented and compared to the manufacturer's data below. **T**

## II. EXPERIMENTAL

## *A. Heat Treatment*

 Strands approximately 1 meter long were cut from verification sample lengths sent from Fermilab for the *Ic* tests. The

This work is financially supported by DOE through Fermi National Accelerator Laboratory. This work was performed at the National High Magnetic Field Laboratory, which is supported by National Science Foundation Cooperative Agreement No. DMR-1644779 and the State of Florida.

J. Levitan and J. Lu are with Magnet Science and Technology, National High Magnetic Field Laboratory, Tallahassee, FL 32310 USA, (Corresponding author: Jun Lu, junlu@magnet.fsu.edu.) L. D. Cooley is with the Applied Superconductivity Center, National High Magnetic Field Laboratory, Tallahassee, FL 32310 USA. Vito Lombardo is with Technical Division of the Fermi National Accelerator Laboratory, Batavia, IL 60510 USA.

strand samples were wound with 1 kg of tension on Ti6Al4V barrels used for heat treating and testing ITER Nb3Sn wires, with no modification of the barrels or mounting procedures from the description in [2]. Straight wires approximately 25 cm long were inserted into 1 mm interior diameter quartz tubes to create RRR samples. The ends of the Nb3Sn wire for both the *Ic* and RRR samples were welded shut using a spot welder to prevent tin leak during the heat treatment [3], [4]. Groups of samples were wrapped in stainless steel foil before being placed in the center of the furnaces.

TABLE I PARAMETERS FROM THE LARP Nb3Sn SPECIFICATIONS [1]

| Description          | Specification |
|----------------------|---------------|
| Ic 4.22 K, 12 T (A)  | ≥ 632         |
| Ic 4.22 K, 15 T (A)  | ≥ 331         |
| n-value 4.22 K, 15 T | > 30          |
| RRR                  | ≥ 150         |
| Diameter (mm)        | 0.85          |
| Sub-elements         | 108/127       |
| Nb:Sn ratio          | 3.6:1         |

Both the *Ic* and RRR samples were heat treated in UHP argon (Ar) gas using following schedule,

|               |  | Ramp at 25 °C/h to 210 °C, soak for 48 h at 210 °C, |
|---------------|--|-----------------------------------------------------|
|               |  | Ramp at 50 °C/h to 400 °C, soak for 48 h at 400 °C, |
|               |  | Ramp at 50 °C/h to 665 °C, soak for 75 h at 665 °C, |
| Furnace cool. |  |                                                     |

The cross-section of a heat treated wire is shown in Fig. 1. It should be noted that the heat treatment at the supplier is 50 hours at 665 °C, 25 hours shorter than ours.

![](_page_0_Picture_18.jpeg)

**Caption:** Figure 1 shows the cross-section of an MQXFA Nb3Sn wire after heat-treatment, as part of the verification testing under the LHC Accelerator R&D Program (LARP). The image, likely obtained via electron microscopy, reveals the microstructural characteristics of the superconducting wire. The arrangement and quality of filaments in the cross-section are critical for achieving high critical current (Ic) and preventing instability or defects in superconducting applications .


Fig. 1. Cross-section of MQXFA Nb3Sn after heat-treatment.

## *B. Verification Tests*

## *1) Critical Current (Ic)*

After the heat treatment, each wire was detached at one end and gently loosened from the ITER barrel. It was then tightened by applying downward pressure with a finger, which moved along the helical groove of the ITER barrel. The wire was reattached and soldered to the copper rings. This process removed approximately 3 mm of slack from the wire. The barrel was then mounted on a test probe, with pressure contacts to the copper terminals, as shown in Fig. 2. Two voltage taps with lengths of 25 and 50 cm were soldered across the center turns. The 50 cm length was used for the measurement. *Ic* was determined by measuring voltage in a liquid helium bath as current was increased by 1 A steps in a superconducting magnet. *Ic* was recorded at 12, 13, 14, and 15 T with a criterion of 10 μV/m. The current-voltage curves were also used to determine the transition index *n* by a power law fit for data between 10 μV/m and 100 μV/m.

![](_page_1_Picture_3.jpeg)

**Caption:** Figure 2 depicts a prepared Ic sample mounted on a test probe used in NHMFL's verification testing. The setup is crucial for measurements of critical current under superconducting conditions (12 to 15 T). Accurate alignment and attachment to the copper terminals ensure reliable data acquisition on superconducting properties .


Fig. 2. A prepared *Ic* sample attached to a probe for testing.

## *2) Residual Resistance Ratio (RRR)*

The set up used for RRR measurements was the same as previously reported [3], [4]. Unlike for the ITER project, which defined RRR as the ratio of resistances at 273 K and 20 K, the LARP specification called for the use of the IEC standard [5], which is defined as the ratio of resistance at room temperature (295 K) to that determined by the intersection of two straight lines extrapolated from the resistance versus temperature curve, as shown on Fig. 3. A LabVIEW program was developed to measure resistance versus temperature during a slow warm-up from 4.2 K. The temperature was measured by a Cernox temperature sensor, and the typical warm-up rate was 10 mK per second. The resistance was obtained by measuring voltage when -1 and +1 A current was applied using a Keithley 2400 SourceMeter. The resistance vs. temperature curve was then analyzed by a second LabVIEW program, which fitted the two sections near the normal-tosuperconducting transition of the *R* vs. *T* curve with straight lines as shown in Fig. 3. The low temperature resistance used to determine RRR was then defined by the intercept of the two straight lines (point A in Fig. 3) [5]. Since the RRR measured by the manufacturer was defined as R295K/R20K, resistance data at 20 K was also obtained to calculate R295K/R20K and compare with the manufacturer's data.

![](_page_1_Figure_7.jpeg)

**Caption:** Figure 3 presents a resistance versus temperature curve used to determine the Residual Resistance Ratio (RRR) of the LHC Nb3Sn wires. RRR is crucial for evaluating the wire's performance, as it provides insights into purity and defect levels. The resistance value at point A, where the two extrapolated lines intersect, is used alongside the resistance at 295 K for this calculation.


Fig. 3. A resistance versus temperature curve. RRR is obtained from the ratio of resistance at 295 K (not shown) and point (A) on the curve.

## III. RESULTS

## *A. Critical Current (Ic)*

 The final step of the supplier's heat treatment was 665 °C for 50 hours instead of 75 hours as used by the NHMFL. An investigation was performed by Fermilab and the vendor to determine the impact of the extended heat treatment. A comparison was made of the vendor data, using the average of the 187 samples measured for *Ic* at 12, 13, 14, and 15 T. 50 samples were used for 665°C/75h and 137 were used for 665°C/50h. This resulted in a 2.9% difference for *Ic* data obtained at 12 T, and a 4.9% difference for *Ic* data obtained at 15 T. Results of the investigation are shown in Table II.

TABLE II HEAT TREAMENT COMPARISON RESULTS

|                                  | Ic at 12 T | Ic at 15 T | RRR   |
|----------------------------------|------------|------------|-------|
| Vendor 665°C/75h average         | 713 A      | 388 A      | 300   |
| Vendor 665°C/50h average         | 693 A      | 370 A      | 301   |
| Vendor 665°C/75h : 665°C/50h     | 102.9%     | 104.9%     | 99.7% |
| Lab 665°C/75h : Vendor 665°C/75h | 100.3%     | 102.1%     | N/A   |
| Lab 665°C/75h : Vendor 665°C/50h | 103.1%     | 107.1%     | N/A   |

The average and standard deviation of our results are summarized in Table III. Fig. 4 shows our *Ic* results at 12 and 15 T after they have been corrected for the difference in heat treatments between the NHMFL and the supplier. The corresponding *n*-values at 15 T are shown in Fig. 5. The results measured by the supplier are also plotted in each figure for comparison.

TABLE III SUMMARY OF RESULTS

|            | NHMFL |      | Supplier |      |  |  |  |
|------------|-------|------|----------|------|--|--|--|
|            | Mean  | σ    | Mean     | σ    |  |  |  |
| Ic 12T (A) | 712.7 | 16.5 | 694.9    | 14.3 |  |  |  |
| Ic 13T (A) | 593.1 | 13.2 | 572.8    | 11.5 |  |  |  |
| Ic 14T (A) | 488.1 | 9.8  | 465.6    | 9.7  |  |  |  |
| Ic 15T (A) | 395.5 | 8.5  | 371.2    | 8.5  |  |  |  |
| n 12T      | 43.8  | 2.8  | 44.3     | 4.4  |  |  |  |
| n 15T      | 38.4  | 2.6  | 38.3     | 5.8  |  |  |  |
| RRR-IEC    | 375.1 | 69.8 | N/A      | N/A  |  |  |  |
| R295K/R20K | 323.7 | 55.0 | 295.8    | 46.2 |  |  |  |

![](_page_2_Figure_0.jpeg)

**Caption:** Figure 4 displays Ic data comparison for samples at 12 T and 15 T, corrected for differences in heat treatments. This comparison demonstrates the influence extended heat treatments have on the superconducting properties of the samples, with NHMFL treatments yielding higher Ic values indicative of enhanced Nb3Sn reaction or diffusion barrier improvements .


Fig. 4. *Ic* data comparison between the corrected NHMFL and supplier data at (a) 12 T and (b) 15 T. The NHMFL results have been adjusted by 2.8% and 4.7% for 12 and 15 T respectively to correct for the difference in heat treatments.

![](_page_2_Figure_2.jpeg)

**Caption:** Figure 5 plots the n-values at 15 T for samples prepared by both the NHMFL and the supplier. The n-value, which reflects the sharpness of the transition from the superconducting to the normal state, is crucial for understanding the material's performance in magnetic fields. The figure indicates consistent n-values between the two preparation treatments, thereby verifying the quality and reliability of NHMFL's extended heat treatment method compared to the supplier's process.


Fig. 5. *n*-value at 15 T. Comparison between the NHFML and supplier.

## *B. Residual Resistance Ratio (RRR)*

Fig. 6 shows RRR defined by the IEC standard [5] and R295K/R20K along with the supplier's R295K/R20K data for comparison. The average and standard deviations of each set of data are presented in Table III. From the results, the average RRR-IEC is about 16% higher than R295K/R20K due to the fact that the IEC definition calls for resistance near the normal-tosuperconducting transition, which is at about 17 K for these samples, about 3 K lower than the 20 K method.

![](_page_2_Figure_7.jpeg)

**Caption:** Figure 6 shows a comparison of RRR defined by the IEC standard and R295K/R20K values obtained by NHMFL and the supplier. Differences in RRR values provide insights into variations in heat treatment processes, as well as the wire's quality and consistency. The NHMFL results consistently show higher values compared to the supplier, indicating a more thorough reaction during heat treatment .


Fig. 6. RRR-IEC and R295K/R20K data measured by the NHMFL and R295K/R20K data measured by the supplier.

## IV. DISCUSSION

## *A. Critical Current (Ic)*

It was observed that after the heat treatment, the wire was loose on the ITER barrel due to the thermal contraction mismatch between the wire and the barrel during the cool down to room temperature. This looseness under electromagnetic force during the testing made the sample prone to premature quenches. Therefore, it is reasonable to expect that tightening the wire to the ITER barrel after heat treatment would reduce the probability of premature sample quench. Our previous experience on ITER internal-tin Nb3Sn wire, however, indicated that tightening ITER Nb3Sn wire to the barrel causes more frequent premature sample quenches.

In order to study the effect of the wire tightening process, we experimented with and without the tightening process and the results are shown in Fig. 7. The label 'tightened hot' means that after the 'not tightened' sample was soldered to the copper terminal rings and tested, an attempt was made to tighten the sample on the same barrel while the solder was melted (hot). The label 'tightened' refers to new samples that were tightened before soldered to the copper rings. These were created after the success of the 'tightened hot' samples. From Fig. 7, it is evident that the tightening process produces significantly more consistent and higher *Ic*.

Sample 9 was the only sample that did not increase between 'not tightened' and 'tightened hot'. The sample quenched prematurely at 601 A which was most likely caused from damage during the tightening process. The risk of this damage was reduced greatly by tightening the wire before soldering to the barrel in the 'tightened' samples.

![](_page_3_Figure_0.jpeg)

**Caption:** Figure 7 illustrates the impact of the tightening process on Ic at 12 T for samples not tightened, tightened at soldering temperature, and fully tightened samples. The tightening of Nb3Sn wires to an ITER barrel shows higher and more consistent Ic values. The tightening reduces premature quenching incidents due to mechanical damage and helps achieve more reliable superconducting properties .


Fig. 7. *Ic* at 12 T for samples not tightened, tightened at soldering temperature after initial soldering to Cu terminals, and tightened before soldering.

There is an obvious difference between the LARP wire and ITER wire in response to the tightening process. While ITER wires are very sensitive to damage by the tightening process, LARP wires can tolerate some handling (slightly bending) such as the tightening process. It is speculated that the ITER wire design of a single diffusion barrier might be responsible for its sensitivity to the tightening process (slightly bending). LARP wires with distributed diffusion barriers for each subelement makes the diameter of the non-copper region much smaller than that of ITER wires. With soft copper between the sub-elements, the LARP samples are more forgiving to some post-heat-treatment handling. The higher *Ic* values from the tightened samples can be explained by strain sensitivity of Nb3Sn where compressive intrinsic thermal strain is reduced as a result of the tightening process.

Our systematically higher *Ic* than that of the supplier can be explained by our 25 hour longer heat treatment time during the 665 °C stage. This could lead to more thorough Nb and Sn reaction or partial reaction of Nb diffusion barriers to form Nb3Sn and, therefore, to higher *Ic* [6].

## *B. Residual Resistance Ratio (RRR)*

Despite the fact that our heat treatment time is 25 hours longer than that of the manufacturer, which could lead to lower RRR as seen in Table II, the R295K/R20K by NHMFL is slightly higher than that of the supplier (Fig. 6). Since a previous benchmarking by the two labs using the same set RRR sample showed very good agreement, this slight but systematic difference might be attributed to slight differences in the heat treatment environment and temperature control between the two labs.

## V. CONCLUSION

The *Ic* and RRR test results for the verification of the LARP wires are presented. A comparison between the supplier and the NHMFL is shown. Our results are in reasonable agreement with the supplier's quality control data. The reasons for the small but systematically higher *Ic* and RRR values by NHMFL compared with those of the manufacturer are discussed.

## ACKNOWLEDGMENT

We would like to acknowledge Bruker OST for development and fabrication of the Nb3Sn wire for LARP program and for providing the quality control measurement data.

## REFERENCES

- [1] L. D. Cooley, A. K Ghosh, D. R. Dietderich, and I. Pong, "Conductor Specification and Validation for High-Luminosity LHC Quadrupole Magnets" *IEEE Trans. Appl. Supercond.*, VOL. 27, NO. 4, 2017 pp. 6000505
- [2] I. Pong et al., "Worldwide benchmarking of ITER internal Tin Nb3Sn and NbTi strands test facilities," IEEE Trans. Appl. Supercond., vol. 22, 2012, pp. 4802606.
- [3] D. McGuire, et al., "Verification Testing of ITER Nb3Sn strand at the NHMFL" *IEEE Trans. Appl. Supercond.* Vol. 25, No. 3, 2015 pp. 9500304
- [4] J. Lu, S. Hill, D. McGuire, K. Dellinger, A. Nijhuis, W. A. J. Wessel, H. J. G. Krooshoop, K. Chan, and N. Martovetsky, "Comparative Measurements of ITER Nb3Sn Strands Between Two Laboratories" *IEEE Trans. Appl. Supercond.*, VOL. 27, NO. 5, 2017 pp. 6001004
- [5] International standard "Residual resistance ratio measurement Residual resistance ratio of Nb3Sn composite superconductors" CEI/IEC 61788- 11:2003
- [6] Jeffrey A. Parrell, Youzhu Zhang, Michael B. Field, and Seung Hong, "Development of Internal Tin Nb3Sn Conductor for Fusion and Particle Accelerator Applications" *IEEE Trans. Appl. Supercond.*, VOL. 17, NO. 2, 2007 pp. 2560.