clear all
set seed 98765

// ================================================
// Simulation: Hospital-level binary indicator
// ================================================
// Purpose: Simulate hierarchical data where patients are nested in trusts
//          Estimate trust-level performance using Empirical Bayes
//          Account for patient case-mix in risk adjustment

// ========================================
// Simulate from data generating-mechanism
// ========================================

local n_trusts = 120        // Number of NHS trusts

// Fixed effects on log-odds scale
local beta0 = -4.595    // baseline risk ≈ 1%
local beta_stage2 = 0.25
local beta_stage3 = 0.50
local beta_stage4 = 0.80
local beta_comorbid1 = 0.20
local beta_comorbid2 = 0.45
local beta_comorbid3 = 0.70

// Between-trust SD on log-odds scale
local tau = 0.35

// ========================================
// Generate trust-level data
// ========================================

set obs `n_trusts'
gen trust_id = _n

// True trust random effects
gen u_j_true = rnormal(0, `tau')

// Varying sample sizes: 80-350 patients per trust
gen n_patients = floor(runiform(80, 351))
summarize n_patients
display _newline "Sample sizes: " r(min) " to " r(max) " patients per trust"

// ========================================
// Expand to patient-level
// ========================================

expand n_patients
bysort trust_id: gen patient_id = _n

// Patient-level covariates
gen stage = ceil(runiform()*4)              // Stage 1-4 (uniform)
gen comorbidity = min(floor(rpoisson(1)), 3) // 0-3+ comorbidities (Poisson)

// Generate linear predictor: β₀ + β×X + u_j
gen xb = `beta0' + ///
    `beta_stage2'*(stage==2) + `beta_stage3'*(stage==3) + `beta_stage4'*(stage==4) + ///
    `beta_comorbid1'*(comorbidity==1) + `beta_comorbid2'*(comorbidity==2) + ///
    `beta_comorbid3'*(comorbidity==3) + ///
    u_j_true

// Convert to probability	
gen p = invlogit(xb)
gen y = rbinomial(1, p)
display "Total patients: " _N


// ========================================
// Fit multilevel
// ========================================

melogit y i.stage i.comorbidity || trust_id:, nolog

// Between-trust variance from melogit random intercept SD
scalar tau2_hat = _b[/:var(_cons[trust_id])]

// ========================================
// Empirical Bayes estimation
// ========================================
// EB Formula: û_j^EB = λ_j × r̄_j
// where:
//   r̄_j = mean residual for trust j (observed - fixed effects)
//   λ_j = τ²/(τ² + σ²/n_j) = shrinkage factor
//
// Small trusts → small λ → more shrinkage toward zero
// Large trusts → large λ → less shrinkage

predict u_eb, reffects reses(u_eb_se)    // EB estimates and s.e.


// Mean linear predictor across all patients (fixed effects only)

predict prob_adj_EB, fixedonly mu


predict linear_pred, fixedonly xb
summ linear_pred
scalar mean_xb = r(mean)

// EB predicted probability for each trust centered on national event rate 
summ y
gen mean = r(mean)
scalar mean_xb = ln(mean / (1 - mean))
display mean_xb /* log odds of national event rate */

gen logit_EB  = mean_xb + u_eb
gen p_EB    = invlogit(logit_EB)
gen p_EB_lo = invlogit(logit_EB - 1.96*u_eb_se)
gen p_EB_hi = invlogit(logit_EB + 1.96*u_eb_se)


gen pct_predicted = 100*p_EB
gen pct_ci_lower  = 100*p_EB_lo
gen pct_ci_upper  = 100*p_EB_hi


// ========================================
// Expected probabilities from case-mix model
// ========================================
// Fit model without trust random effect to get expected risk under case mix only

logit y i.stage i.comorbidity
predict p_exp, pr



// ========================================
// Aggregate to trust level
// ========================================

collapse (sum) O=y E=p_exp ///
         (mean) u_eb u_eb_se ///
                logit_EB p_EB p_EB_lo p_EB_hi ///
                pct_predicted pct_ci_lower pct_ci_upper mean ///
         (count) n=y, ///
         by(trust_id)

// Calculate EB confidence intervals
gen ci_lower = u_eb - 1.96*u_eb_se
gen ci_upper = u_eb + 1.96*u_eb_se

// Flag statistical outliers for EB estimate (CI excludes zero)
gen outlier = (ci_lower > 0 | ci_upper < 0)

// Rank trusts by EB estimate
egen rank = rank(u_eb)

// Calulate risk-adjusted hospital performance before shrinkage
gen before_shrinkage_prop = O/E*mean
gen before_shrinkage = O/E*mean*100

// ========================================
// Random-effect variance and rankability
// ========================================

display _newline "=========================================="
display "Random-effect variance"
display "=========================================="
display "Random-effect variance = " %6.2f tau2_hat

// ========================================
// 1. Raw rankability
// ========================================

// Raw observed trust proportion
gen p_hat = O / n

// Avoid 0 or 1 proportions
replace p_hat = 0.001 if p_hat<=0.001
replace p_hat = 0.999 if p_hat>=0.999

// Sampling variance on logit scale
gen s2_logit_raw = 1 / (n * p_hat * (1 - p_hat))

summ s2_logit_raw, detail
scalar med_s2_logit_raw = r(p50)

// Raw rankability
scalar R_raw = tau2_hat / (tau2_hat + med_s2_logit_raw)


// ==========================================================
// 2. Rankability (with case-mix adjustment on s2 variances)
// ==========================================================

// Overall observed event rate
summ O, meanonly
scalar Otot = r(sum)

summ n, meanonly
scalar Ntot = r(sum)

scalar pbar = Otot / Ntot

// Indirectly standardized adjusted probability
gen p_adj = pbar * (O / E)

// Avoid 0 or 1 adjusted probabilities
replace p_adj = 0.001 if p_adj<=0.001
replace p_adj = 0.999 if p_adj>=0.999

// Sampling variance on logit scale
gen s2_logit_adj = 1 / (n * p_adj * (1 - p_adj))

summ s2_logit_adj, detail
scalar med_s2_logit_adj = r(p50)

// Rankability (using case-mix adjusted s2)
scalar R_adj = tau2_hat / (tau2_hat + med_s2_logit_adj)


// ========================================
// Print both rankability metrics
// ========================================

display _newline "=========================================="
display "Rankability summaries"
display "=========================================="

display "Raw observed proportions:"
display "  Median sampling variance (logit scale) = " %8.4f med_s2_logit_raw
display "  Rankability R_raw                      = " %8.4f R_raw
display "  Rankability R_raw (%)                  = " %6.2f (100*R_raw)

display _newline "Case-mix adjusted (O/E):"
display "  Overall event rate pbar                = " %8.4f pbar
display "  Median sampling variance (logit scale) = " %8.4f med_s2_logit_adj
display "  Rankability R_adj                      = " %8.4f R_adj
display "  Rankability R_adj (%)                  = " %6.2f (100*R_adj)



// =========================================================
// Shrinkage factor
// =========================================================

// The shrinkage formula:
//   B_j = tau2 / (tau2 + s2_j)
// pi^2/3 approx: s2_j = (pi^2/3)/n_j

// pi^2/3 approximation
scalar sigma2_const = 3.14159265^2 / 3
gen B_pi = tau2_hat / (tau2_hat + sigma2_const/n)
label variable B_pi "shrinkage, pi^2/3 approx"
summ B_pi, detail
scalar med_B_pi = r(p50)


display _newline "=========================================="
display "Shrinkage factor summaries (median across trusts)"
display "=========================================="
display "  pi^2/3 approx    B_pi        = " %8.4f med_B_pi
display ""
display "  Rankability R_adj            = " %8.4f R_adj

*save "..../test_data_set.dta", replace

// ============================================================
//  Empirical Bayes Probabilistic Ranking
// ============================================================

clear all
*use "..../test_data_set.dta", clear

keep trust_id u_eb u_eb_se

* parameters 
local n_sim   100000
local seed    12501
local top_pct 10 20 

quietly count
local n_providers = r(N)

* percentage thresholds 
foreach pct of local top_pct {
    local threshold_`pct' = ceil(`pct' / 100 * `n_providers')
}

* expand to n_sim iterations
expand `n_sim'
bysort trust_id: gen iteration = _n

* simulate posterior draws
set seed `seed'
gen u_eb_sim = rnormal(u_eb, u_eb_se)

* rank within each iteration (lower = better)
bysort iteration: egen rank = rank(u_eb_sim)

* flag top X% within each iteration 
foreach pct of local top_pct {
    gen top`pct'pct = (rank <= `threshold_`pct'')
}

*  collapse to trust level
collapse ///
    (first) u_eb u_eb_se          ///
    (mean)  expected_rank = rank  ///
    (sd)    sd_rank = rank        ///
    (mean)  top10pct top20pct  ///
    , by(trust_id)

sort trust_id 
*export "..../probabilistic_ranking.dta", replace 

count if top10pct>0.8
count if top20pct>0.8


// ==========================================================================
// Fig. 1: Hospital performance before and after empirical Bayes shrinkage
// ==========================================================================

clear all
*use "..../test_data_set.dta", clear

gen x_before = 1
gen x_after  = 2

// Before shrinkage: before_shrinkage


// After shrinkage: pct_predicted

 twoway ///
 (pcspike before_shrinkage x_before pct_predicted x_after, ///
     lcolor(gs10) lwidth(thin)) ///
 (scatter before_shrinkage x_before, msymbol(o) msize(vsmall) mcolor(black)) ///
 (scatter pct_predicted x_after,  msymbol(o) msize(vsmall) mcolor(black)) ///
 , ///
 xlabel(1 `" "Before empirical Bayes" "shrinkage" "' ///
        2 `" "After empirical Bayes" "shrinkage" "', labsize(medlarge)) ///
 xtitle("") ///
 ytitle("Event probability (%)", size(medlarge) margin(r+2)) ///
 title("Hospital performance" , size(medlarge)) ///
 subtitle("Risk-adjusted for patient case-mix", size(medlarge)) ///
 legend(off) ///
 xscale(range(0.8 2.2)) ///
 yscale(range(0 10)) ///
 ylabel(0 (2) 10, labsize(medium)) 
 
// ====================================================================================
// Suppl Fig. 1: Hospital performance in highest and lowest hospital volume quintiles
// ====================================================================================

// Lowest-volume quintile
 xtile volume_q = n, nq(5)
 
 twoway ///
 (pcspike before_shrinkage x_before pct_predicted x_after if volume_q==1, ///
     lcolor(gs10) lwidth(thin)) ///
 (scatter before_shrinkage x_before if volume_q==1, msymbol(o) msize(vsmall) mcolor(gs10)) ///
 (scatter pct_predicted x_after if volume_q==1, msymbol(o) msize(vsmall) mcolor(gs10)) ///
 , ///
 title("Lowest-volume quintile", size(large)) ///
 xlabel(1 `""Before empirical Bayes" "shrinkage""' 2 `""After empirical Bayes" "shrinkage""',    labsize(medlarge)) ///
 xscale(range(0.8 2.3))	///
 ytitle("Event probability (%)", size(medlarge)) ///
 yscale(range(0 10)) ///
 ylabel(0(2)10, labsize(medlarge) angle(horizontal)) ///
 legend(off) ///
 plotregion(lcolor(none)) ///
 graphregion(color(white) margin(r+12)) ///
 scheme(s1mono) ///
 name(g_q1, replace)


// Highest-volume quintile
 twoway ///
 (pcspike before_shrinkage x_before pct_predicted x_after if volume_q==5, ///
     lcolor(black) lwidth(thin)) ///
 (scatter before_shrinkage x_before if volume_q==5, msymbol(o) msize(vsmall) mcolor(black)) ///
 (scatter pct_predicted x_after if volume_q==5, msymbol(o) msize(vsmall) mcolor(black)) ///
 , ///
 title("Highest-volume quintile", size(large)) ///
 xlabel(1 `""Before empirical Bayes" "shrinkage""' 2 `""After empirical Bayes" "shrinkage""', labsize(medlarge)) ///
 xscale(range(0.8 2.3))	///
 xtitle("") ///
 ytitle("") ///
 yscale(range(0 10)) ///
 ylabel(0(2)10, labsize(medlarge) angle(horizontal)) ///	
 legend(off) ///
 plotregion(lcolor(none)) ///
 graphregion(color(white) margin(r+12)) ///
 scheme(s1mono) ///
 name(g_q5, replace)

 graph combine g_q1 g_q5, ///
    cols(2) ///
    ycommon ///
    imargin(0 0 0 0) ///
    xsize(9) ysize(5) ///
    title("Hospital performance", size(large)) ///
    subtitle("Risk-adjusted for patient case-mix", size(med)) ///
    graphregion(color(white)) ///
    scheme(s1mono)
	
// ==========================================================================
// Fig. 2: Identification of hospital outliers
// ==========================================================================
 
// Funnel plot
 
 codebook n		//max 349
 set obs 350
 
// Calculate 95% contol limits  for volumes up to 350 using formulae in appendix of Spiegelhalter paper: Spiegelhalter, D.J. (2005), Funnel plots for comparing institutional performance. Statist. Med., 24: 1185-1202. https://doi.org/10.1002/sim.1970

 gen target=mean
 replace target = target[1] if missing(target)
 
 
 keep target

 gen n = _n


// Upper 95% limit
qui gen ru = .
qui forvalue i = 0/ 350 {			 
replace ru = `i' if binomialtail(n,`i',target)<0.025 & ru==.
}
*
												
gen ru_1 = ru
replace ru_1 = ru-1 if ru>0
gen u1bin = binomialtail(n,ru,target)
gen u2bin = binomialtail(n,ru-1,target)

gen alpha = .
replace alpha = 0 if ru==ru_1
replace alpha = (u1bin - 0.025) / (u1bin-u2bin) if ru!=ru_1
												
gen ul95 = (ru-alpha)/n

drop ru ru_1 u1bin u2bin alpha
											
// Lower 95% limit

qui gen rl = .
qui forvalue i = 0/ 350 {			 
replace rl = `i' if binomial(n,`i',target)>0.025 & rl==.
}
*

gen rl_1 = rl
replace rl_1 = rl-1 if rl>0
gen a1bin = binomial(n,rl,target)
gen a2bin = binomial(n,rl-1,target)

gen alpha = .
replace alpha = 0 if rl==rl_1
replace alpha = (a1bin - 0.025) / (a1bin-a2bin) if rl!=rl_1

gen ll95 = (rl-alpha)/n

drop rl rl_1 a1bin a2bin alpha

// Merge with hospital level data
 
merge 1:m n using "..../test_data_set.dta", keep(match master using)
gen data=_merge==3
drop _merge

// Convert to percentages
foreach var in  target ll95  ul95{
    replace `var' = `var'*100
}


replace ll95 =. if ll95<=0
replace ul95 =. if ul95>16


// Create different colour based on outlier category

//Funnel plot outliers 
gen out_funnel = (before_shrinkage>ul95&data==1)

*Tag catepilllar outliers 
gen out_catepillar = (ci_lower >0&data==1)



lab define out 0 "Not an outlier" 1 "Outlier"
lab values out_catepillar out_funnel out 

tab out_funnel out_catepillar if data==1

*Change in status 
gen out_change = .

replace out_change = 1 if out_funnel==0 & out_catepillar==0
replace out_change = 2 if out_funnel==1 & out_catepillar==0
replace out_change = 3 if out_funnel==0 & out_catepillar==1
replace out_change = 4 if out_funnel==1 & out_catepillar==1

label define out_change 1 "Neither" 2 "Funnel only" 3 "EB only" 4 "Both"
label values out_change out_change
tab out_change if data==1


******************************************************
*** ADJUSTED MORTALITY ******************************************************************
*Funnel with legend
#delimit ;
twoway ///
(scatter before_shrinkage n if out_change==1, msymbol(o) mcolor(gs10)) ///
(scatter before_shrinkage n if out_change==2, msymbol(o) mcolor(blue)) ///
(scatter before_shrinkage n if out_change==3, msymbol(o) mcolor(orange)) ///
(scatter before_shrinkage n if out_change==4, msymbol(o) mcolor(red)) ///
(line target n, sort lpattern(solid) lwidth(medthin) lcolor(black)) ///
(line ll95 n, sort lpattern(shortdash_dot) lwidth(medthin) lcolor(black)) ///
(line ul95 n, sort lpattern(shortdash_dot) lwidth(medthin) lcolor(black)) ///
if n < 460, ///
ytitle("Event probability (%)", margin(medsmall) size(vlarge)) ///
ylabel(0(2)16, labsize(large) grid glcolor(gs14) glwidth(vthin)) ///
yscale(range(0 16)) ///
xtitle("Number of surgical resections", margin(medium) size(vlarge)) ///
xscale(range(0 350)) xlabel(0(50)350, labsize(large)) ///
graphregion(color(white)) ///
legend(order(5 6 2 4) ///
       lab(2 "Funnel plot outlier") ///
       lab(4 "Funnel plot & empirical Bayes outlier") ///
       lab(6 "95% control limits") ///
       lab(5 "National average") ///
       size(large) cols(2) symxsize(6) keygap(1) colgap(4) position(1) ring(0) ///
       region(fcolor(white) lcolor(black) lwidth(thin))) ///
plotregion(color(white)) ///
name(g_funnel_legend, replace);
#delimit cr

*Create rank based on EB estimate 
capture drop rank
sort pct_predicted
egen rank = rank(pct_predicted),  unique


*Catepillar with legend
twoway (line target rank, lpattern(solid) lwidth(medium) lcolor(black)) || /*
*/  (rspike pct_ci_lower pct_ci_upper rank, lwidth(medthin) lcolor(black%40)) || /*
*/  (scatter pct_predicted rank if out_change==1, msymbol(o) msize(small) mcolor(gs10)) || /*
*/  (scatter pct_predicted rank if out_change==2, msymbol(o) msize(small) mcolor(blue)) || /*
*/  (scatter pct_predicted rank if out_change==3, msymbol(o) msize(small) mcolor(orange)) || /*
*/  (scatter pct_predicted rank if out_change==4, msymbol(o) msize(small) mcolor(red)), /*
*/  xtitle("Ranking of hospitals on empirical Bayes estimates", margin(medium) size(vlarge)) /*
*/  graphregion(color(white)) /*
*/	xlabel(0(50)100, labsize(large) labcolor(white) tlcolor(white))  /*
*/ plotregion(color(white) lcolor(none)) xscale(line) yscale(line) /*
*/  yscale(range(0 16)) ylabel(0(2)16, labsize(large) grid glcolor(gs14) glwidth(vthin)) /*
*/  legend(order(1 2 4  6) /*
*/         lab(1 "National average") /*
*/         lab(2 "95% uncertainty interval") /*
*/         lab(4 "Funnel plot outlier") /*
*/         lab(6 "Funnel plot & empirical Bayes outlier") /*
*/         size(large) /*
*/       cols(2) /*
*/       symxsize(6) /*
*/       keygap(1) /*
*/       colgap(4) /*
*/       position(1) ring(0)) /*
*/  name(g_caterpillar_legend, replace) 


	
	graph combine g_funnel_legend g_caterpillar_legend, ///
    cols(2) ///
	ycommon ///
    imargin(0 0 0 0) ///
    xsize(11) ysize(4) ///
    graphregion(color(white)) ///
	title("Hospital performance", size(vlarge) margin(b=1)) ///
	subtitle("Risk-adjusted for patient case-mix", size(large) margin(t=0 b=1))
	


cd "W:\Bowel Projects\BOWELaudit\Adil_PhD\Emperical Bayes Estimation\figures"