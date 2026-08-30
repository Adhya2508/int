# Task 5: Scenario and Stress Simulation Report

## 1. Scope and Modeling Limitation Statement
> [!IMPORTANT]
> The scenario projections in this report focus **exclusively on prepayment behaviors** (`next_12m_prepayment_flag`). The credit vintage data contains **zero positive delinquency or default cases** during the observation window. Therefore, meaningful delinquency/default stress testing modeling is not feasible on this dataset and has been omitted.

## 2. Simulation Methodology
We simulated prepayment trajectories for the test cohort (Dec 2025) over a 12-month horizon (Jan 2026 – Dec 2026) under three macro scenarios:
1. **Base Scenario (Actual Model Projection)**: Borrower characteristics and interest rate spreads remain at current levels.
2. **Adverse Credit Scenario (Actual Model Projection)**: Simulates a severe economic downturn where borrower credit scores drop by 50 points and debt-to-income (DTI) ratios increase by 10 points (credit-locking the cohort).
3. **High Prepayment Scenario (Scenario Approximation)**: Simulates a drop in market interest rates by 2.0% (represented by increasing the interest rate spread by +200 bps), creating a strong refinance incentive.

## 3. Cohort Projections
* **Projection Chart**: Saved as [scenario_projections.png](file:///e:/intain/data_final/outputs/scenario_projections.png)
* **12-Month Cumulative Prepayment Rates**:
  - **Base Scenario**: 5.95% of the cohort prepays.
  - **Adverse Credit Scenario**: 4.57% of the cohort prepays (prepayments drop due to credit constraints).
  - **High Prepayment Scenario**: 15.14% of the cohort prepays (significant increase due to spread drops).

## 4. Segment-Level Impact Breakdown (Base Scenario)
* **By Credit Band**:
  - Subprime (<660): 7.00% avg prepay probability
  - Near-Prime (660-720): 5.18% avg prepay probability
  - Prime (>720): 6.27% avg prepay probability
* **By Vintage**:
  - 12024.0: 6.54% avg prepay probability
  - 12025.0: 6.87% avg prepay probability
  - 22024.0: 9.99% avg prepay probability
  - 32024.0: 13.07% avg prepay probability
  - 42024.0: 12.43% avg prepay probability
  - 52024.0: 11.75% avg prepay probability
  - 62023.0: 6.93% avg prepay probability
  - 62024.0: 8.54% avg prepay probability
  - 72023.0: 19.32% avg prepay probability
  - 72024.0: 6.39% avg prepay probability
  - 82023.0: 10.75% avg prepay probability
  - 82024.0: 4.36% avg prepay probability
  - 92023.0: 11.96% avg prepay probability
  - 92024.0: 3.80% avg prepay probability
  - 102023.0: 3.71% avg prepay probability
  - 102024.0: 3.88% avg prepay probability
  - 112023.0: 11.47% avg prepay probability
  - 112024.0: 5.00% avg prepay probability
  - 122023.0: 14.09% avg prepay probability
  - 122024.0: 6.45% avg prepay probability
* **By Top 5 Property States**:
  - RI: 10.00% avg prepay probability
  - CA: 9.37% avg prepay probability
  - MA: 8.75% avg prepay probability
  - WA: 8.66% avg prepay probability
  - MT: 8.37% avg prepay probability

## 5. Top Drivers behind Scenario Movement
1. **Refinance Incentive (Interest Rate Spread)**: The spread between the borrower's rate and market rates is the strongest driver of prepayment. Lowering market rates (High Prepay) triggers a large wave of refinancing.
2. **Credit constraints**: Dropping credit scores lock borrowers out of refinance channels, reducing prepayments in the adverse scenario.
