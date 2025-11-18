# M.Sc. Assignment: Developing an LSTM System for Frequency Extraction from a Mixed Signal

**Dr. Yoram Segal**  
© All Rights Reserved  
November 2025

---

## Table of Contents

1. **Background and Goal**
   - 1.1 Problem Statement
   - 1.2 The Principle
   - 1.3 Usage Example

2. **Dataset Creation**
   - 2.1 General Parameters
   - 2.2 Noisy Signal Creation (S - Mixed and Noisy)
   - 2.3 Ground Truth Targets Creation
   - 2.4 Differences Between Train vs. Test Sets

3. **Training Dataset Structure**

4. **Pedagogical Highlights: Internal State and Sequence Length**
   - 4.1 The Internal State of LSTM
   - 4.2 Critical Implementation Requirements (L=1)
   - 4.3 Alternative and Justification

5. **Performance Evaluation**
   - 5.1 Success Metrics
   - 5.2 Recommended Graphs

6. **Assignment Summary**

7. **References**

---

## 1. Background and Goal

### 1.1 Problem Statement

Given a mixed noisy signal **S** - composed of **4 ideal sine waves** at different frequencies, where the noise changes randomly at each sample.

**The goal** is to develop a **Long Short-Term Memory (LSTM) network** capable of extracting each pure frequency separately from the mixed signal, while decisively isolating it from the noise.

### 1.2 The Principle

The system is required to perform **Conditional Regression**:

**Table 1: System Input/Output Structure**

| Input | Description | Target Output / Required |
|-------|-------------|--------------------------|
| S[t] | Sample from the mixed noisy signal | Target_i[t] (Pure sine wave, no noise) |
| C | Selection vector (One-Hot) for frequency selection | — |

### 1.3 Usage Example

If the selection vector is **C = [0, 1, 0, 0]**, we want to extract the pure frequency **f₂**:

```
Input: S[t] + C → LSTM → Output: Pure Sinus₂[t]
```

**Examples:**
1. S[0] (noisy) + C → LSTM → Pure Sinus₂[0] (clean)
2. S[1] (noisy) + C → LSTM → Pure Sinus₂[1] (clean)

---

## 2. Dataset Creation

### 2.1 General Parameters

- **Frequencies:** f₁ = 1Hz; f₂ = 3Hz; f₃ = 5Hz; f₄ = 7Hz
- **Time Domain:** 0-10 seconds
- **Sampling Rate (Fs):** 1000 Hz
- **Total Samples:** 10,000

### 2.2 Noisy Signal Creation (S - Mixed and Noisy)

**Critical Point:** The amplitude **Aᵢ(t)** and phase **φᵢ(t)** must change randomly at each sample **t**.

**1. Noisy sine wave at sample t:**

```
Amplitude: Aᵢ(t) ~ Uniform(0.8, 1.2)
Phase: φᵢ(t) ~ Uniform(0, 2π)

Sinusⁿᵒⁱˢʸ_i(t) = Aᵢ(t) · sin(2π·fᵢ·t + φᵢ(t))
```

**2. Normalized sum (system input):**

```
S(t) = (1/4) · Σ(i=1 to 4) Sinusⁿᵒⁱˢʸ_i(t)
```

### 2.3 Ground Truth Targets Creation (No Noise)

The pure target for each frequency **i** is:

```
Targetᵢ(t) = sin(2π·fᵢ·t)
```

### 2.4 Differences Between Train vs. Test Sets

- **Training Set:** Uses random seed **#1**
- **Test Set:** Uses random seed **#2**
  - ⚠️ **Important:** Same frequencies, completely different noise!

---

## 3. Training Dataset Structure

**Total rows in training set:** 40,000 (10,000 samples × 4 frequencies)

**Data Format:** Each row represents a single sample. The network input is a vector of size **5**:

```
[S[t], C₁, C₂, C₃, C₄]
```

**Table 2: Data Format Example (Training Set)**

| Row | t(sec) | S[t] (Noisy Input) | C (Selection) | Target (Pure Output) |
|-----|--------|-------------------|---------------|---------------------|
| 1 | 0.000 | 0.8124 | [1,0,0,0] | 0.0000 |
| ... | ... | ... | [1,0,0,0] | ... |
| 10001 | 0.000 | 0.8124 | [0,1,0,0] | 0.0000 |
| 10002 | 0.001 | 0.7932 | [0,1,0,0] | 0.0188 |
| ... | ... | ... | ... | ... |
| 40000 | 9.999 | 0.6543 | [0,0,0,1] | 0.0440 |

---

## 4. Pedagogical Highlights: Internal State and Sequence Length

Within the framework of this assignment, we define the **Sequence Length** as **L = 1** by default (single sample mode).

### 4.1 The Internal State of LSTM

The internal state of LSTM consists of:
- **Hidden State (hₜ)**
- **Cell State (cₜ)**

This state enables the network to learn **temporal dependency** between samples.

### 4.2 Critical Implementation Requirements (L = 1)

When working with **L = 1**, we must **manually manage the internal state** during training so the network can utilize its memory:

✅ **The internal state (hₜ and cₜ) must NOT be reset between consecutive samples.**

**Table 3: State Management Comparison in LSTM Model**

| Scenario | Required Action | Essential Explanation |
|----------|----------------|----------------------|
| Regular LSTM (L > 1 sequence) | Reset state at each batch. No batch connection. | The network assumes no sequential relationship between sequences. |
| **This assignment (L = 1)** | **Preserve and pass state to next step as input.** | The network **CAN** learn temporal patterns through state management. |

### 4.3 Alternative and Justification

**Recommendation:** Training with longer sequences **(L > 1)** has pedagogical and computational efficiency advantages due to the full power of LSTM.

- Students are welcome to work with **L ≠ 1** (Sliding Window of size **L = 10** or **L = 50**) instead of **L = 1**.

**Justification Requirement:** If choosing **L ≠ 1**, the work must include:
  - Detailed justification for the choice
  - How it contributes to temporal learning advantage of LSTM
  - How the output is handled

---

## 5. Performance Evaluation

### 5.1 Success Metrics

**1. MSE on Training Set (with noise seed #1):**

```
MSE_train = (1/40000) · Σ(j=1 to 40000) (LSTM(S_train[t], C) - Target[t])²
```

**2. MSE on Test Set (with noise seed #2):**

```
MSE_test = (1/40000) · Σ(j=1 to 40000) (LSTM(S_test[t], C) - Target[t])²
```

**3. Generalization Check:**

If **MSE_test ≈ MSE_train**, then the system generalizes well! ✓

### 5.2 Recommended Graphs

Display a visual comparison on the **test set (with noise seed #2)**, such as:

**Graph 1:** Comparison for a selected frequency (e.g., f₂):
- Display three components on the same graph:
  1. **Target₂** (pure, line)
  2. **LSTM Output** (dots)
  3. **S** (mixed noisy, as background, chaotic)

**Graph 2:** Four sub-graphs showing all 4 extracted frequencies:
- Each sub-graph displays the extraction for one frequency **fᵢ** separately.

---

## 6. Assignment Summary

**Students are required to:**

✅ **Generate Data:** Create 2 datasets (training and testing) with noise that changes at each sample.

✅ **Build Model:** Construct an LSTM network that receives `[S[t], C]` and returns the pure sample `Targetᵢ[t]`.

✅ **State Management:** Ensure the internal state is preserved between consecutive samples (Sequence Length L = 1) for temporal learning.

✅ **Evaluation:** Evaluate performance using MSE and graphs, and analyze the system's generalization to new noise.

---

## 🎯 Key to Success

**The key to success is proper internal state management** and learning the periodic frequency structure of **Targetᵢ** while being immune to the random noise!

---

## 7. References

(In Hebrew - Original Document)

---

**© Dr. Yoram Segal - All Rights Reserved**

