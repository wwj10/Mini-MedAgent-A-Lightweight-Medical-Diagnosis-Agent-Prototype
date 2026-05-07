# Mini-MedAgent-A-Lightweight-Medical-Diagnosis-Agent-Prototype

## 1. Project Overview

Mini-MedAgent is a lightweight prototype for studying medical LLM agents.  
The system simulates a multi-turn diagnostic environment where a DoctorAgent cannot directly access the complete patient case. Instead, it must interact with a PatientAgent and an ExamAgent to collect information, order tests, and submit a final diagnosis.

This project is inspired by medical agent research such as interactive diagnosis, process-level feedback, and experience-based agent evolution.

## 2. Core Idea

The project implements a simplified Diagnose–Grade–Evolve loop:

1. **Diagnose**  
   DoctorAgent performs multi-turn clinical reasoning through:
   - AskQuestion
   - OrderTest
   - SubmitDiagnosis

2. **Grade**  
   JudgeAgent evaluates the final diagnosis.  
   ProcessGrader evaluates each intermediate action.

3. **Evolve**  
   MemoryManager converts process-level feedback into reusable experience memory, which can be retrieved in future cases.

## 3. System Components

### DoctorAgent

Calls an LLM through an OpenAI-compatible API, currently tested with DeepSeek.  
It decides the next action based on chief complaint, previous interactions, test results, cost, and retrieved memory.

### PatientAgent

Simulates a patient by answering questions based on hidden case information.

### ExamAgent

Simulates medical tests and returns test results with associated costs.

### JudgeAgent

Evaluates whether the final diagnosis is medically correct or semantically equivalent to the gold diagnosis.

### ProcessGrader

Evaluates each diagnostic action with labels:

- HIGH_YIELD
- REASONABLE
- LOW_YIELD
- INEFFICIENT
- CRITICAL_ERROR

### MemoryManager

Stores process-level feedback as reusable memory and supports multiple retrieval strategies:

- Keyword retrieval
- TF-IDF retrieval
- Embedding retrieval with caching

## 4. Current Dataset

The current version uses 5 toy medical cases:

1. Community-acquired pneumonia
2. Acute myocardial infarction
3. Type 2 diabetes mellitus
4. Acute appendicitis
5. Pulmonary embolism

These cases are designed for prototyping agent workflows rather than clinical validation.

## 5. Main Experiments

The project currently includes the following experiments:

### 5.1 No Memory vs Memory

Compares agent performance with and without experience memory.

### 5.2 Memory Type Ablation

Compares:

- No Memory
- Positive Memory only
- Negative Memory only
- All Memory

### 5.3 Retrieval Ablation

Compares:

- No Memory
- Keyword Memory
- Embedding Memory

### 5.4 Embedding Configuration Ablation

Compares different values of:

- memory_top_k
- memory_similarity_threshold

### 5.5 Memory Compression Ablation

Compares:

- Full Memory
- Balanced Memory
- Compact Memory

## 6. Key Findings

Preliminary results suggest:

1. Memory can reduce low-yield actions and improve diagnostic efficiency.
2. Embedding-based memory retrieval performs better than keyword retrieval in reducing cost and low-value actions.
3. Over-compressing memory reduces token usage but may harm diagnostic quality.
4. Full memory is currently the safest option for medical diagnosis, while balanced memory may be useful for cost-sensitive settings.
5. Repeated experiments show that single-run results can be unstable, so repeated evaluation is necessary.

## 7. Recommended Current Configuration

Based on repeated experiments, the current recommended setting is:

```python
memory_retrieval_mode = "embedding"
memory_top_k = 5
memory_similarity_threshold = 0.35
memory_prompt_style = "full"
