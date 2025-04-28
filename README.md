# CSF364 - Assignment 1

**By:**
* **Rishabh Goyal** (2021B1A72320H)
* **Yash Gupta** (2021B4A72699H)
* **Soham Mangle** (2021B3A71068H)
* **Vani Jain** (2021B1A73126H)

Each team member contributed significantly to different aspects of the project, ensuring a well-rounded and comprehensive approach.

Webpage can be accessed on: https://rish12311.github.io/CSF364_Assignment1/webpage.html

This repository contains implementations of two algorithms for finding the **Maximum h-Clique Density**:
* **Algorithm 1**
* **Algorithm 2**

These implementations are written in **C++** and optimized for performance.

## 📌 Setup Requirements

To run these algorithms, ensure you have:
* A **C++ compiler** with C++17 support
* A **Unix-like environment** (Linux, macOS, WSL, etc.)

## 🚀 Compilation

Each algorithm is implemented in a separate C++ file. Compile them as follows:

**Algorithm 1**
```
g++ -std=c++17 -O3 -o algo1 algo1.cpp
```

**Algorithm 2**
```
g++ -std=c++17 -O3 -o algo2 algo2.cpp
```

## ▶️ Execution

Run each algorithm with the input file and h-value:

**Algorithm 1**
```
./algo1 <filename>.mtx <h_value>
```

**Algorithm 2**
```
./algo2 <filename>.mtx <h_value>
```

Replace `<filename>.mtx` with the path to your matrix format data file and `<h_value>` with the desired h value for calculating h-clique density.

## 📖 Algorithm Descriptions

**1️⃣ Algorithm 1**
An algorithm for finding the **maximum h-clique density** in a graph using optimized graph processing techniques.

**2️⃣ Algorithm 2**
An alternative approach to computing **maximum h-clique density** with different performance characteristics.

## 📂 Input Format

The input file should be in Matrix Market (.mtx) format, containing edges of the graph where each line represents an edge between two vertices.

## 📊 Output

Each algorithm outputs the following:
* **Maximum h-clique density** of the graph
* **Execution time**
* Additional metrics relevant to the h-clique density calculation

## Contributions

| **Name** | **Responsibilities** |
|----------|----------------------|
| **Rishabh Goyal** | Website development & results analysis |
| **Yash Gupta** | Report writing & implementation of Algorithm 2 |
| **Vani Jain** | Performance optimization & testing |
| **Soham Mangle** | Implementation of Algorithm 1 |
