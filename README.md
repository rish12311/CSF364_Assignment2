# CSF364 - Assignment 1

**By:**
* **Rishabh Goyal** (2021B1A72320H)
* **Yash Gupta** (2021B4A72699H)
* **Soham Mangle** (2021B3A71068H)
* **Vani Jain** (2021B1A73126H)

Each team member contributed significantly to different aspects of the project, ensuring a well-rounded and comprehensive approach.

Webpage can be accessed on: https://rish12311.github.io/CSF364_Assignment2/webpage.html

This repository contains implementations of two algorithms for finding the **Maximum h-Clique Density**:
* **Exact Algorithm**
* **Core-Exact Algorithm**

These implementations are written in **C++** and optimized for performance.

## 📌 Setup Requirements

To run these algorithms, ensure you have:
* A **C++ compiler** with C++17 support
* A **Unix-like environment** (Linux, macOS, WSL, etc.)

## 🚀 Compilation

Each algorithm is implemented in a separate C++ file. Compile them as follows:

**Exact Algorithm**
```
g++ -std=c++17 -O3 -o algo1 algo1.cpp
```

**Core-Exact Algorithm**
```
g++ -std=c++17 -O3 -o algo2 algo2.cpp
```

## ▶️ Execution

Run each algorithm with the input file and h-value:

**Exact Algorithm**
```
./algo1 <filename>.mtx <h_value>
```

**Core-Exact Algorithm**
```
./algo2 <filename>.mtx <h_value>
```

Replace `<filename>.mtx` with the path to your matrix format data file and `<h_value>` with the desired h value for calculating h-clique density.

## 📖 Algorithm Descriptions

**1️⃣ Exact Algorithm**
A precise algorithm that guarantees finding the exact **maximum h-clique density** in a graph. This approach provides optimal results but may require more computational resources for larger graphs.

**2️⃣ Core-Exact Algorithm**
An optimized approach that leverages core decomposition techniques to efficiently compute the **maximum h-clique density**. This method provides exact results while potentially offering better performance on certain graph structures.

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
| **Yash Gupta** | Report writing & implementation of Core-Exact algorithm |
| **Vani Jain** | Performance optimization & testing |
| **Soham Mangle** | Implementation of Exact algorithm |
