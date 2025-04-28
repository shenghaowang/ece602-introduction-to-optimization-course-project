# 📚 ECE 602 Introduction to Optimization Course Project

Multi-objective optimization of manure management and emissions reduction using the ε-constraint method and Gurobi.

---

## 🚀 Key Features

- ✅ ε-constraint method for multi-objective optimization
- ✅ Mixed-integer programming formulation
- ✅ Gurobi solver integration
- ✅ Geospatial grid-based modeling of manure site distribution
- ✅ Comparative analysis with existing research results

---

## 📦 Installation

The repository can be installed using a Python virtual environment.

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ece602-introduction-to-optimization-course-project.git
cd text-classification-cnn-lstm
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### Part 1: Energy Production & Cost Optimization

```bash
source venv/bin/activate # On Windows: venv\Scripts\activate
python part1.py
```

### Part 2: Emissions of AD Plants Optimization

```bash
source venv/bin/activate # On Windows: venv\Scripts\activate
python part2_manure.py
```

---

## References

* Mukherjee, U., Tolson, B., Granito, C., Basu, N., Moreno, J., Saari, R., Bindas, S. (2025) A Landscape approach to Waste Management at the Food-Water-Energy Nexus: Spatial optimization Modelling for Biodigester Locations [Manuscript in preparation]. 
* Mavrotas, G. (2009). "Effective implementation of the ε-constraint method in multi-objective mathematical programming problems." Applied Mathematics and Computation, 213(2), 455–465. https://doi.org/10.1016/j.amc.2009.03.037
* Bindas, S. (2024). Atmospheric emissions associated with the use of biogas in Ontario (Master’s thesis, University of Waterloo). University of Waterloo's Institutional Repository. https://uwspace.uwaterloo.ca/items/d0ff0f30-4651-497d-8e08-6c56e2f6532b
