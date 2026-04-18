# GHOST: Graph-based Hierarchical On-the-fly Self-correcting Threat Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GHOST** is an innovative zero-shot supply chain disruption prediction framework that operates without requiring historical disruption labels. It uses only standard operational metrics to predict and prevent supply chain failures through autonomous learning and adaptive self-correction.

## 🚀 Key Features

- **Zero-Shot Learning**: Predicts disruptions without labeled historical data
- **Multi-Scale Risk Modeling**: Captures both global and local risk propagation patterns
- **Adaptive Self-Correction**: Automatically adapts to changing operational conditions
- **Real-Time Processing**: Processes 180k+ orders in ~15 minutes on standard hardware
- **Stability Mechanism**: Prevents performance degradation during self-training

## 📊 Performance Highlights

- **MSE**: 0.008067 (exceptional accuracy)
- **High-Risk Detection**: 30,943 samples identified (17.2% of dataset)
- **Zero-Shot Mapping**: 100% accuracy on scenario classification
- **Runtime**: 15.3 minutes on NVIDIA T4 GPU
- **Scalability**: Linear scaling with dataset size

## 🏗️ Architecture

GHOST operates through three core pillars:

1. **Data Autonomy**: Autonomous risk inference and synthetic scenario generation
2. **Network Intelligence**: Multi-scale LSTM and GNN risk modeling
3. **Self-Correction**: Drift detection and adaptive model updates

## 📁 Repository Structure

```
GHOST/
├── src/                          # Source code
│   ├── core/                     # Core algorithms
│   ├── models/                   # ML models (LSTM, GNN)
│   ├── data/                     # Data processing utilities
│   ├── decision/                 # Decision support system
│   └── ghost_complete.py         # Main GHOST implementation
├── notebooks/                    # Jupyter notebooks and examples
├── docs/                         # Documentation
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/rxbinsingh/GHOST.git
cd GHOST
pip install -r requirements.txt
```

### Basic Usage

```python
from src.ghost_complete import GHOST

# Initialize GHOST framework
ghost = GHOST()

# Load your supply chain data
data = ghost.load_data('path/to/your/data.csv')

# Run complete pipeline
results = ghost.run_pipeline(data)

# Get risk predictions
risk_scores = results['risk_predictions']
high_risk_nodes = results['high_risk_nodes']
```

### Google Colab

For a quick demo, try our [Google Colab notebook](notebooks/GHOST_Demo.py):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rxbinsingh/GHOST/blob/main/notebooks/GHOST_Demo.py)

## 🔬 Methodology

### Zero-Shot Risk Inference

GHOST infers risk scores from operational anomalies using statistical methods:

```
r_i^(t) = σ((||z_i^(t)||_2 + λ·δ_i^(t)) / (1 + ||z_i^(t)||_2 + λ·δ_i^(t)))
```

### Multi-Scale Graph Neural Network

- **Global Risk Injection**: System-wide risk propagation
- **Local Risk Attention**: Edge-specific risk modeling with multi-head attention

### Bootstrap Self-Distillation

- Generates synthetic disruption scenarios
- Iterative self-improvement with stability controls
- Prevents performance degradation during training

## 📊 Experimental Results

Tested on DataCo Smart Supply Chain dataset (180,519 orders):

| Metric | Value |
|--------|-------|
| Prediction MSE | 0.008067 |
| High-Risk Samples | 30,943 (17.2%) |
| Bootstrap Stability | 100% (3/3 iterations) |
| Drift Correction | 0.000014 → 0.000 |
| Zero-Shot Accuracy | 100% (5/5 scenarios) |
| Runtime (T4 GPU) | 15.3 minutes |

## 🛠️ Requirements

- Python 3.8+
- PyTorch 1.12+
- NetworkX 2.8+
- NumPy 1.21+
- SciPy 1.9+
- scikit-learn 1.1+
- pandas 1.4+

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Tutorial](docs/tutorial.md)
- [Mathematical Formulations](docs/mathematical_formulations.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Robin Singh**
- Institution: Bennett University, Noida, India
- Email: robinsingh4889@gmail.com
- Student ID: e23cseu0797@bennett.edu.in
- GitHub: [@rxbinsingh](https://github.com/rxbinsingh)

## 🙏 Acknowledgments

- Bennett University for research support
- DataCo for providing the Smart Supply Chain dataset
- EUREKA Research Hackathon for the platform to showcase this work

## 📈 Future Work

- Multi-modal risk integration (news, weather, geopolitical data)
- Hierarchical risk modeling for multi-tier supply networks
- Causal risk analysis for root cause identification
- Federated learning for collaborative risk intelligence
- Real-time deployment studies

---

⭐ **Star this repository if you find it useful!**

For questions or support, please open an issue or contact [robinsingh4889@gmail.com](mailto:robinsingh4889@gmail.com).