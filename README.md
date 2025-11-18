# Low-Cost Image Processing Sensor for Airborne Particle Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs24196425-blue)](https://doi.org/10.3390/s24196425)

> **PhD Dissertation Project** | University of Deusto, Spain (2023-2025)  
> **Author:** Syed Mohsin Ali Shah 

## 📋 Table of Contents

- [Overview](#overview)
- [Research Problem](#research-problem)
- [Key Contributions](#key-contributions)
- [Publications](#publications)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔬 Overview

This repository contains the complete implementation of a **low-cost image processing-based sensor system** for detecting and quantifying airborne particulate matter (PM10) using petroleum jelly-coated substrates and smartphone imaging. The research addresses the critical gap in accessible, scalable air quality monitoring for citizen science applications.

### Key Features

- ✅ **Low-cost DIY sensor**: Petroleum jelly-coated paper substrates (~€0.50 per sample)
- ✅ **Smartphone-based**: No specialized equipment required
- ✅ **Automated detection**: Eliminates manual particle counting
- ✅ **Validated accuracy**: R² = 0.73 correlation with NILU reference stations
- ✅ **Scalable deployment**: Tested with 10,000+ users across 140 schools
- ✅ **Three-iteration methodology**: Progressive refinement from clustering to morphological feature extraction

---

## 🎯 Research Problem

### The Monitoring Gap

Traditional air quality monitoring systems (BAM, TEOM, reference stations) face significant limitations:

- **Limited spatial coverage**: High costs (€20,000-50,000 per station) restrict deployment density
- **Geographic "black spots"**: Schools, playgrounds, and residential areas remain unmonitored
- **Accessibility barriers**: Technical expertise required for operation and maintenance
- **Scalability challenges**: Cannot capture hyperlocal pollution variations

### Previous Citizen Science Limitations

NILU's citizen science approach (Castell et al., 2021) used petroleum jelly passive samplers with **manual particle counting**:

- ⏱️ Time-intensive: ~10-15 minutes per sample
- 👥 Subjective: Observer bias and human error
- 📈 Non-scalable: Limited throughput for large campaigns

### Our Solution

**Automated image processing algorithms** that enable:
- Objective, reproducible particle detection and quantification
- Rapid processing of thousands of images
- PM10 mass concentration estimation (µg/m³)
- Scientific validation against reference monitoring standards

---


## 🏆 Key Contributions

### 1. **Methodological Innovation**
- First automated image processing approach for petroleum jelly passive sampling
- Three-iteration progressive refinement methodology
- Synthetic data generation with realistic smartphone camera noise models

### 2. **Scientific Validation**
- Comprehensive validation against 6 NILU reference stations
- R² = 0.73 correlation demonstrates scientific validity
- 48 comparative samples across diverse environmental conditions

### 3. **Citizen Science Platform**
- **AmiAire**: Web-based platform enabling non-expert participation
- 10,000+ users engaged across 140 educational institutions
- 2,000+ PSP images processed with automated analysis
- Interactive mapping and WHO-aligned air quality visualization

### 4. **Practical Impact**
- Photography guidelines for citizen scientists based on synthetic data findings
- Scalable, accessible methodology for resource-constrained communities
- Expanded spatial and temporal coverage beyond traditional monitoring networks

---

## 💻 Installation

### Prerequisites

```bash
Python 3.8+
OpenCV 4.5+
NumPy 1.20+
scikit-learn 0.24+
scikit-image 0.18+
Pandas 1.3+
Matplotlib 3.4+
```


### Clone Repository

```bash
git clone https://github.com/thisismohsinsyed/AmiAire-Project-App
cd AmiAire-Project-App
```



### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---


## 🌐 AmiAire Platform

### Overview

AmiAire is a web-based citizen science platform that democratizes air quality monitoring through accessible, low-cost image processing technology.

### Key Features

- 📤 **Image Upload**: Simple interface for PSP image submission
- 🤖 **Automated Processing**: CNN-based PSP validation + particle detection pipeline
- 📊 **Real-time Analysis**: Instant PM10 concentration estimation
- 🗺️ **Interactive Mapping**: Visualize air quality across participating locations
- 🎨 **WHO-Aligned Visualization**: Color-coded air quality levels
- 📱 **Mobile-Responsive**: Accessible on smartphones and tablets

### Impact

- **10,000+ users** engaged across **140 schools**
- **2,000+ PSP images** processed
- **Expanded coverage** in Oslo, Norway beyond traditional monitoring networks
- **Educational outcomes**: Improved science literacy and environmental awareness




## 📚 Publications

### Journal Articles

1. **Shah, S. M. A.**, Casado-Mansilla, D., & López-de Ipiña, D. (2024). An Image-Based Sensor System for Low-Cost Airborne Particle Detection in Citizen Science Air Quality Monitoring. *Sensors*, 24(19), 6425. [https://doi.org/10.3390/s24196425](https://doi.org/10.3390/s24196425)

### Conference Papers

2. **Shah, S. M. A.**, Casado-Mansilla, D., López-de Ipiña, D., Hassani, A. H., & Illueca Fernández, E. (2024). An Image Processing-Based Approach for Precision Detection of Coarse Particle Deposition Rate. In *Proceedings of the 14th International Conference on Air Quality*, Helsinki, Finland. (Abstract)

3. **Shah, S. M. A.**, Casado-Mansilla, D., López-de Ipiña, D., Illueca Fernández, E., Hassani, A., & Pujante Pérez, A. (2024). A Low-Cost Image Sensor for Particulate Matter Detection to Streamline Citizen Science Campaigns on Air Quality Monitoring. In *2024 9th International Conference on Smart and Sustainable Technologies (SpliTech)* (pp. 1-6). IEEE.

4. Gómez Vazquez, I., **Shah, S. M. A.**, Casado-Mansilla, D., & López-de Ipiña, D. (2025). AmiAire: A Citizen Science and Action Research Initiative for Air Quality Monitoring and Behavioral Change. In *2025 10th International Conference on Smart and Sustainable Technologies (SpliTech)* (pp. 1-5). IEEE.

### Related Work

5. Castell, N., Grossberndt, S., Gray, L., Fredriksen, M. F., Skaar, J. S., & Høiskar, B. A. K. (2021). Implementing Citizen Science in Primary Schools: Engaging Young Children in Monitoring Air Pollution. *Frontiers in Climate*, 3, 639128. [https://doi.org/10.3389/fclim.2021.639128](https://doi.org/10.3389/fclim.2021.639128)

---

## 📖 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{shah2024image,
  title={An Image-Based Sensor System for Low-Cost Airborne Particle Detection in Citizen Science Air Quality Monitoring},
  author={Shah, Syed Mohsin Ali and Casado-Mansilla, Diego and L{\'o}pez-de Ipi{\~n}a, Diego},
  journal={Sensors},
  volume={24},
  number={19},
  pages={6425},
  year={2024},
  publisher={MDPI},
  doi={10.3390/s24196425}
}

@phdthesis{shah2025lowcost,
  title={A Low Cost Image Processing Sensor for Air Borne Particle Detection},
  author={Shah, Syed Mohsin Ali},
  year={2025},
  school={University of Deusto},
  address={Bilbao, Spain}
}
```

---


### Future Work

- Deep learning-based particle segmentation for overlapping particles
- Real-time IoT integration for continuous monitoring
- Multi-season, multi-location validation campaigns
- Cloud-based scalable processing infrastructure
- Enhanced preprocessing for extreme weather conditions

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Areas

- 🐛 Bug fixes and issue reports
- 📝 Documentation improvements
- ✨ New feature implementations
- 🧪 Additional test coverage
- 🌍 Translations and localization

---

## 📧 Contact

**Syed Mohsin Ali Shah**  
PhD Candidate, University of Deusto  
📧 Email: [thisismohsinsyed@gmail.com]  
🔗 LinkedIn: [[linkedin.com/in/yourprofile](https://www.linkedin.com/in/syed-mohsin-ali-shah/)]  


---

## 🙏 Acknowledgments

This research was conducted as part of the PhD program in **Engineering for Information Society and Sustainable Development** at the University of Deusto, Spain (2023-2025).

### Collaborations

- **NILU - Norwegian Institute for Air Research**: Validation support and reference station data
- **Oslo Municipality**: Support for citizen science campaigns
- **AmiAire Project**: Platform development and deployment
- **140 participating schools**: Citizen science engagement and data collection

### Funding
Minitry of Science and Technology Spain

### Special Thanks

- All participating schools and citizen scientists
- University of Deusto research community

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Syed Mohsin Ali Shah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---


*Making air quality monitoring accessible to everyone, one image at a time.* 🌍💚















