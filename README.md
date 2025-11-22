# GPS Project in TBA4565

This repository contains two projects completed in the course **Geomatics, Specialization Course (TBA4565)** at NTNU.  
Together, the projects explore two major approaches to GNSS positioning: **absolute positioning** and **relative high-precision positioning**.

---

### Project 1
Project 1 examines absolute point positioning using code pseudoranges, where satellite orbits, signal delays, and receiver coordinates are estimated through a least-squares adjustment. 

### Project 2
Project 2 addresses high-precision relative positioning using carrier-phase observations, emphasizing double differencing and integer ambiguity resolution. 

---

## Installation and setup
```bash
# 1) Clone the repository
git clone https://github.com/PederUndheim/TBA4565_GPS_Project
cd TBA4565_GPS_Project

# 2) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# 3) Install project dependencies
pip install -r requirements.txt

# 4) Run Project 1 or Project 2
python project1/main.py
python project2/main.py
```
