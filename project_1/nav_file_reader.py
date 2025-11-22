import pandas as pd
import re

def read_nav_file(filepath: str):
    eph_data = {}
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Skip header
    idx = 0
    while "END OF HEADER" not in lines[idx]:
        idx += 1
    idx += 1

    while idx < len(lines):
        if not lines[idx].strip():
            break
        block = lines[idx:idx+8]
        block = [ln.replace("D", "E") for ln in block]
        line1 = re.sub(r"(?<=\d)-", " -", block[0])
        parts = line1.split()
        sv = parts[0]       
        year = int(parts[1])
        month = int(parts[2])
        day = int(parts[3])
        hour = int(parts[4])
        minute = int(parts[5])
        second = int(float(parts[6]))
        epoch = pd.Timestamp(year, month, day, hour, minute, second)

        # --- collect numeric values
        nums = parts[7:] 
        for ln in block[1:]:
            ln = re.sub(r"(?<=\d)-", " -", ln)  
            nums += ln.split()
        nums = [float(x) for x in nums]

        params = {
            "af0": nums[0], "af1": nums[1], "af2": nums[2],
            "IODE": nums[3], "Crs": nums[4], "DeltaN": nums[5], "M0": nums[6],
            "Cuc": nums[7], "Eccentricity": nums[8], "Cus": nums[9], "sqrtA": nums[10],
            "TransTime": nums[11], "Cic": nums[12], "Omega0": nums[13], "Cis": nums[14],
            "Io": nums[15], "Crc": nums[16], "omega": nums[17], "OmegaDot": nums[18],
            "IDOT": nums[19]
        }

        eph_data.setdefault(sv, []).append((epoch, params))
        idx += 8

    return eph_data

if __name__ == "__main__":
    filepath = "/Users/pederundheim/Desktop/GPS_project/project_1/data/gps_data.txt"
    eph_data = read_nav_file(filepath)
    for sv, records in eph_data.items():
        print(f"SV: {sv}")
        for epoch, params in records:
            print(f"  Epoch: {epoch}, Params: {params}")