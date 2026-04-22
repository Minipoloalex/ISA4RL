import argparse
from datetime import datetime

def parse_date(date_string):
    if date_string.lower() == "today":
        return datetime.now()

    formats = [
        "%Y-%m-%d", 
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
            
    raise ValueError(f"Unrecognized date format: {date_string}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="today")
    
    args = parser.parse_args()
    
    dt_obj = parse_date(args.date)
    print(int(dt_obj.timestamp()))
