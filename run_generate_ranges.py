CHUNK_SIZE = 5

segments = [
    (104, 110), (113, 120), (123, 130), (133, 140), 
    (142, 150), (156, 160), (164, 170), (173, 180), 
    (182, 190), (193, 200), (202, 210), (212, 220), 
    (225, 230), (233, 240), (245, 250), (252, 594)
]

# 1. Flatten all untrained segments into a single list of individual IDs
missing_ids = []
for start, end in segments:
    missing_ids.extend(range(start, end))

# 2. Grab boundaries every 5 missing elements
boundaries = []
for i in range(0, len(missing_ids), CHUNK_SIZE):
    boundaries.append(missing_ids[i])

# Add the final absolute end boundary (the very last missing ID + 1)
if missing_ids:
    boundaries.append(missing_ids[-1] + 1)

# 3. Create ranges from those boundaries
ranges = []
for i in range(len(boundaries) - 1):
    ranges.append((boundaries[i], boundaries[i+1]))

# 4. Pair them up for Task A and Task B on each node
with open("ranges_gemini.txt", "w") as f:
    for i in range(0, len(ranges), 2):
        a_start, a_end = ranges[i]
        
        # Check if there is a Task B for this node
        if i + 1 < len(ranges):
            b_start, b_end = ranges[i+1]
        else:
            b_start, b_end = 0, 0 # Fallback if there's an odd number of chunks
            
        f.write(f"{a_start} {a_end} {b_start} {b_end}\n")

total_lines = (len(ranges) + 1) // 2
print(f"Success! Generated ranges.txt with {total_lines} lines.")
