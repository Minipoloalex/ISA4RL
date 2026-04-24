import json

# Load configuration from the JSON file
with open("ranges_params.json", "r") as f:
    config = json.load(f)

chunk_size = config["chunk_size"]
segments = config["segments"]

# 1. Flatten all untrained segments into a single list of individual IDs
missing_ids = []
for start, end in segments:
    missing_ids.extend(range(start, end))

# 2. Grab boundaries every X missing elements
boundaries = []
for i in range(0, len(missing_ids), chunk_size):
    boundaries.append(missing_ids[i])

# Add the final absolute end boundary (the very last missing ID + 1)
if missing_ids:
    boundaries.append(missing_ids[-1] + 1)

# 3. Create ranges from those boundaries
ranges = []
for i in range(len(boundaries) - 1):
    ranges.append((boundaries[i], boundaries[i+1]))

# # 4. Pair them up for Task A and Task B on each node
# with open("ranges_gemini.txt", "w") as f:
#     for i in range(0, len(ranges), 2):
#         a_start, a_end = ranges[i]
        
#         # Check if there is a Task B for this node
#         if i + 1 < len(ranges):
#             b_start, b_end = ranges[i+1]
#         else:
#             b_start, b_end = 0, 0 # Fallback if there's an odd number of chunks
            
#         f.write(f"{a_start} {a_end} {b_start} {b_end}\n")

with open("ranges.txt", "w") as f:
    for start, end in ranges:
        f.write(f"{start} {end}\n")

print(f"Success! Generated ranges.txt")
