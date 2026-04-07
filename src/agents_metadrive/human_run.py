from metadrive.envs.metadrive_env import MetaDriveEnv

# ==========================================
#              CONFIGURATION
# ==========================================

# Define your map using block characters. 
# Common blocks:
# 'S' = Straight
# 'C' = Curve (Circular)
# 'T' = T-Intersection
# 'X' = Crossroad
# 'O' = Roundabout
# 'R' = Ramp (entry/exit)
# 'r' = Ramp (straight)

# Change this string to test different map combinations!
MAP_CONFIG = "SXC"

# Control settings
MANUAL_CONTROL = True    # Set to True to drive with W/A/S/D
TRAFFIC_DENSITY = 0.0    # Set to 0.0 to test just the road, >0.0 to add cars
NUM_STEPS = 5000         # How long the simulation runs before auto-closing

# ==========================================

def run_map_test():
    # Setup the environment configuration
    config = {
        "map": MAP_CONFIG,
        "manual_control": MANUAL_CONTROL,
        "use_render": True,
        "traffic_density": TRAFFIC_DENSITY,
    }

    env = MetaDriveEnv(config)
    
    try:
        obs, info = env.reset()
        print(f"\n--- Successfully loaded map: '{MAP_CONFIG}' ---")
        if MANUAL_CONTROL:
            print("Manual control is ON. Use W/A/S/D to drive and test the blocks.")
        else:
            print("Manual control is OFF. The vehicle will take random actions.")
        
        # Simulation loop
        for step in range(NUM_STEPS):
            # If manual control is on, the action passed to step() is ignored 
            # and keyboard input is used instead.
            action = env.action_space.sample() if not MANUAL_CONTROL else [0, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset the environment if the car crashes or finishes the map
            if terminated or truncated:
                print("Episode ended (crash or destination reached). Resetting...")
                env.reset()
                
    finally:
        env.close()
        print("Simulation closed.")

if __name__ == "__main__":
    run_map_test()
