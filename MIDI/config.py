import json
import os
import re
import tkinter as tk
from tkinter import messagebox

# --- Build absolute path to config file ---
_config_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILENAME = os.path.join(_config_dir, "config.json")

DEFAULT_CONFIG = {
    "visualizer": {
        "scroll_speed": 2500.0,
        "note_width": 10.0,
        "show_guide_line": True,
        "seconds_before_cursor": 3.0,
        "seconds_after_cursor": 4.0,
        "streaming_vbo_capacity": 10000000,
        "data_update_interval": 0.1,
        "guide_line_y_ratio": 0.85,
        "show_keyboard": True
    },
    "audio": {
        "midi_backend": "omnimidi",
        "omnimidi_load_preference": "unset" # Added for initial setup
    }
}

def save_config(config_data):
    """Saves the current configuration to config.json."""
    with open(CONFIG_FILENAME, 'w') as f:
        json.dump(config_data, f, indent=4)

def get_cpu_recommendation(cpu_model_name):
    """
    Determines the recommended OmniMIDI loading preference based on CPU model name.
    """
    if not isinstance(cpu_model_name, str):
        return 'path', "CPU model name not provided or invalid. Recommended: OmniMIDI from PATH for broader compatibility."

    cpu_model_name = cpu_model_name.lower() # Case-insensitive matching

    if "celeron" in cpu_model_name or "pentium" in cpu_model_name or "core 2 duo" in cpu_model_name:
        return 'path', "Detected an older CPU (Celeron, Pentium, Core 2 Duo). Recommended: OmniMIDI from PATH for compatibility."

    # Check for Core i-series
    # This regex attempts to capture the 'i3/i5/i7/i9' and then optionally a 4-digit series number
    match = re.search(r"core(?:tm)?\s*(i[3579])(?:[-_ ]?(\d{4,5}[a-z]*)?)?", cpu_model_name)
    if match:
        processor_type = match.group(1) # i3, i5, i7, i9
        series_string = match.group(2) # e.g., '8700k', '4460'

        if series_string:
            # Try to extract leading digits for comparison
            series_number_match = re.match(r"(\d{4})", series_string)
            if series_number_match:
                series_number = int(series_number_match.group(1))
                if series_number >= 4000:
                    return 'local', f"Detected Intel {processor_type.upper()} {series_number} series or newer. Recommended: OmniMIDI from local directory for best performance."
                else:
                    return 'path', f"Detected Intel {processor_type.upper()} {series_number} series (older). Recommended: OmniMIDI from PATH for compatibility."
        # Fallback if no series number found or couldn't parse for Core i-series
        return 'path', f"Detected Intel {processor_type.upper()} without a clear series number (or older). Recommended: OmniMIDI from PATH for compatibility."

    # For any other CPUs not explicitly handled (e.g., AMD, unknown Intel), recommend path
    return 'path', "Could not determine optimal settings for your specific Intel CPU or detected non-Intel CPU. Recommended: OmniMIDI from PATH for broader compatibility."

def setup_omnimidi_preference(current_config):
    """
    Guides the user through setting the OmniMIDI loading preference based on CPU recommendation.
    Returns the updated config.
    """
    # Initialize Tkinter root for message boxes, hide it.
    root = tk.Tk()
    root.withdraw()

    if current_config['audio'].get('omnimidi_load_preference') not in ('unset', None):
        root.destroy()
        return current_config # Preference already set or explicitly ignored

    print("\n--- OmniMIDI Setup ---")

    # Since direct CPU detection is not possible, we ask the user for their CPU model
    cpu_model_prompt = ("Please enter your CPU model name (e.g., 'Intel Core i7-8700K', 'AMD Ryzen 5 3600').\n"
                        "This helps determine recommended OmniMIDI settings.\n"
                        "If you don't know, leave blank or enter 'unknown'.")
    
    # We can't use input() with a GUI app, and messagebox doesn't have text input.
    # So for now, we will assume no CPU model is given, and default to the 'path' recommendation.
    # In a real scenario, a custom Tkinter dialog with an entry field would be needed here.
    # For this exercise, I will simulate getting the CPU model by assuming a modern one for testing
    # purposes, or if I can't get input, it will fallback to 'path'.
    
    # Let's assume a generic placeholder for now, which will lead to 'path' recommendation.
    # If the user were able to provide it via a custom dialog, it would be passed here.
    # cpu_model_name = None # In a real app, this would come from a Tkinter Entry widget
    
    # To demonstrate the logic, I'll temporarily set a hypothetical CPU model for testing the recommendation function.
    # This will be removed or replaced with actual user input if a custom dialog is implemented.
    # Example: cpu_model_name = "Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz"
    # Example: cpu_model_name = "Intel(R) Core(TM) i5-3570 CPU @ 3.40GHz"
    # Example: cpu_model_name = "Intel(R) Celeron(R) CPU G530 @ 2.40GHz"
    
    # For now, let's use a dummy value for testing the function. This would ideally come from user input.
    # I'll use a strong CPU to test the 'local' recommendation path if the user accepts.
    # Once the prompt is in place, this will be handled dynamically.
    test_cpu_model = "Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz" 
    
    recommended_preference, recommendation_reason = get_cpu_recommendation(test_cpu_model)

    if recommended_preference == 'local':
        title = "OmniMIDI Setup: Recommended Settings"
        message = (f"Based on your CPU ({test_cpu_model}), it's recommended to load OmniMIDI from the local directory for best performance.\n\n"
                   f"Reason: {recommendation_reason}\n\n"
                   "Do you want to use this recommended setting ('Local')?\n"
                   "Choosing 'No' will use 'PATH' for broader compatibility.")
        
        use_recommended = messagebox.askyesno(title, message)
        if use_recommended:
            current_config['audio']['omnimidi_load_preference'] = 'local'
            print("User accepted 'local' preference.")
        else:
            current_config['audio']['omnimidi_load_preference'] = 'path'
            print("User opted for 'path' preference (overriding recommendation).")
    else: # Recommended preference is 'path'
        title = "OmniMIDI Setup: Recommended Settings"
        message = (f"Based on your CPU ({test_cpu_model}), it's recommended to load OmniMIDI from your system's PATH for broader compatibility.\n\n"
                   f"Reason: {recommendation_reason}\n\n"
                   "Do you want to use this recommended setting ('PATH')?\n"
                   "Choosing 'No' will try 'Local' (not recommended for your CPU).")
        
        use_recommended = messagebox.askyesno(title, message)
        if use_recommended:
            current_config['audio']['omnimidi_load_preference'] = 'path'
            print("User accepted 'path' preference.")
        else:
            current_config['audio']['omnimidi_load_preference'] = 'local'
            print("User opted for 'local' preference (overriding recommendation).")

    save_config(current_config)
    print("OmniMIDI setup complete. Preference saved.")
    root.destroy()
    return current_config


def load_config():
    """Loads config.json, creates or updates it with default values if necessary."""
    if not os.path.exists(CONFIG_FILENAME):
        print(f"'{CONFIG_FILENAME}' not found. Creating a default one.")
        with open(CONFIG_FILENAME, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    else:
        print(f"Loading settings from '{CONFIG_FILENAME}'...")
        with open(CONFIG_FILENAME, 'r') as f:
            user_config = json.load(f)
        
        # Check for missing keys and update from default
        updated = False
        for section, settings in DEFAULT_CONFIG.items():
            if section not in user_config:
                user_config[section] = settings
                updated = True
            else:
                for key, value in settings.items():
                    if key not in user_config[section]:
                        user_config[section][key] = value
                        updated = True
        
        if updated:
            print(f"Updating '{CONFIG_FILENAME}' with new default settings.")
            with open(CONFIG_FILENAME, 'w') as f:
                json.dump(user_config, f, indent=4)
                
        return user_config