import json
import os

# --- Build absolute path to config file ---
_config_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILENAME = os.path.join(_config_dir, "config.json")
BUNDLED_OMNIMIDI_DLL = os.path.join(_config_dir, "OmniMIDI.dll")

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
        "omnimidi_load_preference": "unset"
    }
}


def save_config(config_data):
    """Saves the current configuration to config.json."""
    with open(CONFIG_FILENAME, 'w') as f:
        json.dump(config_data, f, indent=4)


def has_bundled_omnimidi():
    return os.path.exists(BUNDLED_OMNIMIDI_DLL)


def setup_omnimidi_preference(current_config):
    """
    Sets the initial OmniMIDI loading preference without any CPU-based prompt.
    Returns the updated config.
    """
    current_preference = current_config['audio'].get('omnimidi_load_preference')
    if current_preference not in ('unset', None):
        if current_preference == 'local' and not has_bundled_omnimidi():
            current_config['audio']['omnimidi_load_preference'] = 'path'
            save_config(current_config)
        return current_config

    current_config['audio']['omnimidi_load_preference'] = 'local' if has_bundled_omnimidi() else 'path'
    save_config(current_config)
    print("OmniMIDI setup complete. Preference saved.")
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
