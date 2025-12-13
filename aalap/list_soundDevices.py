import sounddevice as sd

def list_sound_devices():
    """Lists all available sound devices on the system."""
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(f"[{idx:2d}]: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")

if __name__ == "__main__":
    list_sound_devices()