import os

def is_sound_available(sargah: str, bait: str) -> bool:
    sounds_dir = os.path.join(os.path.dirname(__file__), "../sounds")
    file_name = f"{sargah}-{bait}.mp3"
    file_path = os.path.join(sounds_dir, file_name)

    # Check if file exists
    return os.path.isfile(file_path)