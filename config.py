import os
import sys
import shutil
import glob
from typing import Union

# Set this to your custom path if you want to override auto-detection.
# Example: blender_executable = "/Applications/Blender.app/Contents/MacOS/Blender"
# This file will try to find the blender executable file if you don't provide it here,
# but it is not guaranteed to find it.
blender_executable: Union[str, None] = None


# 1. If the user defined a path manually, verify it exists and use it.
if blender_executable is not None:
    if not os.path.isfile(blender_executable):
        print(f"Warning: Provided blender_executable path does not exist: {blender_executable}")
        blender_executable = None

if blender_executable is None:
    guesses = []

    # 2. Add OS-specific default paths to our guesses
    if sys.platform == "darwin":  # macOS
        guesses.extend([
            "/Applications/Blender.app/Contents/MacOS/Blender",
            os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender")
        ])

    elif sys.platform.startswith("win"):  # Windows
        # Windows often appends version numbers to the folder (e.g., Blender 4.0)
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        blender_base = os.path.join(program_files, "Blender Foundation")

        # Use glob to find any version folder containing blender.exe
        if os.path.exists(blender_base):
            guesses.extend(glob.glob(os.path.join(blender_base, "Blender*", "blender.exe")))

    elif sys.platform.startswith("linux"):  # Linux
        guesses.extend([
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/opt/blender/blender",
            "/snap/bin/blender",
            os.path.expanduser("~/blender/blender"),
            os.path.expanduser("~/software/blender/blender"),
        ])

    # 3. Check which guessed path actually exists and is executable
    for guess in guesses:
        if os.path.isfile(guess) and os.access(guess, os.X_OK):
            blender_executable = guess

    # 4. Fallback: Check if Blender is in the system PATH
    system_path_blender = shutil.which("blender")
    if system_path_blender:
        blender_executable = system_path_blender

if blender_executable is None:
    ImportError("Please define assign the path to the blender executable to the variable"
                "'blender_executable' in the config.py file at the top level of this project."
                "Alternatively, you can define the system command 'blender' that executes blender.")