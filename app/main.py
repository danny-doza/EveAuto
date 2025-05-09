import os
import time
from multiprocessing import Process
import numpy as np
from PIL import Image

from ui_elements.calibrator import UIElementCalibrator, Point, Box, TextField
from startup import launch_or_focus, start_eve, select_profile
from ui_elements.app_context import get_app

from actions import Actions

def main():
    get_app() # Initialize the app context for Qt

    # overlay_proc = Process(target=show_mouse_overlay, daemon=True)
    # overlay_proc.start()

    # Ensure you are logged in, the EVE client is running, and you have selected a profile
    # before running this script.
    app_name = "EVE"  # Replace with your app name
    launch_or_focus(app_name)

    # start_eve()
    # select_profile(1)  # Select profile #1

    ui_spec = {
        #"Inventory": Point,
        #"Username Field": TextField,
        "fullscreen": Box,
        "__STEP1__": "Navigate to the Overview pane.",
        "overview-destination-details": Box,
        "overview-actions": Box,
        "overview-tabs": Box,
        "overview-list": Box,
        "__STEP2__": "Select a destination and begin orbiting it with w + click.",
        "live-action": TextField,
        "mine-button": Box,
        "__STEP3__": "Open the System Info pane.",
        "system-name": TextField,
        # "__STEP2__": "Navigate to the Agency pane.",
        # "agency-tabs": Box,
        # "__STEP3__": "Navigate to the Resource Harvesting tab.",
        # "resource-sources": Box,
        # "resource-filters-pane": Box,
        # "resource-results-pane": Box,
        # "resource-info-pane": Box,
    }
    elements = UIElementCalibrator.get_ui_elements(ui_spec)
    
    try:
        while True:
            print("\nMain script running...")

            # if elements["overview-list"].click_button_by_color(lower_hsv, upper_hsv):
            #     print("=^D")

            actions = Actions(elements)
            actions.mine("Veldspar")
            #actions.click_element_by_name("fullscreen", "Mining Hold")

            time.sleep(7)
    except KeyboardInterrupt:
        print("Shutting down.")
    

    # You can now access:
    # elements["Login Button"].click()
    # elements["Information Pane"].screenshot()

if __name__ == "__main__":
    main()