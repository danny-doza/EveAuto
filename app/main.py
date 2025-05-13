import os
import time
from multiprocessing import Process
import numpy as np
from PIL import Image

from ui_elements.calibrator import UIElementCalibrator, Point, Box, TextField
from startup import launch_or_focus, start_eve, select_profile
from ui_elements.app_context import get_app
from actions import Actions, AgentStateMachine
from image_processing import ImageProcessor

from pynput.mouse import Controller as MouseController
from pynput.keyboard import Controller as KeyboardController, Listener as KeyboardListener, Key
mouse = MouseController()
controller = KeyboardController()

from rich_log import get_logger
logger = get_logger("main")

from threading import Event

pause_event    = Event()    # when set, “paused”
shutdown_event = Event()    # when set, “shutdown”

def on_press(key):
    if key == Key.space:
        if pause_event.is_set(): pause_event.clear()
        else:                  pause_event.set()
        logger.info(f"Paused: {pause_event.is_set()}")
    elif key == Key.esc:
        shutdown_event.set()
        logger.info("Shutdown requested")

def main():
    get_app() # Initialize the app context for Qt

    # Ensure you are logged in, the EVE client is running, and you have selected a profile
    # before running this script.
    app_name = "EVE"  # Replace with your app name
    launch_or_focus(app_name)

    # start_eve()
    # select_profile(1)  # Select profile #1

    listener = KeyboardListener(on_press=on_press)
    listener.daemon = True
    listener.start()

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
        "__STEP3__": "Open the System Info pane and leave it open.",
        "system-name": TextField,
        "__STEP4__": "Open the Inventory pane and leave it open.",
        "inventory": Box,
        # "__STEP2__": "Navigate to the Agency pane.",
        # "agency-tabs": Box,
        # "__STEP3__": "Navigate to the Resource Harvesting tab.",
        # "resource-sources": Box,
        # "resource-filters-pane": Box,
        # "resource-results-pane": Box,
        # "resource-info-pane": Box,
    }
    elements = UIElementCalibrator.get_ui_elements(ui_spec)

    time.sleep(2)  # Wait for the UI to stabilize

    # Instantiate Actions once
    lower_blue = np.array([60, 25, 70])
    upper_blue = np.array([130, 110, 255])
    actions = Actions(elements, lower_blue, upper_blue)

    agent_sm = AgentStateMachine(actions, lower_blue, upper_blue)
    agent_sm.draw_graph()

    while not shutdown_event.is_set():

        # ImageProcessor.compute_fill_percentage(elements["inventory"].screenshot(), debug=True)
        # time.sleep(3)

        if not pause_event.is_set():
            
            # === Start your desired worker process using the Actions controller ===
            worker = Process(target=actions.sell, args=(["Veldspar", "Scordite"],))
            worker.start()
            # ======================================================================
            
            # Monitor the worker process
            while worker.is_alive():
                if shutdown_event.is_set():
                    worker.terminate()
                    break
                if pause_event.is_set():
                    logger.info("Pause requested; terminating mining process")
                    worker.terminate()
                    # Wait until unpaused before continuing
                    while pause_event.is_set() and not shutdown_event.is_set():
                        time.sleep(0.1)
                    break
                time.sleep(0.3)
        else:
            # When paused, just sleep briefly until unpaused
            while pause_event.is_set() and not shutdown_event.is_set():
                time.sleep(0.3)

    listener.stop()
    logger.info("Shutting down.")
    

    # You can now access:
    # elements["Login Button"].click()
    # elements["Information Pane"].screenshot()

if __name__ == "__main__":
    main()