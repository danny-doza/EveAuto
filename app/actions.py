import cv2
from humancursor import SystemCursor
import json
import numpy as np
import os
import pyautogui
from pyautogui import ImageNotFoundException
from pathlib import Path
from PIL import Image
from pynput.keyboard import Key
from pynput.mouse import Button
import time
import random

from input.emulator import emulator, mouse, keyboard
from ui_elements.overlays.overlays import Overlays
from image_processing import ImageProcessor

def get_dir() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent


class Actions:

    cursor: SystemCursor = SystemCursor()
    elements: dict = {}
    known_positions: dict = {} # Cache for known positions, takes "<system_name>-<element_name>"

    def __init__(self, elements: dict):
        self.elements = elements
        self.known_positions = {}


    @staticmethod
    def click(x: int, y: int, button: str = "LEFT", keys: str = None):
        """Clicks on the point."""
        print(f"Attempting to click {button} mouse button at ({x}, {y})...")

        successful = False

        if x is None or y is None:
            print("Invalid coordinates for click.")
            return successful

        if keys:
            successful = Actions.key_and_click(x, y, keys)
        else:
            #pyautogui.moveTo(self.x, self.y, duration=random.uniform(0.5, 0.9))
            if emulator.online:
                #pyautogui.moveTo(center_x, center_y, duration=random.uniform(0.5, 0.9))
                emulator.move_and_click(x, y, button=button)
                successful = True
            else:
                target = (x, y)
                Actions.cursor.move_to(target)
                time.sleep(random.uniform(0, 1.5))
                Actions.cursor.click_on(target)
                successful = True
        
        if successful:
            debug = False
            if debug:
                label = "" # keep label empty
                Overlays.show_highlight_overlay(x, y, 20, 20, label=label, duration=9, shape="rect")

        return successful

    @staticmethod
    def key_and_click(x: int, y: int, keys: str, button: str = "LEFT"):
        """Press a key or set of keys and click on the point."""
        print(f"Attempting to click {button} key with {button} mouse button at ({x}, {y})...")

        successful = False

        if x is None or y is None:
            print("Invalid coordinates for key_and_click.")
            return successful

        key_list = keys.lower().split("-")

        key_map = {
            'ctrl': Key.ctrl,
            'alt': Key.alt,
            'shift': Key.shift,
            'cmd': Key.cmd,
            'enter': Key.enter,
            'esc': Key.esc,
            'tab': Key.tab,
            'space': Key.space,
            'backspace': Key.backspace,
            'delete': Key.delete,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
        }

        key_objs = [key_map.get(k, k) for k in key_list]  # default to char if not in map


        if emulator.online:
            for key in key_objs:
                emulator.press(key)
                
            target = (x, y)
            Actions.cursor.move_to(target)
            time.sleep(random.uniform(0, 1.5))
            Actions.cursor.click_on(target)
            
            for key in reversed(key_objs):
                emulator.release(key)
            successful = True
        else:
            for key in key_objs:
                keyboard.press(key)
            
            target = (x, y)
            Actions.cursor.move_to(target)
            time.sleep(random.uniform(0, 1.5))
            Actions.cursor.click_on(target)
            
            for key in reversed(key_objs):
                keyboard.release(key)
            successful = True
        
        if successful:
            debug = False
            if debug:
                label = "" # keep label empty
                Overlays.show_highlight_overlay(x, y, 20, 20, label=label, duration=9, shape="rect")

        return successful

    def scroll(self, clicks: int = -3, x: int = None, y: int = None):
        """
        Scrolls the mouse wheel within the center of the box.
        
        :param clicks: Number of scroll steps. Negative = up, positive = down.
        :param x: Optional override X position for scroll.
        :param y: Optional override Y position for scroll.
        """
        if x is None or y is None:
            x = self.p1.x + (self.p2.x - self.p1.x) // 2
            y = self.p1.y + (self.p2.y - self.p1.y) // 2

        # Move to the scroll point
        mouse.position = (x, y)
        mouse.scroll(0, clicks)


    def element_active(self, element_key: str):
        # Define green HSV range
        lower_green = np.array([55, 100, 100])
        upper_green = np.array([70, 220, 220])
        detected_halo = self.elements[element_key].detect_halo(lower_green, upper_green)
        if detected_halo:
            print("Element is active.")
            return True
        return False


    def target_aware(self, target: str):
        live_action_key = "live-action"

        if self.elements.get(live_action_key) is None:
            print(f"Element '{live_action_key}' not found.")
            return None

        if target in self.elements.get(live_action_key).read_text():
            print(f"Currently engaging with {target}")
            return True
        return False


    def element_text_highlighted(self, element_key: str, item_text: str):
        highlighted_text = self.elements.get(element_key).extract_highlighted_text()
        print(f"Extracted highlighted text: {highlighted_text}")
        if item_text in highlighted_text:
            print(f"Item text already highlighted.")
            return True
        return False


    def click_element_by_name(self, element_key: str, item_text: str,
                                   x: int = None, y: int = None, keys: str = None):
        print(f"Attempting to retrieve element by name: {item_text}")
        if self.elements.get(element_key) is None:
            print(f"Element {element_key} not found.")
            return None
        
        if self.element_text_highlighted(element_key, item_text):
            return None

        if x is None or y is None:
            return self.elements[element_key].click_button_by_text(item_text, confidence=0.5, keys=keys)
        else:
            self.click(x, y)
            return (x, y)


    def click_element_by_color(self, element_key: str, lower_hsv, upper_hsv,
                                    x: int = None, y: int = None, keys: str = None):
        print("Attempting to retrieve element by color range...")
        if self.elements.get(element_key) is None:
            print(f"Element {element_key} not found.")
            return None

        if x is None or y is None:
            return self.elements[element_key].click_button_by_color(lower_hsv, upper_hsv, keys=keys)
        else:
            self.click(x, y)
            return (x, y)


    def click_element_by_icon(self, element_key: str, icon: Image): # takes img of icon to match
        print("Attempting to retrieve element by icon...")

        if self.elements.get(element_key) is None:
            print(f"Element {element_key} not found.")
            return None

        element_screenshot = self.elements[element_key].screenshot()
        relative_icon_loc = ImageProcessor.match_template_with_color_mask(element_screenshot, icon, confidence=0.5)
        
        if relative_icon_loc:
            x = self.elements[element_key].p1.x
            y = self.elements[element_key].p1.y
            icon_loc_x = x + relative_icon_loc[0]
            icon_loc_y = y + relative_icon_loc[1]
            Overlays.show_highlight_overlay(icon_loc_x, icon_loc_y,
                                            20, 20, label="", duration=9)
            #pyautogui.moveTo(button_location[0], button_location[1], duration=random.uniform(0.5, 0.9))
            
            self.click(icon_loc_x, icon_loc_y)
            return (icon_loc_x, icon_loc_y)
        else:
            print("Icon not found on screen. Ensure the game is running and the button is visible.")
            return None


    def perform_action_on_element_item_by_name(self, element_key: str, item_text: str, action: str,
                                               x: int = None, y: int = None):
        if self.elements.get(element_key) is None:
            print(f"Element {element_key} not found.")
            return None

        if x is None or y is None:
            return self.elements[element_key].click_button_by_text(item_text, keys=action, confidence=0.7)
        else:
            self.click(x, y, keys=action)
            return (x, y)


    def performing_action(self, action: str, target: str = None, confidence: float = 0.5) -> bool:
        print(f"Checking if performing action: {action}...")

        if self.elements.get("live-action") is None:
            print(f"Element 'live-action' not found, unable to retrieve current action.")
            return False
        
        performing_action = self.elements["live-action"].contains(action, confidence=confidence)
        if performing_action:
            print(f"Already {action}!")
        
        if target:
            return performing_action and self.targeting(target)
        else:
            return performing_action


    def targeting(self, target: str, confidence: float = 0.5) -> bool:
        print(f"Checking if targeting: {target}...")

        if self.elements.get("overview-destination-details") is None:
            print(f"Element 'live-action' not found, unable to retrieve current action.")
            return False

        on_target = self.elements["overview-destination-details"].contains(target, confidence=confidence)
        if on_target:
            print(f"Already targeting {target}")
        
        return on_target


    def target_locked(self):
        target_locked_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/target_locked.png'))
        relative_icon_loc = ImageProcessor.match_template_with_color_mask(
            self.elements["overview-actions"].screenshot(), target_locked_img, confidence=0.5
        )

        if relative_icon_loc:
            return True
        else:
            print("Unable to find target locked icon.")
            return False


    def select_target(self, target: str):
        print(f"Attempting to select new target of {target}...")

        if self.targeting(target):
            print(f"Already targeting {target}!")
            return True
    
        if self.click_element_by_name("overview-list", target):
            print(f"Selected new target: {target}...")
            return True

        print(f"Failed to target: {target}")
        return False


    def select_autopilot_route(self):
        # Define yellow color range
        lower_hsv=np.array([20, 100, 100])
        upper_hsv=np.array([35, 255, 255])

        if self.click_element_by_color("overview-list", lower_hsv, upper_hsv):
            print(f"Selected next destination along autopilot route.")
            return True
        
        print(f"Failed to select next autopilot destination.")
        return False


    def move_to(self, target: str):
        if self.performing_action("Docking", target) or self.performing_action("Warping", target):
            print(f"Already moving to {target}.")
            return True
        
        if emulator.online:
            # click on targeted station and attempt to dock (with keyboard)
            if self.click_element_by_name("overview-list", target, key='d'):
                print(f"Beginning approach to {target}...")
                return True
            else:
                print(f"Failed attempt to approach {target}...")
        else:
            if self.select_target(target):
                # click on targeted station and attempt to dock (by action img)
                dock_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/d.png'))
                if self.click_element_by_icon("overview-actions", dock_img):
                    print(f"Beginning approach to {target}...")
                    return True

                warp_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/w.png'))
                if self.click_element_by_icon("overview-actions", warp_img):
                    print(f"Beginning approach to {target}...")
                    return True

                print(f"Failed attempt to approach {target}...")
            else:
                print(f"Failed to select target.")
        
        return False
    

    def orbit(self, target: str):
        if self.performing_action("Orbiting", target):
            print(f"Already orbiting {target}.")
            return True

        if emulator.online:
            # click on targeted station and attempt to dock (with keyboard)
            if self.click_element_by_name("overview-list", target, key='w'):
                print(f"Began attempting to orbit {target}...")
                return True
            else:
                print(f"Failed to orbit {target}...")
        else:
            if self.select_target(target):

                # click on targeted station and attempt to orbit (by action img)
                orbit_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/orbit.png'))
                if self.click_element_by_icon("overview-actions", orbit_img):
                    print(f"Began attempting to orbit {target}...")
                    return True
                else:
                    print(f"Unable to begin orbiting {target}, attempting approach...")
                    
                    # click on targeted station and attempt to approach (by action img)
                    approach_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/q.png'))
                    if self.click_element_by_icon("overview-actions", approach_img):
                        print(f"Began attempting to orbit {target}...")
                        return True
                    else:
                        print(f"Failed approach attempt.")

                print(f"Failed to orbit {target}...")
            else:
                print(f"Failed to select target.")
        
        return False
    

    def target_lock(self, target: str):
        if self.target_locked():
            print(f"Already locked on {target}.")
            return True

        if emulator.online:
            # click on targeted station and attempt to dock (with keyboard)
            if self.click_element_by_name("overview-list", target, key='ctrl'):
                print(f"Began locking on {target}...")
                return True
            else:
                print(f"Failed to lock onto {target}...")
        else:
            if self.select_target(target):

                # click on targeted station and attempt to orbit (by action img)
                target_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/ctrl.png'))
                if self.click_element_by_icon("overview-actions", target_img):
                    print(f"Began locking on {target}...")
                    return True
                else:
                    print(f"Unable lock onto {target}.")

            else:
                print(f"Failed to select target.")
        
        return False


    def mine(self, mineral: str):
        # skip if we're already aware of mineral and mining it
        if self.element_active("mine-button"):
            if not self.target_aware(mineral):
                print("Mineral not detected in live action. Are we sure we're mining and targeting the right mineral?")
            print("Already mining. Skipping.")
            return False
        
        # Find, target lock mineral, and start mining
        print(f"Targeting {mineral} to mine.")
        system_name = self.elements["system-name"].read_text()
        print(f"Current system name is: {system_name}")

        # 1.) Navigate to Mining tab
        if not self.element_text_highlighted("overview-tabs", "Mining"):
            clicked_mining_tab = self.click_element_by_name("overview-tabs", "Mining")
            if clicked_mining_tab:
                print("Navigated to Mining tab.")

        # 2.) Locate mineral and begin approach
        # check if mineral is within local area
        mineral_display_pos = self.click_element_by_name("overview-list", mineral)
        if not mineral_display_pos: # if not, warp to a nearby asteroid belt
            print("Mineral not found in local area, attempting to warp to new asteroid belt...")
            if self.move_to("asteroid belt"):
                print(f"Moving to nearby asteroid belt...")
                while self.elements["live-action"].contains("warping"):
                    print("Warping to asteroid belt...")
                    time.sleep(2)

        # check if mineral is within local area again
        # if not, return
        mineral_display_pos = self.click_element_by_name("overview-list", mineral)
        if not mineral_display_pos:
            print(f"{mineral} not found in region.")
            return
        print(f"Found {mineral} in region at position {mineral_display_pos}.")
        
        # 3.) Begin orbiting mineral to ensure within mining distance
        if self.orbit(mineral):
            print(f"Began moving to orbit {mineral}.")

            # Wait while moving into orbit
            polling_duration = 3
            while self.elements["overview-destination-details"] \
                .has_significant_change(duration=polling_duration):
                
                print("Waiting to enter orbit...")
                # Wait for the destination details to update
                time.sleep(1)
            
            # Check if the target is locked
            if self.target_lock(mineral):
                print("Target locked. Beginning attempt to mine...")

                # Press the mining button
                self.elements["mine-button"].click()                


    def dock(self, station: str, prefer_autopilot_route: bool = False):
        print(f"Targeting {station} to dock.")
        system_name = self.elements["system-name"].read_text()
        print(f"Current system name is: {system_name}")

        # 1.) Navigate to General tab
        if not self.element_text_highlighted("overview-tabs", "General"):
            clicked_general_tab = self.click_element_by_name("overview-tabs", "General")
            if clicked_general_tab:
                print("Navigated to General tab.")

        # 2.) Select destination
        if prefer_autopilot_route:
            if self.select_autopilot_route():
                print(f"Selected autopilot route destination and began approach.")
            else:
                print(f"Failed to select next autopilot route destination.")
        else:
            if self.click_element_by_name("overview-list", station):
                print(f"Began approach to {station}...")
            else:
                print(f"Failed to select {station}")

        # 3.) Begin warp / dock sequence
        if self.move_to(station):
            print(f"Beginning approach to {station}...")
        else:
            print(f"Failed attempt to approach {station}...")


    def undock(self):
        undock_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/undock.png'))
        if self.click_element_by_icon("fullscreen", undock_img):
            print("Began undocking sequence...")


