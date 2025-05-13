from typing import List
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

from transitions.extensions import GraphMachine

from input.emulator import emulator, mouse, keyboard
from ui_elements.overlays.overlays import Overlays
from image_processing import ImageProcessor

from rich_log import get_logger
logger = get_logger("Actions")

def get_dir() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent


class Actions:

    cursor: SystemCursor = SystemCursor()
    elements: dict = {}
    known_positions: dict = {} # Cache for known positions, takes "<system_name>-<element_name>"

    def __init__(self, elements: dict, lower_hsv: np.ndarray, upper_hsv: np.ndarray):
        
        self.elements = elements
        self.accent_lower_hsv = lower_hsv
        self.accent_upper_hsv = upper_hsv

        self.known_positions = {}



    @staticmethod
    def click(x: int, y: int, button: str = "LEFT", keys: str = None, debug: bool = False):
        """Clicks on the point."""
        logger.debug(f"Attempting to click {button} mouse button at ({x}, {y})...")

        successful = False

        if x is None or y is None:
            logger.warning("Coordinates for click are null.")
            return successful

        mouse_map = {
            'left': Button.left,
            'middle': Button.middle,
            'right': Button.right,
        }
        mouse_button = mouse_map.get(button.lower(), Button.left)  # default to left button

        if keys:
            successful = Actions.key_and_click(x, y, keys)
        else:
            #pyautogui.moveTo(self.x, self.y, duration=random.uniform(0.5, 0.9))
            if emulator.online:
                #pyautogui.moveTo(center_x, center_y, duration=random.uniform(0.5, 0.9))
                emulator.move_and_click(x, y, button=button)
                successful = True
            else:

                # Move the cursor to the target position
                target = (x, y)
                Actions.cursor.move_to(target, duration=random.uniform(0.12, 4.1), steady=True)
                mouse.click(button=mouse_button)

                # Move the cursor to a random position off of button after clicking
                rest_spot = (target[0] + random.uniform(-50, 50) - 300,
                             target[1] + random.uniform(-50, 50) - 100)
                Actions.cursor.move_to(rest_spot, duration=random.uniform(0.12, 4.1), steady=True)
                
                successful = True
        
        if debug:
            label = "" # keep label empty
            Overlays.show_highlight_overlay(x, y, 20, 20, label=label, duration=3, shape="rect")

        return successful

    @staticmethod
    def click_and_drag(start: tuple[int, int], end: tuple[int, int],
                          button: str = "LEFT", debug: bool = False):
        """Simulates a click-and-drag operation from one point to another."""
        logger.debug(f"Click-and-drag from ({start}) to ({end}) with {button} button")

        if start is None or end is None:
            logger.warning("Invalid coordinates for click-and-drag.")
            return False

        if emulator.online:
            emulator.mouse_drag(start, end, button=button)
        else:
            Actions.cursor.drag_and_drop(start, end, duration=random.uniform(0.5, 2.0), steady=True)
            
            # Move the cursor to a random position off of button after clicking
            rest_spot = (start[0] + random.uniform(-50, 50) - 300,
                         start[1] + random.uniform(-50, 50) - 100)
            Actions.cursor.move_to(rest_spot, duration=random.uniform(0.12, 4.1), steady=True)
    
        if debug:
            Overlays.show_highlight_overlay(start[0], start[1], 20, 20, duration=3, shape="rect")
            Overlays.show_highlight_overlay(end[0], end[1], 20, 20, duration=3, shape="rect")

        return True


    @staticmethod
    def key_and_click(x: int, y: int, keys: str, button: str = "LEFT", debug: bool = False):
        """Press a key or set of keys and click on the point."""
        logger.debug(f"Attempting to click {button} key with {button} mouse button at ({x}, {y})...")

        successful = False

        if x is None or y is None:
            logger.warning("Invalid coordinates for key_and_click.")
            return successful

        mouse_map = {
            'left': Button.left,
            'middle': Button.middle,
            'right': Button.right,
        }
        mouse_button = mouse_map.get(button.lower(), Button.left)  # default to left button

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
            Actions.cursor.move_to(target, duration=random.uniform(0.12, 4.1), steady=True)
            mouse.click(button=mouse_button)
            
            # Move the cursor to a random position off of button after clicking
            rest_spot = (target[0] + random.uniform(-50, 50) - 300,
                         target[1] + random.uniform(-50, 50) - 100)
            Actions.cursor.move_to(rest_spot, duration=random.uniform(0.12, 4.1), steady=True)
            
            for key in reversed(key_objs):
                emulator.release(key)
            successful = True
        else:
            for key in key_objs:
                keyboard.press(key)
            
            target = (x, y)
            Actions.cursor.move_to(target, duration=random.uniform(0.12, 4.1), steady=True)
            mouse.click(button=mouse_button)
            
            # Move the cursor to a random position off of button after clicking
            rest_spot = (target[0] + random.uniform(-50, 50) - 300,
                         target[1] + random.uniform(-50, 50) - 100)
            Actions.cursor.move_to(rest_spot, duration=random.uniform(0.12, 4.1), steady=True)

            for key in reversed(key_objs):
                keyboard.release(key)
            successful = True
        
        if successful:
            if debug:
                label = "" # keep label empty
                Overlays.show_highlight_overlay(x, y, 20, 20, label=label, duration=3, shape="rect")

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
        lower_green = np.array([35, 50, 50])   # more inclusive of yellowish-greens
        upper_green = np.array([85, 255, 255]) # allows brighter and more saturated greens
        detected_halo = self.elements[element_key].detect_halo(lower_green, upper_green, confidence=0.4)
        if detected_halo:
            return True
        return False


    def element_text_highlighted(self, element_key: str, item_text: str, threshold: int = 124):
        highlighted_text = self.elements.get(element_key).extract_highlighted_text(threshold=threshold)
        logger.debug(f"Extracted highlighted text: {highlighted_text}")
        if item_text in highlighted_text:
            return True
        return False


    def get_text_location(self, element_key: str, item_text: str, keys: str = None,
                          threshold: int = 124, lower_hsv=None, upper_hsv=None,
                          confidence: int = 0.8):
        logger.info(f"Attempting to retrieve element location by name: {item_text}")
        if self.elements.get(element_key) is None:
            logger.warning(f"Element {element_key} not found.")
            return None

        return self.elements[element_key].get_text_location(
            item_text, 
            threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
            confidence=confidence, keys=keys
        )


    def click_element_by_name(self, element_key: str, item_text: str,
                              x: int = None, y: int = None, keys: str = None,
                              threshold: int = 124, lower_hsv=None, upper_hsv=None,
                              confidence: int = 0.8):
        logger.info(f"Attempting to retrieve element by name: {item_text}")
        if self.elements.get(element_key) is None:
            logger.warning(f"Element {element_key} not found.")
            return None

        # if self.element_text_highlighted(element_key, item_text):
        #     logger.info(f"Item text already highlighted.")
        #     return (0, 0)

        if x is None or y is None:
            return self.elements[element_key].click_button_by_text(
                item_text, 
                threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                confidence=confidence, keys=keys
            )
        else:
            self.click(x, y)
            return (x, y)


    def click_element_by_color(self, element_key: str, lower_hsv, upper_hsv,
                                    x: int = None, y: int = None, keys: str = None):
        logger.info("Attempting to retrieve element by color range...")
        if self.elements.get(element_key) is None:
            logger.warning(f"Element {element_key} not found.")
            return None

        if x is None or y is None:
            return self.elements[element_key].click_button_by_color(lower_hsv, upper_hsv, keys=keys)
        else:
            self.click(x, y)
            return (x, y)


    def click_element_by_icon(self, element_key: str, icon: Image): # takes img of icon to match
        logger.info("Attempting to retrieve element by icon...")

        if self.elements.get(element_key) is None:
            logger.warning(f"Element {element_key} not found.")
            return None

        element_screenshot = self.elements[element_key].screenshot()
        relative_icon_loc = ImageProcessor.match_template(element_screenshot, icon, confidence=0.36,
                                                          match_white=True,
                                                          debug=True)
        
        if relative_icon_loc:
            x = self.elements[element_key].p1.x
            y = self.elements[element_key].p1.y
            icon_loc_x = x + relative_icon_loc[0]
            icon_loc_y = y + relative_icon_loc[1]
            #pyautogui.moveTo(button_location[0], button_location[1], duration=random.uniform(0.5, 0.9))
            
            self.click(icon_loc_x, icon_loc_y)
            return (icon_loc_x, icon_loc_y)
        else:
            logger.warning("Icon not found on screen. Ensure the game is running and the button is visible.")
            return None


    def performing_action(self, action: str, target: str = None, confidence: float = 0.5) -> bool:
        logger.info(f"Checking if performing action: {action}...")

        if self.elements.get("live-action") is None:
            logger.warning(f"Element 'live-action' not found, unable to retrieve current action.")
            return False
        
        performing_action = self.elements["live-action"].contains(action, confidence=confidence)
        if performing_action:
            logger.info(f"Already {action}!")
        
        if target:
            return performing_action and self.targeting(target)
        else:
            return performing_action


    def targeting(self, target: str, confidence: float = 0.5) -> bool:
        logger.info(f"Checking if targeting: {target}...")

        if self.elements.get("overview-destination-details") is None:
            logger.warning(f"Element 'live-action' not found, unable to retrieve current action.")
            return False

        on_target = self.elements["overview-destination-details"].contains(target, confidence=confidence)
        if on_target:
            logger.info(f"Already targeting {target}")
        
        return on_target


    def target_locked(self):
        target_locked_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/target_locked.png'))
        relative_icon_loc = ImageProcessor.match_template(
            self.elements["overview-actions"].screenshot(), target_locked_img,
            confidence=0.38,
            debug=True
        )

        if relative_icon_loc:
            return True
        else:
            logger.warning("Unable to find target locked icon.")
            return False


    def select_target(self, target: str):
        logger.info(f"Attempting to select new target of {target}...")

        if self.targeting(target):
            logger.info(f"Already targeting {target}!")
            return True
    
        if self.click_element_by_name("overview-list", target):
            logger.info(f"Selected new target: {target}...")
            return True

        logger.warning(f"Failed to target: {target}")
        return False


    def select_autopilot_route(self):
        # Define yellow color range
        lower_hsv=np.array([20, 100, 100])
        upper_hsv=np.array([35, 255, 255])

        if self.click_element_by_color("overview-list", lower_hsv, upper_hsv):
            logger.info(f"Selected next destination along autopilot route.")
            return True
        
        logger.warning(f"Failed to select next autopilot destination.")
        return False


    def move_to(self, target: str):
        if self.performing_action("Docking", target) or self.performing_action("Warping", target):
            logger.info(f"Already moving to {target}.")
            return True
        
        if emulator.online:
            # click on targeted station and attempt to dock (with keyboard)
            if self.click_element_by_name("overview-list", target, key='d'):
                logger.info(f"Beginning approach to {target}...")
                return True
            else:
                logger.warning(f"Failed attempt to approach {target}...")
        else:
            if self.select_target(target):
                # click on targeted station and attempt to dock (by action img)
                dock_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/d.png'))
                if self.click_element_by_icon("overview-actions", dock_img):
                    logger.info(f"Beginning approach to {target}...")
                    return True

                warp_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/w.png'))
                if self.click_element_by_icon("overview-actions", warp_img):
                    logger.info(f"Beginning warp to {target}...")
                    return True

                logger.warning(f"Failed attempt to approach {target}...")
            else:
                logger.warning(f"Failed to select target.")
        
        return False
    

    def orbit(self, target: str):
        if self.performing_action("Orbiting", target):
            logger.info(f"Already orbiting {target}.")
            return True

        if emulator.online:
            # click on targeted station and attempt to dock (with keyboard)
            if self.click_element_by_name("overview-list", target, key='w'):
                logger.info(f"Began attempting to orbit {target}...")
                return True
            else:
                logger.warning(f"Failed to orbit {target}...")
        else:
            if self.select_target(target):

                # click on targeted station and attempt to orbit (by action img)
                orbit_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/orbit.png'))
                if self.click_element_by_icon("overview-actions", orbit_img):
                    logger.info(f"Began attempting to orbit {target}...")
                    return True
                else:
                    logger.warning(f"Unable to begin orbiting {target}, attempting approach...")
                    
                    # # click on targeted station and attempt to approach (by action img)
                    # approach_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/q.png'))
                    # if self.click_element_by_icon("overview-actions", approach_img):
                    #     logger.info(f"Began attempting to approach {target}...")
                    #     return True
                    # else:
                    #     logger.warning(f"Failed approach attempt.")

                logger.warning(f"Failed to orbit {target}...")
            else:
                logger.warning(f"Failed to select target.")
        
        return False
    

    def target_lock(self, target: str):
        if self.target_locked():
            logger.info(f"Already locked on {target}.")
            return True

        if emulator.online:
            # click on targeted station and attempt to dock (with keyboard)
            if self.click_element_by_name("overview-list", target, key='ctrl'):
                logger.info(f"Began locking onto {target}...")
                return True
            else:
                logger.warning(f"Failed to lock onto {target}...")
        else:
            if self.select_target(target):

                # click on targeted station and attempt to orbit (by action img)
                target_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/actions/ctrl.png'))
                if self.click_element_by_icon("overview-actions", target_img):
                    logger.info(f"Began locking onto {target}...")
                    return True
                else:
                    logger.warning(f"Unable to lock onto {target}.")

            else:
                logger.warning(f"Failed to select target.")
        
        return False


    def mine(self, mineral: str):
        # skip if we're already aware of mineral and mining it
        if self.element_active("mine-button"):
            if not self.targeting(mineral):
                logger.warning("Mineral not detected in live action. Are we sure we're mining and targeting the right mineral?")
            logger.info("Already mining. Skipping.")
            return False
        
        # Find, target lock mineral, and start mining
        logger.info(f"Targeting {mineral} to mine.")
        system_name = self.elements["system-name"].read_text()
        logger.info(f"Current system name is: {system_name}")

        # 1.) Navigate to Mining tab
        if not self.element_text_highlighted("overview-tabs", "Mining"):
            clicked_mining_tab = self.click_element_by_name("overview-tabs", "Mining")
            if clicked_mining_tab:
                logger.info("Navigated to Mining tab.")

        # 2.) Locate mineral and begin approach
        # check if mineral is within local area
        if not self.targeting(mineral): # skip check if already targeting mineral
            mineral_display_pos = self.select_target(mineral)
            if not mineral_display_pos: # if not, warp to a nearby asteroid belt
                logger.warning("Mineral not found in local area, attempting to warp to new asteroid belt...")
                if self.move_to("asteroid belt"):
                    logger.info(f"Moving to nearby asteroid belt...")
                    while self.performing_action("warping"):
                        logger.info("Warping to asteroid belt...")
                        time.sleep(2)

            # check if mineral is within local area again
            # if not, return
            mineral_display_pos = self.select_target(mineral)
            if not mineral_display_pos:
                logger.warning(f"{mineral} not found in region.")
                return
            logger.info(f"Found {mineral} in region at position {mineral_display_pos}.")
        
        # 3.) Begin orbiting mineral to ensure within mining distance
        if self.orbit(mineral):
            logger.info(f"Began moving to orbit {mineral}.")

            # Wait while moving into orbit
            polling_duration = 3
            while self.elements["overview-destination-details"] \
                .has_significant_change(duration=polling_duration):
                
                logger.info("Waiting to enter orbit...")
                # Wait for the destination details to update
                time.sleep(1)
            
            # Check if the target is locked
            if self.target_lock(mineral):
                logger.info("Target locked. Beginning attempt to mine...")

                # Press the mining button
                self.elements["mine-button"].click()                


    def dock(self, station: str, prefer_autopilot_route: bool = False):
        logger.info(f"Targeting {station} to dock.")
        system_name = self.elements["system-name"].read_text()
        logger.info(f"Current system name is: {system_name}")

        # 1.) Navigate to General tab
        if not self.element_text_highlighted("overview-tabs", "General"):
            clicked_general_tab = self.click_element_by_name("overview-tabs", "General")
            if clicked_general_tab:
                logger.info("Navigated to General tab.")

        # 2.) Select destination
        if prefer_autopilot_route:
            if self.select_autopilot_route():
                logger.info(f"Selected autopilot route destination and began approach.")
            else:
                logger.error(f"Failed to select next autopilot route destination.")
        else:
            if self.select_target(station):
                logger.info(f"Began approach to {station}...")
            else:
                logger.error(f"Failed to select {station}")

        # 3.) Begin warp / dock sequence
        if self.move_to(station):
            logger.info(f"Beginning approach to {station}...")
        else:
            logger.error(f"Failed attempt to approach {station}...")


    def undock(self):
        undock_img = Image.open(os.path.join(os.path.dirname(__file__), 'resources/undock.png'))
        if self.click_element_by_icon("fullscreen", undock_img):
            logger.info("Began undocking sequence...")
        else:
            logger.error(f"Failed to begin undocking sequence.")


    def unload(self, minerals: str | List[str]):
        if not self.get_text_location("inventory", "mining hold", confidence=0.8,
                                      lower_hsv=self.accent_lower_hsv, upper_hsv=self.accent_upper_hsv):
            inv_mining_hold_loc = self.get_text_location("inventory", "mining hold", confidence=0.8)
        
        if not isinstance(minerals, list):
            minerals = [minerals]

        for mineral in minerals:
            inv_mineral_loc = self.get_text_location("inventory", mineral,
                                                     threshold=100,
                                                     confidence=0.9)
            while inv_mineral_loc:
                logger.info(f"Attempting to unload {mineral}...")
                
                if not self.get_text_location("inventory", "mining hold", confidence=0.8,
                                              lower_hsv=self.accent_lower_hsv, upper_hsv=self.accent_upper_hsv):
                    inv_mining_hold_loc = self.click_element_by_name("inventory", "mining hold",
                                                                     threshold=100,
                                                                     confidence=0.8)

                inv_mineral_loc = self.get_text_location("inventory", mineral,
                                                         threshold=100,
                                                         confidence=0.9)
                if inv_mineral_loc is None:
                    logger.warning(f"Unable to find {mineral} in inventory.")
                    break

                inv_item_hangar_loc = self.get_text_location("inventory", "item hangar",
                                                             threshold=100,
                                                             confidence=0.8)
                if inv_item_hangar_loc is None:
                    logger.warning("Unable to find item hangar in inventory.")
                    break

                logger.info(f"Unloading {mineral} to item hangar...")
                self.click_and_drag(inv_mineral_loc, inv_item_hangar_loc, debug=True)


    def sell(self, minerals: str | List[str]):
        # check if item hangar is open and open if not
        if not self.get_text_location("inventory", "item hangar", confidence=0.8,
                                      lower_hsv=self.accent_lower_hsv, upper_hsv=self.accent_upper_hsv):
            inv_item_hangar_loc = self.get_text_location("inventory", "item hangar", confidence=0.8)
        
        if not isinstance(minerals, list):
            minerals = [minerals]

        for mineral in minerals:
            inv_mineral_loc = self.get_text_location("inventory", mineral,
                                                     threshold=100,
                                                     confidence=0.9)
            while inv_mineral_loc:
                logger.info(f"Attempting to sell {mineral}...")
                
                # check if item hangar is open and open if not
                if not self.get_text_location("inventory", "item hangar", confidence=0.8,
                                              lower_hsv=self.accent_lower_hsv, upper_hsv=self.accent_upper_hsv):
                    inv_item_hangar_loc = self.click_element_by_name("inventory", "item hangar",
                                                                    threshold=100,
                                                                    confidence=0.8)

                inv_mineral_loc = self.get_text_location("inventory", mineral,
                                                         threshold=100,
                                                         confidence=0.9)
                if inv_mineral_loc is None:
                    logger.warning(f"Unable to find {mineral} in inventory.")
                    break

                if self.click(inv_mineral_loc[0], inv_mineral_loc[1], button="right"):
                    logger.info(f"Right-clicked on {mineral} in inventory.")

                    inv_market_loc = self.click_element_by_name("fullscreen", "Sell this item",
                                                                threshold=100,
                                                                confidence=0.6)
                    if inv_market_loc is None:
                        logger.warning("Unable to find market.")
                        break

                    sell_button = self.click_element_by_name(
                        "fullscreen", "Sell",
                        lower_hsv=self.accent_lower_hsv, upper_hsv=self.accent_upper_hsv,
                        confidence=0.6
                    )
                    if sell_button is None:
                        logger.warning("Unable to find sell button.")
                        break

                    ok_button = self.click_element_by_name(
                        "fullscreen", "OK",
                        lower_hsv=self.accent_lower_hsv, upper_hsv=self.accent_upper_hsv,
                        confidence=0.4
                    )
                    if ok_button is None:
                        logger.warning("No confirmation screen found or ok button not found.")

                    logger.info(f"Clicked on {sell_button} to sell {mineral}...")

                logger.info(f"Selling {mineral}...")

# State machine for agent actions
class AgentStateMachine:
    
    states = ['idle', 'mining', 'docking', 'undocking']

    def __init__(self, actions: Actions, accent_lower_hsv, accent_upper_hsv):
        
        self.actions = actions
        self.accent_lower_hsv = accent_lower_hsv
        self.accent_upper_hsv = accent_upper_hsv

        # Initialize the state machine
        self.machine = GraphMachine(
            model=self,
            states=AgentStateMachine.states,
            initial='idle',
            auto_transitions=False,
            show_conditions=True
        )


        # To 'Mining' state
        self.machine.add_transition(
            trigger='mine',
            source=['idle', 'undocking'],
            dest='mining',
            after='start_mining'
        )

        # To 'Docking' state
        self.machine.add_transition(
            trigger='check_fill',
            source='mining',
            dest='docking',
            conditions='is_full',
            after='start_docking'
        )

        # To 'Unloading' state
        self.machine.add_transition(
            trigger='unload',
            source='docking',
            dest='unloading',
            after='start_unloading'
        )

        # Continue 'Unloading' state if not empty
        self.machine.add_transition(
            trigger='unload',
            source='unloading',
            dest='unloading',
            conditions='is_not_empty',
            after='start_unloading'
        )

        # To 'Undocking' state
        self.machine.add_transition(
            trigger='undock',
            source=['unloading', 'docking'],
            dest='undocking',
            conditions='is_empty',
            after='start_undocking'
        )
        
        # Reset trigger, sets agent to 'Idle' state
        self.machine.add_transition(
            trigger='reset',
            source='*',
            dest='idle'
        )


    def start_mining(self, **kwargs):
        mineral = kwargs.get("mineral", "MINERAL MISSING!")
        self.actions.mine(mineral)


    def start_docking(self, **kwargs):
        station = kwargs.get("station", "STATION MISSING!")
        self.actions.dock(station)


    def start_unloading(self, **kwargs):
        minerals = kwargs.get("minerals", "MINERALS MISSING!")
        self.actions.unload(minerals)


    def start_undocking(self, **kwargs):
        self.actions.undock()


    def can_dock(self, **kwargs):
        """
        Guard condition for docking transition.
        Return True if docking is allowed, False otherwise.
        """
        return self.actions.performing_action("Warping") is False


    def is_full(self, **kwargs):
        """
        Guard condition for mining -> docking transition.
        Return True if the cargo hold is full, False otherwise.
        """
        full_threshold = kwargs.get("full_threshold", 0.8)

        fill_percentage = ImageProcessor.compute_fill_percentage(
            self.actions.elements["inventory"].screenshot(),
            lower_blue, upper_blue, 
            debug=True
        )

        return fill_percentage >= full_threshold


    def is_empty(self, **kwargs):
        """
        Guard condition for unloading -> undocking transition.
        Return True if the cargo hold is empty, False otherwise.
        """
        empty_threshold = kwargs.get("empty_threshold", 0.1)

        fill_percentage = ImageProcessor.compute_fill_percentage(
            self.actions.elements["inventory"].screenshot(),
            lower_blue, upper_blue, 
            debug=True
        )

        return fill_percentage <= empty_threshold


    def is_not_empty(self, **kwargs):
        """
        Guard condition for unloading -> undocking transition.
        Return True if the cargo hold is empty, False otherwise.
        """
        empty_threshold = kwargs.get("empty_threshold", 0.1)

        fill_percentage = ImageProcessor.compute_fill_percentage(
            self.actions.elements["inventory"].screenshot(),
            self.accent_lower_hsv, self.accent_upper_hsv, 
            debug=True
        )

        return fill_percentage >= empty_threshold


    def draw_graph(self, path="agent_state_machine.png"):
        """
        Draws and saves the state machine graph to a PNG file.
        Requires graphviz and pygraphviz.
        """
        try:
            graph = self.machine.get_graph()
            graph.draw(path, prog="dot")
            logger.info(f"State machine graph saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to draw graph: {e}")
            # Fallback: write DOT file
            try:
                graph = self.machine.get_graph()
                dot_path = str(Path(path).with_suffix('.dot'))
                graph.write(dot_path)
                logger.info(
                    f"State machine DOT file saved to {dot_path}. "
                    f"You can render it manually: dot -Tpng {dot_path} -o {str(Path(path).with_suffix('.png'))}"
                )
            except Exception as write_err:
                logger.error(f"Failed to write DOT file: {write_err}")
