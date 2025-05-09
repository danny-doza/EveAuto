import serial
import time
import glob
import pyautogui

from pynput.mouse import Controller as MouseController
from pynput.keyboard import Controller as KeyboardController

mouse = MouseController()
keyboard = KeyboardController()

class ProMicroEmulator:

    def __init__(self, baud=9600, connect_delay=2):
        self.baud = baud
        self.connect_delay = connect_delay
        self.port = self._find_port()
        self.screen_width, self.screen_height = pyautogui.size()
        self.online = False


    def _find_port(self):
        candidates = glob.glob("/dev/tty.usbmodem*")
        if not candidates:
            print("No USB serial device found matching /dev/tty.usbmodem*. Disabling Pro Micro emulator.")
            self.online = False
            return None
        
        print(f"[INFO] Found Pro Micro on {candidates[0]}")
        self.online = True
        return candidates[0]


    def send(self, message: str):
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                time.sleep(self.connect_delay)
                ser.write((message.strip() + "\n").encode("utf-8"))
                print(f"[Sent] {message}")
        except serial.SerialException as e:
            print(f"[ERROR] Could not open serial port {self.port}: {e}")


    def press(self, key: str):
        self.send(f"keyboard_down {key.upper()}")


    def release(self, key: str):
        self.send(f"keyboard_up {key.upper()}")


    def tap(self, key: str):
        self.send(f"keyboard {key.upper()}")


    def click(self, button: str = "LEFT_CLICK"):
        self.send(f"mouse {button.upper()}")


    def _to_hid_coords(self, x: int, y: int) -> tuple[int, int]:
        # Clamp and scale to 0–32767 HID range
        hid_x = min(max(int(x / self.screen_width * 32767), 0), 32767)
        hid_y = min(max(int(y / self.screen_height * 32767), 0), 32767)
        return hid_x, hid_y


    def move_mouse(self, x: int, y: int):
        x_hid, y_hid = self._to_hid_coords(x, y)
        self.send(f"mouse_move {x_hid} {y_hid}")


    def move_and_click(self, x: int, y: int, button: str = "LEFT"):
        button = button.strip().upper()
        if button not in ["LEFT", "RIGHT", "MIDDLE"]:
            print(f"[WARN] Unsupported button '{button}', defaulting to LEFT")
            button = "LEFT"
        x_hid, y_hid = self._to_hid_coords(x, y)
        self.send(f"mouse_click {x_hid} {y_hid} {button}")


    def scroll(self, x: int, y: int, amount: int):
        """
        Scrolls the mouse wheel at the given screen position.

        :param x: Screen X coordinate
        :param y: Screen Y coordinate
        :param amount: Number of scroll steps. Positive = scroll down, negative = scroll up
        """
        x_hid, y_hid = self._to_hid_coords(x, y)
        self.send(f"mouse_scroll {x_hid} {y_hid} {amount}")

emulator = ProMicroEmulator()

# # === Example Usage ===
# emulator = ProMicroEmulator()

# # Tap D
# emulator.tap("d")

# # Hold SHIFT, press D, release SHIFT
# emulator.press("shift")
# emulator.tap("d")
# emulator.release("shift")

# # Click at screen center
# emulator.move_and_click(960, 540) # defaults to left click
# emulator.move_and_click(960, 540, button="LEFT")
# emulator.move_and_click(960, 540, button="MIDDLE")
# emulator.move_and_click(960, 540, button="RIGHT")

# # Scroll down 5 steps at screen center
# emulator.scroll(960, 540, 5)

# # Scroll up 3 steps
# emulator.scroll(960, 540, -3)