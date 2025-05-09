from abc import ABC, abstractmethod

class UIElement(ABC):
    @abstractmethod
    def click(self):
        pass

    @abstractmethod
    def key_and_click(self):
        pass

    @abstractmethod
    def show_debug_overlay(self, label: str = "", duration: int = 2):
        pass


class UIBox(UIElement):
    @abstractmethod
    def screenshot(self):
        pass
