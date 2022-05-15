import numpy as np
import datetime
from orders import Order

class Executed_Trade():
    def __init__(self, volume:int, stock:str, price:float, time:datetime, parent:Order) -> None:
        '''
        executed trade to track history and state of the parent order
        volume :
        stock :
        time :
        price :
        '''
        self.volume = volume
        self.stock = stock
        self.price = price
        self.time = time
        self.parent = parent