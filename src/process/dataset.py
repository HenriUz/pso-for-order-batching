import os

class Problem():
    """
    Reads and processes the data contained in a dataset file.

    Attributes:
        o (int): Total number of orders in the dataset.
        i (int): Total number of items.
        a (int): Total number of available aisles.
        orders (list[dict[int, int]]): List containing dictionaries that represent the orders. Each dictionary lists an item along with its quantity.
        sorted_orders (list[tuple[int, int]]): List sorted in descending order by the number of items in each order. Each element is a tuple in the format (index, number of items).
        aisles (list[dict[int, int]]): List containing dictionaries that represent the aisles. Each dictionary lists an item along with its quantity in the aisle.
        lb (int): Lower bound.
        ub (int): Upper bound.
        result (dict[str, any]): Dictionary that stores the final results, including the lists of selected orders and aisles, the objective function value, and the method's execution time.
    """
    
    def __init__(self, instance_path: str) -> None:
        """
        Args:
            instance_path (str): Path to the instance.
        """

        try:
            with open(instance_path, "rb") as data:
                lines = data.readlines()
        
                first_line = lines[0].strip().split()
                self.o, self.i, self.a = int(first_line[0]), int(first_line[1]), int(first_line[2])

                self.orders = []
                for i in range(self.o):
                    order_line = lines[i + 1].strip().split()
                    order = {int(order_line[1 + k * 2]): int(order_line[2 + k * 2]) for k in range(int(order_line[0]))}
                    self.orders.append(order)

                self.sorted_orders = [(x, sum(d.values())) for x, d in enumerate(self.orders)]
                self.sorted_orders.sort(key = lambda i: i[1], reverse = True)

                self.aisles = []
                for i in range(self.a):
                    aisle_line = lines[i + 1 + self.o].strip().split()
                    aisle = {int(aisle_line[1 + k * 2]): int(aisle_line[2 + k * 2]) for k in range(int(aisle_line[0]))}
                    self.aisles.append(aisle)
                
                last_line = lines[self.o + self.a + 1].strip().split()
                self.lb, self.ub = int(last_line[0]), int(last_line[1])

                self.result = {"orders": [], "aisles": [], "objective": 0, "time": 0}
        
        except FileNotFoundError:
            print(f"Error: file '{instance_path}' not found.")
            exit(1)
        except OSError as e:
            print(f"Error opening the file: {e}")
            exit(1)

    def add_orders(
        self,
        aisles_items: list[int]
    ) -> int:
        """
        Greedily selects the orders with the largest number of items that do not violate capacity and supply constraints.

        To ensure that the aisles always have the best combination of orders, this function assumes that no orders have been selected at the time it is called. And for optimization purposes, this function returns only the quantity of items in the selected orders.

        Args:
            aisles_items (list[int]): Available items.
        
        Returns:
            number_items (int): Number of items in the orders.
        """

        number_items = 0

        available_items = aisles_items.copy()
        for o in self.sorted_orders: 
            if number_items + o[1] <= self.ub:
                valid = True

                orders_items = self.orders[o[0]]        
                for item in orders_items:
                    if orders_items[item] > available_items[item]:
                        valid = False
                        break

                if valid:
                    number_items += o[1]
                    for item in orders_items:
                        available_items[item] -= orders_items[item]

        return number_items
    
    def view_orders(
        self,
        aisles_items: list[int]
    ) -> list[int]:
        """
        Greedily selects the orders with the largest number of items that do not violate capacity and supply constraints.

        To ensure that the aisles always have the best combination of orders, this function assumes that no orders have been selected at the time it is called. Unlike `add_orders`, this function returns a list containing the IDs of the selected orders.

        Args:
            aisles_items (list[int]): Available items.
        
        Returns:
            orders (tuple[int, list[int]]): Order IDs.
        """
        
        orders       = []
        number_items = 0

        available_items = aisles_items.copy()
        for o in self.sorted_orders:
            if number_items + o[1] <= self.ub:
                valid = True

                orders_items = self.orders[o[0]]
                for item in orders_items:
                    if orders_items[item] > available_items[item]:
                        valid = False
                        break
                
                if valid:
                    orders.append(o[0])
                    number_items += o[1]
                    for item in orders_items:
                        available_items[item] -= orders_items[item]
        
        return orders

    def objective_function(
        self,
        number_items: int,
        number_aisles: int
    ) -> float:
        """
        Calculates the value of the objective function and returns it. If the lower bound constraint is violated, or if `number_aisles` is 0, the return value will be 0. 

        Since the methods work directly with the aisles, which have no constraints, and use a greedy function that selects orders while respecting the upper bound and supply constraints (`add_orders`), this function does not check the previous constraints to avoid redundancy.

        Args:
            number_items (int): Number of items in the orders.
            number_aisles (int): Number of selected aisles.
        
        Returns:
            Objective (float): Value of the objective function.
        """
        
        if number_items < self.lb or number_aisles == 0:
            return 0.0

        return number_items / number_aisles

    def save_solution(
        self,
        result_path: str
    ) -> None:
        """
        Creates or overwrites a file in the specified path. This file contains the orders and aisles in the solution in the following format:
        - First line: an integer `o` representing the number of orders in the solution;
        - Next `o` lines: one integer per line, representing the order indices;
        - Next line: an integer `a` representing the number of aisles in the solution;
        - Next `a` lines: one integer per line, representing the aisles indices. 

        Args:
            result_path (str): Path to the result file.
        """

        try:
            with open(result_path, "+w") as file:
                file.write(str(len(self.result["orders"])) + "\n")
                for o in self.result["orders"]:
                    file.write(str(o) + "\n")
                file.write(str(len(self.result["aisles"])) + "\n")
                for a in self.result["aisles"]:
                    file.write(str(a) + "\n")
                file.close()
        except OSError as e:
            print(f"Error creating the file: {e}")
            exit(1)