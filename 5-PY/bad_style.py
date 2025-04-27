def badFunction(param1, param2):
    x = param1 + param2
    y = x * 2
    if x > 10:
        print("x is greater than 10")
    return {"result": x, "doubled": y}


MY_CONSTANT = 5


class myClass:
    def __init__(self, name):
        self.name = name

    def SayHello(self):
        print("Hello, " + self.name)


result = badFunction(MY_CONSTANT, 10)
