import time


def millis():
    return int(round(time.time() * 1000))


FONT = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
}


class task:
    depth = 0
    previouse_log_time = millis()

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        message = "\r" + task.depth * "  " + f" \u27CD  {self.name}"
        print(message)
        task.depth += 1
        self.start_time = millis()
        task.previouse_log_time = self.start_time
        return self

    def __exit__(self, type, value, traceback):
        elapsed_time = millis() - self.start_time
        if type is not None:
            message = (
                "\r"
                + task.depth * "  "
                + f"|  {FONT['RED']}Task is finished with an error: {value}{FONT['RESET']}"
            )
            print(message)
        task.depth -= 1
        message = (
            "\r"
            + task.depth * "  "
            + f" \u27CB  {self.name} {FONT['BOLD']}[+{elapsed_time}ms]{FONT['RESET']}"
        )
        print(message)
        return False

    def log(message):
        elapsed_time = millis() - task.previouse_log_time
        message = (
            "\r"
            + task.depth * "  "
            + f"|  {message} {FONT['BOLD']}[+{elapsed_time}ms]{FONT['RESET']}"
        )
        print(message)
        task.previouse_log_time = millis()


if __name__ == "__main__":
    with task("Task 1"):
        with task("Task 1.1"):
            time.sleep(0.1)
            task.log("Log 1")
            time.sleep(0.2)
            with task("Task 1.1.1"):
                pass
            with task("Task 1.1.2"):
                time.sleep(0.3)
                task.log("Log 2")
                pass
        with task("Task 1.2"):
            pass
    with task("Task 2"):
        pass
