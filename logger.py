class Logger():
    def __init__(self, filename, option):
        self.fileptr = open(filename, option)
        if option == "r":
            self.lines = self.fileptr.readlines()
        else:
            self.lines = []

    def put(self, string):
        self.fileptr.write(string + "\n")
        self.fileptr.flush()

    def close(self):
        self.fileptr.close()

    def findlast(self, identifier, default=0.):
        for line in self.lines[::-1]:
            if line.lower().startswith(identifier):
                string = line.strip().split("\t")[1]
                if string.replace(".", "").isdigit():
                    return float(string)
                elif string.lower() == "true":
                    return True
                elif string.lower() == "false":
                    return False
                else:
                    return string
        return default

    def contains(self, string):
        for line in self.lines[::-1]:
            if string.lower() in line.lower():
                return True
        return False

    def findlast_log_before(self, before_str):
        loglines = []
        in_line = False
        for line in self.lines[::-1]:
            if line.startswith(before_str):
                in_line = True
            elif in_line:
                loglines.append(line)
            if line.strip() == "" and in_line:
                return "".join(loglines[::-1])
        return "".join(loglines[::-1])
