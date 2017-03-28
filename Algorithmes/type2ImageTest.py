import sys

def expected_return_1(path):
    return "0.72,0.80,0.1"

if __name__ == '__main__':
    path = (sys.argv[1])
    print(expected_return_1(path))