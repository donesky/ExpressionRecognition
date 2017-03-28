import sys

def expected_return_1(path):
    return "(-2.0;3.0),(4;-2.1),(-1;2.1)"

if __name__ == '__main__':
    path = (sys.argv[1])
    print(expected_return_1(path))