from ctypes import string_at
import signal
import os

def sig_handler(signum, frame):
    print ("found segfault")
    return
    
signal.signal(signal.SIGSEGV, sig_handler)

os.kill(os.getpid(), signal.SIGSEGV)

def fault():
    text = string_at(1, 10)
    print("text = {0!r}".format(text))

def test():
    print("test: 1")
    try:
        fault()
    except (MemoryError, err):
        print ("ooops!")
        print (err)

    print("test: 2")
    try:
        fault()
    except (MemoryError, err):
        print ("ooops!")
        print (err)

    print("test: end")

def main():
    test()

if __name__ == "__main__":
    main()