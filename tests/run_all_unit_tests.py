import os

for dir in os.listdir('./unit'):
    os.system(f'cd {os.path.join("./unit", dir)}')
    os.system('python3 -m unittest discover -p "*_test.py"')
    os.system('python3 -m unittest discover -p "test_*.py"')
    os.system('cd ..')
