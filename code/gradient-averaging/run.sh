#! /bin/zsh

python main.py --average-gradients=1 --batch-size=100 --iterations=1000

python main.py --average-gradients=5 --batch-size=20 --iterations=5000

python main.py --average-gradients=10 --batch-size=10 --iterations=10000

python main.py --average-gradients=20 --batch-size=5 --iterations=20000

